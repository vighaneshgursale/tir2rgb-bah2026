"""Download paired Landsat-9 TIR/RGB tiles from Microsoft Planetary Computer.

Drop-in replacement for download_gee.py that needs NO account or authentication:
Planetary Computer serves USGS Landsat Collection-2 Level-2 as public COGs with
anonymous, auto-signed URLs.

Output format is identical (.npz per tile):
  tir_200m : ST_B10 Kelvin, 256x256 @200 m/px
  tir_100m : ST_B10 Kelvin, 512x512 @100 m/px
  rgb_100m : SR_B4/B3/B2 reflectance [0,1], 512x512x3 @100 m/px

All three are windowed reads of the SAME scene over the SAME 51.2 km bounding
box on the same grid, so they are co-registered by construction.

Usage:
    python -m src.data.download_mpc --out data/raw --grid 3
"""

import argparse
import itertools
import math
import os

import numpy as np

from src.data.download_gee import CITIES, TILE_KM

STAC_URL = 'https://planetarycomputer.microsoft.com/api/stac/v1'

# Landsat C2L2 scale factors (raw DN -> physical units)
ST_SCALE, ST_OFFSET = 0.00341802, 149.0
SR_SCALE, SR_OFFSET = 0.0000275, -0.2


def lonlat_box(lon, lat, km):
    """Approximate km x km box centered on (lon, lat), in degrees."""
    dlat = km / 111.32 / 2
    dlon = km / (111.32 * math.cos(math.radians(lat))) / 2
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def read_window(href, bbox4326, out_size, resampling):
    """Windowed, decimated read of one COG band over a lon/lat bbox."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds
    with rasterio.open(href) as src:
        bounds = transform_bounds('EPSG:4326', src.crs, *bbox4326)
        win = from_bounds(*bounds, src.transform)
        return src.read(1, window=win, out_shape=(out_size, out_size),
                        resampling=Resampling[resampling], boundless=True,
                        fill_value=0).astype(np.float32)


def download_region(catalog, name, lon, lat, out_dir, max_cloud):
    import planetary_computer as pc
    bbox = lonlat_box(lon, lat, TILE_KM)
    items = list(catalog.search(
        collections=['landsat-c2-l2'], bbox=bbox,
        query={'platform': {'eq': 'landsat-9'},
               'eo:cloud_cover': {'lt': max_cloud}},
        max_items=20).items())
    if not items:
        print(f'  !! no cloud-free landsat-9 scene for {name}')
        return False
    items.sort(key=lambda i: i.properties.get('eo:cloud_cover', 100))

    for item in items[:5]:  # try up to 5 least-cloudy scenes
        signed = pc.sign(item)
        try:
            tir_dn_100 = read_window(signed.assets['lwir11'].href, bbox, 512, 'bilinear')
        except Exception as e:
            print(f'  .. {item.id}: read failed ({e}), trying next')
            continue
        # scene must actually cover the box (0 = nodata DN)
        if (tir_dn_100 == 0).mean() > 0.05:
            continue

        tir_dn_200 = read_window(signed.assets['lwir11'].href, bbox, 256, 'average')
        rgb_dn = np.stack([read_window(signed.assets[b].href, bbox, 512, 'average')
                           for b in ('red', 'green', 'blue')], axis=-1)
        if (rgb_dn == 0).all(axis=-1).mean() > 0.05:
            continue

        tir_100 = tir_dn_100 * ST_SCALE + ST_OFFSET
        tir_200 = tir_dn_200 * ST_SCALE + ST_OFFSET
        rgb_100 = np.clip(rgb_dn * SR_SCALE + SR_OFFSET, 0.0, 1.0).astype(np.float32)

        np.savez_compressed(os.path.join(out_dir, f'{name}.npz'),
                            tir_100m=tir_100.astype(np.float32),
                            tir_200m=tir_200.astype(np.float32),
                            rgb_100m=rgb_100)
        print(f'  ok {name} <- {item.id} '
              f'(cloud {item.properties.get("eo:cloud_cover", -1):.0f}%)')
        return True

    print(f'  !! {name}: no scene fully covers the box')
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data/raw')
    ap.add_argument('--grid', type=int, default=3)
    ap.add_argument('--max-cloud', type=float, default=15)
    args = ap.parse_args()

    from pystac_client import Client
    catalog = Client.open(STAC_URL)
    os.makedirs(args.out, exist_ok=True)

    dlon, dlat = 0.50, 0.46
    offsets = list(itertools.product(range(args.grid), repeat=2))
    center = (args.grid - 1) / 2

    n_ok = 0
    for city, (lon, lat) in CITIES.items():
        for i, j in offsets:
            name = f'{city}_{i}{j}'
            if os.path.exists(os.path.join(args.out, f'{name}.npz')):
                print(f'  -- {name} exists, skipping')
                n_ok += 1
                continue
            print(f'{name} ...')
            try:
                n_ok += int(download_region(catalog, name,
                                            lon + (i - center) * dlon,
                                            lat + (j - center) * dlat,
                                            args.out, args.max_cloud))
            except Exception as e:
                print(f'  !! {name} failed: {e}')
    print(f'\nDone: {n_ok} tiles in {args.out}')


if __name__ == '__main__':
    main()
