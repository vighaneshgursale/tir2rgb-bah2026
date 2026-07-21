"""Download paired Landsat-9 TIR/RGB training tiles from Google Earth Engine.

For each (city, grid-offset) region this fetches three co-registered arrays:
  tir_200m : ST_B10 at 200 m/px, 256x256   (model INPUT)
  tir_100m : ST_B10 at 100 m/px, 512x512   (SR ground truth)
  rgb_100m : SR_B4/B3/B2 at 100 m/px, 512x512x3 (colorization ground truth)

All three cover the SAME 51.2 km square, so they are perfectly co-registered by
construction (GEE does the reprojection). Values are stored in physical units
(Kelvin / reflectance) as .npz — normalization happens later, identically for
every tile (see normalization.py).

Usage (Colab, after `ee.Authenticate()`):
    python -m src.data.download_gee --project YOUR_GEE_PROJECT --out data/raw --grid 3
"""

import argparse
import io
import itertools
import os

import numpy as np
import requests

CITIES = {
    # coastal / water-heavy
    'Vizag': (83.2185, 17.6868), 'Kochi': (76.2673, 9.9312),
    'Goa': (73.8278, 15.4909), 'PortBlair': (92.7265, 11.6234),
    'Alleppey': (76.3388, 9.4981), 'Kolkata': (88.3639, 22.5726),
    'Chennai': (80.2707, 13.0827), 'Mumbai': (72.8777, 19.0760),
    'Mangalore': (74.8560, 12.9141), 'Puri': (85.8312, 19.8135),
    # arid
    'Jaisalmer': (70.9083, 26.9157), 'Bikaner': (73.3119, 28.0229),
    'Jodhpur': (73.0243, 26.2389), 'Kutch': (69.8597, 23.7337),
    # mountainous
    'Shimla': (77.1734, 31.1048), 'Darjeeling': (88.2663, 27.0410),
    'Manali': (77.1892, 32.2432), 'Dehradun': (78.0322, 30.3165),
    # plains / urban / agricultural
    'Nagpur': (79.0882, 21.1458), 'Bhopal': (77.4126, 23.2599),
    'Ludhiana': (75.8573, 30.9010), 'Amritsar': (74.8723, 31.6340),
    'Varanasi': (82.9739, 25.3176), 'Patna': (85.1376, 25.5941),
    'Hyderabad': (78.4867, 17.3850), 'Pune': (73.8567, 18.5204),
    'Delhi': (77.1025, 28.7041), 'Ahmedabad': (72.5714, 23.0225),
    'Raipur': (81.6296, 21.2514), 'Vijayawada': (80.6480, 16.5062),
}

TILE_KM = 51.2          # 512 px * 100 m
HALF_WIDTH_M = TILE_KM * 1000 / 2


def get_scene(lon, lat, start, end, max_cloud):
    import ee
    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(HALF_WIDTH_M).bounds()
    col = (ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
           .filterBounds(region)
           .filterDate(start, end)
           .filter(ee.Filter.lt('CLOUD_COVER', max_cloud)))
    if col.size().getInfo() == 0:
        return None, None
    img = col.median()
    thermal = img.select('ST_B10').multiply(0.00341802).add(149.0)          # Kelvin
    optical = (img.select(['SR_B4', 'SR_B3', 'SR_B2'])
               .multiply(0.0000275).add(-0.2))                              # reflectance
    return thermal, optical, region


def fetch(img, region, bands, dimensions):
    url = img.select(bands).getDownloadURL({
        'region': region,
        'dimensions': dimensions,
        'format': 'NPY',
    })
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    return np.load(io.BytesIO(r.content))


def download_region(name, lon, lat, out_dir, start, end, max_cloud):
    res = get_scene(lon, lat, start, end, max_cloud)
    if res[0] is None:
        print(f'  !! no cloud-free scene for {name}')
        return False
    thermal, optical, region = res

    tir_100 = fetch(thermal, region, ['ST_B10'], '512x512')['ST_B10'].astype(np.float32)
    tir_200 = fetch(thermal, region, ['ST_B10'], '256x256')['ST_B10'].astype(np.float32)
    opt = fetch(optical, region, ['SR_B4', 'SR_B3', 'SR_B2'], '512x512')
    rgb_100 = np.stack([opt['SR_B4'], opt['SR_B3'], opt['SR_B2']], axis=-1).astype(np.float32)
    rgb_100 = np.clip(rgb_100, 0.0, 1.0)

    # sanity: reject tiles that are mostly nodata (fill temp ~ 149 K)
    if (tir_100 < 200).mean() > 0.05:
        print(f'  !! {name}: >5% nodata thermal pixels, skipped')
        return False

    np.savez_compressed(os.path.join(out_dir, f'{name}.npz'),
                        tir_100m=tir_100, tir_200m=tir_200, rgb_100m=rgb_100)
    print(f'  ok {name}: tir200 {tir_200.shape} tir100 {tir_100.shape} rgb {rgb_100.shape}')
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', required=True, help='GEE cloud project id')
    ap.add_argument('--out', default='data/raw')
    ap.add_argument('--grid', type=int, default=3,
                    help='NxN grid of adjacent tiles per city (3 -> 9 tiles/city)')
    ap.add_argument('--start', default='2024-01-01')
    ap.add_argument('--end', default='2026-01-01')
    ap.add_argument('--max-cloud', type=float, default=15)
    args = ap.parse_args()

    import ee
    ee.Initialize(project=args.project)
    os.makedirs(args.out, exist_ok=True)

    # degrees per tile (~51.2 km); coarse but fine for offsetting tile centers
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
                ok = download_region(name, lon + (i - center) * dlon,
                                     lat + (j - center) * dlat,
                                     args.out, args.start, args.end, args.max_cloud)
                n_ok += int(ok)
            except Exception as e:  # GEE quota / transient network errors
                print(f'  !! {name} failed: {e}')
    print(f'\nDone: {n_ok} tiles in {args.out}')


if __name__ == '__main__':
    main()
