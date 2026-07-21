"""Turn raw .npz tiles into training sets for BOTH pipeline stages.

Stage 1 (super-resolution), .npy pairs:
    data/sr/{train,val}/<tile>_lq.npy   tir_200m 256x256  float32 [0,1]
    data/sr/{train,val}/<tile>_gt.npy   tir_100m 512x512  float32 [0,1]

Stage 2 (colorization), pix2pix "AB" side-by-side PNGs (junyanz repo format):
    data/tir2rgb/{train,val}/<tile>_<k>.png   512x256 (A=TIR gray | B=RGB)
    Four 256x256 patches per 512x512 tile, cut from tir_100m / rgb_100m.

Correctness notes (fixes to the earlier notebook attempt):
  - Colorization pairs use 100m TIR (not 200m) as input, per the official workflow.
  - Normalization is the FIXED Kelvin window from normalization.py, never per-patch.

Usage:
    python -m src.data.prepare_patches --raw data/raw --out data --val-frac 0.1
"""

import argparse
import os
import random

import numpy as np
from PIL import Image

from src.data.normalization import (kelvin_to_unit, reflectance_to_uint8,
                                    tir_unit_to_uint8)


def make_ab_image(tir_u8_256, rgb_u8_256):
    """A|B side-by-side pair image expected by pytorch-CycleGAN-and-pix2pix."""
    ab = np.zeros((256, 512, 3), dtype=np.uint8)
    ab[:, :256] = tir_u8_256[..., None]
    ab[:, 256:] = rgb_u8_256
    return Image.fromarray(ab)


def process_tile(path, name, split, out):
    d = np.load(path)
    tir100_unit = kelvin_to_unit(d['tir_100m'])          # 512x512 [0,1]
    tir200_unit = kelvin_to_unit(d['tir_200m'])          # 256x256 [0,1]
    rgb_u8 = reflectance_to_uint8(d['rgb_100m'])         # 512x512x3

    # -- SR pair (whole tile) --
    sr_dir = os.path.join(out, 'sr', split)
    np.save(os.path.join(sr_dir, f'{name}_lq.npy'), tir200_unit.astype(np.float32))
    np.save(os.path.join(sr_dir, f'{name}_gt.npy'), tir100_unit.astype(np.float32))

    # -- 4 colorization patches --
    tir_u8 = tir_unit_to_uint8(tir100_unit)
    cz_dir = os.path.join(out, 'tir2rgb', split)
    for k, (y, x) in enumerate([(0, 0), (0, 256), (256, 0), (256, 256)]):
        ab = make_ab_image(tir_u8[y:y + 256, x:x + 256],
                           rgb_u8[y:y + 256, x:x + 256])
        ab.save(os.path.join(cz_dir, f'{name}_{k}.png'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', default='data/raw')
    ap.add_argument('--out', default='data')
    ap.add_argument('--val-frac', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    tiles = sorted(f for f in os.listdir(args.raw) if f.endswith('.npz'))
    if not tiles:
        raise SystemExit(f'no .npz tiles found in {args.raw}')

    # split by CITY, not by tile, so val tiles are from unseen geography
    cities = sorted({t.split('_')[0] for t in tiles})
    random.Random(args.seed).shuffle(cities)
    n_val = max(1, int(len(cities) * args.val_frac))
    val_cities = set(cities[:n_val])
    print(f'val cities: {sorted(val_cities)}')

    for sub in ('sr/train', 'sr/val', 'tir2rgb/train', 'tir2rgb/val'):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    counts = {'train': 0, 'val': 0}
    for t in tiles:
        name = t[:-4]
        split = 'val' if name.split('_')[0] in val_cities else 'train'
        process_tile(os.path.join(args.raw, t), name, split, args.out)
        counts[split] += 1

    print(f"tiles: {counts['train']} train / {counts['val']} val")
    print(f"colorization pairs: {counts['train'] * 4} train / {counts['val'] * 4} val")


if __name__ == '__main__':
    main()
