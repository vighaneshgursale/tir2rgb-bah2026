"""Evaluation suite: PSNR / SSIM per stage, FID for colorization realism,
water-consistency hallucination check, per-tile inference timing.

Runs the full pipeline over every val tile and writes a metrics table +
before/after gallery.

Usage:
    python -m src.eval.metrics --raw data/raw --val-list data/val_cities.txt \
        --sr-ckpt checkpoints/sr/sr_best.pth \
        --p2p-ckpt checkpoints/tir2rgb/latest_net_G.pth --out results/eval
    (--val-list: file with one city name per line; defaults to all tiles)
"""

import argparse
import glob
import os

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from src.data.normalization import kelvin_to_unit, reflectance_to_uint8, tir_unit_to_uint8
from src.infer.pipeline import Tir2RgbPipeline


def water_consistency(rgb_pred_u8, rgb_gt_u8, thresh=30):
    """Compare 'blueish water' pixel fraction between prediction and ground
    truth. Large disagreement = likely water/land hallucination."""
    def water_frac(img):
        img = img.astype(int)
        return ((img[:, :, 2] - img[:, :, 0]) > thresh).mean()
    wp, wg = water_frac(rgb_pred_u8), water_frac(rgb_gt_u8)
    return wp, wg, abs(wp - wg) < 0.10   # flag if >10 p.p. disagreement


def gallery_row(imgs, path):
    """Save side-by-side comparison strip (input | SR | pred RGB | GT RGB)."""
    h = max(im.height for im in imgs)
    strip = Image.new('RGB', (sum(im.width for im in imgs), h), 'white')
    x = 0
    for im in imgs:
        strip.paste(im, (x, 0))
        x += im.width
    strip.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', default='data/raw')
    ap.add_argument('--val-list', default=None)
    ap.add_argument('--sr-ckpt', required=True)
    ap.add_argument('--p2p-ckpt', required=True)
    ap.add_argument('--out', default='results/eval')
    ap.add_argument('--fid', action='store_true', help='also compute FID (needs pytorch-fid)')
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, 'gallery'), exist_ok=True)
    fid_dirs = {k: os.path.join(args.out, f'fid_{k}') for k in ('real', 'fake')}
    for d in fid_dirs.values():
        os.makedirs(d, exist_ok=True)

    tiles = sorted(glob.glob(os.path.join(args.raw, '*.npz')))
    if args.val_list:
        cities = {c.strip() for c in open(args.val_list) if c.strip()}
        tiles = [t for t in tiles
                 if os.path.basename(t).split('_')[0] in cities]
    if not tiles:
        raise SystemExit('no evaluation tiles found')

    pipe = Tir2RgbPipeline(args.sr_ckpt, args.p2p_ckpt)
    rows, times = [], []
    for path in tiles:
        name = os.path.basename(path)[:-4]
        d = np.load(path)
        tir200 = kelvin_to_unit(d['tir_200m'])
        tir100_gt = kelvin_to_unit(d['tir_100m'])
        rgb_gt = reflectance_to_uint8(d['rgb_100m'])

        sr, rgb_pred, t = pipe(tir200)
        times.append(t['total_s'])

        psnr_sr = peak_signal_noise_ratio(tir100_gt, sr, data_range=1.0)
        ssim_sr = structural_similarity(tir100_gt, sr, data_range=1.0)
        psnr_cz = peak_signal_noise_ratio(rgb_gt, rgb_pred, data_range=255)
        ssim_cz = structural_similarity(rgb_gt, rgb_pred, data_range=255, channel_axis=2)
        wp, wg, water_ok = water_consistency(rgb_pred, rgb_gt)

        rows.append((name, psnr_sr, ssim_sr, psnr_cz, ssim_cz, wp, wg, water_ok,
                     t['total_s']))
        gallery_row([Image.fromarray(tir_unit_to_uint8(tir200)).resize((512, 512), Image.NEAREST),
                     Image.fromarray(tir_unit_to_uint8(sr)),
                     Image.fromarray(rgb_pred),
                     Image.fromarray(rgb_gt)],
                    os.path.join(args.out, 'gallery', f'{name}.png'))
        Image.fromarray(rgb_gt).save(os.path.join(fid_dirs['real'], f'{name}.png'))
        Image.fromarray(rgb_pred).save(os.path.join(fid_dirs['fake'], f'{name}.png'))
        print(f'{name}: SR {psnr_sr:.2f}dB/{ssim_sr:.3f}  '
              f'CZ {psnr_cz:.2f}dB/{ssim_cz:.3f}  water_ok={water_ok}')

    # ---- summary table ----
    arr = np.array([[r[1], r[2], r[3], r[4], r[8]] for r in rows])
    n_flag = sum(1 for r in rows if not r[7])
    lines = [
        '| metric | mean |', '|---|---|',
        f'| SR PSNR (dB) | {arr[:, 0].mean():.2f} |',
        f'| SR SSIM | {arr[:, 1].mean():.3f} |',
        f'| Colorization PSNR (dB) | {arr[:, 2].mean():.2f} |',
        f'| Colorization SSIM | {arr[:, 3].mean():.3f} |',
        f'| Inference time / tile (s) | {arr[:, 4].mean():.2f} |',
        f'| Water-consistency flags | {n_flag}/{len(rows)} |',
    ]
    if args.fid:
        import subprocess
        import sys
        r = subprocess.run([sys.executable, '-m', 'pytorch_fid',
                            fid_dirs['real'], fid_dirs['fake']],
                           capture_output=True, text=True)
        lines.append(f"| FID | {r.stdout.strip() or r.stderr.strip()} |")

    report = '\n'.join(lines)
    with open(os.path.join(args.out, 'metrics.md'), 'w') as f:
        f.write(report + '\n')
    print('\n' + report)
    print(f"\ngallery + metrics.md written to {args.out}/")


if __name__ == '__main__':
    main()
