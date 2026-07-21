"""End-to-end inference: 200 m TIR tile -> SR TIR @100 m -> colorized RGB @100 m.

Loads both trained models once, then runs tiles through the two stages with
per-stage timing. The pix2pix U-Net is fully convolutional with 8 downsamplings,
so it accepts the 512x512 SR output directly — no lossy resize back to 256.

Input formats: .npy / .npz (Kelvin or already [0,1]) or single-band GeoTIFF
(raw ST_B10 DNs or Kelvin — detected from value range).

Usage:
    python -m src.infer.pipeline --tir tile.npy \
        --sr-ckpt checkpoints/sr/sr_best.pth \
        --p2p-ckpt checkpoints/tir2rgb/latest_net_G.pth \
        --out results/tile01
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

from src.data.normalization import dn_to_kelvin, kelvin_to_unit, tir_unit_to_uint8

P2P_REPO = 'third_party/pytorch-CycleGAN-and-pix2pix'


def load_tir(path):
    """Any supported input -> float32 [0,1] TIR array (fixed Kelvin window)."""
    if path.endswith('.npz'):
        arr = np.load(path)['tir_200m']
    elif path.endswith('.npy'):
        arr = np.load(path)
    else:  # GeoTIFF
        import rasterio
        with rasterio.open(path) as src:
            arr = src.read(1)
    arr = arr.astype(np.float32)
    if arr.max() > 1000:      # raw ST_B10 digital numbers
        arr = kelvin_to_unit(dn_to_kelvin(arr))
    elif arr.max() > 1.5:     # Kelvin
        arr = kelvin_to_unit(arr)
    return np.clip(arr, 0.0, 1.0)


def load_sr_model(ckpt_path, device):
    from basicsr.archs.rrdbnet_arch import RRDBNet
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=2)
    state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    state = state.get('model', state.get('params_ema', state.get('params', state)))
    model.load_state_dict(state)
    return model.to(device).eval()


def load_p2p_generator(ckpt_path, device, repo=P2P_REPO):
    from src.train.train_pix2pix import ensure_repo
    ensure_repo(repo)  # clone on demand - inference sessions may not have trained here
    sys.path.insert(0, os.path.abspath(repo))
    import inspect
    from models.networks import define_G
    # upstream repo periodically changes this signature (e.g. dropped gpu_ids);
    # pass only the kwargs the installed version accepts
    kwargs = dict(input_nc=3, output_nc=3, ngf=64, netG='unet_256', norm='batch',
                  use_dropout=True, init_type='normal', init_gain=0.02, gpu_ids=[])
    accepted = inspect.signature(define_G).parameters
    netG = define_G(**{k: v for k, v in kwargs.items() if k in accepted})
    state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    netG.load_state_dict(state)
    return netG.to(device).eval()


class Tir2RgbPipeline:
    def __init__(self, sr_ckpt, p2p_ckpt, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.sr = load_sr_model(sr_ckpt, self.device)
        self.netG = load_p2p_generator(p2p_ckpt, self.device)

    @torch.no_grad()
    def __call__(self, tir_unit):
        """tir_unit: HxW float32 in [0,1] at 200 m/px.
        Returns (sr_tir [0,1] 2Hx2W, rgb uint8 2Hx2Wx3, timings dict)."""
        t0 = time.time()
        x = torch.from_numpy(tir_unit).float().to(self.device)
        x3 = x.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)          # 1x3xHxW

        sr = self.sr(x3).clamp(0, 1)                                  # 1x3x2Hx2W
        t_sr = time.time() - t0

        rgb = self.netG(sr * 2 - 1)                                   # [-1,1] in/out
        rgb_u8 = ((rgb.clamp(-1, 1) + 1) / 2 * 255).squeeze(0)\
            .permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        t_total = time.time() - t0

        sr_np = sr.squeeze(0).mean(dim=0).cpu().numpy()
        return sr_np, rgb_u8, {'sr_s': t_sr, 'colorize_s': t_total - t_sr,
                               'total_s': t_total}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tir', required=True)
    ap.add_argument('--sr-ckpt', required=True)
    ap.add_argument('--p2p-ckpt', required=True)
    ap.add_argument('--out', default='results/out')
    args = ap.parse_args()

    from PIL import Image
    os.makedirs(args.out, exist_ok=True)

    pipe = Tir2RgbPipeline(args.sr_ckpt, args.p2p_ckpt)
    tir = load_tir(args.tir)
    sr, rgb, t = pipe(tir)

    Image.fromarray(tir_unit_to_uint8(tir)).save(f'{args.out}/input_tir_200m.png')
    Image.fromarray(tir_unit_to_uint8(sr)).save(f'{args.out}/sr_tir_100m.png')
    Image.fromarray(rgb).save(f'{args.out}/colorized_rgb_100m.png')
    np.save(f'{args.out}/sr_tir_100m.npy', sr)

    print(f"input {tir.shape} -> SR {sr.shape} -> RGB {rgb.shape}")
    print(f"timings: SR {t['sr_s']:.2f}s | colorize {t['colorize_s']:.2f}s "
          f"| total {t['total_s']:.2f}s")
    print(f'outputs written to {args.out}/')


if __name__ == '__main__':
    main()
