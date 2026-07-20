"""Fine-tune a x2 RRDBNet (Real-ESRGAN architecture) on TIR 200m -> 100m pairs.

Deliberately a plain PyTorch loop (not the BasicSR config system) so it runs
unmodified on Colab/Kaggle free tiers. Loss is L1-dominant to minimize texture
hallucination — judges explicitly penalize invented detail.

Starts from photo-pretrained RealESRGAN_x2plus weights (download once):
  wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth

Usage:
    python -m src.train.train_sr --data data/sr --pretrained weights/RealESRGAN_x2plus.pth \
        --out checkpoints/sr --iters 20000
Resume:
    python -m src.train.train_sr --data data/sr --out checkpoints/sr --resume \
        checkpoints/sr/sr_latest.pth --iters 40000
"""

import argparse
import glob
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from basicsr.archs.rrdbnet_arch import RRDBNet


class TirSrDataset(Dataset):
    def __init__(self, root, split='train', crop=None):
        self.lq_paths = sorted(glob.glob(os.path.join(root, split, '*_lq.npy')))
        self.crop = crop  # LQ-space crop size (GT crop = 2x)
        if not self.lq_paths:
            raise FileNotFoundError(f'no *_lq.npy under {root}/{split}')

    def __len__(self):
        return len(self.lq_paths)

    def __getitem__(self, i):
        lq = np.load(self.lq_paths[i])                       # 256x256 [0,1]
        gt = np.load(self.lq_paths[i].replace('_lq', '_gt')) # 512x512 [0,1]
        if self.crop:
            c = self.crop
            y = np.random.randint(0, lq.shape[0] - c + 1)
            x = np.random.randint(0, lq.shape[1] - c + 1)
            lq = lq[y:y + c, x:x + c]
            gt = gt[2 * y:2 * (y + c), 2 * x:2 * (x + c)]
            if np.random.rand() < 0.5:                       # h-flip aug
                lq, gt = lq[:, ::-1], gt[:, ::-1]
        # single-channel TIR replicated to 3ch (matches pretrained weights)
        to3 = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float().unsqueeze(0).repeat(3, 1, 1)
        return to3(lq), to3(gt)


def build_model(pretrained=None, device='cuda'):
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=2)
    if pretrained:
        state = torch.load(pretrained, map_location='cpu', weights_only=True)
        model.load_state_dict(state.get('params_ema', state.get('params', state)))
        print(f'loaded pretrained weights: {pretrained}')
    return model.to(device)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    psnrs = []
    for lq, gt in loader:
        pred = model(lq.to(device)).clamp(0, 1)
        mse = F.mse_loss(pred, gt.to(device)).item()
        psnrs.append(10 * np.log10(1.0 / max(mse, 1e-10)))
    model.train()
    return float(np.mean(psnrs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/sr')
    ap.add_argument('--out', default='checkpoints/sr')
    ap.add_argument('--pretrained', default=None)
    ap.add_argument('--resume', default=None)
    ap.add_argument('--iters', type=int, default=20000)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--crop', type=int, default=128, help='LQ crop size')
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--val-every', type=int, default=1000)
    ap.add_argument('--save-every', type=int, default=1000)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out, exist_ok=True)

    model = build_model(args.pretrained, device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    scaler = torch.amp.GradScaler(enabled=(device == 'cuda'))
    start_iter = 0
    if args.resume:
        ck = torch.load(args.resume, map_location='cpu', weights_only=True)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        start_iter = ck['iter']
        print(f'resumed from {args.resume} @ iter {start_iter}')

    train_loader = DataLoader(TirSrDataset(args.data, 'train', crop=args.crop),
                              batch_size=args.batch, shuffle=True,
                              num_workers=2, drop_last=True)
    val_loader = DataLoader(TirSrDataset(args.data, 'val'), batch_size=1)

    def save(tag, it):
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'iter': it},
                   os.path.join(args.out, f'sr_{tag}.pth'))

    model.train()
    it, t0, best_psnr = start_iter, time.time(), 0.0
    while it < args.iters:
        for lq, gt in train_loader:
            if it >= args.iters:
                break
            it += 1
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device, enabled=(device == 'cuda')):
                loss = F.l1_loss(model(lq.to(device)), gt.to(device))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            if it % 100 == 0:
                print(f'iter {it:6d}  l1 {loss.item():.4f}  '
                      f'{100 / (time.time() - t0):.1f} it/s', flush=True)
                t0 = time.time()
            if it % args.val_every == 0:
                psnr = validate(model, val_loader, device)
                print(f'iter {it:6d}  val PSNR {psnr:.2f} dB', flush=True)
                if psnr > best_psnr:
                    best_psnr = psnr
                    save('best', it)
            if it % args.save_every == 0:
                save('latest', it)

    save('latest', it)
    print(f'done. best val PSNR: {best_psnr:.2f} dB. checkpoints in {args.out}')


if __name__ == '__main__':
    main()
