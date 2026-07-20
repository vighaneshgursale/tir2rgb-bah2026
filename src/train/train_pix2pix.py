"""Thin wrapper around junyanz/pytorch-CycleGAN-and-pix2pix training.

Clones the repo if missing, symlinks our dataset into its expected layout, and
launches train.py with the flags that match this project (AtoB = TIR -> RGB,
256x256 pairs, U-Net256 + PatchGAN). Checkpoints land in --ckpt so they can be
synced to Drive between epochs.

Usage:
    python -m src.train.train_pix2pix --data data/tir2rgb --ckpt checkpoints \
        --name tir2rgb --epochs 150 --epochs-decay 50
Resume:
    ... --continue-train --epoch-count 80
"""

import argparse
import os
import subprocess
import sys

P2P_REPO = 'https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git'


def ensure_repo(dst):
    if not os.path.isdir(dst):
        subprocess.run(['git', 'clone', '--depth', '1', P2P_REPO, dst], check=True)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'dominate'], check=True)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/tir2rgb')
    ap.add_argument('--ckpt', default='checkpoints')
    ap.add_argument('--name', default='tir2rgb')
    ap.add_argument('--repo', default='third_party/pytorch-CycleGAN-and-pix2pix')
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--epochs-decay', type=int, default=50)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--continue-train', action='store_true')
    ap.add_argument('--epoch-count', type=int, default=1)
    args = ap.parse_args()

    repo = ensure_repo(args.repo)
    data = os.path.abspath(args.data)
    ckpt = os.path.abspath(args.ckpt)
    os.makedirs(ckpt, exist_ok=True)

    cmd = [sys.executable, 'train.py',
           '--dataroot', data,
           '--name', args.name,
           '--checkpoints_dir', ckpt,
           '--model', 'pix2pix',
           '--direction', 'AtoB',
           '--netG', 'unet_256',
           '--load_size', '256', '--crop_size', '256',
           '--preprocess', 'none',          # patches are exactly 256, flips only
           '--no_flip',                     # geographic orientation is meaningful
           '--input_nc', '3', '--output_nc', '3',
           '--lambda_L1', '100',
           '--batch_size', str(args.batch),
           '--n_epochs', str(args.epochs),
           '--n_epochs_decay', str(args.epochs_decay),
           '--save_epoch_freq', '5',
           '--display_id', '-1',            # no visdom server on Colab/Kaggle
           ]
    if args.continue_train:
        cmd += ['--continue_train', '--epoch_count', str(args.epoch_count)]

    print(' '.join(cmd))
    subprocess.run(cmd, cwd=repo, check=True)


if __name__ == '__main__':
    main()
