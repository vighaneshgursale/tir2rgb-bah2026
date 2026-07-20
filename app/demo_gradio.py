"""Gradio demo console: upload a 200 m TIR tile -> SR TIR @100 m + colorized RGB.

Mirrors the judging criteria on screen: input/intermediate/final side by side,
per-stage timings, and an explicit hallucination-check line in the run log.

Usage (Colab):
    python -m app.demo_gradio --sr-ckpt checkpoints/sr/sr_best.pth \
        --p2p-ckpt checkpoints/tir2rgb/latest_net_G.pth --share
"""

import argparse
import time

import numpy as np
from PIL import Image

from src.data.normalization import tir_unit_to_uint8
from src.infer.pipeline import Tir2RgbPipeline, load_tir


def build_app(pipe):
    import gradio as gr

    def run(file):
        if file is None:
            return None, None, None, 'Upload a TIR tile first.'
        tir = load_tir(file.name)
        sr, rgb, t = pipe(tir)

        rgb_i = rgb.astype(int)
        water_pct = ((rgb_i[:, :, 2] - rgb_i[:, :, 0]) > 30).mean() * 100
        halluc = ('⚠ possible water over-prediction'
                  if water_pct > 40 else '✓ no hallucination flag')
        log = (f'[{time.strftime("%H:%M:%S")}] input tile: {tir.shape}\n'
               f'stage 1  SR x2 (fine-tuned RRDBNet): {t["sr_s"]:.2f}s -> {sr.shape}\n'
               f'stage 2  Pix2Pix colorize:           {t["colorize_s"]:.2f}s\n'
               f'total inference:                     {t["total_s"]:.2f}s\n'
               f'water pixels: {water_pct:.1f}%  {halluc}')

        return (Image.fromarray(tir_unit_to_uint8(tir)),
                Image.fromarray(tir_unit_to_uint8(sr)),
                Image.fromarray(rgb), log)

    with gr.Blocks(title='TIR->RGB Pipeline · BAH2026 PS10') as demo:
        gr.Markdown('## Thermal IR Super-Resolution + Colorization — PS10 demo')
        with gr.Row():
            inp = gr.File(label='TIR tile @200m (.npy / .npz / .tif)')
            btn = gr.Button('Run inference', variant='primary')
        with gr.Row():
            im_in = gr.Image(label='Input TIR @200m')
            im_sr = gr.Image(label='Super-resolved TIR @100m')
            im_rgb = gr.Image(label='Colorized RGB @100m')
        log = gr.Textbox(label='Run log', lines=6)
        btn.click(run, inputs=inp, outputs=[im_in, im_sr, im_rgb, log])
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sr-ckpt', required=True)
    ap.add_argument('--p2p-ckpt', required=True)
    ap.add_argument('--share', action='store_true')
    args = ap.parse_args()

    pipe = Tir2RgbPipeline(args.sr_ckpt, args.p2p_ckpt)
    build_app(pipe).launch(share=args.share)


if __name__ == '__main__':
    main()
