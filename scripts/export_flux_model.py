#!/usr/bin/env python3
"""Export diffusion model components (UNet, VAE) to TorchScript and ONNX where feasible.
Usage: python3 export_flux_model.py --model-dir /path/to/tiny-sd --out-dir exports
Note: Exporting diffusion models requires careful handling; this script attempts to find UNet/VAE modules and trace them.
"""
import argparse
import os
from pathlib import Path
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--model-dir', type=str, default=None)
parser.add_argument('--out-dir', type=str, default='exports')
args = parser.parse_args()

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
model_dir = Path(args.model_dir) if args.model_dir else Path('models/segmind_tiny-sd_model')

print('Model dir:', model_dir)

try:
    from diffusers import UNet2DConditionModel, AutoencoderKL
except Exception as e:
    print('diffusers not available or components missing:', e)

# Attempt to load UNet/vae from diffusers structure
try:
    from diffusers import DiffusionPipeline
    pipe = DiffusionPipeline.from_pretrained(str(model_dir))
    # Try export UNet
    unet = pipe.unet.eval()
    vae = pipe.vae.eval()
    # Create dummy inputs for UNet: sample, timestep, encoder_hidden_states
    import torch
    sample = torch.randn(1, unet.in_channels, 32, 32)
    timestep = torch.tensor([10])
    # encoder hidden states shape depends on text encoder; use zeros
    enc = torch.randn(1, 64, 32)
    try:
        print('Tracing UNet...')
        traced_unet = torch.jit.trace(unet, (sample, timestep, enc))
        traced_unet.save(out_dir / 'unet.pt')
        print('Saved UNet TorchScript')
    except Exception as e:
        print('UNet TorchScript trace failed:', e)
    try:
        print('Exporting UNet ONNX...')
        torch.onnx.export(unet, (sample, timestep, enc), str(out_dir / 'unet.onnx'), opset_version=17)
        print('Saved UNet ONNX')
    except Exception as e:
        print('UNet ONNX export failed:', e)
    # VAE encode/decode can be exported similarly if needed
except Exception as e:
    print('Failed to load pipeline or export components:', e)

print('Flux export script finished (may require manual tuning).')
