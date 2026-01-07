#!/usr/bin/env python3
"""Export MusicGen (music) model to TorchScript and ONNX where feasible.
Usage: python3 export_musicgen_model.py --model-dir /path/to/musicgen --out-dir exports
Note: MusicGen exports are heavy and may not succeed; this script attempts to export small submodules if present.
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

# Try to load MusicGen components via transformers or huggingface
try:
    from transformers import AutoProcessor
    # MusicGen model class may not be in transformers; user may need the original musicgen repo
except Exception as e:
    print('transformers not available or MusicGen class missing:', e)

# Heuristic: try to load model files from model_dir (e.g., model.safetensors) and export if a Torch model is found
model_dir = args.model_dir or 'facebook/musicgen-small'
print('Using model dir', model_dir)

# This script provides a scaffold; actual MusicGen export requires model-specific code.
print('Note: MusicGen export requires repository-specific export logic. Please run a tailored export in Python with the MusicGen codebase.')

# Attempt trivial export if a torch .pt/.pth is present
for fn in Path(model_dir).rglob('*.pt'):
    try:
        print('Found candidate torch file', fn)
        m = torch.load(fn, map_location='cpu')
        if isinstance(m, torch.nn.Module):
            m.eval()
            ts_path = out_dir / 'musicgen_module.pt'
            torch.jit.trace(m, torch.zeros(1, 1))
            torch.jit.save(m, ts_path)
            print('Saved TorchScript to', ts_path)
    except Exception as e:
        print('Skipping file', fn, 'due to', e)

print('MusicGen export script finished (manual steps likely required).')
