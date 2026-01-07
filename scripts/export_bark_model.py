#!/usr/bin/env python3
"""Export Bark (or compatible TTS) model to TorchScript and ONNX.
Usage: python3 export_bark_model.py --model-dir /path/to/model --out-dir exports
"""
import argparse
import os
import torch
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--model-dir', type=str, default=None)
parser.add_argument('--out-dir', type=str, default='exports')
args = parser.parse_args()

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

# Try to import transformers Bark model
try:
    from transformers import AutoProcessor, BarkModel
except Exception as e:
    print('transformers or BarkModel not available:', e)
    print('Install transformers and the Bark dependencies before running this script.')
    raise

model_dir = args.model_dir
if model_dir is None:
    # try common local paths
    for p in ['models/suno-bark_model', 'models/suno_bark_model', 'models/suno--bark', 'suno/bark']:
        if os.path.isdir(p):
            model_dir = p
            break
    if model_dir is None:
        model_dir = 'suno/bark'

print('Loading Bark model from', model_dir)
processor = AutoProcessor.from_pretrained(model_dir)
model = BarkModel.from_pretrained(model_dir, torch_dtype=torch.float32)
model.eval()

# Example text to trace with
text = "In a world that was once crisp and dry, a strange phenomenon has begun."
inputs = processor(text, return_tensors='pt', padding=True, truncation=True, return_attention_mask=True)
input_ids = inputs['input_ids']

# TorchScript export (trace)
try:
    print('Tracing model for TorchScript...')
    traced = torch.jit.trace(model, (input_ids,))
    ts_path = out_dir / 'bark_model.pt'
    traced.save(str(ts_path))
    print('Saved TorchScript to', ts_path)
except Exception as e:
    print('TorchScript export failed:', e)

# ONNX export
try:
    print('Exporting ONNX...')
    onnx_path = str(out_dir / 'bark_model.onnx')
    torch.onnx.export(model, (input_ids,), onnx_path, input_names=['input_ids'], output_names=['audio'], opset_version=17)
    print('Saved ONNX to', onnx_path)
except Exception as e:
    print('ONNX export failed:', e)
