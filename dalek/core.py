import torch
import gc
import os

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def get_dtype(device=None):
    if device is None:
        device = get_device()
    return torch.float16 if device != "cpu" else torch.float32

def flush():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except:
            pass

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
