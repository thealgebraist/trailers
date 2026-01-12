import time
import torch
from diffusers import StableDiffusionPipeline
import logging
import sys
import os

# Suppress logging
logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

def worker(worker_id):
    model_id = "runwayml/stable-diffusion-v1-5"
    
    # Simple logic to avoid all workers trying to use all CPU cores
    torch.set_num_threads(1) 

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, 
            torch_dtype=dtype,
            use_safetensors=True
        )
    except:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, 
            torch_dtype=dtype
        )

    pipe = pipe.to(device)
    
    # Memory optimizations are crucial for multiple processes
    if device == "cuda":
        pipe.enable_xformers_memory_efficient_attention()
        # pipe.enable_model_cpu_offload() # Might be too slow for benchmarking throughput, but saves VRAM
    elif device == "mps":
        pipe.enable_attention_slicing()

    prompt = "ai sloppy, distorted, glitchy, surreal, melting, weird artifacts"
    
    # Signal ready
    print(f"WORKER_{worker_id}_READY")
    sys.stdout.flush()

    count = 0
    while True:
        try:
            # Generate 1 image at a time
            _ = pipe(prompt, num_inference_steps=32).images[0]
            count += 1
            print(f"WORKER_{worker_id}_IMAGE_DONE")
            sys.stdout.flush()
        except Exception as e:
            print(f"WORKER_{worker_id}_ERROR: {e}")
            sys.stdout.flush()
            time.sleep(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python slop_worker.py <worker_id>")
        sys.exit(1)
    worker(sys.argv[1])
