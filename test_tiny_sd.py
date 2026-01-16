import torch
from diffusers import StableDiffusionPipeline
import os

def test():
    model_id = "segmind/tiny-sd"
    out_path = "test_tiny_sd_py.png"
    prompt = "A high-quality 8k digital masterpiece of a glowing artifact in a dark room"
    
    print(f"--- PyTorch Tiny-SD Test ---")
    print(f"Loading {model_id}...")
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32).to(device)
    
    print(f"Generating on {device}...")
    image = pipe(prompt, num_inference_steps=25).images[0]
    
    image.save(out_path)
    print(f"Success! Saved to {out_path}")

if __name__ == "__main__":
    test()
