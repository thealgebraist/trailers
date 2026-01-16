import torch
import os
from torchvision.utils import save_image

def verify():
    device = "cpu"
    print("--- Verifying C++ Logic in Python (Sequential) ---")
    
    # 1. Encode
    tokens = torch.zeros((1, 77), dtype=torch.long)
    clip = torch.jit.load("tiny_sd_pt/clip.pt")
    cond = clip(tokens)
    del clip
    
    # 2. Denoise
    latents = torch.randn(1, 4, 64, 64)
    unet = torch.jit.load("tiny_sd_pt/unet.pt")
    steps = 20
    for i in range(steps):
        t = torch.tensor([1000.0 * (1.0 - i/steps)])
        noise_pred = unet(latents, t, cond)
        # Use a more realistic DDIM step: x = x - pred * scale
        # For 20 steps, 0.05 is a rough approximation of 1/steps
        latents = latents - 0.05 * noise_pred 
        if i % 5 == 0: print(f"  Step {i}")
    del unet
    
    # 3. Decode
    vae = torch.jit.load("tiny_sd_pt/vae.pt")
    # Scaling factor is CRITICAL
    latents = latents / 0.18215
    pixels = vae(latents)
    del vae
    
    # Save
    pixels = (pixels.clamp(-1, 1) + 1) / 2
    save_image(pixels, "test_cpp_logic_result.png")
    print("Saved test_cpp_logic_result.png")

if __name__ == "__main__":
    verify()