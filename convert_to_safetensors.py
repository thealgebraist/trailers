import torch
from safetensors.torch import save_file
import os

def convert():
    base = "/Users/anders/.cache/huggingface/hub/models--segmind--tiny-sd/snapshots/cad0bd7495fa6c4bcca01b19a723dc91627fe84f"
    
    # Helper to ensure all tensors are contiguous
    def make_contiguous(weights):
        return {k: v.contiguous() for k, v in weights.items()}

    # 1. UNet
    unet_bin = os.path.join(base, "unet/diffusion_pytorch_model.bin")
    if os.path.exists(unet_bin):
        print("Converting UNet...")
        weights = torch.load(unet_bin, map_location="cpu", weights_only=True)
        save_file(make_contiguous(weights), "tiny_sd_unet.safetensors")
        
    # 2. VAE
    vae_bin = os.path.join(base, "vae/diffusion_pytorch_model.bin")
    if os.path.exists(vae_bin):
        print("Converting VAE...")
        weights = torch.load(vae_bin, map_location="cpu", weights_only=True)
        save_file(make_contiguous(weights), "tiny_sd_vae.safetensors")
        
    # 3. CLIP
    clip_bin = os.path.join(base, "text_encoder/pytorch_model.bin")
    if os.path.exists(clip_bin):
        print("Converting CLIP...")
        weights = torch.load(clip_bin, map_location="cpu", weights_only=True)
        save_file(make_contiguous(weights), "tiny_sd_clip.safetensors")
        
    print("Conversion complete!")

if __name__ == "__main__":
    convert()