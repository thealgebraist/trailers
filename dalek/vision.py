import torch
import os
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from .core import get_device, get_dtype, flush

def load_sdxl_lightning(base="stabilityai/stable-diffusion-xl-base-1.0", 
                        repo="ByteDance/SDXL-Lightning", 
                        ckpt="sdxl_lightning_4step_unet.safetensors"):
    device = get_device()
    dtype = get_dtype(device)
    
    print(f"Loading SDXL Lightning UNet from {repo}...")
    unet_config = UNet2DConditionModel.from_config(base, subfolder="unet")
    unet = UNet2DConditionModel.from_config(unet_config).to(device, dtype)
    unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=str(device)))

    pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=dtype, variant="fp16").to(device)
    pipe.vae.to(torch.float32)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    
    if device == "cuda":
        pipe.enable_model_cpu_offload()
    
    return pipe

def load_sdxl_base(base="stabilityai/stable-diffusion-xl-base-1.0"):
    device = get_device()
    dtype = get_dtype(device)
    
    print(f"Loading Base SDXL Pipeline for high-quality generation...")
    pipe = StableDiffusionXLPipeline.from_pretrained(base, torch_dtype=dtype, variant="fp16").to(device)
    pipe.vae.to(torch.float32)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    if device == "mps":
        pipe.enable_attention_slicing()
    if device == "cuda":
        pipe.enable_model_cpu_offload()
        
    return pipe

def generate_image(pipe, prompt, steps=8, guidance=0.0, seed=42):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    image = pipe(
        prompt=prompt,
        guidance_scale=guidance,
        num_inference_steps=steps,
        generator=generator
    ).images[0]
    return image
