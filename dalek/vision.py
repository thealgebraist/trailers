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
    # Support local repo paths: if `repo` is a directory containing the checkpoint, load it directly.
    ckpt_path = None
    try:
        candidate = os.path.join(repo, ckpt)
        if os.path.isdir(repo) and os.path.exists(candidate):
            ckpt_path = candidate
    except Exception:
        ckpt_path = None
    if ckpt_path is None:
        ckpt_path = hf_hub_download(repo, ckpt)
    unet.load_state_dict(load_file(ckpt_path, device=str(device)))

    # Load pipeline, prefer a local base directory if available (supports offline use)
    base_to_use = base
    if os.path.isdir(base) and os.path.exists(os.path.join(base, "config.json")):
        base_to_use = base
    else:
        # check for a base model directory inside the provided repo path
        local_candidate = os.path.join(repo, os.path.basename(base))
        if os.path.isdir(local_candidate) and os.path.exists(os.path.join(local_candidate, "config.json")):
            base_to_use = local_candidate

    pipe = StableDiffusionXLPipeline.from_pretrained(base_to_use, unet=unet, torch_dtype=dtype, variant="fp16").to(device)
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
