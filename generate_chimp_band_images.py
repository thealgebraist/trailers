import torch
import os
import gc
import random
import numpy as np
import PIL.Image
if not hasattr(PIL.Image, "ANTIALIAS"): PIL.Image.ANTIALIAS = PIL.Image.LANCZOS
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download

# --- Configuration ---
OUTPUT_DIR = "assets_chimp_band_64"
os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)

if torch.cuda.is_available(): DEVICE = "cuda"
else: DEVICE = "cpu"

# --- Scene Definitions ---
# Pool of instruments to mix with Bongos
RIDICULOUS_INSTRUMENTS = [
    "banjo", "cowbell", "bass guitar", "kazoo", "tuba", 
    "accordion", "triangle", "keytar", "theremin", "electric violin",
    "slide whistle", "didgeridoo"
]

SCENES = []

# Generate 64 Scenes
for i in range(64):
    sid = f"{i+1:02d}_scene"
    
    roll = random.random()
    if roll < 0.4:
        # Solo Bongo
        visual_prompt = "A minimalist photo of a single chimp playing bongo drums with intense focus. Plain white background, studio lighting, high contrast."
        desc = "Bongo Solo"
    elif roll < 0.7:
        # Duo
        extra = random.choice(RIDICULOUS_INSTRUMENTS)
        visual_prompt = f"A minimalist photo of two chimps on a plain white background. One plays bongos, the other plays {extra}. Serious expressions, studio lighting."
        desc = f"Bongos and {extra}"
    else:
        # Trio (Max 3 instruments)
        extras = random.sample(RIDICULOUS_INSTRUMENTS, 2)
        visual_prompt = f"A minimalist photo of a three-chimp band on a plain white background. Playing bongos, {extras[0]}, and {extras[1]}. Silly but stoic, studio lighting."
        desc = f"Bongos, {extras[0]}, and {extras[1]}"

    SCENES.append({
        "id": sid,
        "visual": visual_prompt,
        "description": desc
    })

def flush():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def generate_images():
    print(f"--- Generating 64 Images (SDXL Lightning) ---")
    base = "stabilityai/stable-diffusion-xl-base-1.0"
    repo = "ByteDance/SDXL-Lightning"
    ckpt = "sdxl_lightning_4step_unet.safetensors"

    try:
        from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        # Load UNet
        print(f"Loading UNet from {repo}...")
        unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(DEVICE, torch.float16)
        unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=str(DEVICE)))

        # Load Pipeline
        pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16").to(DEVICE)
        
        # Ensure scheduler uses trailing timesteps
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

        if DEVICE == "cuda": 
            pipe.enable_model_cpu_offload() 
        
        for scene in SCENES:
            fname = f"{OUTPUT_DIR}/images/{scene['id']}.png"
            if os.path.exists(fname): continue
            
            print(f"Generating image: {scene['id']} ({scene['description']})")
            
            prompt = f"{scene['visual']}, only chimps and animals, no humans, weird exotic creatures, studio lighting, high quality"
            
            pipe(
                prompt=prompt, 
                guidance_scale=0.0, 
                num_inference_steps=8, 
                generator=torch.Generator(device="cpu").manual_seed(100 + int(scene['id'].split('_')[0]))
            ).images[0].save(fname)
            
        del pipe; flush()
    except Exception as e:
        print(f"Image generation failed: {e}")

if __name__ == "__main__":
    generate_images()
