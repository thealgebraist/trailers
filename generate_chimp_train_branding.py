import torch
import os
import gc
import numpy as np
import PIL.Image
if not hasattr(PIL.Image, "ANTIALIAS"): PIL.Image.ANTIALIAS = PIL.Image.LANCZOS
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# --- Configuration ---
OUTPUT_DIR = "assets_chimp_train"
os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)

# Select device: mps for macOS, cuda for NVIDIA, else cpu
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# Determine dtype based on device
DTYPE = torch.float16 if DEVICE != "cpu" else torch.float32

print(f"Using device: {DEVICE} with {DTYPE}")

# Branding and Story Completion Prompts
BRANDING_PROMPTS = [
    {
        "id": "00_title_card",
        "visual": "A cinematic movie title card on a dark wooden background. The text 'CHIMP BANANA TRAIN' is embossed in gold, jungle leaves framing the edges, high quality movie poster style."
    },
    {
        "id": "00_studio_logo",
        "visual": "A professional movie studio logo. A golden silhouette of a chimp's head inside a circular frame, with the text 'UNIVERSAL CHIMP STUDIOS' written in elegant gold lettering below it, dramatic lighting, dark background."
    },
    {
        "id": "33_story_end_1",
        "visual": "The same chimp standing on a jungle ridge overlooking his home, holding a single glowing golden banana towards the sunrise, a sense of completion and peace."
    },
    {
        "id": "34_story_end_2",
        "visual": "A final shot of the jungle steam train disappearing into the distant morning mist, 'The End' written in elegant gold script across the center of the frame, cinematic farewell."
    }
]

def flush():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        try: torch.mps.empty_cache()
        except: pass

def generate_branding_images():
    print("--- Generating Branding and End Images (SDXL Base, 128 steps) ---")
    base = "stabilityai/stable-diffusion-xl-base-1.0"

    try:
        # Load Pipeline (Base model for high-iteration quality)
        print(f"Loading Base SDXL Pipeline for high-iteration quality...")
        pipe = StableDiffusionXLPipeline.from_pretrained(base, torch_dtype=DTYPE, variant="fp16").to(DEVICE)
        
        # Follow deprecation warning: Upcast VAE to float32
        pipe.vae.to(torch.float32)
        
        # Standard scheduler for high step counts
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

        if DEVICE == "mps":
            pipe.enable_attention_slicing()
        
        if DEVICE == "cuda": 
            pipe.enable_model_cpu_offload() 
        
        for scene in BRANDING_PROMPTS:
            fname = f"{OUTPUT_DIR}/images/{scene['id']}.png"
            # Overwrite if exists to ensure the 128-step version is saved
            
            print(f"Generating high-quality image: {scene['id']} (128 steps)")
            
            # Final prompt reinforcement
            full_prompt = f"{scene['visual']}, minimalist style, cinematic lighting, no humans, only animals, 8k photorealistic"
            
            # High-quality settings: 128 steps, 7.5 guidance
            pipe(
                prompt=full_prompt, 
                guidance_scale=7.5, 
                num_inference_steps=128, 
                generator=torch.Generator(device="cpu").manual_seed(2026)
            ).images[0].save(fname)
            
        del pipe; flush()
        print("Success! High-quality branding and end images generated.")
    except Exception as e:
        print(f"Image generation failed: {e}")

if __name__ == "__main__":
    generate_branding_images()
