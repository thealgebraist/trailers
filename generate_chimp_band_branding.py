import torch
import os
import gc
import numpy as np
import PIL.Image
if not hasattr(PIL.Image, "ANTIALIAS"): PIL.Image.ANTIALIAS = PIL.Image.LANCZOS
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler

# --- Configuration ---
OUTPUT_DIR = "assets_chimp_band_64"
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

# Branding and Story Completion Prompts for Bongo Band
BRANDING_PROMPTS = [
    {
        "id": "00_bongo_title_card",
        "visual": "A minimalist studio shot of a title card. The text 'BONGO FRENZY' is written in bold, clean black lettering on a pure white background. Studio lighting, high contrast."
    },
    {
        "id": "00_studio_logo_minimal",
        "visual": "A professional movie studio logo. A clean black silhouette of a chimp's head inside a circular frame on a pure white background, with the text 'UNIVERSAL CHIMP STUDIOS' in modern black lettering below it. Studio lighting."
    },
    {
        "id": "65_bongo_end_1",
        "visual": "A minimalist photo of three chimps standing in a line on a plain white background, holding bongos and kazoos, taking a formal bow. Serious expressions, studio lighting, high contrast."
    },
    {
        "id": "66_bongo_end_2",
        "visual": "A final minimalist shot. The words 'THE END' in large, elegant black serif font centered on a pure white background. Studio lighting, 8k high quality."
    }
]

def flush():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        try: torch.mps.empty_cache()
        except: pass

def generate_branding_images():
    print("--- Generating Bongo Band Branding and End Images (SDXL Base, 128 steps) ---")
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
            
            print(f"Generating high-quality image: {scene['id']} (128 steps)")
            
            # Final prompt reinforcement
            full_prompt = f"{scene['visual']}, only chimps and animals, no humans, weird exotic creatures, 8k photorealistic"
            
            # High-quality settings: 128 steps, 7.5 guidance
            pipe(
                prompt=full_prompt, 
                guidance_scale=7.5, 
                num_inference_steps=128, 
                generator=torch.Generator(device="cpu").manual_seed(2026 + int(scene['id'].split('_')[0]))
            ).images[0].save(fname)
            
        del pipe; flush()
        print(f"Success! Bongo Band branding and end images generated in {OUTPUT_DIR}/images/")
    except Exception as e:
        print(f"Image generation failed: {e}")

if __name__ == "__main__":
    generate_branding_images()
