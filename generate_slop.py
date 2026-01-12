import torch
from diffusers import StableDiffusionPipeline
import os
import warnings
import logging

# Suppress warnings for a cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)

def generate_images(count=32, output_prefix="shining_slop"):
    model_id = "segmind/tiny-sd"

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")

    # Use torch_dtype for the pipeline (diffusers standard)
    # The deprecation warning often comes from sub-modules
    torch_dtype = torch.float32 if device in ["cpu", "mps"] else torch.float16
    
    print(f"Loading model {model_id}...")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        dtype=torch_dtype,
        use_safetensors=False
    )
    pipe = pipe.to(device)

    if device == "mps":
        pipe.enable_attention_slicing()

    slop_prompt = (
        "scene from The Shining movie, Jack Torrance insane face, overlook hotel hallway carpet, "
        "creepy twins holding hands with extra fingers, distorted faces, melting walls, "
        "blood elevator glitchy, bad anatomy, mutated limbs, surreal, ai artifacting, "
        "oversaturated, high contrast, hyper-realistic, 8k, eerie, nightmare fuel"
    )
    
    print(f"Generating {count} images...")

    for i in range(count):
        output_path = f"{output_prefix}_{i}.png"
        if os.path.exists(output_path):
            print(f"Skipping {output_path} (already exists)")
            continue
            
        print(f"Generating image {i+1}/{count}...")
        # Generate image
        # Using a fixed seed per index to ensure variety and reproducibility if restarted
        generator = torch.Generator(device=device).manual_seed(42 + i)
        image = pipe(slop_prompt, generator=generator).images[0]
        image.save(output_path)
        print(f"Image saved to {output_path}")

if __name__ == "__main__":
    generate_images(32)
