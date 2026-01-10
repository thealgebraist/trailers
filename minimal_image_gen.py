#!/usr/bin/env python3
"""
Minimal image generation from text (simplified version without SD 1.5)
Uses a procedural approach similar to our TTS example
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import hashlib

def text_to_image_procedural(prompt, width=512, height=512):
    """
    Generate an image from text using procedural generation
    (This is a placeholder - real SD 1.5 would be neural)
    """
    # Create a hash from the prompt for deterministic generation
    hash_val = int(hashlib.md5(prompt.encode()).hexdigest(), 16)
    np.random.seed(hash_val % (2**32))
    
    # Generate a gradient background
    img_array = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create color gradients based on prompt hash
    r = (hash_val % 256)
    g = ((hash_val >> 8) % 256)
    b = ((hash_val >> 16) % 256)
    
    for y in range(height):
        for x in range(width):
            img_array[y, x] = [
                int(r * (1 - y/height) + 50 * (y/height)),
                int(g * (1 - x/width) + 100 * (x/width)),
                int(b * (x/width) * (y/height) + 80)
            ]
    
    # Add some noise for texture
    noise = np.random.randint(-20, 20, (height, width, 3))
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    
    # Convert to PIL Image
    img = Image.fromarray(img_array)
    
    # Add text overlay
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    except:
        font = ImageFont.load_default()
    
    # Add prompt text at bottom
    text_bbox = draw.textbbox((0, 0), f"Prompt: {prompt[:40]}", font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    # Draw black rectangle for text background
    draw.rectangle([(10, height-text_height-20), (text_width+20, height-10)], fill=(0, 0, 0, 180))
    draw.text((15, height-text_height-15), f"Prompt: {prompt[:40]}", fill=(255, 255, 255), font=font)
    
    return img

def save_image_data_for_cpp(img, prompt, output_prefix="image"):
    """
    Save image data for C++ inference
    """
    img_array = np.array(img)
    
    # Save as raw RGB data
    np.save(f'{output_prefix}_rgb.npy', img_array)
    
    # Save dimensions
    with open(f'{output_prefix}_dims.txt', 'w') as f:
        f.write(f"{img_array.shape[1]} {img_array.shape[0]} {img_array.shape[2]}")
    
    # Save prompt
    with open(f'{output_prefix}_prompt.txt', 'w') as f:
        f.write(prompt)
    
    print(f"✓ Saved {output_prefix}_rgb.npy ({img_array.shape})")
    print(f"✓ Saved {output_prefix}_dims.txt")
    print(f"✓ Saved {output_prefix}_prompt.txt")

def generate_with_local_sd15():
    """
    Try to use a local/cached SD 1.5 model if available
    """
    try:
        from diffusers import StableDiffusionPipeline
        import torch
        
        model_id = "runwayml/stable-diffusion-v1-5"
        
        # Check if model is cached
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()
        
        has_sd15 = any("stable-diffusion-v1-5" in str(repo.repo_id) for repo in cache_info.repos)
        
        if not has_sd15:
            print("SD 1.5 not cached locally. Would need to download ~4GB.")
            return None
        
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        pipe = pipe.to(device)
        
        return pipe
    except Exception as e:
        print(f"Could not load SD 1.5: {e}")
        return None

if __name__ == "__main__":
    prompt = "a beautiful landscape with mountains and a lake, photorealistic, 4k"
    
    print("=== Minimal Image Generation ===\n")
    print(f"Prompt: '{prompt}'")
    
    # Use procedural generation (SD 1.5 requires ~4GB download)
    print("\n--- Generating with procedural method ---")
    print("(Using fast procedural generation - for real SD 1.5, see notes below)")
    
    image = text_to_image_procedural(prompt)
    output_path = "pytorch_procedural_image.png"
    
    # Save image
    image.save(output_path)
    print(f"\n✓ Saved: {output_path}")
    print(f"  Size: {image.size}")
    
    # Save data for C++
    save_image_data_for_cpp(image, prompt, "gen_image")
    
    print("\n=== Image Generation Complete ===")
    print(f"\nNote: This uses procedural generation for demonstration.")
    print(f"For real SD 1.5:")
    print(f"  1. Uncomment the pipe = generate_with_local_sd15() line")
    print(f"  2. First run downloads ~4GB model from HuggingFace")
