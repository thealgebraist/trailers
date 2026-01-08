import os
import random
from dalek.core import get_device, flush, ensure_dir
from dalek.vision import load_sdxl_lightning, generate_image

# --- Configuration ---
OUTPUT_DIR = "assets_chimp_band_64"
ensure_dir(f"{OUTPUT_DIR}/images")

DEVICE = get_device()
print(f"Using device: {DEVICE}")

# --- Scene Definitions ---
RIDICULOUS_INSTRUMENTS = [
    "banjo", "cowbell", "bass guitar", "kazoo", "tuba", 
    "accordion", "triangle", "keytar", "theremin", "electric violin",
    "slide whistle", "didgeridoo"
]

SCENES = []

# Use a fixed seed for reproducible instrument selection (Mirrored by voiceover script)
random.seed(42)

# Generate 64 Scenes matching assets script logic
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

def generate_images():
    print(f"--- Generating 64 Images (SDXL Lightning, 8 steps) ---")
    try:
        pipe = load_sdxl_lightning()
        
        for scene in SCENES:
            fname = f"{OUTPUT_DIR}/images/{scene['id']}.png"
            if os.path.exists(fname): continue
            
            print(f"Generating image: {scene['id']} ({scene['description']})")
            prompt = f"{scene['visual']}, only chimps and animals, no humans, weird exotic creatures, studio lighting, 8k photorealistic"
            
            image = generate_image(pipe, prompt, steps=8, guidance=0.0, seed=100 + int(scene['id'].split('_')[0]))
            image.save(fname)
            
        del pipe; flush()
    except Exception as e:
        print(f"Image generation failed: {e}")

if __name__ == "__main__":
    generate_images()
