import os
from dalek.core import get_device, flush, ensure_dir
from dalek.vision import load_sdxl_base, generate_image

# --- Configuration ---
OUTPUT_DIR = "assets_chimp_train"
ensure_dir(f"{OUTPUT_DIR}/images")

DEVICE = get_device()
print(f"Using device: {DEVICE}")

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

def generate_branding_images():
    print("--- Generating Branding and End Images (SDXL Base, 128 steps) ---")
    try:
        pipe = load_sdxl_base()
        
        for scene in BRANDING_PROMPTS:
            fname = f"{OUTPUT_DIR}/images/{scene['id']}.png"
            print(f"Generating high-quality image: {scene['id']} (128 steps)")
            
            full_prompt = f"{scene['visual']}, minimalist style, cinematic lighting, no humans, only animals, 8k photorealistic"
            
            image = generate_image(pipe, full_prompt, steps=128, guidance=7.5, seed=2026)
            image.save(fname)
            
        del pipe; flush()
        print("Success! High-quality branding and end images generated.")
    except Exception as e:
        print(f"Image generation failed: {e}")

if __name__ == "__main__":
    generate_branding_images()