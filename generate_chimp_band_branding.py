import os
from dalek.core import get_device, flush, ensure_dir
from dalek.vision import load_sdxl_base, generate_image

# --- Configuration ---
OUTPUT_DIR = "assets_chimp_band_64"
ensure_dir(f"{OUTPUT_DIR}/images")

DEVICE = get_device()
print(f"Using device: {DEVICE}")

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

def generate_branding_images():
    print("--- Generating Bongo Band Branding and End Images (SDXL Base, 128 steps) ---")
    try:
        pipe = load_sdxl_base()
        
        for scene in BRANDING_PROMPTS:
            fname = f"{OUTPUT_DIR}/images/{scene['id']}.png"
            print(f"Generating high-quality image: {scene['id']} (128 steps)")
            
            full_prompt = f"{scene['visual']}, only chimps and animals, no humans, weird exotic creatures, 8k photorealistic"
            
            image = generate_image(pipe, full_prompt, steps=128, guidance=7.5, seed=2026 + int(scene['id'].split('_')[0]))
            image.save(fname)
            
        del pipe; flush()
        print(f"Success! Bongo Band branding and end images generated in {OUTPUT_DIR}/images/")
    except Exception as e:
        print(f"Image generation failed: {e}")

if __name__ == "__main__":
    generate_branding_images()