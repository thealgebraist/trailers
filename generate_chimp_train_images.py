import os
from dalek.core import get_device, flush, ensure_dir
from dalek.vision import load_sdxl_lightning, generate_image

# --- Configuration ---
OUTPUT_DIR = "assets_chimp_train"
ensure_dir(f"{OUTPUT_DIR}/images")

DEVICE = get_device()
print(f"Using device: {DEVICE}")

# Simplified, logical narrative prompts for 32 scenes
SCENE_PROMPTS = [
    "A lone chimp in a cozy jungle hut, sitting on a wooden stool, deep in thought, thinking about a glowing golden banana.",
    "Close-up of the same chimp's face, eyes closed, dreaming of a perfect banana.",
    "The same chimp packing a small burlap sack in his jungle hut.",
    "The same chimp walking towards a jungle train station with a steam locomotive.",
    "The same chimp standing on a wooden train platform, holding a train ticket.",
    "The same chimp looking at the approaching steam train.",
    "The same chimp sitting inside a vintage wooden train carriage.",
    "The same chimp looking out of the train window at the jungle passing by.",
    "View from the train window: lush jungle trees blurring past.",
    "The same chimp pressing his face against the train window glass.",
    "The same chimp watching a river from the train window.",
    "The same chimp relaxing in his train seat.",
    "The same chimp stepping off the train onto a remote jungle station platform.",
    "The same chimp looking at a signpost pointing towards 'Banana Market'.",
    "The same chimp walking on a path through a dense, sunlit forest.",
    "The same chimp looking up at the tall forest canopy.",
    "The same chimp crossing a small stream in the forest.",
    "The same chimp seeing the market in the distance.",
    "The same chimp at a bustling banana market run by other chimps.",
    "The same chimp inspecting a huge, glowing golden banana at a market stall.",
    "The same chimp holding the golden banana triumphantly.",
    "The same chimp walking back through the forest at twilight, blue atmosphere.",
    "The same chimp in the forest at night, holding his golden banana, moonlight filtering through trees.",
    "The same chimp navigating the dark forest, fireflies around him.",
    "The same chimp at the jungle train station at night, waiting under a glowing lamp.",
    "The same chimp sitting on a bench at the night station, banana by his side.",
    "The same chimp watching the headlights of the night train arrive.",
    "The same chimp inside the dim, peaceful train carriage at night.",
    "The same chimp looking at the moon through the train window.",
    "The same chimp resting his head against the wooden seat, looking happy.",
    "The same chimp back in his jungle hut at night, tucked into bed.",
    "The same chimp asleep in his bed with the golden banana on a table nearby."
]

SCENES = []
for i, prompt in enumerate(SCENE_PROMPTS):
    sid = f"{i+1:02d}_scene"
    SCENES.append({"id": sid, "visual": prompt})

def generate_images():
    print("--- Generating Images (SDXL Lightning, 8 steps) ---")
    try:
        pipe = load_sdxl_lightning()
        
        for scene in SCENES:
            fname = f"{OUTPUT_DIR}/images/{scene['id']}.png"
            if os.path.exists(fname): continue
            
            print(f"Generating image: {scene['id']}")
            full_prompt = f"{scene['visual']}, minimalist style, cinematic lighting, no humans, only animals, 8k photorealistic"
            
            image = generate_image(pipe, full_prompt, steps=8, guidance=0.0, seed=101 + int(scene['id'].split('_')[0]))
            image.save(fname)
            
        del pipe; flush()
    except Exception as e:
        print(f"Image generation failed: {e}")

if __name__ == "__main__":
    generate_images()
