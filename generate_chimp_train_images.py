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

if torch.cuda.is_available(): DEVICE = "cuda"
else: DEVICE = "cpu"

# Simplified, logical narrative prompts for 32 scenes
# Focusing on the "same chimp" and "no humans"
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

def flush():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def generate_images():
    print("--- Generating Images (SDXL Lightning, 8 steps) ---")
    base = "stabilityai/stable-diffusion-xl-base-1.0"
    repo = "ByteDance/SDXL-Lightning"
    ckpt = "sdxl_lightning_4step_unet.safetensors"

    try:
        # Load UNet via safetensors
        print(f"Loading UNet from {repo}...")
        unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(DEVICE, torch.float16)
        unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=str(DEVICE)))

        # Load Pipeline
        pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, dtype=torch.float16, variant="fp16").to(DEVICE)
        
        # Follow deprecation warning: Upcast VAE to float32
        pipe.vae.to(torch.float32)

        # Ensure scheduler uses trailing timesteps
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

        if DEVICE == "cuda": 
            pipe.enable_model_cpu_offload() 
        
        for scene in SCENES:
            fname = f"{OUTPUT_DIR}/images/{scene['id']}.png"
            if os.path.exists(fname): continue
            
            print(f"Generating image: {scene['id']}")
            
            # Final prompt reinforcement
            full_prompt = f"{scene['visual']}, minimalist style, cinematic lighting, no humans, only animals, 8k photorealistic"
            
            # SDXL Lightning specific settings: 8 steps, 0 guidance
            pipe(
                prompt=full_prompt, 
                guidance_scale=0.0, 
                num_inference_steps=8, 
                generator=torch.Generator(device="cpu").manual_seed(101 + int(scene['id'].split('_')[0]))
            ).images[0].save(fname)
            
        del pipe; flush()
    except Exception as e:
        print(f"Image generation failed: {e}")

if __name__ == "__main__":
    generate_images()