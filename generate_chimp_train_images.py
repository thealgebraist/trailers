import torch
import os
import gc
import numpy as np
import PIL.Image
if not hasattr(PIL.Image, "ANTIALIAS"): PIL.Image.ANTIALIAS = PIL.Image.LANCZOS
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download

# --- Configuration ---
OUTPUT_DIR = "assets_chimp_train"
os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)

if torch.cuda.is_available(): DEVICE = "cuda"
else: DEVICE = "cpu"

VO_SCRIPTS = [
    "Charlie the chimp wakes up in his jungle home, dreaming of the perfect banana.",
    "He finds a golden train ticket under a leaf and knows today is special.",
    "Charlie waves goodbye to his monkey friends and sets off on his adventure.",
    "He arrives at the bustling jungle train station, eyes wide with excitement.",
    "The train pulls in, steam hissing, and Charlie hops aboard with a big grin.",
    "He finds a window seat and presses his face to the glass, watching the trees blur by.",
    "A friendly toucan conductor checks Charlie’s ticket and tips his hat.",
    "The train rattles over a river, where crocodiles wave from the water below.",
    "Charlie shares a snack with a shy lemur sitting beside him.",
    "The train enters a dark tunnel, and everyone holds their breath in the shadows.",
    "Out of the tunnel, sunlight floods the carriage and Charlie laughs with joy.",
    "A family of parrots sings a song, filling the train with cheerful music.",
    "Charlie sketches a banana in his notebook, imagining its sweet taste.",
    "The train stops at a mountain station, and snow monkeys throw snowballs at the windows.",
    "Charlie helps a lost baby elephant find her seat.",
    "The train zooms past fields of wildflowers, colors swirling outside.",
    "A wise old gorilla tells Charlie stories of legendary bananas.",
    "Charlie spots a distant city and wonders if the best bananas are there.",
    "The train slows as it nears Banana Market Station, excitement building.",
    "Vendors wave bunches of bananas as the train comes to a stop.",
    "Charlie leaps off, heart pounding, and races to the biggest fruit stand.",
    "He inspects every banana, searching for the perfect one.",
    "At last, he finds a huge, golden banana shining in the sunlight.",
    "Charlie trades his ticket for the banana and hugs it close.",
    "He takes a big bite, savoring the sweet, creamy flavor.",
    "Other animals gather around, and Charlie shares his prize with new friends.",
    "The sun sets as Charlie sits on the station bench, happy and full.",
    "He waves goodbye to the market and boards the train home, banana in paw.",
    "Charlie dreams of new adventures as the train chugs into the night.",
    "Back in his jungle bed, Charlie smiles, knowing dreams can come true.",
    "The stars twinkle above, and the jungle is peaceful once more.",
    "Charlie’s train adventure is a story he’ll never forget."
]

SCENES = []
for i, voice_line in enumerate(VO_SCRIPTS):
    sid = f"{i+1:02d}_scene"
    framing = "close-up" if i % 3 == 0 else ("wide-angle" if i % 3 == 1 else "medium shot")
    image_prompt = (f"Photorealistic {framing} cinematic still of the scene: {voice_line} "
                    "+ extremely detailed, realistic textures, cinematic lighting, 1024x1024, film grain, movie still, no cartoons")
    SCENES.append({"id": sid, "visual": image_prompt})

def flush():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def generate_images():
    print("--- Generating Images (SDXL Lightning, 4 steps) ---")
    base = "stabilityai/stable-diffusion-xl-base-1.0"
    repo = "ByteDance/SDXL-Lightning"
    ckpt = "sdxl_lightning_4step_unet.safetensors"

    try:
        # Load UNet
        unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(DEVICE, torch.float16)
        unet.load_state_dict(torch.load(hf_hub_download(repo, ckpt), map_location=DEVICE))

        # Load Pipeline
        pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16").to(DEVICE)
        
        # Ensure scheduler uses trailing timesteps
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

        if DEVICE == "cuda": 
            pipe.enable_model_cpu_offload() 
        
        for scene in SCENES:
            fname = f"{OUTPUT_DIR}/images/{scene['id']}.png"
            if os.path.exists(fname): continue
            
            print(f"Generating image: {scene['id']}")
            prompt = scene['visual']
            
            # SDXL Lightning specific settings: 4 steps, 0 guidance
            pipe(
                prompt=prompt, 
                guidance_scale=0.0, 
                num_inference_steps=4, 
                generator=torch.Generator(device="cpu").manual_seed(101)
            ).images[0].save(fname)
            
        del pipe; flush()
    except Exception as e:
        print(f"Image generation failed: {e}")

if __name__ == "__main__":
    generate_images()
