import torch
import scipy.io.wavfile
import os
import gc
import subprocess
import re
import requests
import numpy as np
import PIL.Image
if not hasattr(PIL.Image, "ANTIALIAS"): PIL.Image.ANTIALIAS = PIL.Image.LANCZOS
from diffusers import StableAudioPipeline
from transformers import AutoProcessor

# --- Configuration ---
OUTPUT_DIR = "assets_chimp_train"
os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/voice", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/sfx", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/music", exist_ok=True)

if torch.cuda.is_available(): DEVICE = "cuda"
else: DEVICE = "cpu"

# --- Narrative VO Data ---
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

SCENE_DURATION = 4.0

# Build scene entries.
# Note: The trailer script used a specific structure. We recreate it here based on the original chimp logic 
# but adapted to the trailer script's processing style if needed.
SCENES = []
for i, voice_line in enumerate(VO_SCRIPTS):
    sid = f"{i+1:02d}_scene"
    framing = "close-up" if i % 3 == 0 else ("wide-angle" if i % 3 == 1 else "medium shot")
    image_prompt = (f"Photorealistic {framing} cinematic still of the scene: {voice_line} "
                    "+ extremely detailed, realistic textures, cinematic lighting, 1024x1024, film grain, movie still, no cartoons")
    
    # SFX logic preserved
    text = voice_line.lower()
    if "train" in text or "station" in text:
        sfx = "train station ambience: distant train rumble, soft platform announcements, footsteps, ambient jungle birds"
    elif "banana" in text:
        sfx = "market sounds: vendors calling, rustling fruit, soft crowd murmur, light breeze"
    elif "tunnel" in text:
        sfx = "tunnel ambience: muffled train sounds, low rumble, echoes"
    elif "parrot" in text or "parrots" in text:
        sfx = "colorful birdcalls, chirps, tropical parrots singing"
    elif "snow" in text or "snow monkeys" in text:
        sfx = "cold mountain breeze, distant chatter, soft crunch of snow"
    else:
        sfx = "jungle ambience: distant insect hum, soft birdsong, rustling leaves"
        
    scene = {"id": sid, "visual": image_prompt, "voice_prompt": voice_line, "sfx_prompt": sfx}
    SCENES.append(scene)

MUSIC_THEMES = [
    { "id": "theme_fun", "prompt": "Upbeat, playful orchestral score with jungle percussion and cheerful melodies. High quality." }
]

def flush():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def apply_trailer_voice_effect(file_path):
    """Applies a simple audio effect chain using ffmpeg."""
    temp_path = file_path.replace(".wav", "_temp.wav")
    # Simple compression and EQ for clarity
    filter_complex = "lowshelf=g=5:f=100,acompressor=threshold=-12dB:ratio=4:makeup=4dB"
    cmd = ["ffmpeg", "-y", "-i", file_path, "-af", filter_complex, temp_path]
    try: 
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.replace(temp_path, file_path)
    except Exception as e: 
        print(f"Failed effect: {e}")

def generate_voice_fishspeech():
    """Generate 32 separate voice lines using Fish Speech V1.5 via SDK."""
    print("--- Generating 32 Fish Speech voice lines ---")
    
    try:
        from fish_audio_sdk import Session, TTS
        session = Session() 
        tts = TTS(session) 
        
        for i, scene in enumerate(SCENES):
            txt = scene['voice_prompt']
            out_file = f"{OUTPUT_DIR}/voice/voice_{i+1:02d}.wav"
            meta_file = f"{OUTPUT_DIR}/voice/voice_{i+1:02d}.txt"
            
            if os.path.exists(out_file):
                print(f"Skipping existing {out_file}")
                continue
                
            print(f"Generating voice {i+1}/32: {txt[:60]}...")
            # SDK usage
            audio_bytes = tts.tts(text=txt, reference_id="default")
            
            with open(out_file, "wb") as f:
                f.write(audio_bytes)
                
            with open(meta_file, 'w') as mf:
                mf.write(f"Prompt: {txt}\nModel: fish-speech-v1.5\n")
            print(f"Wrote {out_file}")
            
            # Apply optional effect
            apply_trailer_voice_effect(out_file)
            
    except ImportError:
        print("Error: 'fish-audio-sdk' not found. Please install it with: pip install fish-audio-sdk")
    except Exception as e:
        print(f"Fish Speech generation failed: {e}")

def generate_voice():
    """Wrapper to handle voice generation strategy."""
    # In this script, we default to Fish Speech as requested
    generate_voice_fishspeech()

def generate_images():
    print("--- Generating Images (SDXL Lightning, 4 steps) ---")
    base = "stabilityai/stable-diffusion-xl-base-1.0"
    repo = "ByteDance/SDXL-Lightning"
    ckpt = "sdxl_lightning_4step_unet.safetensors"

    try:
        from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        # Load UNet
        print(f"Loading UNet from {repo}...")
        unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(DEVICE, torch.float16)
        unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=str(DEVICE)))

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
            
            # Refine prompt to exclude humans and specify animal world
            prompt = scene['visual'].replace("Vendors", "Chimp vendors").replace("city", "city of animals")
            prompt = f"{prompt}, only animals, no humans, chimps and exotic creatures, photorealistic, 8k"
            
            # SDXL Lightning specific settings: 8 steps, 0 guidance
            pipe(
                prompt=prompt, 
                guidance_scale=0.0, 
                num_inference_steps=8, 
                generator=torch.Generator(device="cpu").manual_seed(101)
            ).images[0].save(fname)
            
        del pipe; flush()
    except Exception as e:
        print(f"Image generation failed: {e}")

def generate_audio():
    print("\n--- Generating Music & SFX (Stable Audio Open) ---")
    try:
        # Using Stable Audio Open for SFX and Music
        model_id = "stabilityai/stable-audio-open-1.0"
        pipe = StableAudioPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
        if DEVICE == "cuda": pipe.enable_model_cpu_offload()
        else: pipe.to(DEVICE)
        
        neg = "low quality, noise, distortion, artifacts, fillers, talking"
        
        # Generate SFX
        for scene in SCENES:
            filename = f"{OUTPUT_DIR}/sfx/{scene['id']}.wav"
            if os.path.exists(filename): continue
            print(f"Generating SFX: {scene['id']}")
            audio = pipe(
                prompt=scene['sfx_prompt'], 
                negative_prompt=neg, 
                num_inference_steps=100, 
                audio_end_in_s=SCENE_DURATION
            ).audios[0]
            scipy.io.wavfile.write(filename, rate=44100, data=audio.cpu().numpy().T)
            
        # Generate MUSIC
        for theme in MUSIC_THEMES:
            filename = f"{OUTPUT_DIR}/music/{theme['id']}.wav"
            if os.path.exists(filename): continue
            print(f"Generating Music: {theme['id']}")
            audio = pipe(
                prompt=theme['prompt'], 
                negative_prompt=neg, 
                num_inference_steps=100, 
                audio_end_in_s=45.0
            ).audios[0]
            scipy.io.wavfile.write(filename, rate=44100, data=audio.cpu().numpy().T)
            
        del pipe; flush()
    except Exception as e: 
        print(f"Audio generation failed: {e}")

if __name__ == "__main__":
    import sys
    if "voice" in sys.argv: 
        generate_voice()
        sys.exit(0)
    if "images" in sys.argv:
        generate_images()
        sys.exit(0)
    
    # Default: Run all
    generate_images()
    generate_voice()
    generate_audio()