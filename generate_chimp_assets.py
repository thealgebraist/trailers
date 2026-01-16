import torch
import os
import sys
import numpy as np
import scipy.io.wavfile as wavfile
from diffusers import AutoPipelineForText2Image
from transformers import pipeline

# --- Configuration ---
PROJECT_NAME = "chimp"
ASSETS_DIR = f"assets_{PROJECT_NAME}"
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE != "cpu" else torch.float32

# --- Scenes ---
SCENES = [
    ("01_chimp_map", "Close up of a cute chimpanzee wearing a tiny explorer's hat, looking intensely at a map showing a Golden Banana, cinematic lighting, 8k, pixar style"),
    ("02_chimp_packing", "A chimpanzee packing a small vintage leather suitcase with a toothbrush and a magnifying glass, cozy bedroom, cinematic lighting, 8k, pixar style"),
    ("03_chimp_station", "A chimpanzee standing on a train platform as a massive, steam-puffing vintage train pulls into the station, steam everywhere, 8k, pixar style"),
    ("04_chimp_train_window", "A chimpanzee sitting in a plush velvet train seat, looking out at passing green mountains through a window, 8k, pixar style"),
    ("05_chimp_penguin", "A chimpanzee sharing a train seat with a confused penguin wearing a tuxedo, funny interaction, 8k, pixar style"),
    ("06_train_bridge", "A steam train crossing a high precarious stone bridge over a lush tropical jungle, cinematic wide shot, 8k, pixar style"),
    ("07_fruit_city", "A bustling futuristic city where buildings are shaped like giant fruits, pineapple towers, melon domes, 8k, pixar style"),
    ("08_golden_banana", "A glowing golden banana resting on a red velvet cushion in a high-end shop window, magical aura, 8k, pixar style"),
    ("09_chimp_running", "A chimpanzee sprinting through a colorful city street towards a fruit boutique, motion blur, 8k, pixar style"),
    ("10_chimp_reaching", "Close up of a chimpanzee's hand reaching out to touch a glowing golden banana, 8k, pixar style"),
    ("11_title_card", "Movie title card 'THE BANANA QUEST' with a golden banana icon, tropical jungle background, professional typography, 8k"),
    ("12_chimp_slippery", "A chimpanzee trying to peel a golden banana but it's very slippery and flying out of his hands, funny expression, 8k, pixar style")
]

VO_PROMPT = """
One chimp. One dream. And a ticket to the ultimate prize. 
Across the Great Divide, to the city of legends. 
He's not just hungry... he's on a mission. 
Experience the adventure of a lifetime. 
The Banana Quest. Coming this Summer.
"""

def generate_images():
    print(f"--- Generating {len(SCENES)} Images on {DEVICE} ---")
    pipe = AutoPipelineForText2Image.from_pretrained(
        "blanchon/FLUX.1-schnell-4bit", 
        torch_dtype=DTYPE
    ).to(DEVICE)
    num_steps = 64
    guidance = 0.0
    
    os.makedirs(f"{ASSETS_DIR}/images", exist_ok=True)
    
    for s_id, prompt in SCENES:
        out_path = f"{ASSETS_DIR}/images/{s_id}.png"
        # Force regeneration for better quality if it was low-step count
        if os.path.exists(out_path):
            os.remove(out_path)
            
        print(f"Generating: {s_id} ({num_steps} steps)")
        image = pipe(prompt, num_inference_steps=num_steps, guidance_scale=guidance).images[0]
        image.save(out_path)
    del pipe
    if DEVICE == "mps": torch.mps.empty_cache()

def generate_voiceover():
    print(f"--- Generating Voiceover on {DEVICE} ---")
    os.makedirs(f"{ASSETS_DIR}/voice", exist_ok=True)
    out_path = f"{ASSETS_DIR}/voice/voiceover_full.wav"
    
    if os.path.exists(out_path):
        print("Voiceover already exists.")
        return

    try:
        # Note: pipeline() doesn't always support local_files_only as a direct kwarg in some versions
        # but the underlying model loading does. 
        tts = pipeline("text-to-speech", model="suno/bark-small", device=DEVICE)
        print("Synthesizing vocal data...")
        audio = tts(VO_PROMPT)
        wavfile.write(out_path, audio["sampling_rate"], (audio["audio"] * 32767).astype(np.int16))
        print(f"Saved VO to: {out_path}")
        del tts
    except Exception as e:
        print(f"Voiceover generation failed: {e}")

def generate_music():
    print(f"--- Generating Background Music on {DEVICE} ---")
    os.makedirs(f"{ASSETS_DIR}/music", exist_ok=True)
    out_path = f"{ASSETS_DIR}/music/theme_main.wav"
    
    if os.path.exists(out_path):
        print("Music already exists.")
        return

    try:
        print("Attempting to load MusicGen pipeline...")
        synthesiser = pipeline("text-to-audio", "facebook/musicgen-small", device=DEVICE)
        prompt = "upbeat whimsical orchestral adventure theme, funny, lighthearted, cinematic, high quality"
        print("Generating music...")
        audio_output = synthesiser(prompt, forward_params={"max_new_tokens": 1500}) 
        wav_data = audio_output["audio"][0]
        sample_rate = audio_output["sampling_rate"]
        wavfile.write(out_path, sample_rate, (wav_data * 32767).astype(np.int16))
        print(f"Saved Music to: {out_path}")
        del synthesiser
    except Exception as e:
        print(f"Music generation failed: {e}")
        print("Creating a 30s silent placeholder for music.")
        sample_rate = 44100
        duration = 30
        silent_audio = np.zeros(sample_rate * duration, dtype=np.int16)
        wavfile.write(out_path, sample_rate, silent_audio)

if __name__ == "__main__":
    os.makedirs(ASSETS_DIR, exist_ok=True)
    mode = sys.argv[1] if len(sys.argv) > 1 else "all" 
    
    if mode in ["image", "all"]:
        generate_images()
    if mode in ["vo", "all"]:
        generate_voiceover()
    if mode in ["music", "all"]:
        generate_music()
    
    print(f"\n--- Assets ready in {ASSETS_DIR} ---")
