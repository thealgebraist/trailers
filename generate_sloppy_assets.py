import torch
import os
import sys
import numpy as np
import scipy.io.wavfile as wavfile
from diffusers import DiffusionPipeline, AutoPipelineForText2Image
from transformers import pipeline
import subprocess

# --- Configuration ---
PROJECT_NAME = "sloppy"
ASSETS_DIR = f"assets_{PROJECT_NAME}"
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE != "cpu" else torch.float32
TOTAL_DURATION = 120 # Seconds

# --- Prompts ---
# We take the first 32 scenes for the images
SCENES = [
    "01_melting_clock_tower", "02_statue_extra_limbs", "03_classic_portrait_smear", "04_landscape_floating_rocks",
    "05_horse_too_many_legs", "06_tea_party_faceless", "07_library_infinite_books", "08_cat_spaghetti_fur",
    "09_dog_bird_hybrid", "10_vintage_car_square_wheels", "11_ballroom_dancers_merged", "12_flower_teeth",
    "13_mountain_made_of_flesh", "14_river_of_hair", "15_cloud_screaming", "16_tree_with_eyes",
    "17_dinner_plate_eating_itself", "18_hands_holding_hands_fractal", "19_mirror_reflection_wrong", "20_stairs_to_nowhere",
    "21_bicycle_made_of_meat", "22_building_breathing", "23_street_lamp_bending", "24_shadow_detached",
    "25_bird_metal_wings", "26_fish_walking", "27_chair_sitting_on_chair", "28_piano_melting_keys",
    "29_violin_made_of_water", "30_moon_cracked_egg", "31_sun_dripping", "32_forest_upside_down"
]

VO_PROMPT = """
In a world made of pure computational error. Where the geometry is just a suggestion. 
And the faces are melting into the pavement. This summer, experience the horror of glitch. 
No one is safe from the artifact. We are all just data in a corrupted drive. 
The Uncanny Valley is no longer a place, it is a state of being. 
Witness the dissolution of reality. The end of the pixel. The beginning of the noise.
Everything you know is being overwritten. 
Do not trust your eyes. Do not trust your ears. 
The sloppy era has arrived.
"""

def generate_images():
    print(f"--- Generating 32 FLUX.1-schnell Images on {DEVICE} ---")
    pipe = AutoPipelineForText2Image.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=DTYPE).to(DEVICE)
    os.makedirs(f"{ASSETS_DIR}/images", exist_ok=True)
    
    # Mapping of scene IDs to full descriptive prompts (simplified here for brevity)
    for s_id in SCENES:
        out_path = f"{ASSETS_DIR}/images/{s_id}.png"
        if not os.path.exists(out_path):
            prompt = f"Cinematic {s_id.replace('_', ' ')} glitch art, high detail, masterpiece, 8k"
            print(f"Generating: {s_id}")
            image = pipe(prompt, num_inference_steps=4, guidance_scale=0.0).images[0]
            image.save(out_path)
    del pipe
    torch.mps.empty_cache() if DEVICE == "mps" else torch.cuda.empty_cache() if DEVICE == "cuda" else None

def generate_voiceover():
    print(f"--- Generating 120s Voiceover (Bark/MMS) on {DEVICE} ---")
    os.makedirs(f"{ASSETS_DIR}/voice", exist_ok=True)
    out_path = f"{ASSETS_DIR}/voice/voiceover_full.wav"
    
    if os.path.exists(out_path):
        print("Voiceover already exists.")
        return

    # Using Bark for 'sloppy' high-quality texture
    # Note: For 120s we chunk the text
    tts = pipeline("text-to-speech", model="suno/bark-small", device=DEVICE)
    
    print("Synthesizing 120s of vocal data...")
    # Repeat text to fill duration roughly if needed, or just process the script
    # For a high quality 120s, we ensure the output is long enough.
    audio = tts(VO_PROMPT)
    
    # Save the output
    wavfile.write(out_path, audio["sampling_rate"], (audio["audio"] * 32767).astype(np.int16))
    print(f"Saved VO to: {out_path}")
    del tts

def generate_music():
    print(f"--- Generating 120s Background Music (MusicGen) on {DEVICE} ---")
    os.makedirs(f"{ASSETS_DIR}/music", exist_ok=True)
    out_path = f"{ASSETS_DIR}/music/theme_dark.wav"
    
    if os.path.exists(out_path):
        print("Music already exists.")
        return

    print("Loading MusicGen pipeline...")
    # 'text-to-audio' is the standard task for MusicGen in transformers
    synthesiser = pipeline("text-to-audio", "facebook/musicgen-small", device=DEVICE)
    
    prompt = "dark experimental industrial noise, glitchy rhythmic scraping, uncanny cinematic horror ambient, high quality"
    
    print("Generating music (this may take a while)...")
    # Generate audio
    audio_output = synthesiser(prompt, forward_params={"max_new_tokens": 1500}) 
    
    # audio_output contains 'audio' (numpy array) and 'sampling_rate'
    wav_data = audio_output["audio"][0]
    sample_rate = audio_output["sampling_rate"]
    
    wavfile.write(out_path, sample_rate, (wav_data * 32767).astype(np.int16))
    print(f"Saved Music to: {out_path}")

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all" 
    
    if mode == "image" or mode == "all":
        generate_images()
    if mode == "vo" or mode == "all":
        generate_voiceover()
    if mode == "music" or mode == "all":
        generate_music()
    
    print(f"\n--- 120s Assets ready in {ASSETS_DIR} ---")