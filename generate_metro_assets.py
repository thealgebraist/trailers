import torch
import os
import sys
import numpy as np
import scipy.io.wavfile as wavfile
from diffusers import DiffusionPipeline, StableAudioOpenPipeline
from transformers import pipeline
from PIL import Image

# --- Configuration ---
PROJECT_NAME = "metro"
ASSETS_DIR = f"assets_{PROJECT_NAME}"
DEVICE = "cuda" # Targeted for H200
DTYPE = torch.bfloat16 # High precision and performance for H200

# Scene Definitions (Prompts & SFX Prompts)
SCENES = [
    ("01_entrance", "Cinematic shot of a futuristic minimalist brutalist metro entrance, concrete, fog, neon strip light, 8k, photorealistic", "subway station ambience wind howling distant eerie drone"),
    ("02_face_scan", "Close up grotesque biometric face scanner, red laser grid mapping a weeping human face, dystopian technology", "digital scanning noise high pitched beep laser hum"),
    ("03_finger_scan", "Futuristic security device crushing a human finger against a glass plate, green light, macro photography", "mechanical servo whine glass squeak crunch"),
    ("04_smell_detector", "Bizarre nose-shaped mechanical sensor sniffing a person's neck, medical aesthetic, sterile white", "sniffing sound vacuum pump sucking air"),
    ("05_torso_slime", "Person pressing bare chest against a wall of gelatinous bio-luminescent blue slime, imprint visible", "wet squelch slime dripping sticky sound"),
    ("06_tongue_print", "Metal surgical clamp holding a human tongue, scanning laser, saliva dripping, high detail", "wet mouth sound metallic click servo motor"),
    ("07_retina_drill", "Eye scanning device that looks like a surgical drill, red laser beam pointing into pupil, extreme close up", "high pitched drill whine laser zap"),
    ("08_ear_wax_sampler", "Tiny robotic probe entering a human ear, futuristic macro photography, cold lighting", "squishy probing sound mechanical whir"),
    ("09_hair_count", "Robotic tweezers plucking a single hair from a scalp, digital counter display showing numbers", "sharp pluck sound digital counter increment beep"),
    ("10_sweat_analysis", "Person standing in a glass tube sweating profusley under heat lamps, collection drains at feet", "heavy breathing steam hiss dripping water"),
    ("11_bone_crusher", "Hydraulic press gently compressing a human arm to measure density, medical readout, chrome metal", "hydraulic hiss metallic thud bone creak"),
    ("12_spirit_photo", "Ectoplasmic aura camera screen, person looks like a ghost in the viewfinder, distortion, grainy", "static noise ghostly moan electrical crackle"),
    ("13_karma_scale", "Golden mechanical scales weighing a human heart against a feather, futuristic minimalist court", "metallic clinking scales balancing heavy thud"),
    ("14_dream_extract", "Helmet with wires sucking glowing mist from person's head, fiber optic cables, cyberpunk", "vacuum suction electrical humming bubbling liquid"),
    ("15_memory_wipe", "Flash of white light, person looking dazed and empty, pupil dilated, bright overexposed", "camera flash capacitor charge high pitch ring"),
    ("16_genetic_sieve", "Blood sample passing through glowing filter, DNA strands visible, microscopic view", "liquid pumping bubbling biological squish"),
    ("17_final_stamp", "Hot branding iron stamping 'APPROVED' on a forehead, steam rising, skin texture", "sizzling burning sound heavy stamp thud"),
    ("18_platform", "Empty endless subway platform, sleek white tiles, ominous silence, vanishing point", "empty room tone distant train rumble fluorescent hum"),
    ("19_train_interior", "Inside metro train, minimalist grey seats, sad people staring at feet, uniform grey clothing, sterile", "subway train interior rumble wheels on track rhythmic clacking"),
    ("20_title_card", "Text 'METRO' in minimal sans-serif font, glowing white on black background, cinematic typography", "deep bass boom cinematic hit silence"),
]

# Voiceover Script (Sarcastic, Terse, Deep Voice)
VO_SCRIPT = """
Welcome to the Metro. 
The future of transit is secure. 
For your safety, we require a few... verifications.
Face scan. Don't blink. We need to see the fear in your eyes.
Finger scan. Press harder. Until it hurts. Good.
Olfactory analysis. You smell like anxiety. And cheap coffee.
Torso imprint. The slime is sterile. Mostly.
Tongue print. Taste the sensor. It tastes like copper. And submission.
Retina check. Keep your eye open. The laser is warm.
Auricular sampling. We are listening to your thoughts. Through your earwax.
Follicle audit. One hair. Two hair. Three. We are counting.
Sweat extraction. Perspire for the state. Your fluids are data.
Bone density verification. Just a little pressure. To ensure you are solid.
Spirit photography. Your aura is grey. How disappointing.
Karma weighing. Your sins are heavy. You will pay extra.
Dream extraction. Leave your hopes here. You won't need them.
Memory wipe. Forget why you came. Forget who you are.
Genetic sieve. You are filtered. You are processed.
Final stamp. Approved. You are now a passenger.
Welcome to the platform. The wait is eternal.
Board the train. Sit down. Be sad.
This is the Metro.
We are going nowhere.
Fast.
"""

def generate_images():
    print(f"--- Generating 20 FLUX.2-dev Images (64 steps) on {DEVICE} ---")
    pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.2-dev", torch_dtype=DTYPE).to(DEVICE)

    os.makedirs(f"{ASSETS_DIR}/images", exist_ok=True)
    
    for s_id, prompt, _ in SCENES:
        out_path = f"{ASSETS_DIR}/images/{s_id}.png"
        if not os.path.exists(out_path):
            print(f"Generating: {s_id}")
            image = pipe(prompt, num_inference_steps=64, guidance_scale=3.5, width=1280, height=720).images[0]
            image.save(out_path)
    del pipe
    torch.cuda.empty_cache()

def generate_sfx():
    print(f"--- Generating SFX with Stable Audio Open on {DEVICE} ---")
    pipe = StableAudioOpenPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16).to(DEVICE)

    os.makedirs(f"{ASSETS_DIR}/sfx", exist_ok=True)

    for s_id, _, sfx_prompt in SCENES:
        out_path = f"{ASSETS_DIR}/sfx/{s_id}.wav"
        if not os.path.exists(out_path):
            print(f"Generating SFX for: {s_id} -> {sfx_prompt}")
            # Stable Audio Open generates 44.1kHz audio
            audio = pipe(sfx_prompt, num_inference_steps=100, audio_length_in_s=12.0).audios[0]
            # audio is [channels, samples]
            audio_np = audio.T.cpu().numpy()
            wavfile.write(out_path, 44100, (audio_np * 32767).astype(np.int16)) 
            
    del pipe
    torch.cuda.empty_cache()

def generate_voiceover():
    print(f"--- Generating Voiceover with Full Bark on {DEVICE} ---")
    os.makedirs(f"{ASSETS_DIR}/voice", exist_ok=True)
    
    out_path = f"{ASSETS_DIR}/voice/voiceover_full.wav"
    if os.path.exists(out_path):
        return

    tts = pipeline("text-to-speech", model="suno/bark", device=DEVICE)
    
    lines = [l for l in VO_SCRIPT.split('\n') if l.strip()]
    full_audio = []
    
    print(f"Synthesizing {len(lines)} lines...")
    sampling_rate = 24000
    for line in lines:
        print(f"  Speaking: {line[:30]}...")
        output = tts(line, forward_params={"history_prompt": "v2/en_speaker_6"})
        audio_data = output["audio"]
        sampling_rate = output["sampling_rate"]
        
        silence = np.zeros(int(sampling_rate * 0.8))
        full_audio.append(audio_data.flatten())
        full_audio.append(silence)
        
    combined = np.concatenate(full_audio)
    wavfile.write(out_path, sampling_rate, (combined * 32767).astype(np.int16))
    del tts
    torch.cuda.empty_cache()

def generate_music():
    print(f"--- Generating Music (240s) with MusicGen-Large on {DEVICE} ---")
    os.makedirs(f"{ASSETS_DIR}/music", exist_ok=True)
    out_path = f"{ASSETS_DIR}/music/metro_theme.wav"
    
    if os.path.exists(out_path):
        return

    synthesiser = pipeline("text-to-audio", "facebook/musicgen-large", device=DEVICE)
    prompt = "eerie minimal synth drone, dark ambient, sci-fi horror soundtrack, slow pulsing deep bass, cinematic atmosphere, high quality"
    
    clips = []
    sr = 32000
    for i in range(8): # 8 * 30s = 240s
        print(f"Generating music chunk {i+1}/8...")
        output = synthesiser(prompt, forward_params={"max_new_tokens": 1500})
        clips.append(output["audio"][0].flatten())
        sr = output["sampling_rate"]
        
    combined = np.concatenate(clips, axis=0)
    wavfile.write(out_path, sr, (combined * 32767).astype(np.int16))
    del synthesiser
    torch.cuda.empty_cache()

if __name__ == "__main__":
    generate_images()
    generate_sfx()
    generate_voiceover()
    generate_music()
