import torch
import scipy.io.wavfile
import os
import gc
import subprocess
import re
import numpy as np
import PIL.Image
if not hasattr(PIL.Image, "ANTIALIAS"): PIL.Image.ANTIALIAS = PIL.Image.LANCZOS
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor, BarkModel, MusicgenForConditionalGeneration

# --- Configuration ---
OUTPUT_DIR = "assets_chimp_band"
os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/voice", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/sfx", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/music", exist_ok=True)

if torch.cuda.is_available(): DEVICE = "cuda"
else: DEVICE = "cpu"

# --- Scene Definitions ---
# 32 Scenes: Alternating between Band (Odd) and Solo (Even)
# Instruments: Banjo, Bongos, Cowbell, Bass, More Bongos
INSTRUMENTS = ["banjo", "bongos", "cowbell", "bass guitar", "bongos", "bongos", "bongos", "bongos"] # Weighted for bongos
SCENES = []

for i in range(32):
    sid = f"{i+1:02d}_scene"
    
    if i % 2 == 0:
        # BAND SHOT (Even index 0, 2, 4... becomes Scene 1, 3, 5...)
        # Wait, usually 1-based index. Let's say Scene 1 (i=0) is Band.
        # Request: "switch between showing a single chimp playing his instrument and then back to the whole band"
        # Let's do: Odd = Band, Even = Solo
        
        visual_prompt = "A minimalist photo of a chimp band on a plain white background. One plays banjo, one plays bongos, one plays cowbell, one plays bass. High contrast, studio lighting, silly but stoic."
        audio_prompt = "A rhythmic bongo drum loop with a banjo strumming and a cowbell keeping time. [music]"
        action_desc = "The whole band plays together."
        
    else:
        # SOLO SHOT
        # Cycle through instruments
        inst_idx = (i // 2) % len(INSTRUMENTS)
        inst = INSTRUMENTS[inst_idx]
        
        visual_prompt = f"A minimalist photo of a single chimp playing the {inst} on a plain white background. Close up, intense focus, studio lighting."
        
        if inst == "bongos":
            audio_prompt = "Fast, energetic bongo drum solo. [music]"
        elif inst == "banjo":
            audio_prompt = "A quick, twangy banjo riff. [music]"
        elif inst == "cowbell":
            audio_prompt = "A rhythmic cowbell pattern. [music]"
        elif inst == "bass guitar":
            audio_prompt = "A funky bass guitar groove. [music]"
        else:
             audio_prompt = "Bongo drums playing. [music]"

        action_desc = f"A chimp performs a {inst} solo."

    scene = {
        "id": sid,
        "visual": visual_prompt,
        "audio_prompt": audio_prompt,
        "description": action_desc
    }
    SCENES.append(scene)


def flush():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def generate_images():
    print("--- Generating Images (SDXL Lightning) ---")
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
            
            pipe(
                prompt=prompt, 
                guidance_scale=0.0, 
                num_inference_steps=4, 
                generator=torch.Generator(device="cpu").manual_seed(101 + int(scene['id'].split('_')[0]))
            ).images[0].save(fname)
            
        del pipe; flush()
    except Exception as e:
        print(f"Image generation failed: {e}")

def generate_scene_audio():
    print("--- Generating Scene Audio (MusicGen) ---")
    try:
        processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").to(DEVICE)
        
        sample_rate = 32000 # MusicGen default
        
        for scene in SCENES:
            fname = f"{OUTPUT_DIR}/sfx/{scene['id']}.wav"
            if os.path.exists(fname): continue
            
            print(f"Generating Audio for {scene['id']}: {scene['audio_prompt']}")
            
            inputs = processor(
                text=[scene['audio_prompt']],
                padding=True,
                return_tensors="pt",
            ).to(DEVICE)
            
            with torch.no_grad():
                # Generate ~5 seconds (256 tokens is roughly 5s depending on config, let's use max_new_tokens)
                # MusicGen generates 50 tokens/sec approx. So 250 tokens ~ 5s
                audio_values = model.generate(**inputs, max_new_tokens=256)
                
            # Save
            audio_data = audio_values[0, 0].cpu().numpy()
            scipy.io.wavfile.write(fname, rate=sample_rate, data=audio_data)
            
        del model; del processor; flush()
        
    except Exception as e:
        print(f"Scene audio generation failed: {e}")

def generate_soundtrack():
    print("--- Generating 4 Bongo Rich Songs (120s each with MusicGen) ---")
    
    try:
        processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").to(DEVICE)
        sample_rate = 32000
        
        # 4 distinct song themes/structures
        # Note: MusicGen max duration is typically 30s. We will generate 4x30s chunks.
        songs = [
            {
                "filename": "soundtrack_1.wav",
                "prompts": ["Fast paced bongo drums solo with enthusiastic shouting. High quality."] * 4
            },
            {
                "filename": "soundtrack_2.wav",
                "prompts": ["A funky bass line with heavy bongo percussion and cowbell. High quality."] * 4
            },
            {
                "filename": "soundtrack_3.wav",
                "prompts": ["Wild chaotic bongo frenzy with a banjo strumming rapidly. High quality."] * 4
            },
            {
                "filename": "soundtrack_4.wav",
                "prompts": ["Deep rhythmic tribal bongos with a steady cowbell beat. High quality."] * 4
            }
        ]

        for song in songs:
            fname = f"{OUTPUT_DIR}/music/{song['filename']}"
            if os.path.exists(fname): 
                print(f"Skipping {fname}")
                continue

            print(f"Generating {song['filename']}...")
            full_audio = []
            
            for i, p in enumerate(song["prompts"]):
                print(f"  Segment {i+1}/4: {p}")
                inputs = processor(
                    text=[p],
                    padding=True,
                    return_tensors="pt",
                ).to(DEVICE)
                
                with torch.no_grad():
                    # ~30 seconds = 1500 tokens (50 tokens/sec)
                    audio_values = model.generate(**inputs, max_new_tokens=1500)
                    
                full_audio.append(audio_values[0, 0].cpu().numpy())
            
            combined = np.concatenate(full_audio)
            scipy.io.wavfile.write(fname, rate=sample_rate, data=combined)
        
        del model; del processor; flush()

    except Exception as e:
        print(f"Soundtrack generation failed: {e}")

if __name__ == "__main__":
    import sys
    # Default to all
    generate_images()
    generate_scene_audio()
    generate_soundtrack()
