import torch
import scipy.io.wavfile
import os
import gc
import random
import numpy as np
import PIL.Image
if not hasattr(PIL.Image, "ANTIALIAS"): PIL.Image.ANTIALIAS = PIL.Image.LANCZOS
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor, MusicgenForConditionalGeneration

# --- Configuration ---
OUTPUT_DIR = "assets_chimp_band_64"
os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/voice", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/sfx", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/music", exist_ok=True)

if torch.cuda.is_available(): DEVICE = "cuda"
else: DEVICE = "cpu"

# --- Scene Definitions ---
# Pool of instruments to mix with Bongos
RIDICULOUS_INSTRUMENTS = [
    "banjo", "cowbell", "bass guitar", "kazoo", "tuba", 
    "accordion", "triangle", "keytar", "theremin", "electric violin",
    "slide whistle", "didgeridoo"
]

SCENES = []

# Generate 64 Scenes
for i in range(64):
    sid = f"{i+1:02d}_scene"
    
    # Logic: Always Bongos. 
    # 40% Chance of Solo Bongos
    # 30% Chance of Duo (Bongos + 1)
    # 30% Chance of Trio (Bongos + 2)
    
    roll = random.random()
    if roll < 0.4:
        # Solo Bongo
        instruments = ["bongos"]
        visual_prompt = "A minimalist photo of a single chimp playing bongo drums with intense focus. Plain white background, studio lighting, high contrast."
        audio_prompt = "A fast, energetic bongo drum solo. High quality."
        desc = "Bongo Solo"
    elif roll < 0.7:
        # Duo
        extra = random.choice(RIDICULOUS_INSTRUMENTS)
        instruments = ["bongos", extra]
        visual_prompt = f"A minimalist photo of two chimps on a plain white background. One plays bongos, the other plays {extra}. Serious expressions, studio lighting."
        audio_prompt = f"Rhythmic bongo drums accompanying a {extra} melody. High quality."
        desc = f"Bongos and {extra}"
    else:
        # Trio (Max 3 instruments)
        extras = random.sample(RIDICULOUS_INSTRUMENTS, 2)
        instruments = ["bongos"] + extras
        visual_prompt = f"A minimalist photo of a three-chimp band on a plain white background. Playing bongos, {extras[0]}, and {extras[1]}. Silly but stoic, studio lighting."
        audio_prompt = f"A chaotic but rhythmic jam session with bongos, {extras[0]}, and {extras[1]}. High quality."
        desc = f"Bongos, {extras[0]}, and {extras[1]}"

    SCENES.append({
        "id": sid,
        "visual": visual_prompt,
        "audio_prompt": audio_prompt,
        "description": desc
    })

def flush():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def generate_images():
    print(f"--- Generating 64 Images (SDXL Lightning) ---")
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
            
            print(f"Generating image: {scene['id']} ({scene['description']})")
            
            pipe(
                prompt=scene['visual'], 
                guidance_scale=0.0, 
                num_inference_steps=8, 
                generator=torch.Generator(device="cpu").manual_seed(100 + int(scene['id'].split('_')[0]))
            ).images[0].save(fname)
            
        del pipe; flush()
    except Exception as e:
        print(f"Image generation failed: {e}")

def generate_scene_samples():
    print("--- Generating 64 Scene Audio Samples (8s each via MusicGen) ---")
    try:
        processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").to(DEVICE)
        sample_rate = 32000
        
        for scene in SCENES:
            fname = f"{OUTPUT_DIR}/sfx/{scene['id']}.wav"
            if os.path.exists(fname): continue
            
            print(f"Generating Sample for {scene['id']}: {scene['audio_prompt']}")
            
            inputs = processor(
                text=[scene['audio_prompt']],
                padding=True,
                return_tensors="pt",
            ).to(DEVICE)
            
            with torch.no_grad():
                # MusicGen ~50 tokens/sec. 8 seconds ~= 400 tokens.
                audio_values = model.generate(**inputs, max_new_tokens=400)
                
            audio_data = audio_values[0, 0].cpu().numpy()
            scipy.io.wavfile.write(fname, rate=sample_rate, data=audio_data)
            
        del model; del processor; flush()
        
    except Exception as e:
        print(f"Scene audio generation failed: {e}")

def generate_soundtracks():
    print("--- Generating 4 Bongo Rich Songs (120s each via MusicGen) ---")
    
    try:
        processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").to(DEVICE)
        sample_rate = 32000
        
        songs = [
            {
                "filename": "soundtrack_1.wav",
                "prompts": ["Fast paced tribal bongo drum solo with enthusiastic percussion. High quality."] * 4
            },
            {
                "filename": "soundtrack_2.wav",
                "prompts": ["A funky bass line with heavy bongo percussion and cowbell rhythm. High quality."] * 4
            },
            {
                "filename": "soundtrack_3.wav",
                "prompts": ["Wild chaotic bongo frenzy with a banjo strumming rapidly. High quality."] * 4
            },
            {
                "filename": "soundtrack_4.wav",
                "prompts": ["Deep rhythmic hypnotic bongos with a steady tuba bassline. High quality."] * 4
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
                    # MusicGen limit is usually 30s (approx 1500 tokens)
                    audio_values = model.generate(**inputs, max_new_tokens=1500)
                    
                full_audio.append(audio_values[0, 0].cpu().numpy())
            
            combined = np.concatenate(full_audio)
            scipy.io.wavfile.write(fname, rate=sample_rate, data=combined)
        
        del model; del processor; flush()

    except Exception as e:
        print(f"Soundtrack generation failed: {e}")

if __name__ == "__main__":
    generate_images()
    generate_scene_samples()
    generate_soundtracks()