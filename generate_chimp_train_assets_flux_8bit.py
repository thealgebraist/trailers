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
from diffusers import FluxPipeline
import subprocess
from transformers import AutoProcessor

OUTPUT_DIR = "assets_chimp_train"
os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/voice", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/sfx", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/music", exist_ok=True)

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

SCENE_DURATION = 4.0

# Build scene entries with distinct prompts for image, voice, and sfx so each modality is tailored to the storyline
SCENES = []
for i, voice_line in enumerate(VO_SCRIPTS):
    sid = f"{i+1:02d}_scene"
    # Alternate camera framing for visual variety
    framing = "close-up" if i % 3 == 0 else ("wide-angle" if i % 3 == 1 else "medium shot")
    image_prompt = (f"Photorealistic {framing} cinematic still of the scene: {voice_line} "
                    "+ extremely detailed, realistic textures, cinematic lighting, 1024x1024, film grain, movie still, no cartoons")
    # SFX prompts derived by keywords
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
    scene = {"id": sid, "image_prompt": image_prompt, "voice_prompt": voice_line, "sfx_prompt": sfx}
    SCENES.append(scene)

MUSIC_THEMES = [
    { "id": "theme_fun", "prompt": "Upbeat, playful orchestral score with jungle percussion and cheerful melodies. High quality." }
]

def flush():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def generate_images():
    print("\n--- Generating Images (flux.1 schnell 8-bit) ---")
    base_model_id = "black-forest-labs/FLUX.1-schnell"
    pruna_path = "/workspace/.hf_home/hub/models--PrunaAI--FLUX.1-schnell-8bit/snapshots/51f676f44a2720848b5451bab4459a538367bdff"
    
    try:
        from optimum.quanto import freeze, qfloat8, quantize
        from diffusers import FluxPipeline, FluxTransformer2DModel
        from transformers import T5EncoderModel
        
        # Register classes for safe unpickling in PyTorch 2.6+
        if hasattr(torch.serialization, 'add_safe_globals'):
            torch.serialization.add_safe_globals([FluxTransformer2DModel, T5EncoderModel])
        
        print(f"Loading base components from {base_model_id}...")
        pipe = FluxPipeline.from_pretrained(
            base_model_id, 
            transformer=None, 
            text_encoder_2=None, 
            dtype=torch.bfloat16
        )
        
        # 1. Load and Quantize Transformer
        print("Loading and quantizing transformer...")
        transformer = FluxTransformer2DModel.from_pretrained(base_model_id, subfolder="transformer", dtype=torch.bfloat16)
        quantize(transformer, weights=qfloat8)
        freeze(transformer)
        
        transformer_weights_path = os.path.join(pruna_path, "transformer.pt")
        if os.path.exists(transformer_weights_path):
            print(f"Loading transformer weights from {transformer_weights_path}...")
            # If PrunaAI provided the full state dict
            state_dict = torch.load(transformer_weights_path, map_location="cpu", weights_only=False)
            transformer.load_state_dict(state_dict)
        else:
            print(f"Warning: {transformer_weights_path} not found. Using base transformer.")
            
        # 2. Load and Quantize Text Encoder 2 (T5)
        print("Loading and quantizing text_encoder_2...")
        text_encoder_2 = T5EncoderModel.from_pretrained(base_model_id, subfolder="text_encoder_2", dtype=torch.bfloat16)
        quantize(text_encoder_2, weights=qfloat8)
        freeze(text_encoder_2)
        
        te2_weights_path = os.path.join(pruna_path, "text_encoder_2.pt")
        if os.path.exists(te2_weights_path):
            print(f"Loading text_encoder_2 weights from {te2_weights_path}...")
            state_dict_te2 = torch.load(te2_weights_path, map_location="cpu", weights_only=False)
            text_encoder_2.load_state_dict(state_dict_te2)
        else:
            print(f"Warning: {te2_weights_path} not found. Using base text_encoder_2.")

        # 3. Assemble Pipeline
        pipe.transformer = transformer
        pipe.text_encoder_2 = text_encoder_2
        pipe.to(DEVICE)
        
        for scene in SCENES:
            filename = f"{OUTPUT_DIR}/images/{scene['id']}.png"
            txt_filename = f"{OUTPUT_DIR}/images/{scene['id']}.txt"
            if os.path.exists(filename):
                continue
            print(f"Generating image: {scene['id']}")
            full_prompt = scene['image_prompt']
            pipe(
                prompt=full_prompt, 
                height=1024, 
                width=1024, 
                num_inference_steps=4, 
                generator=torch.Generator(device=DEVICE).manual_seed(101)
            ).images[0].save(filename)
            with open(txt_filename, "w") as f:
                f.write(f"Model: PrunaAI 8-bit (path: {pruna_path})\nPrompt: {full_prompt}\n")
        del pipe; flush()
    except Exception as e:
        print(f"Image generation failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    import sys
    if "images" in sys.argv:
        generate_images()
        return
    if "voice" in sys.argv:
        generate_voice_fishspeech()
        return
    # Add calls for sfx, music as implemented

def generate_voice_fishspeech():
    """Generate 32 separate voice lines using Fish Speech V1.5 via SDK.
    Requires `fish-audio-sdk` installed.
    """
    print("--- Generating 32 Fish Speech voice lines ---")
    os.makedirs(f"{OUTPUT_DIR}/voice", exist_ok=True)
    
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
            audio_bytes = tts.tts(text=txt, reference_id="default")
            
            with open(out_file, "wb") as f:
                f.write(audio_bytes)
                
            with open(meta_file, 'w') as mf:
                mf.write(f"Prompt: {txt}\nModel: fish-speech-v1.5\n")
            print(f"Wrote {out_file}")
            
    except ImportError:
        print("Error: 'fish-audio-sdk' not found. Please install it with: pip install fish-audio-sdk")
    except Exception as e:
        print(f"Fish Speech generation failed: {e}")

if __name__ == "__main__":
    main()
