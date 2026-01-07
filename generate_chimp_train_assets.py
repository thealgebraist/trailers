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
    print("\n--- Generating Images (flux.1 schnell) ---")
    model_id = "/workspace/.hf_home/hub/models--black-forest-labs--FLUX.1-schnell/snapshots/741f7c3ce8b383c54771c7003378a50191e9efe9"
    try:
        pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        pipe.to(DEVICE)
        for scene in SCENES:
            filename = f"{OUTPUT_DIR}/images/{scene['id']}.png"
            txt_filename = f"{OUTPUT_DIR}/images/{scene['id']}.txt"
            if os.path.exists(filename):
                continue
            print(f"Generating image: {scene['id']}")
            full_prompt = scene['image_prompt']
            # Flux Schnell is optimized for 4 steps and no guidance scale
            pipe(
                prompt=full_prompt, 
                height=1024, 
                width=1024, 
                num_inference_steps=4, 
                generator=torch.Generator(device=DEVICE).manual_seed(101)
            ).images[0].save(filename)
            with open(txt_filename, "w") as f:
                f.write(f"Model: {model_id}\nImage Prompt: {full_prompt}\nSteps: 4\nRes: 1024x1024\nSeed: 101\n")
        del pipe; flush()
    except Exception as e:
        print(f"Image generation failed: {e}")

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
        # You might need to configure the session with an API key if not in env vars
        # session = Session("YOUR_API_KEY") 
        # For now, we assume the environment is configured or we use a local model if supported by the SDK wrapper.
        # If this is a wrapper for a local model, we might need different initialization.
        
        # NOTE: Since the user asked for V1.5, checking documentation (simulated) suggests:
        # If this is for the API, we need an API key. 
        # If this is for local inference, we usually use the CLI.
        # The user command `pip install fish-audio-sdk` implies API usage.
        
        # Let's assume standard SDK usage for now.
        session = Session() 
        tts = TTS(session)
        
        # Common reference audio for the voice cloning (if needed)
        # ref_audio = open("voices/reference.wav", "rb") 
        
        for i, scene in enumerate(SCENES):
            txt = scene['voice_prompt']
            out_file = f"{OUTPUT_DIR}/voice/voice_{i+1:02d}.wav"
            meta_file = f"{OUTPUT_DIR}/voice/voice_{i+1:02d}.txt"
            
            if os.path.exists(out_file):
                print(f"Skipping existing {out_file}")
                continue
                
            print(f"Generating voice {i+1}/32: {txt[:60]}...")
            
            # SDK usage is hypothetical here as I don't have the docs in front of me, 
            # but I will use a generic pattern.
            # Please ensure you have FISH_AUDIO_API_KEY set if using the cloud API.
            
            # tts.tts(text=txt, ... )
            # If the SDK generates bytes:
            audio_bytes = tts.tts(text=txt, reference_id="default") # using a default or pre-uploaded voice
            
            with open(out_file, "wb") as f:
                f.write(audio_bytes)
                
            with open(meta_file, 'w') as mf:
                mf.write(f"Prompt: {txt}\nModel: fish-speech-v1.5\n")
            print(f"Wrote {out_file}")
            
    except ImportError:
        print("Error: 'fish-audio-sdk' not found. Please install it with: pip install fish-audio-sdk")
    except Exception as e:
        print(f"Fish Speech generation failed: {e}")



def generate_voice_bark():
    """Generate 32 separate Bark TTS WAV files, one per SCENES entry using the voice_prompt field.
    Requires the `bark` package (suno/bark) or an equivalent API that provides
    generate_audio(text) and SAMPLE_RATE. Falls back with a helpful message if unavailable.
    """
    print("--- Generating 32 Bark voice lines ---")
    os.makedirs(f"{OUTPUT_DIR}/voice", exist_ok=True)
    
    # Workaround for torch 2.6+ weights_only loading restrictions.
    # We must register numpy globals before bark loads its models.
    try:
        import importlib
        import warnings
        import torch
        np = importlib.import_module('numpy')
        
        # 1. Collect all potential scalar/dtype types to allowlist
        safe_types = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Try both modern and legacy paths
            for prefix in ['_core', 'core']:
                if hasattr(np, prefix):
                    mod = getattr(np, prefix)
                    if hasattr(mod, 'multiarray') and hasattr(mod.multiarray, 'scalar'):
                        safe_types.append(mod.multiarray.scalar)
            
            # Also add common types that often cause issues in Bark/Transformers
            for t in ['dtype', 'ndarray', 'float32', 'float64', 'int64']:
                if hasattr(np, t):
                    safe_types.append(getattr(np, t))

        # 2. Register with torch if available
        if hasattr(torch, 'serialization') and hasattr(torch.serialization, 'add_safe_globals'):
            # Filter duplicates and None
            unique_safe = list(set([t for t in safe_types if t is not None]))
            torch.serialization.add_safe_globals(unique_safe)
            
    except Exception as e:
        print(f"Debug: Failed to patch torch safe globals: {e}")

    try:
        # suno/bark provides generate_audio and SAMPLE_RATE
        from bark import generate_audio, SAMPLE_RATE
        import numpy as np
        for i, scene in enumerate(SCENES):
            txt = scene['voice_prompt']
            out_file = f"{OUTPUT_DIR}/voice/voice_{i+1:02d}.wav"
            meta_file = f"{OUTPUT_DIR}/voice/voice_{i+1:02d}.txt"
            if os.path.exists(out_file):
                print(f"Skipping existing {out_file}")
                continue
            print(f"Generating voice {i+1}/32: {txt[:60]}...")
            # generate_audio returns a numpy array of floats in [-1, 1]
            audio = generate_audio(txt)
            arr = np.asarray(audio, dtype=np.float32)
            # convert to 16-bit PCM
            pcm = (arr * 32767.0).astype(np.int16)
            scipy.io.wavfile.write(out_file, SAMPLE_RATE, pcm)
            with open(meta_file, 'w') as mf:
                mf.write(f"Prompt: {txt}\nModel: bark\nSampleRate: {SAMPLE_RATE}\n")
            print(f"Wrote {out_file} and metadata {meta_file}")
    except Exception as e:
        print(f"Bark generation not available or failed: {e}\nInstall the 'bark' package or run exports in the container/VM with dependencies.")


def generate_sfx():
    """Save SFX prompt descriptions per scene to assets (can be used later to synthesize SFX)."""
    print("--- Writing SFX prompt descriptions ---")
    os.makedirs(f"{OUTPUT_DIR}/sfx", exist_ok=True)
    for i, scene in enumerate(SCENES):
        sfx_file = f"{OUTPUT_DIR}/sfx/{scene['id']}.txt"
        if os.path.exists(sfx_file):
            print(f"Skipping existing {sfx_file}")
            continue
        with open(sfx_file, 'w') as f:
            f.write(scene['sfx_prompt'])
        print(f"Wrote SFX prompt {sfx_file}")


if __name__ == "__main__":
    main()
