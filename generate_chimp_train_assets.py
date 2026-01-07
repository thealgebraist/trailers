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
from diffusers import StableDiffusionPipeline
import subprocess
from transformers import AutoProcessor, BarkModel

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
    model_id = "flux/flux.1-schnell"
    try:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        pipe.to(DEVICE)
        for scene in SCENES:
            filename = f"{OUTPUT_DIR}/images/{scene['id']}.png"
            txt_filename = f"{OUTPUT_DIR}/images/{scene['id']}.txt"
            if os.path.exists(filename):
                continue
            print(f"Generating image: {scene['id']}")
            full_prompt = scene['image_prompt']
            pipe(prompt=full_prompt, height=1024, width=1024, guidance_scale=7.5, num_inference_steps=16, generator=torch.Generator(device=DEVICE).manual_seed(101)).images[0].save(filename)
            with open(txt_filename, "w") as f:
                f.write(f"Model: {model_id}\nImage Prompt: {full_prompt}\nSteps: 16\nRes: 1024x1024\nSeed: 101\n")
        del pipe; flush()
    except Exception as e:
        print(f"Image generation failed: {e}")

def main():
    import sys
    if "images" in sys.argv:
        generate_images()
        return
    if "voice" in sys.argv:
        try:
            generate_voice_bark()
        except NameError:
            print("Function generate_voice_bark is not defined. Please ensure it is present in the script.")
        return
    # Add calls for sfx, music as implemented

def generate_voice_vibevoice():
    print("--- Generating 120s VibeVoice voiceover ---")
    output_path = f"{OUTPUT_DIR}/voice/voiceover_full.wav"
    full_text = " ".join([scene['visual'] for scene in SCENES])
    try:
        subprocess.run([
            "python3", "-m", "F5TTS_MLX.generate",
            "--text", full_text,
            "--output", output_path,
            "--duration", "120",
            "--model_path", "VibeVoiceModel"
        ], check=True)
    except Exception as e:
        print(f"VibeVoice generation failed: {e}")


def generate_voice_bark():
    """Generate 32 separate Bark TTS WAV files, one per SCENES entry using the voice_prompt field.
    Requires the `bark` package (suno/bark) or an equivalent API that provides
    generate_audio(text) and SAMPLE_RATE. Falls back with a helpful message if unavailable.
    """
    print("--- Generating 32 Bark voice lines ---")
    os.makedirs(f"{OUTPUT_DIR}/voice", exist_ok=True)
    try:
        # Workaround for torch 2.6+ weights_only loading restrictions when Bark checkpoints
        # include numpy scalar globals. Allowlist the numpy scalar for torch.serialization
        # so torch.load used by Bark can succeed (only do this if model files are trusted).
        try:
            import importlib
            import warnings
            np = importlib.import_module('numpy')
            
            scalar_type = None
            
            # 1. Try modern numpy._core first (Numpy 2.0+)
            # We wrap in catch_warnings just in case, though _core shouldn't warn.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if hasattr(np, '_core') and hasattr(np._core, 'multiarray') and hasattr(np._core.multiarray, 'scalar'):
                    scalar_type = np._core.multiarray.scalar
            
            # 2. Fallback to legacy numpy.core if not found (Numpy < 2.0)
            if scalar_type is None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if hasattr(np, 'core') and hasattr(np.core, 'multiarray') and hasattr(np.core.multiarray, 'scalar'):
                        scalar_type = np.core.multiarray.scalar

            # 3. Register with torch
            if scalar_type is not None and hasattr(torch, 'serialization') and hasattr(torch.serialization, 'add_safe_globals'):
                torch.serialization.add_safe_globals([scalar_type])
                
        except Exception as e:
            # Silence errors during this patch attempt to allow script to proceed
            print(f"Debug: Failed to patch torch safe globals: {e}")
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
