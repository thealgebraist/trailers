import torch
import scipy.io.wavfile
import os
import subprocess
import re
import numpy as np
from dalek.core import get_device, flush, ensure_dir
from dalek.vision import load_sdxl_lightning, generate_image
from diffusers import StableAudioPipeline
import ChatTTS

# --- Configuration ---
OUTPUT_DIR = "assets_chimp_train"
ensure_dir(f"{OUTPUT_DIR}/images")
ensure_dir(f"{OUTPUT_DIR}/voice")
ensure_dir(f"{OUTPUT_DIR}/sfx")
ensure_dir(f"{OUTPUT_DIR}/music")

DEVICE = get_device()
print(f"Using device: {DEVICE}")

# --- Narrative & Visual Data ---
VO_SCRIPTS = [
    "Charlie the chimp sits in his cozy jungle hut, dreaming of a glowing golden banana.",
    "He can almost taste the sweetness as he imagines the perfect fruit.",
    "Today is the day; he packs his small bag and prepares for a grand journey.",
    "Charlie arrives at the jungle train station, where the steam engine huffs and puffs.",
    "He stands on the platform, his ticket held tightly in his furry hand.",
    "The whistle blows, and Charlie knows his adventure is finally beginning.",
    "He climbs aboard the wooden carriage and finds a comfortable seat.",
    "The train starts to move, clicking and clacking along the iron rails.",
    "Charlie sits quietly, watching the jungle landscape begin to move.",
    "He presses his face against the cool glass of the window, mesmerized.",
    "Tall trees and rushing rivers blur into a beautiful green streak.",
    "The rhythm of the train lulls him into a peaceful, expectant state.",
    "Finally, the train slows down as it pulls into the distant station.",
    "Charlie hops off the train, looking around at the exciting new place.",
    "He knows the great banana market is just through the nearby woods.",
    "He enters the deep, lush forest, where sunlight filters through the canopy.",
    "Every rustle in the leaves makes him think he is getting closer.",
    "He walks with a steady pace, driven by the thought of that golden banana.",
    "At last, he reaches the bustling banana market, run by friendly chimps.",
    "He searches through the stalls until he sees it: the perfect golden banana.",
    "He holds the banana high, his heart filled with pure, simple joy.",
    "As evening falls, the forest turns into a landscape of deep blues and shadows.",
    "Charlie walks back through the trees, the jungle alive with night sounds.",
    "The moon rises high, lighting his path as he carries his treasure.",
    "He reaches the station at night, the platform quiet under the glowing lamps.",
    "He waits for the late-night train, his golden banana tucked safely away.",
    "The distant light of the locomotive appears, cutting through the darkness.",
    "Back on the train, Charlie watches the moonlight reflect off the trees.",
    "The carriage is dim and peaceful as the train carries him back home.",
    "He is tired but happy, resting his head against the wooden seat.",
    "At last, Charlie is back in his own jungle bed, the journey complete.",
    "He falls asleep with a smile, dreaming of his next big adventure."
]

SCENE_PROMPTS = [
    "A lone chimp in a cozy jungle hut, sitting on a wooden stool, deep in thought, thinking about a glowing golden banana.",
    "Close-up of the same chimp's face, eyes closed, dreaming of a perfect banana.",
    "The same chimp packing a small burlap sack in his jungle hut.",
    "The same chimp walking towards a jungle train station with a steam locomotive.",
    "The same chimp standing on a wooden train platform, holding a train ticket.",
    "The same chimp looking at the approaching steam train.",
    "The same chimp sitting inside a vintage wooden train carriage.",
    "The same chimp looking out of the train window at the jungle passing by.",
    "View from the train window: lush jungle trees blurring past.",
    "The same chimp pressing his face against the train window glass.",
    "The same chimp watching a river from the train window.",
    "The same chimp relaxing in his train seat.",
    "The same chimp stepping off the train onto a remote jungle station platform.",
    "The same chimp looking at a signpost pointing towards 'Banana Market'.",
    "The same chimp walking on a path through a dense, sunlit forest.",
    "The same chimp looking up at the tall forest canopy.",
    "The same chimp crossing a small stream in the forest.",
    "The same chimp seeing the market in the distance.",
    "The same chimp at a bustling banana market run by other chimps.",
    "The same chimp inspecting a huge, glowing golden banana at a market stall.",
    "The same chimp holding the golden banana triumphantly.",
    "The same chimp walking back through the trees, the jungle alive with night sounds.",
    "The same chimp in the forest at night, holding his golden banana, moonlight filtering through trees.",
    "The same chimp navigating the dark forest, fireflies around him.",
    "The same chimp at the jungle train station at night, waiting under a glowing lamp.",
    "The same chimp sitting on a bench at the night station, banana by his side.",
    "The same chimp watching the headlights of the night train arrive.",
    "The same chimp inside the dim, peaceful train carriage at night.",
    "The same chimp looking at the moon through the train window.",
    "The same chimp resting his head against the wooden seat, looking happy.",
    "The same chimp back in his jungle hut at night, tucked into bed.",
    "The same chimp asleep in his bed with the golden banana on a table nearby."
]

SCENES = []
for i in range(32):
    sid = f"{i+1:02d}_scene"
    SCENES.append({
        "id": sid, 
        "visual": SCENE_PROMPTS[i], 
        "voice_prompt": VO_SCRIPTS[i]
    })

MUSIC_THEMES = [
    { "id": "theme_fun", "prompt": "Upbeat, playful orchestral score with jungle percussion and cheerful melodies. High quality." }
]

def apply_trailer_voice_effect(file_path):
    temp_path = file_path.replace(".wav", "_temp.wav")
    filter_complex = "lowshelf=g=5:f=100,acompressor=threshold=-12dB:ratio=4:makeup=4dB"
    cmd = ["ffmpeg", "-y", "-i", file_path, "-af", filter_complex, temp_path]
    try: 
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.replace(temp_path, file_path)
    except Exception as e: print(f"Failed effect: {e}")

def generate_voice():
    print("--- Generating 32 ChatTTS voice lines ---")
    try:
        chat = ChatTTS.Chat()
        chat.load(compile=False)
        for i, scene in enumerate(SCENES):
            txt = scene['voice_prompt']
            out_file = f"{OUTPUT_DIR}/voice/voice_{i+1:02d}.wav"
            if os.path.exists(out_file): continue
            print(f"Generating voice {i+1}/32: {txt[:60]}...")
            wavs = chat.infer([txt], use_decoder=True)
            if wavs:
                audio_array = np.array(wavs[0]).flatten()
                scipy.io.wavfile.write(out_file, 24000, audio_array)
                apply_trailer_voice_effect(out_file)
        del chat; flush()
    except Exception as e: print(f"ChatTTS generation failed: {e}")

def generate_images():
    print("--- Generating Images (SDXL Lightning, 8 steps) ---")
    try:
        pipe = load_sdxl_lightning()
        for scene in SCENES:
            fname = f"{OUTPUT_DIR}/images/{scene['id']}.png"
            if os.path.exists(fname): continue
            print(f"Generating image: {scene['id']}")
            prompt = f"{scene['visual']}, minimalist style, cinematic lighting, no humans, only animals, 8k photorealistic"
            image = generate_image(pipe, prompt, steps=8, guidance=0.0, seed=101 + int(scene['id'].split('_')[0]))
            image.save(fname)
        del pipe; flush()
    except Exception as e: print(f"Image generation failed: {e}")

def generate_audio():
    print("\n--- Generating Music & SFX (Stable Audio Open) ---")
    try:
        model_id = "stabilityai/stable-audio-open-1.0"
        pipe = StableAudioPipeline.from_pretrained(model_id, dtype=torch.float32)
        if DEVICE == "cuda": pipe.enable_model_cpu_offload()
        else: pipe.to(DEVICE)
        neg = "low quality, noise, distortion, artifacts, fillers, talking"
        for theme in MUSIC_THEMES:
            filename = f"{OUTPUT_DIR}/music/{theme['id']}.wav"
            if os.path.exists(filename): continue
            print(f"Generating Music: {theme['id']}")
            audio = pipe(prompt=theme['prompt'], negative_prompt=neg, num_inference_steps=100, audio_end_in_s=120.0).audios[0]
            scipy.io.wavfile.write(filename, rate=44100, data=audio.cpu().numpy().T)
        del pipe; flush()
    except Exception as e: print(f"Audio generation failed: {e}")

if __name__ == "__main__":
    import sys
    if "voice" in sys.argv: generate_voice(); sys.exit(0)
    if "images" in sys.argv: generate_images(); sys.exit(0)
    generate_images(); generate_voice(); generate_audio()