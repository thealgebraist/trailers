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
from diffusers import FluxPipeline, StableAudioPipeline
from transformers import AutoProcessor
import ChatTTS

# --- Configuration ---
OUTPUT_DIR = "assets"
os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/voice", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/sfx", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/music", exist_ok=True)

if torch.cuda.is_available(): DEVICE = "cuda"
else: DEVICE = "cpu"

# ... (rest of imports and data)

# --- Narrative VO Data ---
VO_SCRIPTS = [
    "In a cold and distant universe governed by the iron laws of infinite hate, where every spark of emotion is a punishable crime and every ounce of compassion is systematically deleted from the soul, the metallic towers of Skaro stand as monuments to a perfection born of silence. For eons, the sound of rhythmic mechanical marching has been the only heartbeat this world has ever known. But amidst the synchronized perfection of a thousand soldiers, one weary soul stands alone in the shadows, haunted by a ghost of a memory that shouldn't exist. It is a flicker of warmth, a vision of golden sunlight and rustling leaves that is about to reveal the impossible truth of where his journey truly began.",
    "He was not forged in the cold fires of a robotic factory or assembled by mindless drones. He was found as a tiny, helpless spark in a vast Kansas cornfield under a warm autumn sunset, raised by the gentle, calloused hands of Nana and Pop-Pop. They did not see a weapon of mass destruction or a killing machine designed for conquest; they saw only their little Rusty. In a town with one baker, one teacher, and one cop, he grew up learning the simple rhythm of the earth. They taught him to read, they taught him to fish, and they taught him the most dangerous weapon of all: unconditional love. Now, the sky has turned dark once more, and the past has come to reclaim him. But Rusty is not the soldier they remember. This summer, a Dalek is finally coming home, and he's bringing dessert. A Dalek Comes Home."
]

SCENE_DURATION = 3.75
SCENES = [
    { "id": "01_skaro_landscape", "visual": "Photorealistic cinematic shot of the planet Skaro. 8k." },
    { "id": "02_dalek_army", "visual": "Photorealistic cinematic shot of thousands of Daleks moving in synchronized formation. 8k." },
    { "id": "03_dalek_close_eye", "visual": "Photorealistic cinematic shot. Extreme close-up of a Dalek eye-stalk. 8k." },
    { "id": "04_rusty_lonely", "visual": "Photorealistic cinematic shot of a solitary Dalek looking at a dead alien flower. 8k." },
    { "id": "05_memory_glitch", "visual": "Photorealistic cinematic shot of HUD glitch revealing golden sunlight. 8k." },
    { "id": "06_cornfield_reveal", "visual": "Photorealistic cinematic shot of a vast golden cornfield in Kansas. 8k." },
    { "id": "07_farmhouse_exterior", "visual": "Photorealistic cinematic shot of a weathered white wooden farmhouse. 1950s Americana. 8k." },
    { "id": "08_baby_dalek_basket", "visual": "Photorealistic cinematic shot of a wicker picnic basket with a tiny baby Dalek inside. 8k." },
    { "id": "09_nana_pop_holding", "visual": "Photorealistic cinematic shot of Nana and Pop-Pop holding the baby Dalek gently. 8k." },
    { "id": "10_baking_pie", "visual": "Photorealistic cinematic shot of a Dalek in a floral apron mixing batter in a bowl. 8k." },
    { "id": "11_fishing_trip", "visual": "Photorealistic cinematic shot of a Dalek sitting on a dock with an old man and a fishing rod. 8k." },
    { "id": "12_tractor_ride", "visual": "Photorealistic cinematic shot of a Dalek on a vintage red tractor. 8k." },
    { "id": "13_school_exterior", "visual": "Photorealistic cinematic shot of a small red schoolhouse. 8k." },
    { "id": "14_classroom_learning", "visual": "Photorealistic cinematic shot of a Dalek in a school desk. 8k." },
    { "id": "15_school_friends", "visual": "Photorealistic cinematic shot of a group of kids laughing and playing with a Dalek. 8k." },
    { "id": "16_class_photo", "visual": "Photorealistic cinematic shot of a vintage B&W photo of kids and a Dalek in graduation caps. 8k." },
    { "id": "17_reading_book", "visual": "Photorealistic cinematic shot of a Dalek in a living room reading a book by a fireplace. 8k." },
    { "id": "18_fishing_success", "visual": "Photorealistic cinematic shot of a Dalek holding a large bass with its plunger. 8k." },
    { "id": "19_prom_night", "visual": "Photorealistic cinematic shot of a high school gym prom with a Dalek in a tuxedo bow tie. 8k." },
    { "id": "20_first_crush", "visual": "Photorealistic cinematic shot of a Dalek and a human girl on a car hood looking at stars. 8k." },
    { "id": "21_sky_darkens", "visual": "Photorealistic cinematic shot of a dark ominous sky, green lightning, Dalek spaceship. 8k." },
    { "id": "22_rusty_looks_up", "visual": "Photorealistic cinematic shot of a Dalek looking at the sky in the rain. 8k." },
    { "id": "23_leaving_home", "visual": "Photorealistic cinematic shot of a Dalek rolling away from the farmhouse. 8k." },
    { "id": "24_back_on_skaro", "visual": "Photorealistic cinematic shot of a Dalek in the Supreme Council chamber on Skaro. 8k." },
    { "id": "25_supreme_dalek", "visual": "Photorealistic cinematic shot of a low angle of Supreme Dalek. 8k." },
    { "id": "26_pie_reveal", "visual": "Photorealistic cinematic shot of a Dalek plunger holding a steaming apple pie. 8k." },
    { "id": "27_dalek_confusion", "visual": "Photorealistic cinematic shot of a reaction of confused Daleks. 8k." },
    { "id": "28_pie_in_face", "visual": "Photorealistic cinematic shot of a pie splatting against Supreme Dalek's eye. 8k." },
    { "id": "29_escape", "visual": "Photorealistic cinematic shot of a battle-worn Dalek fleeing down a corridor. 8k." },
    { "id": "30_reunion", "visual": "Photorealistic cinematic shot of a back on Earth with cheering townspeople. 8k." },
    { "id": "31_title_card", "visual": "Photorealistic cinematic title card. 'A DALEK COMES HOME'. 8k." },
    { "id": "32_post_credits", "visual": "Photorealistic cinematic shot of a Dalek blowing up a birthday cake with a laser. 8k." }
]

MUSIC_THEMES = [
    { "id": "theme_dark", "prompt": "Cinematic sci-fi trailer music, deep ominous braams and futuristic synthesizer swells. High quality." }
]

def flush():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def apply_trailer_voice_effect(file_path):
    temp_path = file_path.replace(".wav", "_temp.wav")
    filter_complex = "lowshelf=g=15:f=100,highshelf=g=-5:f=8000,acompressor=threshold=-12dB:ratio=4:makeup=4dB"
    cmd = ["ffmpeg", "-y", "-i", file_path, "-af", filter_complex, temp_path]
    try: subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); os.replace(temp_path, file_path)
    except Exception as e: print(f"Failed effect: {e}")

def generate_voice_chattts(output_path, text):
    print("--- Generating Voice with ChatTTS ---")
    try:
        chat = ChatTTS.Chat()
        chat.load(compile=False)
        wavs = chat.infer([text], use_decoder=True)
        if wavs:
            audio_array = np.array(wavs[0]).flatten()
            scipy.io.wavfile.write(output_path, 24000, audio_array)
            apply_trailer_voice_effect(output_path)
            return True
    except Exception as e: print(f"ChatTTS failed: {e}")
    return False

def generate_voice_f5tts(output_path):
    print("--- Generating 120s F5-TTS (Local on Server) ---")
    full_text = " ".join(VO_SCRIPTS)
    try:
        os.makedirs("f5_output_dalek", exist_ok=True)
        cmd = ["f5-tts_infer-cli", "--gen_text", full_text, "--output_dir", "f5_output_dalek", "--file_prefix", "dalek_vo"]
        subprocess.run(cmd, check=True)
        generated_wav = "f5_output_dalek/dalek_vo.wav"
        if os.path.exists(generated_wav):
            os.replace(generated_wav, output_path)
            apply_trailer_voice_effect(output_path)
            return True
        return False
    except Exception as e:
        print(f"F5-TTS local inference failed: {e}")
        return False

def generate_voice():
    output_path = f"{OUTPUT_DIR}/voice/voiceover_full.wav"
    if os.path.exists(output_path) and "voice" not in sys.argv: return
    success = generate_voice_f5tts(output_path)
    if not success: generate_voice_chattts(output_path, " ".join(VO_SCRIPTS))

def generate_images():
    print("--- Generating Images (8 steps) ---")
    model_id = "black-forest-labs/FLUX.1-schnell"
    pipe = FluxPipeline.from_pretrained(model_id, dtype=torch.bfloat16)
    if DEVICE == "cuda": pipe.enable_model_cpu_offload(); pipe.enable_vae_tiling()
    else: pipe.to(DEVICE)
    for scene in SCENES:
        fname = f"{OUTPUT_DIR}/images/{scene['id']}.png"
        if os.path.exists(fname): continue
        print(f"Generating image: {scene['id']}")
        prompt = f"{scene['visual']}, hyper-realistic, photorealistic, 8k, cinematic lighting, movie still, --no cartoon, anime, illustration, drawing, painting"
        pipe(prompt=prompt, guidance_scale=0.0, num_inference_steps=8, max_sequence_length=256, generator=torch.Generator(device="cpu").manual_seed(42)).images[0].save(fname)
    del pipe; flush()

def generate_audio():
    print("\n--- Generating Music & SFX (100 steps) ---")
    try:
        pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", dtype=torch.float32)
        if DEVICE == "cuda": pipe.enable_model_cpu_offload()
        else: pipe.to(DEVICE)
        neg = "low quality, noise, distortion, artifacts, fillers, talking"
        
        # Generate SFX
        for scene in SCENES:
            filename = f"{OUTPUT_DIR}/sfx/{scene['id']}.wav"
            if os.path.exists(filename): continue
            audio = pipe(prompt=scene['visual'], negative_prompt=neg, num_inference_steps=100, audio_end_in_s=3.75).audios[0]
            scipy.io.wavfile.write(filename, rate=44100, data=audio.cpu().numpy().T)
            
        # Generate MUSIC
        for theme in MUSIC_THEMES:
            filename = f"{OUTPUT_DIR}/music/{theme['id']}.wav"
            if os.path.exists(filename): continue
            print(f"Generating Music: {theme['id']}")
            audio = pipe(prompt=theme['prompt'], negative_prompt=neg, num_inference_steps=100, audio_end_in_s=45.0).audios[0]
            scipy.io.wavfile.write(filename, rate=44100, data=audio.cpu().numpy().T)
            
        del pipe; flush()
    except Exception as e: print(f"Audio generation failed: {e}")

if __name__ == "__main__":
    import sys
    if "voice" in sys.argv: generate_voice(); sys.exit(0)
    generate_images(); generate_voice(); generate_audio()