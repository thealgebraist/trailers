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
from transformers import AutoProcessor, BarkModel

# --- Configuration ---
OUTPUT_DIR = "assets_wait"
os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/voice", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/sfx", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/music", exist_ok=True)

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
print(f"Using device: {DEVICE}")

# --- Expanded Narrative VO Data (Targeting 120s) ---
VO_SCRIPTS = [
    "In the cold, sterile silence of the early morning, before the world has fully awakened to the chaos of the day, a single, terrible choice was made in the quiet of a minimalist kitchen. The handle of the stainless steel kettle was cold to the touch, a silent object of utility that was about to become the center of a mounting psychological storm. There is no turning back once the button is pressed, once the internal element begins its slow, invisible transformation of energy into heat. The pressure is rising within the steel walls, and every single second counts as the atmosphere in the room thickens with anticipation and dread. It is almost time for the truth to be revealed, but time itself seems to have warped, stretching each moment into an eternity of observation. Something is coming, a violent eruption of steam and sound, and it will not be stopped by the simple act of looking away.",
    "But the world stayed cold and indifferent to the struggle taking place upon the stovetop. Failure has a bitter, metallic flavor, a reminder of the times when the spark simply refuses to catch. But Arthur is not a man to accept defeat in the face of a cold cup. There is still one last try, one final push against the inertia of the mundane. This time, it is different. The red glow of the indicator light is more aggressive, the vibration of the water more intense. The heat is absolute, a controlled fury that is overflowing the boundaries of the machine. WITNESS! THE BOILING IS HERE! THE STEAM IS TRIUMPHANT! It never boils when you watch it, they say, but they have never seen a man who can stare into the heart of the sun without blinking. Experience the slow-burn event of the year. The Kettle."
]

SCENE_DURATION = 3.75
SCENES = [
    { "id": "01_kitchen_wide", "visual": "Photorealistic cinematic shot of a modern cold minimalist kitchen. 8k." },
    { "id": "02_the_kettle_still", "visual": "Photorealistic cinematic shot of a stainless steel kettle. 8k." },
    { "id": "03_hand_reaching", "visual": "Photorealistic cinematic shot of a trembling hand reaching for a kettle. 8k." },
    { "id": "04_filling_water", "visual": "Photorealistic cinematic shot of water pouring into a kettle. 8k." },
    { "id": "05_closing_lid", "visual": "Photorealistic cinematic shot of a kettle lid snapping shut. 8k." },
    { "id": "06_pushing_button", "visual": "Photorealistic cinematic shot of a finger pressing an LED button. 8k." },
    { "id": "07_the_wait_begins", "visual": "Photorealistic cinematic shot of a man staring intently at a kettle. 8k." },
    { "id": "08_clock_close", "visual": "Photorealistic cinematic shot of a clock face. 8k." },
    { "id": "09_bubbles_forming", "visual": "Photorealistic cinematic shot inside a kettle with boiling bubbles. 8k." },
    { "id": "10_steam_rising", "visual": "Photorealistic cinematic shot of a single wisp of steam. 8k." },
    { "id": "11_face_extreme_close", "visual": "Photorealistic cinematic close up of a human eye. 8k." },
    { "id": "12_kettle_shaking", "visual": "Photorealistic cinematic shot of a vibrating kettle. 8k." },
    { "id": "13_steam_cloud", "visual": "Photorealistic cinematic shot of a thick eruption of steam. 8k." },
    { "id": "14_orchestra_peak_1", "visual": "Photorealistic cinematic montage of kitchen objects. 8k." },
    { "id": "15_the_almost_click", "visual": "Photorealistic cinematic close up of a kettle switch about to pop. 8k." },
    { "id": "16_the_fakeout", "visual": "Total black screen with realistic film grain. 8k." },
    { "id": "17_the_quiet_return", "visual": "Photorealistic cinematic shot of a man in a dark kitchen. 8k." },
    { "id": "18_empty_cup", "visual": "Photorealistic cinematic shot of an empty porcelain cup. 8k." },
    { "id": "19_arthur_sigh_wait", "visual": "Photorealistic cinematic shot of a man exhaling breath vapor. 8k." },
    { "id": "20_re_pushing_button", "visual": "Photorealistic cinematic shot of a hand pressing a red glowing button. 8k." },
    { "id": "21_red_glow", "visual": "Photorealistic cinematic shot of a kitchen in aggressive red light. 8k." },
    { "id": "22_violent_boil", "visual": "Photorealistic cinematic shot of a violent boiling storm inside a kettle. 8k." },
    { "id": "23_steam_jet", "visual": "Photorealistic cinematic shot of a powerful steam jet. 8k." },
    { "id": "24_arthur_screaming", "visual": "Photorealistic cinematic shot of a man screaming in red light. 8k." },
    { "id": "25_kettle_levitating", "visual": "Photorealistic cinematic shot of a kettle levitating with electrical sparks. 8k." },
    { "id": "26_the_boiling_point", "visual": "Photorealistic cinematic macro shot of glowing white water. 8k." },
    { "id": "27_final_crescendo", "visual": "Photorealistic cinematic rapid montage of boiling. 8k." },
    { "id": "28_the_pop", "visual": "Photorealistic cinematic close up of a switch popping. 8k." },
    { "id": "29_title_card_kettle", "visual": "Photorealistic cinematic title card 'THE KETTLE'. 8k." },
    { "id": "30_slogan_kettle", "visual": "Photorealistic cinematic text 'IT NEVER BOILS'. 8k." },
    { "id": "31_coming_soon_kettle", "visual": "Photorealistic cinematic text 'COMING SOON'. 8k." },
    { "id": "32_post_credits_kettle", "visual": "Photorealistic cinematic shot of a man pouring lukewarm water. 8k." }
]

MUSIC_THEMES = [
    { "id": "theme_build_1", "prompt": "Minimalist ominous electronic underscore, low humming drone, tense anticipation. High quality." },
    { "id": "theme_build_2", "prompt": "Aggressive industrial noise, high frequency screeching and deep sub-bass vibration. High quality." },
    { "id": "theme_silence_bridge", "prompt": "Atmospheric cinematic silence, very sparse high-pitched resonant crystal pings. High quality." }
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

def generate_voice_bark(output_path, text):
    print("---" Falling back to Bark ---")
    try:
        processor = AutoProcessor.from_pretrained("suno/bark")
        model = BarkModel.from_pretrained("suno/bark", torch_dtype=torch.float32).to(DEVICE)
        voice_preset = "v2/en_speaker_9"
        sample_rate = model.generation_config.sample_rate
        total_target_len = int(120.0 * sample_rate)
        full_audio = []
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        for sent in sentences:
            clean_text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?]', '', sent)
            inputs = processor(clean_text, voice_preset=voice_preset, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                audio_array = model.generate(**inputs, min_eos_p=0.05).cpu().numpy().squeeze()
            full_audio.append(audio_array)
            full_audio.append(np.zeros(int(0.2 * sample_rate), dtype=np.float32))
        combined = np.concatenate(full_audio)
        if len(combined) < total_target_len: combined = np.pad(combined, (0, total_target_len - len(combined)))
        else: combined = combined[:total_target_len]
        scipy.io.wavfile.write(output_path, rate=sample_rate, data=combined)
        apply_trailer_voice_effect(output_path)
        del model; del processor; flush()
    except Exception as e: print(f"Bark failed: {e}")

def generate_voice_f5tts(output_path):
    print("---" Generating 120s F5-TTS (Local on Server) ---")
    full_text = " ".join(VO_SCRIPTS)
    try:
        os.makedirs("f5_output_wait", exist_ok=True)
        cmd = ["f5-tts_infer-cli", "--gen_text", full_text, "--output_dir", "f5_output_wait", "--file_prefix", "wait_vo"]
        subprocess.run(cmd, check=True)
        generated_wav = "f5_output_wait/wait_vo.wav"
        if os.path.exists(generated_wav):
            os.replace(generated_wav, output_path)
            apply_trailer_voice_effect(output_path)
            return True
        return False
    except Exception as e: print(f"F5-TTS local inference failed: {e}")
    return False

def generate_voice():
    output_path = f"{OUTPUT_DIR}/voice/voiceover_full.wav"
    if os.path.exists(output_path) and "voice" not in sys.argv: return
    success = generate_voice_f5tts(output_path)
    if not success: generate_voice_bark(output_path, " ".join(VO_SCRIPTS))

def generate_images():
    print("\n---" Generating Images (8 steps) ---")
    model_id = "black-forest-labs/FLUX.1-schnell"
    try:
        pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        if DEVICE == "cuda": pipe.enable_model_cpu_offload(); pipe.enable_vae_tiling()
        else: pipe.to(DEVICE)
        for scene in SCENES:
            filename = f"{OUTPUT_DIR}/images/{scene['id']}.png"
            txt_filename = f"{OUTPUT_DIR}/images/{scene['id']}.txt"
            if os.path.exists(filename): continue
            print(f"Generating image: {scene['id']}")
            full_prompt = f"{scene['visual']}, hyper-realistic, photorealistic, 8k, cinematic lighting, movie still, --no cartoon, anime, illustration, drawing, painting"
            pipe(prompt=full_prompt, height=512, width=512, guidance_scale=0.0, num_inference_steps=8, max_sequence_length=256, generator=torch.Generator(device="cpu").manual_seed(12345)).images[0].save(filename)
            with open(txt_filename, "w") as f: f.write(f"Model: {model_id}\nPrompt: {full_prompt}\nSteps: 8\nRes: 512x512\nSeed: 12345\n")
        del pipe; flush()
    except Exception as e: print(f"Image generation failed: {e}")

def generate_audio():
    print("\n---" Generating Music & SFX (100 steps) ---")
    try:
        pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float32)
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
    if "voice" in sys.argv:
        generate_voice()
        sys.exit(0)
    generate_images(); generate_voice(); generate_audio()
