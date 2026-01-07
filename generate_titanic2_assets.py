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
OUTPUT_DIR = "assets_titanic2"
os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/voice", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/sfx", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/music", exist_ok=True)

if torch.cuda.is_available(): DEVICE = "cuda"
else: DEVICE = "cpu"

# --- Expanded Narrative VO Data (Targeting 120s) ---
VO_SCRIPTS = [
    "For one hundred years, the most famous wreck in history has slept in the crushing, lightless depths of the North Atlantic, a rusted tomb for the dreams of a bygone era. But billionaire visionary Sir Alistair Moneybags has a plan that defies both nature and common sense. He is bringing the legend back to the surface, raising the original hull from the silt and restoring it to its former, dangerous glory. Titanic 2 is bigger, stronger, and more expensive than every other ship on the ocean combined, featuring a New Jack and a New Rose walking hand-in-hand into the exact same old mistake.",
    "But deep in the Arctic circle, a legendary enemy has been waiting for its second chance, and this time, it is personal. Radar says it is impossible, the Captain says, Not again, but the hubris of man has raised the dead to sink them one more time. Investment terminated, the Iceberg is back with a robotic vengeance, and the cold water is calling for its prize. Prepare for the resinkening, a disaster so massive it had to happen twice. This July, the unsinkable becomes the unthinkable. Titanic 2."
]

SCENE_DURATION = 3.75
SCENES = [
    { "id": "01_deep_ocean", "visual": "Photorealistic cinematic shot of the dark North Atlantic ocean floor. Rusted Titanic hull. 8k." },
    { "id": "02_billionaire_reveal", "visual": "Photorealistic cinematic shot of Sir Alistair Moneybags in a sharp suit. 8k." },
    { "id": "03_raising_the_ship", "visual": "Photorealistic cinematic shot of massive cranes lifting the Titanic hull from the sea. 8k." },
    { "id": "04_dry_dock_restoration", "visual": "Photorealistic cinematic shot of Titanic hull in a high-tech dry dock. Welders. 8k." },
    { "id": "05_titanic_2_full", "visual": "Photorealistic cinematic shot of the complete Titanic 2 at sea during sunset. 8k." },
    { "id": "06_grand_staircase_modern", "visual": "Photorealistic cinematic shot of the modern Grand Staircase with crystal chandeliers. 8k." },
    { "id": "07_jack_clone", "visual": "Photorealistic cinematic shot of a Jack Dawson clone in 1912 clothing. 8k." },
    { "id": "08_rose_clone", "visual": "Photorealistic cinematic shot of a Rose DeWitt Bukater clone in an elegant dress. 8k." },
    { "id": "09_north_atlantic_night", "visual": "Photorealistic cinematic shot of the ship at night in cold moonlight. 8k." },
    { "id": "10_iceberg_waiting", "visual": "Photorealistic cinematic shot of a massive iceberg in the Arctic. 8k." },
    { "id": "11_iceberg_tracking", "visual": "Photorealistic cinematic shot of the iceberg with glowing robotic sensors. 8k." },
    { "id": "12_radar_warning", "visual": "Photorealistic cinematic shot of a high-tech radar screen with fast beeping alerts. 8k." },
    { "id": "13_captain_panic", "visual": "Photorealistic cinematic shot of the Captain's face in panic. 8k." },
    { "id": "14_the_hit", "visual": "Photorealistic cinematic shot of the ship hitting the iceberg. 8k." },
    { "id": "15_water_everywhere", "visual": "Photorealistic cinematic shot of the boiler room flooding. 8k." },
    { "id": "16_ship_tilting", "visual": "Photorealistic cinematic shot of the ship tilting as it sinks. 8k." },
    { "id": "17_jet_ski_escape", "visual": "Photorealistic cinematic shot of billionaires escaping on jet skis. 8k." },
    { "id": "18_billionaire_regret", "visual": "Photorealistic cinematic shot of Sir Moneybags underwater. 8k." },
    { "id": "19_iceberg_transformation", "visual": "Photorealistic cinematic shot of the iceberg revealing giant robotic arms. 8k." },
    { "id": "20_underwater_fight", "visual": "Photorealistic cinematic shot of the Titanic hull fighting the robot iceberg. 8k." },
    { "id": "21_door_scene_2", "visual": "Photorealistic cinematic shot of Jack and Rose on a floating door. 8k." },
    { "id": "22_propeller_guy_2", "visual": "Photorealistic cinematic shot of a man hitting the ship's propeller. 8k." },
    { "id": "23_orchestra_rock", "visual": "Photorealistic cinematic shot of the band playing electric guitars on the sinking deck. 8k." },
    { "id": "24_ship_snapping", "visual": "Photorealistic cinematic shot of the Titanic snapping in half. 8k." },
    { "id": "25_lifeboat_selfie", "visual": "Photorealistic cinematic shot of survivors taking a selfie in a lifeboat. 8k." },
    { "id": "26_iceberg_salute", "visual": "Photorealistic cinematic shot of the iceberg making a peace sign. 8k." },
    { "id": "27_rescue_ship_iceberg", "visual": "Photorealistic cinematic shot of the rescue ship Carpathia hitting another iceberg. 8k." },
    { "id": "28_jack_rose_underwater", "visual": "Photorealistic cinematic shot of Jack and Rose as skeletons on the ocean floor. 8k." },
    { "id": "29_title_card_t2", "visual": "Photorealistic cinematic title card 'TITANIC 2'. 8k." },
    { "id": "30_coming_july", "visual": "Photorealistic cinematic text 'SINKING THIS JULY'. 8k." },
    { "id": "31_iceberg_name", "visual": "Photorealistic cinematic shot of the iceberg with 'ICE-BORG' written on it. 8k." },
    { "id": "32_post_credits_t2", "visual": "Photorealistic cinematic shot of an iceberg in a bathtub. 8k." }
]

MUSIC_THEMES = [
    { "id": "theme_dark", "prompt": "Epic action disaster score, booming orchestral hits and low frequency mechanical hums. High quality." }
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
    print("-- Falling back to Bark --")
    try:
        processor = AutoProcessor.from_pretrained("suno/bark")
        model = BarkModel.from_pretrained("suno/bark", dtype=torch.float32).to(DEVICE)
        voice_preset = "v2/en_speaker_9"
        sample_rate = model.generation_config.sample_rate
        total_target_len = int(120.0 * sample_rate)
        full_audio = []
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        for sent in sentences:
            clean_text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?]', '', sent)
            inputs = processor(clean_text, voice_preset=voice_preset, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                audio_array = model.generate(**inputs, attention_mask=inputs.get("attention_mask"), min_eos_p=0.05).cpu().numpy().squeeze()
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
    print("-- Generating 120s F5-TTS (Local on Server) --")
    full_text = " ".join(VO_SCRIPTS)
    try:
        os.makedirs("f5_output_titanic2", exist_ok=True)
        cmd = ["f5-tts_infer-cli", "--gen_text", full_text, "--output_dir", "f5_output_titanic2", "--file_prefix", "titanic2_vo"]
        subprocess.run(cmd, check=True)
        generated_wav = "f5_output_titanic2/titanic2_vo.wav"
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
    if not success: generate_voice_bark(output_path, " ".join(VO_SCRIPTS))

def generate_images():
    print("\n-- Generating Images (8 steps) --")
    model_id = "black-forest-labs/FLUX.1-schnell"
    try:
        pipe = FluxPipeline.from_pretrained(model_id, dtype=torch.bfloat16)
        if DEVICE == "cuda": pipe.enable_model_cpu_offload(); pipe.enable_vae_tiling()
        else: pipe.to(DEVICE)
        for scene in SCENES:
            filename = f"{OUTPUT_DIR}/images/{scene['id']}.png"
            txt_filename = f"{OUTPUT_DIR}/images/{scene['id']}.txt"
            if os.path.exists(filename): continue
            print(f"Generating image: {scene['id']}")
            full_prompt = f"{scene['visual']}, hyper-realistic, photorealistic, 8k, cinematic lighting, movie still, --no cartoon, anime, illustration, drawing, painting"
            pipe(prompt=full_prompt, height=512, width=512, guidance_scale=0.0, num_inference_steps=8, max_sequence_length=256, generator=torch.Generator(device="cpu").manual_seed(42)).images[0].save(filename)
            with open(txt_filename, "w") as f: f.write(f"Model: {model_id}\nPrompt: {full_prompt}\nSteps: 8\nRes: 512x512\nSeed: 42\n")
        del pipe; flush()
    except Exception as e: print(f"Image generation failed: {e}")

def generate_audio():
    print("\n-- Generating Music & SFX (100 steps) --")
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
    if "voice" in sys.argv:
        generate_voice()
        sys.exit(0)
    generate_images(); generate_voice(); generate_audio()
