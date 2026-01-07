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
OUTPUT_DIR = "assets_snakes"
os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/voice", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/sfx", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/music", exist_ok=True)

if torch.cuda.is_available(): DEVICE = "cuda"
else: DEVICE = "cpu"

# --- Narrative VO Data (Targeting 120s) ---
VO_SCRIPTS = [
    "The Titanic. A name synonymous with grandeur, a floating palace of steam and steel that was meant to be the absolute pinnacle of human engineering. In the spring of 1912, it set sail as the height of luxury, carrying the dreams and the weight of a generation across the Atlantic. From the opulent wood of the Grand Staircase to the sparkling crystal of the dining halls, every detail was a testament to man's mastery over the ocean. But deep in the dark, forgotten corners of the cargo hold, amidst the dust and the heavy crates, something cold, hungry, and ancient was waiting for its moment to strike.",
    "They were watching the horizon for the white ghost of an iceberg, but they were about to hit something far more terrifying than any frozen wall. Draw me like one of your French reptiles, because no one is safe from the monkey-fighting snakes on this Monday-to-Friday boat. From the first-class cabins to the boiler room floors, history is being rewritten in blood and scales. Prepare for the greatest tragedy ever hissed, a chaotic battle for survival where the ocean is the least of their worries. Ice was not the only cold thing in the water that night. This July, experience the resinkening of a legend. Snakes on Titanic."
]

SCENE_DURATION = 3.75
SCENES = [
    { "id": "01_titanic_dock", "visual": "Photorealistic cinematic shot of the Titanic docked at Southampton. 8k." },
    { "id": "02_luxury_interior", "visual": "Photorealistic cinematic shot of the Grand Staircase. 8k." },
    { "id": "03_captain_pride", "visual": "Photorealistic cinematic shot of Captain Smith on the bridge. 8k." },
    { "id": "04_cargo_hold", "visual": "Photorealistic cinematic shot of the dark cargo hold. 8k." },
    { "id": "05_crate_shake", "visual": "Photorealistic cinematic shot of a crate shaking. 8k." },
    { "id": "06_iceberg_lookout", "visual": "Photorealistic cinematic shot of lookouts in the crow's nest at night. 8k." },
    { "id": "07_iceberg_impact", "visual": "Photorealistic cinematic shot of Titanic striking the iceberg. 8k." },
    { "id": "08_crates_break", "visual": "Photorealistic cinematic shot of crates exploding open. 8k." },
    { "id": "09_snakes_hallway", "visual": "Photorealistic cinematic shot of a narrow flooded corridor. 8k." },
    { "id": "10_tea_time_terror", "visual": "Photorealistic cinematic shot of a wealthy lady sipping tea with a cobra. 8k." },
    { "id": "11_ballroom_chaos", "visual": "Photorealistic cinematic shot of the ballroom with snakes everywhere. 8k." },
    { "id": "12_draw_me", "visual": "Photorealistic cinematic shot of a sketch pad with a charcoal snake. 8k." },
    { "id": "13_samuel_captain", "visual": "Photorealistic cinematic shot of Samuel L. Jackson in a tuxedo. 8k." },
    { "id": "14_snakes_on_bow", "visual": "Photorealistic cinematic shot of two giant anacondas on the bow. 8k." },
    { "id": "15_smokestack_wrap", "visual": "Photorealistic cinematic shot of a giant snake wrapping around a smokestack. 8k." },
    { "id": "16_flooded_stairs", "visual": "Photorealistic cinematic shot of the Grand Staircase flooding. 8k." },
    { "id": "17_lifeboat_full", "visual": "Photorealistic cinematic shot of a lifeboat full of snakes. 8k." },
    { "id": "18_propeller_guy", "visual": "Photorealistic cinematic shot of a man caught by a vine snake. 8k." },
    { "id": "19_door_scene", "visual": "Photorealistic cinematic shot of Rose on a floating door with a python. 8k." },
    { "id": "20_violin_band", "visual": "Photorealistic cinematic shot of the band playing with snakes. 8k." },
    { "id": "21_underwater_swim", "visual": "Photorealistic cinematic underwater shot of sea snakes. 8k." },
    { "id": "22_hero_shot", "visual": "Photorealistic cinematic shot of an action hero on a shark. 8k." },
    { "id": "23_ship_snap", "visual": "Photorealistic cinematic shot of the Titanic snapping with snakes. 8k." },
    { "id": "24_car_scene", "visual": "Photorealistic cinematic shot of a steamy car window with a snake tail. 8k." },
    { "id": "25_survivors", "visual": "Photorealistic cinematic shot of dawn at sea with snake rafts. 8k." },
    { "id": "26_old_rose", "visual": "Photorealistic cinematic shot of a snake catching a necklace. 8k." },
    { "id": "27_celine_dion_snake", "visual": "Photorealistic cinematic shot of a snake with a wig on the bow. 8k." },
    { "id": "28_snake_plane", "visual": "Photorealistic cinematic shot of a biplane with a pilot snake. 8k." },
    { "id": "29_iceberg_face", "visual": "Photorealistic cinematic shot of an iceberg with a face. 8k." },
    { "id": "30_title_card", "visual": "Photorealistic cinematic title card 'SNAKES ON TITANIC'. 8k." },
    { "id": "31_slogan", "visual": "Photorealistic cinematic text 'ICE WAS NOT THE ONLY COLD THING'. 8k." },
    { "id": "32_post_credits", "visual": "Photorealistic cinematic shot of a snake in a tuxedo. 8k." }
]

MUSIC_THEMES = [
    { "id": "theme_disaster", "prompt": "Epic disaster movie score, intense suspenseful violins, deep orchestral hits and sharp hissing sounds. High quality." }
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
    print("--- Falling back to Bark ---")
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
    print("--- Generating 120s F5-TTS (Local on Server) ---")
    full_text = " ".join(VO_SCRIPTS)
    try:
        os.makedirs("f5_output_snakes", exist_ok=True)
        cmd = ["f5-tts_infer-cli", "--gen_text", full_text, "--output_dir", "f5_output_snakes", "--file_prefix", "snakes_vo"]
        subprocess.run(cmd, check=True)
        generated_wav = "f5_output_snakes/snakes_vo.wav"
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
        pipe(prompt=prompt, guidance_scale=0.0, num_inference_steps=8, max_sequence_length=256, generator=torch.Generator(device="cpu").manual_seed(123)).images[0].save(fname)
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
