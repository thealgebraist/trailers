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
OUTPUT_DIR = "assets_boring"
os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/voice", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/sfx", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/music", exist_ok=True)

if torch.cuda.is_available(): DEVICE = "cuda"
else: DEVICE = "cpu"

# --- Expanded Narrative VO Data (Targeting 120s) ---
VO_SCRIPTS = [
    "In a world where time has simply given up and the clocks have forgotten how to tick, every single second stretches out into a lifetime of beige indifference. Arthur is a man with a vision, but it is not a vision of grandeur or a dream of change. It is a vision of moisture, a slow and deliberate study of the dark damp patch spreading across his grey plaster wall. There is no escape from the beige, no relief from the steady, rhythmic hum of the ancient refrigerator. Hunger is just a distant memory, replaced by the reality of cold, lumpy porridge on a wooden table. The heat never comes to this room, and time is a luxury that Arthur simply cannot afford to spend on anything but staring.",
    "Action is a concept reserved for the young and the restless. Here, drama is found in the slow, agonizing peeling of floral wallpaper. Adventure is out there in the world, but so is the rain, and Arthur prefers the static consistency of his chipped ceramic mug. Refreshment is a myth, a story told by those who still believe in the sun. Some stories have already ended long before they reach the final page. Light is a choice that he did not make, and so he sits in the gathering shadows of his own existence. The tension is imperceptible, a silent vibration in the air. Nothing happens. And then, it happens again. This season, experience the event of the century. The Damp Patch."
]

SCENE_DURATION = 3.75
SCENES = [
    { "id": "01_the_wall", "visual": "Photorealistic cinematic shot of a grey plaster wall. 8k." },
    { "id": "02_dripping_tap", "visual": "Photorealistic cinematic shot of a single water drop on a chipped porcelain tap. 8k." },
    { "id": "03_arthur_staring", "visual": "Photorealistic cinematic shot of a middle-aged man staring blankly. 8k." },
    { "id": "04_the_patch_macro", "visual": "Photorealistic cinematic macro shot of a damp patch on a wall. 8k." },
    { "id": "05_beige_curtains", "visual": "Photorealistic cinematic shot of heavy beige curtains. 8k." },
    { "id": "06_grey_porridge", "visual": "Photorealistic cinematic shot of a bowl of cold lumpy porridge. 8k." },
    { "id": "07_the_radiator", "visual": "Photorealistic cinematic shot of an old iron radiator with peeling white paint. 8k." },
    { "id": "08_arthur_watch", "visual": "Photorealistic cinematic close up of a wristwatch. 8k." },
    { "id": "09_dust_mote", "visual": "Photorealistic cinematic shot of a single dust mote in a light beam. 8k." },
    { "id": "10_peeling_wallpaper", "visual": "Photorealistic cinematic shot of a strip of floral wallpaper peeling. 8k." },
    { "id": "11_arthur_sigh", "visual": "Photorealistic cinematic shot of Arthur closing heavy eyelids. 8k." },
    { "id": "12_the_window_rain", "visual": "Photorealistic cinematic shot through a dirty window with rain. 8k." },
    { "id": "13_chipped_mug", "visual": "Photorealistic cinematic shot of a chipped ceramic mug. 8k." },
    { "id": "14_dead_fly", "visual": "Photorealistic cinematic shot of a dead fly on a windowsill. 8k." },
    { "id": "15_the_ceiling_light", "visual": "Photorealistic cinematic shot of a bare lightbulb. 8k." },
    { "id": "16_arthur_sitting", "visual": "Photorealistic cinematic wide shot of Arthur on a wooden chair. 8k." },
    { "id": "17_the_patch_grows", "visual": "Photorealistic cinematic shot of a dark damp patch growing. 8k." },
    { "id": "18_close_up_eye", "visual": "Photorealistic cinematic close-up of a human eye with a tear. 8k." },
    { "id": "19_empty_bookshelf", "visual": "Photorealistic cinematic shot of an empty wooden bookshelf. 8k." },
    { "id": "20_shadow_movement", "visual": "Photorealistic cinematic shot of a shadow moving on floorboards. 8k." },
    { "id": "21_arthur_standing", "visual": "Photorealistic cinematic shot of Arthur standing up slowly. 8k." },
    { "id": "22_touching_the_wall", "visual": "Photorealistic cinematic shot of a finger approaching a wet wall. 8k." },
    { "id": "23_the_contact", "visual": "Photorealistic cinematic shot of a finger touching wet plaster. 8k." },
    { "id": "24_reaction_shot", "visual": "Photorealistic cinematic close up of a face feeling nothing. 8k." },
    { "id": "25_the_kettle", "visual": "Photorealistic cinematic shot of a rusted metal kettle. 8k." },
    { "id": "26_staring_again", "visual": "Photorealistic cinematic shot of a man staring at a wall. 8k." },
    { "id": "27_fading_out", "visual": "Photorealistic cinematic shot fading to dark grey. 8k." },
    { "id": "28_title_card_boring", "visual": "Photorealistic cinematic title card 'THE DAMP PATCH'. 8k." },
    { "id": "29_slogan_boring", "visual": "Photorealistic cinematic text 'WATCH WHILE YOU WAIT FOR DEATH'. 8k." },
    { "id": "30_coming_whenever", "visual": "Photorealistic cinematic text 'COMING EVENTUALLY'. 8k." },
    { "id": "31_critic_quote", "visual": "Photorealistic cinematic text shot of critic quote. 8k." },
    { "id": "32_post_credits_boring", "visual": "Photorealistic cinematic shot of a fly landing on a damp patch. 8k." }
]

MUSIC_THEMES = [
    { "id": "theme_dark", "prompt": "Sorrowful minimal industrial ambient score, depressing cello and cold drones. High quality." }
]

def flush():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def apply_trailer_voice_effect(file_path):
    temp_path = file_path.replace(".wav", "_temp.wav")
    filter_complex = "lowshelf=g=10:f=100,highshelf=g=-5:f=8000,acompressor=threshold=-15dB:ratio=2:makeup=4dB"
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
        os.makedirs("f5_output_boring", exist_ok=True)
        cmd = ["f5-tts_infer-cli", "--gen_text", full_text, "--output_dir", "f5_output_boring", "--file_prefix", "boring_vo"]
        subprocess.run(cmd, check=True)
        generated_wav = "f5_output_boring/boring_vo.wav"
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
            pipe(prompt=full_prompt, height=512, width=512, guidance_scale=0.0, num_inference_steps=8, max_sequence_length=256, generator=torch.Generator(device="cpu").manual_seed(101)).images[0].save(filename)
            with open(txt_filename, "w") as f: f.write(f"Model: {model_id}\nPrompt: {full_prompt}\nSteps: 8\nRes: 512x512\nSeed: 101\n")
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