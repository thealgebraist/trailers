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
OUTPUT_DIR = "assets_romcom"
os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/voice", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/sfx", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/music", exist_ok=True)

if torch.cuda.is_available(): DEVICE = "cuda"
else: DEVICE = "cpu"

# --- Expanded Narrative VO Data (Targeting 120s) ---
VO_SCRIPTS = [
    "In the soot-stained industrial heart of the North, where the sky is a permanent shade of grey and the air is thick with the scent of coal and grease, life moved with the cold, unyielding precision of a factory loom. Rusty was a working-class drone, a weathered soul in a metallic shell, existing in a world that had long ago forgotten how to feel anything but the grind of the gears. He spent his long, repetitive days assembling misery one bolt at a time, until the moment he looked across the steam of the laundry room and saw Her. Betty was a spark of color in a monochrome world, a melody in a landscape of industrial noise. It was a love that defied every law of logic, a connection that shouldn't exist in a society built on efficiency and cold calculation.",
    "But Rusty's father, a black-clad traditionalist who believed that emotion was the ultimate malfunction, had far darker plans for his son's future. Love is illogical, he screamed into the damp Manchester night, but Rusty had a secret he could no longer suppress. He wasn't just a soldier; he was a born hoofer with music in his base and a rhythm in his soul. This Christmas, experience the most heartwarming and unlikely musical since Singin' in the Rain. He has the beat, he has the girl, and he has one last chance to show the world what it truly means to be alive. Reality has a way of trying to exterminate dreams, but even a plunger can play a flute if the heart is right. Join Rusty and Betty for the event of the season. The Plunger's Refrain."
]

SCENE_DURATION = 3.75
SCENES = [
    { "id": "01_industrial_north", "visual": "Photorealistic cinematic shot of 1950s Manchester. 8k." },
    { "id": "02_dalek_flat_cap", "visual": "Photorealistic cinematic shot of a lone weathered Dalek in a wool flat cap. 8k." },
    { "id": "03_factory_line", "visual": "Photorealistic cinematic shot of a textile mill interior. 8k." },
    { "id": "04_the_meeting", "visual": "Photorealistic cinematic shot of a Dalek and a human woman lock eye-stalks. 8k." },
    { "id": "05_shared_tea", "visual": "Photorealistic cinematic shot of a Dalek and Betty drinking tea. 8k." },
    { "id": "06_stern_father", "visual": "Photorealistic cinematic shot of a massive black Dalek in a vest. 8k." },
    { "id": "07_father_shouting", "visual": "Photorealistic cinematic close up of a red eye-stalk. 8k." },
    { "id": "08_back_alley", "visual": "Photorealistic cinematic shot of a damp back alley at night. 8k." },
    { "id": "09_tap_dancing_reveal", "visual": "Photorealistic cinematic shot of a Dalek base tapping cobblestones. 8k." },
    { "id": "10_gene_kelly_homage", "visual": "Photorealistic cinematic shot of a Dalek swinging on a lamppost in the rain. 8k." },
    { "id": "11_technicolor_shift", "visual": "Photorealistic cinematic shot of a colorful street with Betty in a yellow dress. 8k." },
    { "id": "12_dance_on_stairs", "visual": "Photorealistic cinematic shot of a Dalek and Betty dancing on marble stairs. 8k." },
    { "id": "13_tuxedo_dalek", "visual": "Photorealistic cinematic shot of a Dalek in a tuxedo spinning Betty. 8k." },
    { "id": "14_chorus_line", "visual": "Photorealistic cinematic shot of a chorus line of Daleks in top hats. 8k." },
    { "id": "15_plunger_flute", "visual": "Photorealistic cinematic shot of a Dalek playing a plunger flute. 8k." },
    { "id": "16_back_to_reality", "visual": "Photorealistic cinematic shot of color fading to grey. 8k." },
    { "id": "17_the_breakup", "visual": "Photorealistic cinematic shot of Betty crying against a brick wall. 8k." },
    { "id": "18_rusty_drinking", "visual": "Photorealistic cinematic shot of a Dalek in a pub with an oil pint. 8k." },
    { "id": "19_talent_poster", "visual": "Photorealistic cinematic shot of a wet poster on a brick wall. 8k." },
    { "id": "20_entering_hall", "visual": "Photorealistic cinematic shot of a Dalek entering a grand hall. 8k." },
    { "id": "21_taking_stage", "visual": "Photorealistic cinematic shot of a Dalek on stage in a spotlight. 8k." },
    { "id": "22_first_tap", "visual": "Photorealistic cinematic shot of a Dalek base tapping on wood. 8k." },
    { "id": "23_music_explodes", "visual": "Photorealistic cinematic shot of a Dalek doing high speed tap dance. 8k." },
    { "id": "24_crowd_cheering", "visual": "Photorealistic cinematic shot of a cheering crowd. 8k." },
    { "id": "25_betty_joins", "visual": "Photorealistic cinematic shot of Betty dancing with a Dalek. 8k." },
    { "id": "26_the_lift", "visual": "Photorealistic cinematic shot of a Dalek lifting Betty. 8k." },
    { "id": "27_award_ceremony", "visual": "Photorealistic cinematic shot of a Dalek holding a golden trophy. 8k." },
    { "id": "28_walking_home", "visual": "Photorealistic cinematic shot of a Dalek and Betty walking down a wet street at sunset. 8k." },
    { "id": "29_title_card_romcom", "visual": "Photorealistic cinematic title card 'THE PLUNGER'S REFRAIN'. 8k." },
    { "id": "30_critics_quotes", "visual": "Photorealistic cinematic text shot of reviews. 8k." },
    { "id": "31_slogan_romcom", "visual": "Photorealistic cinematic text 'EXTERMINATE THE SADNESS'. 8k." },
    { "id": "32_post_credits_romcom", "visual": "Photorealistic cinematic shot of a Dalek with a pancake on its eye. 8k." }
]

MUSIC_THEMES = [
    { "id": "theme_dark", "prompt": "Heartwarming romantic orchestral score, swelling strings and whimsical woodwinds. High quality." }
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
    print("--- Generating 120s F5-TTS (Local on Server) ---")
    full_text = " ".join(VO_SCRIPTS)
    try:
        os.makedirs("f5_output_romcom", exist_ok=True)
        cmd = ["f5-tts_infer-cli", "--gen_text", full_text, "--output_dir", "f5_output_romcom", "--file_prefix", "romcom_vo"]
        subprocess.run(cmd, check=True)
        generated_wav = "f5_output_romcom/romcom_vo.wav"
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
    print("\n--- Generating Images (8 steps) ---")
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
            pipe(prompt=full_prompt, height=512, width=512, guidance_scale=0.0, num_inference_steps=8, max_sequence_length=256, generator=torch.Generator(device="cpu").manual_seed(777)).images[0].save(filename)
            with open(txt_filename, "w") as f: f.write(f"Model: {model_id}\nPrompt: {full_prompt}\nSteps: 8\nRes: 512x512\nSeed: 777\n")
        del pipe; flush()
    except Exception as e: print(f"Image generation failed: {e}")

def generate_audio():
    print("\n--- Generating Music & SFX (100 steps) ---")
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
