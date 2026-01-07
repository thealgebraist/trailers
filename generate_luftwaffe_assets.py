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
OUTPUT_DIR = "assets_luftwaffe"
os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/voice", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/sfx", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/music", exist_ok=True)

if torch.cuda.is_available(): DEVICE = "cuda"
else: DEVICE = "cpu"

# --- Expanded Narrative VO Data (Targeting 120s) ---
VO_SCRIPTS = [
    "Major Rick Savage was the most decorated pilot in the Allied forces, a legend of the skies who had never met a dogfight he couldn't win. But during a top-secret mission over the heart of the Black Forest, his P-51 Mustang suffered a catastrophic engine failure, sending him spiraling into a dense, primordial canopy. When he finally clawed his way from the wreckage, he didn't wake up in the Germany he had spent years studying on maps. He woke up in a world where the very laws of nature had been subverted, a twisted reflection of history known as the Planet Of The Luftwaffe.",
    "In this nightmare landscape, evolution took a dark and unexpected turn, and a new order of highly intelligent primates has risen to absolute power. Led by the ruthless and tactical General Kong, the master race of apes has enslaved the remaining human population, treating us as nothing more than primitive animals. Take your stinking paws off me, you damn dirty Nazi! But Rick Savage is not a man to be tamed. Deep in the sewers of Berlin, the human resistance is finally ready to strike back. They are building the ultimate weapon, the Potassium Bomb, and it is finally time to go bananas. You maniacs! You blew it up! God damn you all to hell! Planet of the Luftwaffe."
]

SCENE_DURATION = 3.75
SCENES = [
    { "id": "01_dogfight", "visual": "Photorealistic cinematic shot of a WWII aerial dogfight. 8k." },
    { "id": "02_crash_landing", "visual": "Photorealistic cinematic shot of a crashed plane in a jungle. 8k." },
    { "id": "03_waking_up", "visual": "Photorealistic cinematic POV shot. Silhouette of an ape soldier against a sun. 8k." },
    { "id": "04_reveal_ape", "visual": "Photorealistic cinematic shot of a Chimpanzee in a WWII officer uniform with a monocle. 8k." },
    { "id": "05_title_splash", "visual": "Photorealistic cinematic title shot 'PLANET OF THE LUFTWAFFE'. 8k." },
    { "id": "06_berlin_jungle", "visual": "Photorealistic cinematic wide shot of Berlin overgrown with realistic jungle vines. 8k." },
    { "id": "07_marching_chimps", "visual": "Photorealistic cinematic shot of thousands of chimpanzees marching in uniform. 8k." },
    { "id": "08_gorilla_guards", "visual": "Photorealistic cinematic shot of Rick Savage with Gorilla guards. 8k." },
    { "id": "09_hitler_kong", "visual": "Photorealistic cinematic shot of a giant Silverback Gorilla with a mustache at a podium. 8k." },
    { "id": "10_rick_defiant", "visual": "Photorealistic cinematic shot of Rick Savage with an intense expression. 8k." },
    { "id": "11_dirty_ape_line", "visual": "Photorealistic cinematic shot of Rick Savage shouting at a Gorilla. 8k." },
    { "id": "12_human_pets", "visual": "Photorealistic cinematic shot of apes walking humans on leashes. 8k." },
    { "id": "13_resistance_meeting", "visual": "Photorealistic cinematic shot of Rick Savage in a sewer meeting. 8k." },
    { "id": "14_secret_weapon_plan", "visual": "Photorealistic cinematic shot of a map showing a banana rocket. 8k." },
    { "id": "15_banana_rocket", "visual": "Photorealistic cinematic shot of a massive metallic V2 rocket shaped like a banana. 8k." },
    { "id": "16_poop_grenade", "visual": "Photorealistic cinematic action shot of Rick jumping over sandbags. 8k." },
    { "id": "17_dogfight_monkeys", "visual": "Photorealistic cinematic aerial battle between Rick and ape pilots. 8k." },
    { "id": "18_tank_battle", "visual": "Photorealistic cinematic shot of a Tiger Tank rolling. 8k." },
    { "id": "19_bananas", "visual": "Photorealistic cinematic shot of Rick firing a machine gun. 8k." },
    { "id": "20_bunker_assault", "visual": "Photorealistic cinematic shot of Rick storming a bunker. 8k." },
    { "id": "21_interrogation", "visual": "Photorealistic cinematic shot of an ape officer eating a cigarette. 8k." },
    { "id": "22_rocket_launch", "visual": "Photorealistic cinematic shot of the V-Banana igniting. 8k." },
    { "id": "23_plane_wing_fight", "visual": "Photorealistic cinematic shot of a fight on a bomber wing. 8k." },
    { "id": "24_romantic_kiss", "visual": "Photorealistic cinematic shot of a passionate kiss with an explosion behind. 8k." },
    { "id": "25_statue_of_liberty", "visual": "Photorealistic cinematic shot of a giant monkey statue on a beach. 8k." },
    { "id": "26_rick_screaming", "visual": "Photorealistic cinematic close up of Rick Savage screaming. 8k." },
    { "id": "27_monkey_laugh", "visual": "Photorealistic cinematic shot of a Gorilla laughing. 8k." },
    { "id": "28_hero_walk", "visual": "Photorealistic cinematic shot of Rick putting on sunglasses. 8k." },
    { "id": "29_title_card_2", "visual": "Photorealistic cinematic title card 'PLANET OF THE LUFTWAFFE'. 8k." },
    { "id": "30_coming_soon", "visual": "Photorealistic cinematic text 'COMING SOON'. 8k." },
    { "id": "31_final_joke", "visual": "Photorealistic cinematic shot of an ape slipping on a banana peel. 8k." },
    { "id": "32_post_credits", "visual": "Photorealistic cinematic shot of two red eyes glowing in the dark. 8k." }
]

MUSIC_THEMES = [
    { "id": "theme_dark", "prompt": "Bombastic military march with primitive jungle drums and high-pitched monkey screeches. High quality." }
]

def flush():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def apply_trailer_voice_effect(file_path):
    temp_path = file_path.replace(".wav", "_temp.wav")
    filter_complex = "lowshelf=g=10:f=100,highshelf=g=-5:f=8000,acompressor=threshold=-12dB:ratio=4:makeup=4dB"
    cmd = ["ffmpeg", "-y", "-i", file_path, "-af", filter_complex, temp_path]
    try: subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); os.replace(temp_path, file_path)
    except Exception as e: print(f"Failed effect: {e}")

def generate_voice_bark(output_path, text):
    print("---"Falling back to Bark---")
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
    print("---"Generating 120s F5-TTS (Local on Server)---")
    full_text = " ".join(VO_SCRIPTS)
    try:
        os.makedirs("f5_output_luftwaffe", exist_ok=True)
        cmd = ["f5-tts_infer-cli", "--gen_text", full_text, "--output_dir", "f5_output_luftwaffe", "--file_prefix", "luftwaffe_vo"]
        subprocess.run(cmd, check=True)
        generated_wav = "f5_output_luftwaffe/luftwaffe_vo.wav"
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
    print("---"Generating Images (8 steps)---")
    model_id = "black-forest-labs/FLUX.1-schnell"
    pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    if DEVICE == "cuda": pipe.enable_model_cpu_offload(); pipe.enable_vae_tiling()
    else: pipe.to(DEVICE)
    for scene in SCENES:
        fname = f"{OUTPUT_DIR}/images/{scene['id']}.png"
        if os.path.exists(fname): continue
        print(f"Generating image: {scene['id']}")
        prompt = f"{scene['visual']}, hyper-realistic, photorealistic, 8k, cinematic lighting, movie still, --no cartoon, anime, illustration, drawing, painting"
        pipe(prompt=prompt, guidance_scale=0.0, num_inference_steps=8, max_sequence_length=256, generator=torch.Generator(device="cpu").manual_seed(666)).images[0].save(fname)
    del pipe; flush()

def generate_audio():
    print("\n---"Generating Music & SFX (100 steps)---")
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
    if "voice" in sys.argv: generate_voice(); sys.exit(0)
    generate_images(); generate_voice(); generate_audio()