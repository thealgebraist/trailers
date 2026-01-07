import PIL.Image
if not hasattr(PIL.Image, "ANTIALIAS"): PIL.Image.ANTIALIAS = PIL.Image.LANCZOS
import torch
import scipy.io.wavfile
import os
import gc
import subprocess
import re
import requests
import numpy as np
from diffusers import WanPipeline, AudioLDM2Pipeline
from transformers import AutoProcessor, BarkModel
from moviepy.editor import *

# --- ElevenLabs Config ---
ELEVEN_API_KEY = open("eleven_key.txt").read().strip() if os.path.exists("eleven_key.txt") else os.getenv("ELEVEN_API_KEY", "")
VOICE_ID = "EXAVITQu4vr4xnSDxMaL" # Bella
MODEL_ID = "eleven_multilingual_v2"

# --- Configuration ---
OUTPUT_DIR = "assets_wan"
os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/voice", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/sfx", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/music", exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

MOVIES = [
    {
        "title": "A Dalek Comes Home",
        "file_prefix": "dalek",
        "music_prompt": "Emotional cinematic trailer music, acoustic guitar swelling into orchestral strings, hopeful, sci-fi drama.",
        "vo": "In a cold and distant universe governed by the iron laws of infinite hate, one weary soldier stands amidst the rusted ruins of Skaro, haunted by a ghost of a memory that shouldn't exist. He was found as a tiny, helpless spark in a vast Kansas cornfield, raised by the gentle, calloused hands of Nana and Pop-Pop. This summer, a Dalek is finally coming home, and he's bringing dessert.",
        "visual_prompts": [
            "A Dalek standing in a golden wheat field at sunset, cinematic lighting, photorealistic, 4k.",
            "Close up of a Dalek eye stalk looking at a human hand, emotional, macro.",
            "A Dalek wearing a propeller hat surrounded by laughing children in a schoolyard.",
            "A Dalek rolling down a country road, dust kicking up, triumphant.",
            "Title card text 'A DALEK COMES HOME' floating in space."
        ]
    },
    {
        "title": "Snakes on Titanic",
        "file_prefix": "snakes",
        "music_prompt": "Intense disaster movie score, orchestral hits, suspenseful violins, deep bass.",
        "vo": "The unsinkable ship was once the height of luxury, but in the dark depths of the cargo hold, something was waiting. They were watching the horizon for ice, but they were about to hit something far worse than any frozen wall. History will be rewritten in blood and scales. Prepare for the greatest tragedy ever hissed. Snakes on Titanic.",
        "visual_prompts": [
            "The Titanic ship sailing at night, vibrant lights, cinematic.",
            "A green snake slithering over a fancy dinner plate in a luxury dining room.",
            "Panic in the ballroom, people running, snakes everywhere.",
            "Samuel L Jackson in a tuxedo holding a flare gun on the deck.",
            "The Titanic sinking while giant snakes wrap around the hull."
        ]
    },
    {
        "title": "Planet of the Luftwaffe",
        "file_prefix": "luftwaffe",
        "music_prompt": "Military march with jungle drums, bizarre, comedic, dramatic.",
        "vo": "Major Rick Savage was the best pilot in the Allied forces, until he crashed behind enemy lines and woke up in a Germany he didn't recognize. Evolution took a dark and unexpected turn, and a new order of primates has risen to power. This summer, experience the war for the planet. Planet of the Luftwaffe.",
        "visual_prompts": [
            "A chimpanzee wearing a WWII general uniform screaming, cinematic, gritty.",
            "A P-51 Mustang plane crash landed in a dense jungle, smoke rising.",
            "Gorilla soldiers marching in formation holding bananas like rifles.",
            "A giant banana rocket launching into the sky, smoke, fire.",
            "The Statue of Liberty but it is a monkey, apocalyptic landscape."
        ]
    }
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
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        full_audio = []
        for sent in sentences:
            clean_text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?]', '', sent)
            inputs = processor(clean_text, voice_preset=voice_preset, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                audio_array = model.generate(**inputs, attention_mask=inputs.get("attention_mask"), min_eos_p=0.05).cpu().numpy().squeeze()
            full_audio.append(audio_array)
            full_audio.append(np.zeros(int(0.2 * sample_rate), dtype=np.float32))
        combined = np.concatenate(full_audio)
        scipy.io.wavfile.write(output_path, rate=sample_rate, data=combined)
        apply_trailer_voice_effect(output_path)
        del model; del processor; flush()
    except Exception as e: print(f"Bark failed: {e}")

def generate_voice(movie):
    output_path = f"{OUTPUT_DIR}/voice/{movie['file_prefix']}_vo.wav"
    if os.path.exists(output_path) and "voice" not in sys.argv: return
    if not ELEVEN_API_KEY or ELEVEN_API_KEY == "":
        generate_voice_bark(output_path, movie['vo']); return
    print(f"--- Generating ElevenLabs VO for {movie['title']} ---")
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
    headers = { "Accept": "audio/mpeg", "Content-Type": "application/json", "xi-api-key": ELEVEN_API_KEY }
    data = { "text": movie['vo'], "model_id": MODEL_ID, "voice_settings": { "stability": 0.5, "similarity_boost": 0.8 } }
    try:
        res = requests.post(url, json=data, headers=headers)
        if res.status_code == 200:
            with open(output_path, 'wb') as f: f.write(res.content)
            apply_trailer_voice_effect(output_path)
        else:
            print(f"ElevenLabs Quota Exceeded or Error ({res.status_code}).")
            generate_voice_bark(output_path, movie['vo'])
    except Exception as e:
        print(f"ElevenLabs connection failed: {e}")
        generate_voice_bark(output_path, movie['vo'])

def generate_music(pipe, prompt, filename):
    if os.path.exists(filename): return
    print(f"Generating Music: {prompt[:30]}...")
    audio = pipe(prompt, num_inference_steps=80, audio_length_in_s=20.0).audios[0]
    scipy.io.wavfile.write(filename, rate=16000, data=audio)

def generate_video_clip(pipe, prompt, filename):
    if os.path.exists(filename): return
    print(f"Generating Video Clip (Wan2.1): {prompt[:30]}...")
    try:
        video_output = pipe(prompt, width=832, height=480, num_frames=81, num_inference_steps=30, guidance_scale=6.0).frames[0]
        np_frames = [np.array(f) for f in video_output]
        clip = ImageSequenceClip(np_frames, fps=4)
        clip.write_videofile(filename, codec="libx264", audio=False, logger=None)
    except Exception as e: print(f"Error during video generation: {e}")

def assemble_trailer(movie, assets_path):
    print(f"Assembling trailer for {movie['title']}...")
    clips = []
    for i in range(len(movie['visual_prompts'])):
        v_path = f"{assets_path}/{movie['file_prefix']}_clip_{i}.mp4"
        if os.path.exists(v_path): clips.append(VideoFileClip(v_path))
    if not clips: return
    final_video = concatenate_videoclips(clips, method="compose")
    vo_path = f"{assets_path}/voice/{movie['file_prefix']}_vo.wav"
    if os.path.exists(vo_path):
        vo = AudioFileClip(vo_path)
        final_video = final_video.set_audio(vo)
    audio_path = f"{assets_path}/{movie['file_prefix']}_music.wav"
    if os.path.exists(audio_path):
        music = AudioFileClip(audio_path).volumex(0.4)
        if music.duration < final_video.duration: music = afx.audio_loop(music, duration=final_video.duration)
        else: music = music.subclip(0, final_video.duration)
        if final_video.audio: final_video = final_video.set_audio(CompositeAudioClip([final_video.audio, music]))
        else: final_video = final_video.set_audio(music)
    output_filename = f"{movie['file_prefix']}_wan_trailer.mp4"
    final_video.write_videofile(output_filename, fps=24, codec="libx264", audio_codec="aac")

def main():
    for movie in MOVIES: generate_voice(movie)
    print("--- Loading AudioLDM2 ---")
    audio_pipe = AudioLDM2Pipeline.from_pretrained("cvssp/audioldm2", dtype=torch.float32).to(DEVICE)
    for movie in MOVIES:
        fname = f"{OUTPUT_DIR}/{movie['file_prefix']}_music.wav"
        generate_music(audio_pipe, movie['music_prompt'], fname)
    del audio_pipe; flush()
    print("\n--- Loading Wan2.1-T2V-1.3B-Diffusers ---")
    try:
        video_pipe = WanPipeline.from_pretrained("Wan-Video/Wan2.1-T2V-1.3B-Diffusers", dtype=torch.bfloat16)
        if DEVICE == "cuda": video_pipe.enable_model_cpu_offload()
        else: video_pipe.to(DEVICE)
        for movie in MOVIES:
            for i, prompt in enumerate(movie['visual_prompts']):
                fname = f"{OUTPUT_DIR}/{movie['file_prefix']}_clip_{i}.mp4"
                generate_video_clip(video_pipe, prompt, fname)
                flush()
        del video_pipe; flush()
    except Exception as e: print(f"Failed to load or run Wan2.1: {e}")
    for movie in MOVIES: assemble_trailer(movie, OUTPUT_DIR)

if __name__ == "__main__":
    import sys
    if "voice" in sys.argv:
        for m in MOVIES: generate_voice(m)
        sys.exit(0)
    main()
