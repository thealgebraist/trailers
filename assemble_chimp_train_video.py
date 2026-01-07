import PIL.Image
if not hasattr(PIL.Image, "ANTIALIAS"): PIL.Image.ANTIALIAS = PIL.Image.LANCZOS
import os
from moviepy.editor import *

ASSETS_DIR = "assets_chimp_train"
OUTPUT_FILE = "charlies_train_banana_adventure.mpg"

SCENE_IDS = [
    "01_charlie_intro", "02_train_arrival", "03_charlie_boarding", "04_train_ride",
    "05_passengers", "06_market_station", "07_charlie_banana", "08_victory", "09_title_card"
]

def assemble():
    clips = []
    for scene_id in SCENE_IDS:
        print(f"Processing Scene: {scene_id}")
        img_path = f"{ASSETS_DIR}/images/{scene_id}.png"
        vo_path = f"{ASSETS_DIR}/voice/{scene_id}.wav"
        sfx_path = f"{ASSETS_DIR}/sfx/{scene_id}.wav"
        duration = 4.0
        if not os.path.exists(img_path): continue
        audio_clips = []
        if os.path.exists(vo_path) and os.path.getsize(vo_path) > 0:
            vo_clip = AudioFileClip(vo_path).set_start(0.2)
            audio_clips.append(vo_clip)
            if vo_clip.duration + 0.8 > duration: duration = vo_clip.duration + 0.8
        if os.path.exists(sfx_path) and os.path.getsize(sfx_path) > 0:
            sfx_clip = AudioFileClip(sfx_path).volumex(0.5).subclip(0, duration)
            audio_clips.append(sfx_clip)
        img_clip = ImageClip(img_path).set_duration(duration)
        img_clip = img_clip.resize(lambda t: 1 + 0.02 * t)
        if audio_clips:
            img_clip = img_clip.set_audio(CompositeAudioClip(audio_clips))
        clips.append(img_clip)
    print("Concatenating...")
    final_video = concatenate_videoclips(clips, method="compose", padding=-1.0)
    total = final_video.duration
    m1 = AudioFileClip(f"{ASSETS_DIR}/music/theme_fun.wav").volumex(0.7)
    bg_music = afx.audio_loop(m1, duration=total)
    final_video = final_video.set_audio(CompositeAudioClip([final_video.audio, bg_music]))
    final_video.write_videofile(OUTPUT_FILE, codec="mpeg2video", bitrate="5000k", fps=24, audio_codec="mp3")

if __name__ == "__main__":
    assemble()
