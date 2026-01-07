import PIL.Image
if not hasattr(PIL.Image, "ANTIALIAS"): PIL.Image.ANTIALIAS = PIL.Image.LANCZOS
import os
from moviepy.editor import *

ASSETS_DIR = "assets_luftwaffe"
OUTPUT_FILE = "planet_of_the_luftwaffe.mpg"

SCENE_IDS = [
    "01_dogfight", "02_crash_landing", "03_waking_up", "04_reveal_ape", "05_title_splash",
    "06_berlin_jungle", "07_marching_chimps", "08_gorilla_guards", "09_hitler_kong", "10_rick_defiant",
    "11_dirty_ape_line", "12_human_pets", "13_resistance_meeting", "14_secret_weapon_plan", "15_banana_rocket",
    "16_poop_grenade", "17_dogfight_monkeys", "18_tank_battle", "19_bananas", "20_bunker_assault",
    "21_interrogation", "22_rocket_launch", "23_plane_wing_fight", "24_romantic_kiss", "25_statue_of_liberty",
    "26_rick_screaming", "27_monkey_laugh", "28_hero_walk", "29_title_card_2", "30_coming_soon",
    "31_final_joke", "32_post_credits"
]

def assemble():
    clips = []
    
    for scene_id in SCENE_IDS:
        print(f"Processing Scene: {scene_id}")
        img_path = f"{ASSETS_DIR}/images/{scene_id}.png"
        vo_path = f"{ASSETS_DIR}/voice/{scene_id}.wav"
        sfx_path = f"{ASSETS_DIR}/sfx/{scene_id}.wav"
        
        if not os.path.exists(img_path):
            print(f"Skipping {scene_id} (No Image)")
            continue
            
        audio_clips = []
        duration = 3.5
        
        if os.path.exists(vo_path) and os.path.getsize(vo_path) > 0:
            vo_clip = AudioFileClip(vo_path).set_start(0.5)
            audio_clips.append(vo_clip)
            if vo_clip.duration + 1.0 > duration:
                duration = vo_clip.duration + 1.0
        
        if os.path.exists(sfx_path) and os.path.getsize(sfx_path) > 0:
            sfx_clip = AudioFileClip(sfx_path).volumex(0.6).subclip(0, duration)
            audio_clips.append(sfx_clip)
            
        img_clip = ImageClip(img_path).set_duration(duration)
        img_clip = img_clip.resize(lambda t: 1 + 0.03 * t) # Slow zoom
        
        if audio_clips:
            img_clip = img_clip.set_audio(CompositeAudioClip(audio_clips))
            
        clips.append(img_clip)

    print("Concatenating...")
    final_video = concatenate_videoclips(clips, method="compose", padding=-0.8)
    
    # Music
    total = final_video.duration
    m1 = AudioFileClip(f"{ASSETS_DIR}/music/theme_march.wav").volumex(0.6)
    m2 = AudioFileClip(f"{ASSETS_DIR}/music/theme_action.wav").volumex(0.7)
    m3 = AudioFileClip(f"{ASSETS_DIR}/music/theme_absurd.wav").volumex(0.7)
    
    t1, t2 = total * 0.33, total * 0.66
    
    bg_music = CompositeAudioClip([
        afx.audio_loop(m1, duration=t1+5).set_start(0).fx(afx.audio_fadeout, 3),
        afx.audio_loop(m2, duration=(t2-t1)+5).set_start(t1).fx(afx.audio_fadein, 3).fx(afx.audio_fadeout, 3),
        afx.audio_loop(m3, duration=(total-t2)).set_start(t2).fx(afx.audio_fadein, 3)
    ])
    
    final_video = final_video.set_audio(CompositeAudioClip([final_video.audio, bg_music]))
    
    final_video.write_videofile(OUTPUT_FILE, codec="mpeg2video", bitrate="8000k", fps=24, audio_codec="mp3")

if __name__ == "__main__":
    assemble()
