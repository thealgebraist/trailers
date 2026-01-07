import PIL.Image
if not hasattr(PIL.Image, "ANTIALIAS"): PIL.Image.ANTIALIAS = PIL.Image.LANCZOS
import os
import glob
from moviepy.editor import *
from moviepy.config import change_settings

ASSETS_DIR = "assets_snakes"
OUTPUT_FILE = "snakes_on_titanic.mpg"

SCENE_IDS = [
    "01_titanic_dock", "02_luxury_interior", "03_captain_pride", "04_cargo_hold",
    "05_crate_shake", "06_iceberg_lookout", "07_iceberg_impact", "08_crates_break",
    "09_snakes_hallway", "10_tea_time_terror", "11_ballroom_chaos", "12_draw_me",
    "13_samuel_captain", "14_snakes_on_bow", "15_smokestack_wrap", "16_flooded_stairs",
    "17_lifeboat_full", "18_propeller_guy", "19_door_scene", "20_violin_band",
    "21_underwater_swim", "22_hero_shot", "23_ship_snap", "24_car_scene",
    "25_survivors", "26_old_rose", "27_celine_dion_snake", "28_snake_plane",
    "29_iceberg_face", "30_title_card", "31_slogan", "32_post_credits"
]

TRANSITION_DURATION = 1.0

def assemble():
    clips = []
    
    for i, scene_id in enumerate(SCENE_IDS):
        print(f"Processing Scene: {scene_id}")
        
        img_path = f"{ASSETS_DIR}/images/{scene_id}.png"
        vo_path = f"{ASSETS_DIR}/voice/{scene_id}.wav"
        sfx_path = f"{ASSETS_DIR}/sfx/{scene_id}.wav"
        
        if not os.path.exists(img_path):
            print(f"Missing image for {scene_id}, skipping.")
            continue
            
        audio_clips = []
        duration = 3.5 
        
        if os.path.exists(vo_path) and os.path.getsize(vo_path) > 0:
            vo_clip = AudioFileClip(vo_path)
            audio_clips.append(vo_clip.set_start(0.5))
            if vo_clip.duration + 1.0 > duration:
                duration = vo_clip.duration + 1.0
        
        if os.path.exists(sfx_path) and os.path.getsize(sfx_path) > 0:
            sfx_clip = AudioFileClip(sfx_path).volumex(0.6)
            if sfx_clip.duration < duration:
                pass 
            else:
                sfx_clip = sfx_clip.subclip(0, duration)
            audio_clips.append(sfx_clip)
            
        img_clip = ImageClip(img_path).set_duration(duration)
        img_clip = img_clip.resize(lambda t: 1 + 0.04 * t) 
        
        if audio_clips:
            scene_audio = CompositeAudioClip(audio_clips)
            img_clip = img_clip.set_audio(scene_audio)
        
        clips.append(img_clip)

    print(f"Concatenating {len(clips)} clips...")
    final_video = concatenate_videoclips(clips, method="compose", padding=-TRANSITION_DURATION)
    
    # --- Background Music ---
    total_duration = final_video.duration
    
    music_suspense = f"{ASSETS_DIR}/music/theme_suspense.wav"
    music_disaster = f"{ASSETS_DIR}/music/theme_disaster.wav"
    music_funny = f"{ASSETS_DIR}/music/theme_romantic_snake.wav"
    
    music_clips = []
    
    # 0-30% Suspense (Act 1-2)
    # 30-70% Disaster (Act 3-4)
    # 70-100% Funny/Romantic (Finale)
    
    t1 = total_duration * 0.30
    t2 = total_duration * 0.70
    
    if os.path.exists(music_suspense):
        m1 = AudioFileClip(music_suspense).volumex(0.6)
        m1 = afx.audio_loop(m1, duration=t1 + 5)
        m1 = m1.set_start(0).fx(afx.audio_fadeout, 5)
        music_clips.append(m1)
        
    if os.path.exists(music_disaster):
        m2 = AudioFileClip(music_disaster).volumex(0.8)
        m2 = afx.audio_loop(m2, duration=(t2 - t1) + 5)
        m2 = m2.set_start(t1).fx(afx.audio_fadein, 5).fx(afx.audio_fadeout, 5)
        music_clips.append(m2)

    if os.path.exists(music_funny):
        m3 = AudioFileClip(music_funny).volumex(0.7)
        m3 = afx.audio_loop(m3, duration=(total_duration - t2))
        m3 = m3.set_start(t2).fx(afx.audio_fadein, 5).fx(afx.audio_fadeout, 5)
        music_clips.append(m3)

    if music_clips:
        bg_music = CompositeAudioClip(music_clips)
        final_audio = CompositeAudioClip([final_video.audio, bg_music])
        final_video = final_video.set_audio(final_audio)

    print("Writing Video File...")
    final_video.write_videofile(
        OUTPUT_FILE, 
        codec="mpeg2video", 
        bitrate="8000k", 
        fps=24,
        audio_codec="mp3"
    )
    print(f"Done: {OUTPUT_FILE}")

if __name__ == "__main__":
    assemble()
