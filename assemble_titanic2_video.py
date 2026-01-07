import PIL.Image
if not hasattr(PIL.Image, "ANTIALIAS"): PIL.Image.ANTIALIAS = PIL.Image.LANCZOS
import os
from moviepy.editor import *

ASSETS_DIR = "assets_titanic2"
OUTPUT_FILE = "titanic2_resinkening.mpg"

SCENE_IDS = [
    "01_deep_ocean", "02_billionaire_reveal", "03_raising_the_ship", "04_dry_dock_restoration",
    "05_titanic_2_full", "06_grand_staircase_modern", "07_jack_clone", "08_rose_clone",
    "09_north_atlantic_night", "10_iceberg_waiting", "11_iceberg_tracking", "12_radar_warning",
    "13_captain_panic", "14_the_hit", "15_water_everywhere", "16_ship_tilting",
    "17_jet_ski_escape", "18_billionaire_regret", "19_iceberg_transformation", "20_underwater_fight",
    "21_door_scene_2", "22_propeller_guy_2", "23_orchestra_rock", "24_ship_snapping",
    "25_lifeboat_selfie", "26_iceberg_salute", "27_rescue_ship_iceberg", "28_jack_rose_underwater",
    "29_title_card_t2", "30_coming_july", "31_iceberg_name", "32_post_credits_t2"
]

def assemble():
    clips = []
    for scene_id in SCENE_IDS:
        print(f"Processing Scene: {scene_id}")
        img_path = f"{ASSETS_DIR}/images/{scene_id}.png"
        vo_path = f"{ASSETS_DIR}/voice/{scene_id}.wav"
        sfx_path = f"{ASSETS_DIR}/sfx/{scene_id}.wav"
        
        if not os.path.exists(img_path): continue
            
        audio_clips = []
        duration = 3.5
        
        if os.path.exists(vo_path) and os.path.getsize(vo_path) > 0:
            vo_clip = AudioFileClip(vo_path).set_start(0.5)
            audio_clips.append(vo_clip)
            if vo_clip.duration + 1.0 > duration: duration = vo_clip.duration + 1.0
        
        if os.path.exists(sfx_path) and os.path.getsize(sfx_path) > 0:
            sfx_clip = AudioFileClip(sfx_path).volumex(0.6).subclip(0, duration)
            audio_clips.append(sfx_clip)
            
        img_clip = ImageClip(img_path).set_duration(duration)
        img_clip = img_clip.resize(lambda t: 1 + 0.03 * t) 
        
        if audio_clips:
            img_clip = img_clip.set_audio(CompositeAudioClip(audio_clips))
            
        clips.append(img_clip)

    print("Concatenating...")
    final_video = concatenate_videoclips(clips, method="compose", padding=-0.8)
    
    total = final_video.duration
    m1 = AudioFileClip(f"{ASSETS_DIR}/music/theme_epic_salvage.wav").volumex(0.6)
    m2 = AudioFileClip(f"{ASSETS_DIR}/music/theme_horror_ice.wav").volumex(0.8)
    m3 = AudioFileClip(f"{ASSETS_DIR}/music/theme_action_resink.wav").volumex(0.7)
    
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
