import PIL.Image
if not hasattr(PIL.Image, "ANTIALIAS"): PIL.Image.ANTIALIAS = PIL.Image.LANCZOS
import os
from moviepy.editor import *

ASSETS_DIR = "assets_wait"
OUTPUT_FILE = "the_kettle_240s.mpg"

SCENE_IDS = [
    "01_kitchen_wide", "02_the_kettle_still", "03_hand_reaching", "04_filling_water",
    "05_closing_lid", "06_pushing_button", "07_the_wait_begins", "08_clock_close",
    "09_bubbles_forming", "10_steam_rising", "11_face_extreme_close", "12_kettle_shaking",
    "13_steam_cloud", "14_orchestra_peak_1", "15_the_almost_click", "16_the_fakeout",
    "17_the_quiet_return", "18_empty_cup", "19_arthur_sigh_wait", "20_re_pushing_button",
    "21_red_glow", "22_violent_boil", "23_steam_jet", "24_arthur_screaming",
    "25_kettle_levitating", "26_the_boiling_point", "27_final_crescendo", "28_the_pop",
    "29_title_card_kettle", "30_slogan_kettle", "31_coming_soon_kettle", "32_post_credits_kettle"
]

SCENE_DURATION = 240.0 / len(SCENE_IDS) # Exactly 7.5 seconds per scene

def assemble():
    clips = []
    for i, scene_id in enumerate(SCENE_IDS):
        print(f"Scene {i+1}/32: {scene_id}")
        img_path = f"{ASSETS_DIR}/images/{scene_id}.png"
        vo_path = f"{ASSETS_DIR}/voice/{scene_id}.wav"
        sfx_path = f"{ASSETS_DIR}/sfx/{scene_id}.wav"
        
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_path} not found.")
            continue
            
        audio_layers = []
        
        if os.path.exists(vo_path) and os.path.getsize(vo_path) > 0:
            vo_clip = AudioFileClip(vo_path).set_start(1.0) # Start VO 1s into scene
            audio_layers.append(vo_clip)
        
        if os.path.exists(sfx_path) and os.path.getsize(sfx_path) > 0:
            sfx_clip = AudioFileClip(sfx_path).volumex(0.6)
            if sfx_clip.duration > SCENE_DURATION:
                sfx_clip = sfx_clip.subclip(0, SCENE_DURATION)
            audio_layers.append(sfx_clip)
            
        img_clip = ImageClip(img_path).set_duration(SCENE_DURATION)
        # Slow cinematic zoom
        img_clip = img_clip.resize(lambda t: 1 + 0.02 * t) 
        
        if audio_layers:
            img_clip = img_clip.set_audio(CompositeAudioClip(audio_layers))
            
        clips.append(img_clip)

    print("Concatenating 240s video...")
    # Minimal overlap to maintain rhythm
    final_video = concatenate_videoclips(clips, method="compose")
    
    # --- Background Music / Tension Layers ---
    # Phase 1: 0 - 120s (Build 1)
    # Phase 2: 120 - 150s (Silence/Quiet)
    # Phase 3: 150 - 230s (Build 2)
    # Phase 4: 230 - 240s (End)
    
    m1_path = f"{ASSETS_DIR}/music/theme_build_1.wav"
    m2_path = f"{ASSETS_DIR}/music/theme_silence_bridge.wav"
    m3_path = f"{ASSETS_DIR}/music/theme_build_2.wav"
    
    music_layers = []
    
    if os.path.exists(m1_path):
        m1 = AudioFileClip(m1_path).volumex(0.7)
        m1 = afx.audio_loop(m1, duration=120.0).set_start(0).fx(afx.audio_fadeout, 2)
        music_layers.append(m1)
        
    if os.path.exists(m2_path):
        m2 = AudioFileClip(m2_path).volumex(0.4)
        m2 = afx.audio_loop(m2, duration=30.0).set_start(120.0).fx(afx.audio_fadein, 2).fx(afx.audio_fadeout, 2)
        music_layers.append(m2)

    if os.path.exists(m3_path):
        m3 = AudioFileClip(m3_path).volumex(0.9)
        m3 = afx.audio_loop(m3, duration=80.0).set_start(150.0).fx(afx.audio_fadein, 2).fx(afx.audio_fadeout, 1)
        music_layers.append(m3)

    if music_layers:
        bg_music = CompositeAudioClip(music_layers)
        final_video = final_video.set_audio(CompositeAudioClip([final_video.audio, bg_music]))

    print("Writing 4-minute HQ Video...")
    final_video.write_videofile(OUTPUT_FILE, codec="mpeg2video", bitrate="10000k", fps=24, audio_codec="mp3")

if __name__ == "__main__":
    assemble()
