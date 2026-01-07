import PIL.Image
if not hasattr(PIL.Image, "ANTIALIAS"): PIL.Image.ANTIALIAS = PIL.Image.LANCZOS
import os
from moviepy.editor import *

ASSETS_DIR = "assets_romcom"
OUTPUT_FILE = "plungers_refrain.mpg"

SCENE_IDS = [
    "01_industrial_north", "02_dalek_flat_cap", "03_factory_line", "04_the_meeting",
    "05_shared_tea", "06_stern_father", "07_father_shouting", "08_back_alley",
    "09_tap_dancing_reveal", "10_gene_kelly_homage", "11_technicolor_shift", "12_dance_on_stairs",
    "13_tuxedo_dalek", "14_chorus_line", "15_plunger_flute", "16_back_to_reality",
    "17_the_breakup", "18_rusty_drinking", "19_talent_poster", "20_entering_hall",
    "21_taking_stage", "22_first_tap", "23_music_explodes", "24_crowd_cheering",
    "25_betty_joins", "26_the_lift", "27_award_ceremony", "28_walking_home",
    "29_title_card_romcom", "30_critics_quotes", "31_slogan_romcom", "32_post_credits_romcom"
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
    m1 = AudioFileClip(f"{ASSETS_DIR}/music/theme_gritty.wav").volumex(0.6)
    m2 = AudioFileClip(f"{ASSETS_DIR}/music/theme_tap.wav").volumex(0.8)
    m3 = AudioFileClip(f"{ASSETS_DIR}/music/theme_romantic_finale.wav").volumex(0.7)
    
    t1, t2 = total * 0.33, total * 0.75 # Transitions
    
    bg_music = CompositeAudioClip([
        afx.audio_loop(m1, duration=t1+5).set_start(0).fx(afx.audio_fadeout, 3),
        afx.audio_loop(m2, duration=(t2-t1)+5).set_start(t1).fx(afx.audio_fadein, 3).fx(afx.audio_fadeout, 3),
        afx.audio_loop(m3, duration=(total-t2)).set_start(t2).fx(afx.audio_fadein, 3)
    ])
    
    final_video = final_video.set_audio(CompositeAudioClip([final_video.audio, bg_music]))
    final_video.write_videofile(OUTPUT_FILE, codec="mpeg2video", bitrate="8000k", fps=24, audio_codec="mp3")

if __name__ == "__main__":
    assemble()
