import PIL.Image
if not hasattr(PIL.Image, "ANTIALIAS"): PIL.Image.ANTIALIAS = PIL.Image.LANCZOS
import os
from moviepy.editor import *

ASSETS_DIR = "assets_boring"
OUTPUT_FILE = "the_damp_patch.mpg"

SCENE_IDS = [
    "01_the_wall", "02_dripping_tap", "03_arthur_staring", "04_the_patch_macro",
    "05_beige_curtains", "06_grey_porridge", "07_the_radiator", "08_arthur_watch",
    "09_dust_mote", "10_peeling_wallpaper", "11_arthur_sigh", "12_the_window_rain",
    "13_chipped_mug", "14_dead_fly", "15_the_ceiling_light", "16_arthur_sitting",
    "17_the_patch_grows", "18_close_up_eye", "19_empty_bookshelf", "20_shadow_movement",
    "21_arthur_standing", "22_touching_the_wall", "23_the_contact", "24_reaction_shot",
    "25_the_kettle", "26_staring_again", "27_fading_out", "28_title_card_boring",
    "29_slogan_boring", "30_coming_whenever", "31_critic_quote", "32_post_credits_boring"
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
        duration = 4.0 # Slightly longer for boring effect
        
        if os.path.exists(vo_path) and os.path.getsize(vo_path) > 0:
            vo_clip = AudioFileClip(vo_path).set_start(0.5)
            audio_clips.append(vo_clip)
            if vo_clip.duration + 1.0 > duration: duration = vo_clip.duration + 1.0
        
        if os.path.exists(sfx_path) and os.path.getsize(sfx_path) > 0:
            sfx_clip = AudioFileClip(sfx_path).volumex(0.4).subclip(0, duration)
            audio_clips.append(sfx_clip)
            
        img_clip = ImageClip(img_path).set_duration(duration)
        # Slow, almost imperceptible zoom
        img_clip = img_clip.resize(lambda t: 1 + 0.01 * t) 
        
        if audio_clips:
            img_clip = img_clip.set_audio(CompositeAudioClip(audio_clips))
            
        clips.append(img_clip)

    print("Concatenating...")
    # Long crossfades for maximum depression
    final_video = concatenate_videoclips(clips, method="compose", padding=-1.5)
    
    total = final_video.duration
    m1 = AudioFileClip(f"{ASSETS_DIR}/music/theme_hum.wav").volumex(0.5)
    m2 = AudioFileClip(f"{ASSETS_DIR}/music/theme_clock.wav").volumex(0.5)
    m3 = AudioFileClip(f"{ASSETS_DIR}/music/theme_depressing_cello.wav").volumex(0.6)
    
    t1, t2 = total * 0.4, total * 0.7
    
    bg_music = CompositeAudioClip([
        afx.audio_loop(m1, duration=t1+5).set_start(0).fx(afx.audio_fadeout, 5),
        afx.audio_loop(m2, duration=(t2-t1)+5).set_start(t1).fx(afx.audio_fadein, 5).fx(afx.audio_fadeout, 5),
        afx.audio_loop(m3, duration=(total-t2)).set_start(t2).fx(afx.audio_fadein, 5)
    ])
    
    final_video = final_video.set_audio(CompositeAudioClip([final_video.audio, bg_music]))
    final_video.write_videofile(OUTPUT_FILE, codec="mpeg2video", bitrate="5000k", fps=24, audio_codec="mp3")

if __name__ == "__main__":
    assemble()
