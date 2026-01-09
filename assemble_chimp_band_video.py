#!/Users/anders/projects/dalek-comes-home/.venv/bin/python3
"""
Assemble the Chimp Band 64 video.
64 scenes with image + corresponding SFX.
Background music mixed in.
Outputs MP4 and MPEG2 at 95% quality.
"""

import PIL.Image
if not hasattr(PIL.Image, "ANTIALIAS"):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

import os
import re
from moviepy.editor import ImageClip, AudioFileClip, CompositeAudioClip, concatenate_videoclips, afx

ASSETS_DIR = "assets_chimp_band_64"
MP4_OUTPUT = "chimp_band_64.mp4"
MPG_OUTPUT = "chimp_band_64.mpg"

def get_scene_files():
    images_dir = os.path.join(ASSETS_DIR, "images")
    sfx_dir = os.path.join(ASSETS_DIR, "sfx")
    
    # Match files like 01_scene.png and 01_scene.wav
    pattern = re.compile(r"(\d+)_scene\.png")
    
    scenes = []
    for f in sorted(os.listdir(images_dir)):
        match = pattern.match(f)
        if match:
            idx = match.group(1)
            img_path = os.path.join(images_dir, f)
            sfx_path = os.path.join(sfx_dir, f"{idx}_scene.wav")
            scenes.append((img_path, sfx_path))
    return scenes

def assemble():
    scenes = get_scene_files()
    if not scenes:
        print("No scenes found!")
        return

    clips = []
    print(f"Assembling {len(scenes)} scenes...")

    for img_path, sfx_path in scenes:
        duration = 3.0  # default duration
        audio_clips = []
        
        if os.path.exists(sfx_path):
            sfx_clip = AudioFileClip(sfx_path)
            audio_clips.append(sfx_clip)
            if sfx_clip.duration > duration:
                duration = sfx_clip.duration
        
        img_clip = ImageClip(img_path).set_duration(duration)
        # Subtle zoom effect
        img_clip = img_clip.resize(lambda t: 1 + 0.03 * t)
        
        if audio_clips:
            img_clip = img_clip.set_audio(CompositeAudioClip(audio_clips))
        
        clips.append(img_clip)

    print("Concatenating clips...")
    final_video = concatenate_videoclips(clips, method="compose")
    
    # Add background music
    music_dir = os.path.join(ASSETS_DIR, "music")
    music_files = [os.path.join(music_dir, f) for f in sorted(os.listdir(music_dir)) if f.endswith(".wav")]
    
    if music_files:
        # Just use the first one and loop it
        bg_music = AudioFileClip(music_files[0]).volumex(0.3)
        bg_music = afx.audio_loop(bg_music, duration=final_video.duration)
        final_video = final_video.set_audio(CompositeAudioClip([final_video.audio, bg_music]))

    print(f"Writing MP4: {MP4_OUTPUT}")
    # libx264 crf 18 is roughly 95% quality (range is 0-51, 18-28 is typical)
    final_video.write_videofile(MP4_OUTPUT, codec="libx264", audio_codec="aac", fps=24, ffmpeg_params=["-crf", "18"])
    
    print(f"Writing MPEG2: {MPG_OUTPUT}")
    # -q:v 2 is very high quality for mpeg2 (1-31, lower is better)
    final_video.write_videofile(MPG_OUTPUT, codec="mpeg2video", audio_codec="mp2", fps=24, bitrate="15000k", ffmpeg_params=["-q:v", "2"])

if __name__ == "__main__":
    assemble()
