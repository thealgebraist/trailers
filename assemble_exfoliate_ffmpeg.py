#!/usr/bin/env python3
"""
Simple ffmpeg assembler for EXFOLIATE.
Ensures that "image N" (the during image) is shown ONLY while "speak N" (the voice) is playing.
"""

import os
import subprocess
import re

ROOT = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(ROOT, "assets_exfoliate")
IMG_DIR = os.path.join(ASSETS_DIR, "images")
VOICE_DIR = os.path.join(ASSETS_DIR, "voice")
SFX_DIR = os.path.join(ASSETS_DIR, "sfx")
MUSIC_DIR = os.path.join(ASSETS_DIR, "music")

OUTPUT_MP4 = "EXFOLIATE.mp4"
OUTPUT_MPG = "EXFOLIATE.mpg"

NUM_SCENES = 66
BEFORE_DUR = 1.5  # Fixed duration for the "before" images

def exists_and_nonzero(p):
    return p and os.path.exists(p) and os.path.getsize(p) > 0

def probe_duration(path):
    try:
        out = subprocess.run([
            'ffprobe', '-v', 'error', '-select_streams', 'a:0',
            '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', path
        ], capture_output=True, text=True, check=True)
        return float(out.stdout.strip())
    except Exception:
        return 2.0  # Default fallback

def build_and_run():
    cmd = ["ffmpeg", "-y"]

    # We will build a complex filter to handle all synchronization.
    # First, gather all durations and add all inputs.
    
    video_durations = []
    inputs_count = 0
    
    # 1. Add Video Inputs
    for i in range(NUM_SCENES):
        idx = f"{i:02d}"
        v_path = os.path.join(VOICE_DIR, f"voice_{idx}.wav")
        vd = probe_duration(v_path) if exists_and_nonzero(v_path) else 2.0
        
        if i < 2:
            # Intro scenes: Single image
            # USER SPEC: 00: fade in 2s, keep 2s, fade out 3s (Total 7s)
            #           01: fade in 2s, keep 2s, fade out 4s (Total 8s)
            target_dur = 7.0 if i == 0 else 8.0
            vd = max(vd, target_dur)
            img_path = os.path.join(IMG_DIR, f"{idx}_scene.png")
            if exists_and_nonzero(img_path):
                cmd += ["-loop", "1", "-t", str(vd), "-i", img_path]
            else:
                cmd += ["-f", "lavfi", "-t", str(vd), "-i", "color=c=black:s=1280x720"]
            video_durations.append(vd)
            inputs_count += 1
        else:
            # Exfoliation scenes: Before and During
            # USER SPEC: before for half duration, during for last half
            d_half = vd / 2.0
            
            before_path = os.path.join(IMG_DIR, f"{idx}_before.png")
            during_path = os.path.join(IMG_DIR, f"{idx}_during.png")
            
            # Before
            if exists_and_nonzero(before_path):
                cmd += ["-loop", "1", "-t", str(d_half), "-i", before_path]
            else:
                cmd += ["-f", "lavfi", "-t", str(d_half), "-i", "color=c=black:s=1280x720"]
            video_durations.append(d_half)
            
            # During
            if exists_and_nonzero(during_path):
                cmd += ["-loop", "1", "-t", str(vd - d_half), "-i", during_path]
            else:
                cmd += ["-f", "lavfi", "-t", str(vd - d_half), "-i", "color=c=black:s=1280x720"]
            video_durations.append(vd - d_half)
            inputs_count += 2

    # Total video inputs: 2 (intro) + 64*2 = 130
    Nvid = inputs_count
    
    # 2. Add Voice Inputs
    for i in range(NUM_SCENES):
        idx = f"{i:02d}"
        v_path = os.path.join(VOICE_DIR, f"voice_{idx}.wav")
        if exists_and_nonzero(v_path):
            cmd += ["-i", v_path]
        else:
            vd = video_durations[1 + (i-2)*2] if i >= 2 else video_durations[i]
            cmd += ["-f", "lavfi", "-t", str(vd), "-i", "anullsrc=r=44100:cl=stereo"]
    
    # 3. Add SFX Inputs
    for i in range(NUM_SCENES):
        idx = f"{i:02d}"
        s_path = os.path.join(SFX_DIR, f"{idx}_exfoliate.wav")
        if exists_and_nonzero(s_path):
            cmd += ["-i", s_path]
        else:
            vd = video_durations[1 + (i-2)*2] if i >= 2 else video_durations[i]
            cmd += ["-f", "lavfi", "-t", str(vd), "-i", "anullsrc=r=44100:cl=stereo"]

    # 4. Add Music Input
    music_file = os.path.join(MUSIC_DIR, "theme_elevator.wav")
    if exists_and_nonzero(music_file):
        cmd += ["-i", music_file]
    else:
        cmd += ["-f", "lavfi", "-t", "600", "-i", "anullsrc=r=44100:cl=stereo"]

    # Filter Complex
    v_filters = ""
    for i in range(Nvid):
        base = f"[{i}:v]scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p"
        if i == 0:
            # Studios: in 2s, out 3s at 4s (total 7s)
            base += ",fade=t=in:st=0:d=2,fade=t=out:st=4:d=3"
        elif i == 1:
            # Title: in 2s, out 4s at 4s (total 8s)
            base += ",fade=t=in:st=0:d=2,fade=t=out:st=4:d=4"
        v_filters += f"{base}[v{i}];"
    v_concat = "".join([f"[v{i}]" for i in range(Nvid)])
    v_filters += f"{v_concat}concat=n={Nvid}:v=1:a=0[vout];"

    audio_prep = ""
    vo_start = Nvid
    sfx_start = vo_start + NUM_SCENES
    music_start = sfx_start + NUM_SCENES

    # For each scene, prepare the audio segment
    # Intro scenes: voice + sfx (BUT user requested no speak for title/logo)
    # Exfoliation scenes: voice + sfx
    for i in range(NUM_SCENES):
        v_idx = vo_start + i
        s_idx = sfx_start + i
        
        # Prepare voice (silence for first two)
        if i < 2:
            audio_prep += f"anullsrc=r=44100:cl=stereo:d={video_durations[i]}[vo{i}];"
        else:
            audio_prep += f"[{v_idx}:a]aresample=44100,aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo[vo{i}];"
        
        # Prepare SFX
        audio_prep += f"[{s_idx}:a]aresample=44100,aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo[sf{i}];"
        
        # Mix voice and sfx
        audio_prep += f"[vo{i}][sf{i}]amix=inputs=2:duration=first:normalize=0[mix{i}];"
        
        if i < 2:
            # Pad to video duration
            dur = video_durations[i]
            audio_prep += f"[mix{i}]apad=whole_len={int(dur * 44100)}[aseg{i}];"
        else:
            # Direct mix (plays during both before/during images)
            audio_prep += f"[mix{i}]acopy[aseg{i}];"

    # Concat all audio segments
    a_concat = "".join([f"[aseg{i}]" for i in range(NUM_SCENES)])
    audio_prep += f"{a_concat}concat=n={NUM_SCENES}:v=0:a=1[full_content];"

    # Prepare music
    audio_prep += f"[{music_start}:a]aresample=44100,aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,volume=0.08[musicv];"
    
    # Mix content and music
    audio_prep += "[full_content]volume=1.0[contentv];"
    audio_prep += "[contentv][musicv]amix=inputs=2:duration=first:normalize=0[aout]"

    cmd += ["-filter_complex", v_filters + audio_prep]
    cmd += ["-map", "[vout]", "-map", "[aout]"]

    # Output settings
    cmd += [
        "-c:v", "libx264", "-preset", "slow", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
        "-r", "24", OUTPUT_MP4
    ]

    print(f"Running FFmpeg to create {OUTPUT_MP4}...")
    subprocess.run(cmd, check=True)
    print(f"Created {OUTPUT_MP4}")

    # Convert to MPG
    mpg_cmd = [
        "ffmpeg", "-y", "-i", OUTPUT_MP4,
        "-c:v", "mpeg2video", "-q:v", "2", "-b:v", "15000k", "-maxrate", "20000k", "-bufsize", "10000k",
        "-c:a", "mp3", "-b:a", "384k", "-ar", "44100", "-ac", "2",
        "-r", "24", OUTPUT_MPG
    ]
    print(f"Converting to {OUTPUT_MPG}...")
    subprocess.run(mpg_cmd, check=True)
    print(f"Created {OUTPUT_MPG}")

if __name__ == "__main__":
    build_and_run()