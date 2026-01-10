#!/usr/bin/env python3
"""
Assembler for Soviet Alf Trailer.
- Grainy B&W style.
- Bark VO with fake laughs.
- Soviet sitcom music.
"""

import os
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(ROOT, "assets_soviet_alf")
IMG_DIR = os.path.join(ASSETS_DIR, "images")
VOICE_DIR = os.path.join(ASSETS_DIR, "voice")
MUSIC_DIR = os.path.join(ASSETS_DIR, "music")

OUTPUT_FILE = "soviet_alf_trailer.mp4"
NUM_SCENES = 6

def probe_duration(path):
    try:
        out = subprocess.run([
            'ffprobe', '-v', 'error', '-select_streams', 'a:0',
            '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', path
        ], capture_output=True, text=True, check=True)
        return float(out.stdout.strip())
    except:
        return 5.0

def build():
    # 1. Collect inputs and durations
    voice_durations = []
    for i in range(NUM_SCENES):
        v_path = os.path.join(VOICE_DIR, f"voice_{i:02d}.wav")
        voice_durations.append(probe_duration(v_path))

    cmd = ["ffmpeg", "-y"]
    
    # Image inputs
    for i in range(NUM_SCENES):
        path = os.path.join(IMG_DIR, f"scene_{i:02d}.png")
        cmd += ["-loop", "1", "-t", str(voice_durations[i]), "-i", path]
        
    # Voice inputs
    for i in range(NUM_SCENES):
        path = os.path.join(VOICE_DIR, f"voice_{i:02d}.wav")
        cmd += ["-i", path]
        
    # Music input
    music_path = os.path.join(MUSIC_DIR, "theme.wav")
    if os.path.exists(music_path):
        cmd += ["-i", music_path]
    else:
        cmd += ["-f", "lavfi", "-t", "60", "-i", "anullsrc=r=44100:cl=stereo"]

    # Filter Complex
    # Visual: scale, grain, B&W, and concat
    v_filters = ""
    for i in range(NUM_SCENES):
        # Apply black and white, grain, and noise for that soviet feel
        v_filters += (
            f"[{i}:v]scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1,"
            f"format=gray,noise=alls=20:allf=t+u,curves=vintage[v{i}];"
        )
    v_concat = "".join([f"[v{i}]" for i in range(NUM_SCENES)])
    v_filters += f"{v_concat}concat=n={NUM_SCENES}:v=1:a=0[vout];"
    
    # Audio: concat voices and mix music
    vo_start = NUM_SCENES
    music_start = NUM_SCENES * 2
    
    a_filters = ""
    for i in range(NUM_SCENES):
        a_filters += f"[{vo_start+i}:a]aresample=44100,aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo[a{i}];"
    a_concat = "".join([f"[a{i}]" for i in range(NUM_SCENES)])
    a_filters += f"{a_concat}concat=n={NUM_SCENES}:v=0:a=1[vofull];"
    
    # Layer music
    a_filters += f"[{music_start}:a]aresample=44100,volume=0.4[musicv];"
    a_filters += "[vofull][musicv]amix=inputs=2:duration=first:normalize=0[aout]"
    
    cmd += ["-filter_complex", v_filters + a_filters]
    cmd += ["-map", "[vout]", "-map", "[aout]"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "24", OUTPUT_FILE]
    
    print("--- Executing Final Assembly ---")
    subprocess.run(cmd, check=True)
    print(f"Trailer created: {OUTPUT_FILE}")

if __name__ == "__main__":
    build()
