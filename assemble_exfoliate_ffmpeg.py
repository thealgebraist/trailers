#!/usr/bin/env python3
"""
Simple ffmpeg assembler for an "EXFOLIATE" movie built from assets_exfoliate.

Expect structure:
  assets_exfoliate/
    images/
      01_before.png
      01_during.png
      02_before.png
      02_during.png
      ... up to 64
    voice/
      voiceover_01.wav ... voiceover_64.wav
    sfx/
      01_exfoliate.wav ... 64_exfoliate.wav
    music/
      theme_elevator.wav   (should include whisper/moan in the file)

This script concatenates 64 sections, each showing two images (before->during),
plays the per-section voiceover and sfx (loud), mixes in low-volume background music,
and outputs EXFOLIATE.mp4 and EXFOLIATE.mpg.

This is a template: actual assets must be provided by the user.
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

NUM = 64
DEFAULT_BEFORE_DUR = 1.5  # seconds for the "before" image
DEFAULT_EXTRA = 0.5       # extra padding after voice for the "during" image


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
        return 0.0


def build_and_run():
    cmd = ["ffmpeg", "-y"]

    # Collect expected inputs
    video_inputs = []  # pairs of images per section
    voice_files = []
    sfx_files = []

    for i in range(1, NUM + 1):
        idx = f"{i:02d}"
        before = os.path.join(IMG_DIR, f"{idx}_before.png")
        during = os.path.join(IMG_DIR, f"{idx}_during.png")
        video_inputs.append((before, during))

        vf = os.path.join(VOICE_DIR, f"voiceover_{idx}.wav")
        voice_files.append(vf if exists_and_nonzero(vf) else None)

        sf = os.path.join(SFX_DIR, f"{idx}_exfoliate.wav")
        sfx_files.append(sf if exists_and_nonzero(sf) else None)

    # music
    music_file = os.path.join(MUSIC_DIR, "theme_elevator.wav")
    if not exists_and_nonzero(music_file):
        music_file = None

    # Add image inputs: each image is looped for its duration (before image fixed, during depends on voice)
    # We'll add all images as inputs in order: for each section first before then during.
    video_durations = []
    voice_durations = []

    for i, (before, during) in enumerate(video_inputs):
        # probe voice duration
        vd = 0.0
        if voice_files[i]:
            vd = probe_duration(voice_files[i])
        voice_durations.append(vd)

        # durations
        before_dur = DEFAULT_BEFORE_DUR
        during_dur = vd + DEFAULT_EXTRA if vd > 0.001 else 4.0
        video_durations.extend([before_dur, during_dur])

        # add inputs
        if exists_and_nonzero(before):
            cmd += ["-loop", "1", "-t", str(before_dur), "-i", before]
        else:
            # transparent placeholder if missing
            cmd += ["-f", "lavfi", "-t", str(before_dur), "-i", "color=size=1280x720:color=black"]

        if exists_and_nonzero(during):
            cmd += ["-loop", "1", "-t", str(during_dur), "-i", during]
        else:
            cmd += ["-f", "lavfi", "-t", str(during_dur), "-i", "color=size=1280x720:color=black"]

    # Add voice inputs (one per section)
    for i in range(NUM):
        vf = voice_files[i]
        if vf and exists_and_nonzero(vf):
            cmd += ["-i", vf]
        else:
            # silence matching during image duration
            dur = video_durations[2 * i + 1]
            cmd += ["-f", "lavfi", "-t", str(dur), "-i", "anullsrc=r=44100:cl=stereo"]

    # Add SFX inputs (one per section)
    for i in range(NUM):
        sf = sfx_files[i]
        if sf and exists_and_nonzero(sf):
            cmd += ["-i", sf]
        else:
            dur = video_durations[2 * i + 1]
            cmd += ["-f", "lavfi", "-t", str(dur), "-i", "anullsrc=r=44100:cl=stereo"]

    # Add music input (single)
    if music_file:
        cmd += ["-i", music_file]
    else:
        cmd += ["-f", "lavfi", "-t", "120", "-i", "anullsrc=r=44100:cl=stereo"]

    # Build video filter chain: scale/pad each input and concat
    Nvid = 2 * NUM
    v_filters = ""
    for i in range(Nvid):
        dur = video_durations[i]
        base = f"[{i}:v]scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1"
        # small crossfade between the pair boundary could be added, but keep simple
        base += ",format=yuv420p"
        v_filters += f"{base}[v{i}];"
    v_concat_inputs = "".join([f"[v{i}]" for i in range(Nvid)])
    v_filters += f"{v_concat_inputs}concat=n={Nvid}:v=1:a=0[vout];"

    # Audio preparation
    audio_prep = ""
    # voices start after their corresponding image start; voices were appended after all videos, so compute index
    vo_start = Nvid
    sfx_start = vo_start + NUM
    music_start = sfx_start + NUM

    # Prepare voices (resample/format) and silence where appropriate
    for i in range(NUM):
        audio_prep += f"[{vo_start + i}:a]aresample=44100,aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo[vo{i}];"
    vo_inputs = "".join([f"[vo{i}]" for i in range(NUM)])
    audio_prep += f"{vo_inputs}concat=n={NUM}:v=0:a=1[voices];"

    # Prepare sfx
    for i in range(NUM):
        audio_prep += f"[{sfx_start + i}:a]aresample=44100,aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo[sfx{i}];"
    sfx_inputs = "".join([f"[sfx{i}]" for i in range(NUM)])
    audio_prep += f"{sfx_inputs}concat=n={NUM}:v=0:a=1[sfxs];"

    # Prepare music
    audio_prep += f"[{music_start}:a]aresample=44100,aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo[music];"

    # Levels: voices normal, sfx loud, music quiet
    audio_prep += "[voices]volume=1.0[voicesv];"
    audio_prep += "[sfxs]volume=2.0[sfxsv];"  # loud SFX
    audio_prep += "[music]volume=0.15[musicv];"

    # Mix all three
    audio_prep += "[voicesv][sfxsv][musicv]amix=inputs=3:duration=first:normalize=0[aout]"

    cmd += ["-filter_complex", v_filters + audio_prep]
    cmd += ["-map", "[vout]", "-map", "[aout]"]

    # Output mp4
    mp4_cmd = cmd + [
        "-c:v", "libx264", "-preset", "slow", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
        "-r", "24", OUTPUT_MP4
    ]

    print('Running ffmpeg to create MP4...')
    subprocess.run(mp4_cmd, check=True)
    print(f'Created {OUTPUT_MP4}')

    # Convert MP4 to MPG
    mpg_cmd = [
        "ffmpeg", "-y", "-i", OUTPUT_MP4,
        "-c:v", "mpeg2video", "-q:v", "2", "-b:v", "15000k", "-maxrate", "20000k", "-bufsize", "10000k",
        "-c:a", "mp3", "-b:a", "384k", "-ar", "44100", "-ac", "2",
        "-r", "24", OUTPUT_MPG
    ]
    print('Converting MP4 to MPEG2 MPG...')
    subprocess.run(mpg_cmd, check=True)
    print(f'Created {OUTPUT_MPG}')


if __name__ == '__main__':
    build_and_run()
