#!/usr/bin/env python3
import os
import subprocess
import re

ASSETS_DIR = "assets_chimp_train"
img_dir = os.path.join(ASSETS_DIR, "images")
if os.path.isdir(img_dir):
    files = [f for f in os.listdir(img_dir) if re.match(r"^[0-9]{2}_.*\.png$", f) and not f.startswith("00_")]
    files_sorted = sorted(files, key=lambda s: int(s[:2]))
    SCENES = [os.path.splitext(f)[0] for f in files_sorted]
else:
    SCENES = [f"{i:02d}_scene" for i in range(1, 10)]
SCENE_DURATION = 4.0
OUTPUT_MP4 = "charlies_train_banana_adventure.mp4"
OUTPUT_MPG = "charlies_train_banana_adventure.mpg"

def exists_and_nonzero(p):
    return os.path.exists(p) and os.path.getsize(p) > 0

def build_and_run():
    cmd = ["ffmpeg", "-y"]
    scene_count = len(SCENES)

    # Intro definitions
    logo_img = f"{ASSETS_DIR}/images/00_studio_logo.png"
    # logo: fade in 2s, hold 2s, fade out 3s => total 7s
    logo_in = 2.0
    logo_hold = 2.0
    logo_out = 3.0
    logo_dur = logo_in + logo_hold + logo_out

    title_img = f"{ASSETS_DIR}/images/00_title_card.png"
    # title: fade in 1s, hold 2s, fade out 3s => total 6s
    title_in = 1.0
    title_hold = 2.0
    title_out = 3.0
    title_dur = title_in + title_hold + title_out

    # delay audio until after logo + title
    audio_start_delay_ms = int((logo_dur + title_dur) * 1000)

    # Helper to probe duration of audio files
    def probe_duration(path):
        try:
            out = subprocess.run([
                'ffprobe', '-v', 'error', '-select_streams', 'a:0',
                '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', path
            ], capture_output=True, text=True, check=True)
            return float(out.stdout.strip())
        except Exception:
            return 0.0

    # compute voice durations and per-scene video durations
    voice_durations = []
    for i in range(1, scene_count+1):
        vo = f"{ASSETS_DIR}/voice/parler_voiceover_{i}.wav"
        d = probe_duration(vo) if exists_and_nonzero(vo) else 0.0
        voice_durations.append(d)

    # video durations: intro logo, title, then for each scene (voice duration + 1s pause if voice present, else SCENE_DURATION)
    video_durations = [logo_dur, title_dur]
    for vd in voice_durations:
        if vd > 0.001:
            video_durations.append(vd + 1.0)
        else:
            video_durations.append(SCENE_DURATION)
    # Overrides for specific end-story images (if present):
    # image 33: fade in 2s, hold 2s, fade out 2s => total 6s
    # image 34: fade in 2s, hold 3s, fade out 5s => total 10s
    for idx, base in enumerate(SCENES):
        prefix = base.split('_', 1)[0]
        if prefix == '33':
            video_durations[2 + idx] = 2.0 + 2.0 + 2.0
        if prefix == '34':
            video_durations[2 + idx] = 2.0 + 3.0 + 5.0

    # Image inputs: first logo and title, then scene images with computed durations
    # Add logo
    cmd += ["-loop", "1", "-t", str(logo_dur), "-i", logo_img]
    # Add title
    cmd += ["-loop", "1", "-t", str(title_dur), "-i", title_img]
    # Add scene images (use actual filenames from SCENES)
    for idx, base in enumerate(SCENES):
        img = f"{ASSETS_DIR}/images/{base}.png"
        cmd += ["-loop", "1", "-t", str(video_durations[2 + idx]), "-i", img]

    # VO inputs: for each scene add the voice (or silence) then a 1s silence after it
    for i in range(1, scene_count+1):
        vo = f"{ASSETS_DIR}/voice/parler_voiceover_{i}.wav"
        if exists_and_nonzero(vo):
            cmd += ["-i", vo]
        else:
            # fill with silence equal to default scene duration
            cmd += ["-f", "lavfi", "-t", str(SCENE_DURATION), "-i", "anullsrc=r=44100:cl=stereo"]
        # always add 1s pause after each voice
        cmd += ["-f", "lavfi", "-t", "1", "-i", "anullsrc=r=44100:cl=stereo"]

    # SFX inputs
    for i in range(1, scene_count+1):
        sfx = f"{ASSETS_DIR}/sfx/{i:02d}_scene.wav"
        if exists_and_nonzero(sfx):
            cmd += ["-i", sfx]
        else:
            # silence matching the scene image duration
            cmd += ["-f", "lavfi", "-t", str(video_durations[2 + i - 1]), "-i", "anullsrc=r=44100:cl=stereo"]

    # Music (use theme_fun.wav if present, otherwise any .wav in music folder)
    music_files = []
    theme = f"{ASSETS_DIR}/music/theme_fun.wav"
    if exists_and_nonzero(theme):
        music_files = [theme]
    else:
        if os.path.isdir(f"{ASSETS_DIR}/music"):
            for fn in os.listdir(f"{ASSETS_DIR}/music"):
                if fn.lower().endswith('.wav'):
                    music_files.append(f"{ASSETS_DIR}/music/{fn}")
    if not music_files:
        music_files = [None]

    for m in music_files:
        if m and exists_and_nonzero(m):
            cmd += ["-i", m]
        else:
            cmd += ["-f", "lavfi", "-t", "120", "-i", "anullsrc=r=44100:cl=stereo"]

    # Filters
    Nvid = 2 + scene_count
    v_filters = ""
    for i in range(Nvid):
        dur = video_durations[i]
        # use regular fade in/out (no alpha)
        base = f"[{i}:v]scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1"
        if i == 0:
            # logo: fade in (logo_in), hold, fade out (logo_out)
            in_d = logo_in
            out_d = logo_out
            out_st = max(0, dur - out_d)
            # apply regular fade in and fade out
            base += f",fade=t=in:st=0:d={in_d},fade=t=out:st={out_st}:d={out_d}"
        elif i == 1:
            # title: fade in (title_in), hold, fade out (title_out)
            in_d = title_in
            out_d = title_out
            out_st = max(0, dur - out_d)
            base += f",fade=t=in:st=0:d={in_d},fade=t=out:st={out_st}:d={out_d}"
        else:
            # per-scene special fades (e.g., scenes 33 and 34)
            if i >= 2:
                scene_idx = i - 2
                base_name = SCENES[scene_idx]
                prefix = base_name.split('_', 1)[0]
                if prefix == '33':
                    in_d = 2.0
                    out_d = 2.0
                    out_st = max(0, dur - out_d)
                    base += f",fade=t=in:st=0:d={in_d},fade=t=out:st={out_st}:d={out_d}"
                elif prefix == '34':
                    in_d = 2.0
                    out_d = 5.0
                    out_st = max(0, dur - out_d)
                    base += f",fade=t=in:st=0:d={in_d},fade=t=out:st={out_st}:d={out_d}"
        # ensure output is in yuv420p for the encoder
        base += ",format=yuv420p"
        v_filters += f"{base}[v{i}];"
    v_concat = "".join([f"[v{i}]" for i in range(Nvid)])
    v_filters += f"{v_concat}concat=n={Nvid}:v=1:a=0[vout];"

    # Audio indices
    vo_start = Nvid
    sfx_start = vo_start + 2 * scene_count
    music_start = sfx_start + scene_count

    audio_prep = ""
    # VO inputs doubled (voice + 1s pause each)
    for i in range(2 * scene_count):
        audio_prep += f"[{vo_start + i}:a]aresample=44100,aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo[vo{i}];"
    vo_concat = "".join([f"[vo{i}]" for i in range(2 * scene_count)])
    audio_prep += f"{vo_concat}concat=n={2 * scene_count}:v=0:a=1[vo_full];"
    # delay VO until after logo+title
    audio_prep += f"[vo_full]adelay={audio_start_delay_ms}|{audio_start_delay_ms}[vo_del];"

    # SFX Concatenation
    for i in range(scene_count):
        audio_prep += f"[{sfx_start + i}:a]aresample=44100,aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo[sfx{i}];"
    sfx_concat = "".join([f"[sfx{i}]" for i in range(scene_count)])
    audio_prep += f"{sfx_concat}concat=n={scene_count}:v=0:a=1[sfx_full];"
    audio_prep += f"[sfx_full]adelay={audio_start_delay_ms}|{audio_start_delay_ms}[sfx_del];"

    # Music Mixing
    m_count = len(music_files)
    m_prep = ""
    for i in range(m_count):
        m_prep += f"[{music_start + i}:a]aresample=44100,aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo[m{i}];"
    m_inputs = "".join([f"[m{i}]" for i in range(m_count)])
    audio_prep += m_prep + f"{m_inputs}amix=inputs={m_count}:duration=first:normalize=0[music_mix];"
    audio_prep += f"[music_mix]adelay={audio_start_delay_ms}|{audio_start_delay_ms}[music_del];"

    # Final levels and mix (use delayed streams)
    audio_prep += "[vo_del]volume=1.0[vov];"
    audio_prep += "[sfx_del]volume=0.8[sfxv];"
    audio_prep += "[music_del]volume=0.3[musicv];"
    audio_prep += "[vov][sfxv][musicv]amix=inputs=3:duration=first:normalize=0[aout]"

    cmd += ["-filter_complex", v_filters + audio_prep]
    cmd += ["-map", "[vout]", "-map", "[aout]"]

    # First create MP4 (H.264 + AAC)
    mp4_cmd = cmd + [
        "-c:v", "libx264", "-preset", "slow", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
        "-r", "24", OUTPUT_MP4
    ]

    print('Running ffmpeg to create MP4...')
    subprocess.run(mp4_cmd, check=True)
    print(f'Created {OUTPUT_MP4}')

    # Convert MP4 to MPEG2 MPG (audio as mp3)
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
