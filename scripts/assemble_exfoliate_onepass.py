#!/usr/bin/env python3
"""
Single-pass FFmpeg assembler for EXFOLIATE assets producing one MP4 and one MPG
- Builds a single ffmpeg invocation (filter_complex) so no per-segment temp files are created
- Flow: logo -> title -> for each of 4 subject groups: subject card (10s with subject_{idx}_elevator.wav) -> 16 scenes (each scene: body, close, doctor images shown sequentially while the scene voice plays split across them)

Output: exfoliate_single.mp4 (H.264 high quality) and exfoliate_single.mpg (MPEG-2)

Requires ffmpeg/ffprobe on PATH.
"""

import os
import subprocess
from pathlib import Path
import math

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets_exfoliate_positive"
CARD_DIR = ASSETS / "cards"
IMG_DIR = ASSETS / "images"
VOICE_DIR = ASSETS / "voice"
OUT_MP4 = "exfoliate_single.mp4"
OUT_MPG = "exfoliate_single.mpg"

# subjects are groups of 16 scenes
GROUP_SIZE = 16
TOTAL_SCENES = 64
SUBJECT_PROMPTS_IDX = [15, 31, 47, 63]

# durations
LOGO_IN = 2.0
LOGO_HOLD = 2.0
LOGO_OUT = 4.0
LOGO_DUR = LOGO_IN + LOGO_HOLD + LOGO_OUT
TITLE_IN = 2.0
TITLE_HOLD = 2.0
TITLE_OUT = 4.0
TITLE_DUR = TITLE_IN + TITLE_HOLD + TITLE_OUT
SUBJECT_CARD_DUR = 10.0

# helpers

def exists(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 0


def probe_duration(path: Path) -> float:
    try:
        out = subprocess.run([
            'ffprobe', '-v', 'error', '-select_streams', 'a:0',
            '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', str(path)
        ], capture_output=True, text=True, check=True)
        return float(out.stdout.strip())
    except Exception:
        return 0.0


def build():
    cmd = ["ffmpeg", "-y"]
    video_inputs = []
    audio_inputs = []
    v_durations = []

    # 1) Logo
    # try find any logo image in repo
    logo_candidates = list((ROOT / 'asset').rglob('*logo*.png')) + list((ROOT).rglob('*chimp*'))
    logo_img = None
    for p in logo_candidates:
        if p.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            logo_img = p
            break
    if not logo_img:
        logo_img = CARD_DIR / 'intro_doctor_card.png'
    if not exists(logo_img):
        # use color source
        cmd += ["-f", "lavfi", "-t", str(LOGO_DUR), "-i", "color=c=black:s=1280x720"]
        video_inputs.append(None)
    else:
        cmd += ["-loop", "1", "-t", str(LOGO_DUR), "-i", str(logo_img)]
        video_inputs.append(logo_img)
    v_durations.append(LOGO_DUR)

    # 2) Title
    title_img = CARD_DIR / 'scene_00_card.png'
    if not exists(title_img):
        cards = sorted(CARD_DIR.glob('*.png')) if CARD_DIR.exists() else []
        title_img = cards[0] if cards else None
    if not title_img or not exists(title_img):
        cmd += ["-f", "lavfi", "-t", str(TITLE_DUR), "-i", "color=c=black:s=1280x720"]
        video_inputs.append(None)
    else:
        cmd += ["-loop", "1", "-t", str(TITLE_DUR), "-i", str(title_img)]
        video_inputs.append(title_img)
    v_durations.append(TITLE_DUR)

    # 3) For each subject group: subject card + 16 scenes * 3 images
    for g, subj_idx in enumerate(SUBJECT_PROMPTS_IDX):
        # subject card
        subj_card = CARD_DIR / f"subject_{subj_idx:02d}_card.png"
        subj_music = VOICE_DIR / f"subject_{subj_idx:02d}_elevator.wav"
        if exists(subj_card):
            cmd += ["-loop", "1", "-t", str(SUBJECT_CARD_DUR), "-i", str(subj_card)]
            video_inputs.append(subj_card)
        else:
            cmd += ["-f", "lavfi", "-t", str(SUBJECT_CARD_DUR), "-i", "color=c=black:s=1280x720"]
            video_inputs.append(None)
        v_durations.append(SUBJECT_CARD_DUR)
        # audio input for subject card: looped and trimmed by input '-t SUBJECT_CARD_DUR'
        if exists(subj_music):
            cmd += ["-stream_loop", "-1", "-t", str(SUBJECT_CARD_DUR), "-i", str(subj_music)]
            audio_inputs.append(subj_music)
        else:
            # silent audio input
            cmd += ["-f", "lavfi", "-t", str(SUBJECT_CARD_DUR), "-i", "anullsrc=r=44100:cl=stereo"]
            audio_inputs.append(None)

        # scenes for this group: scenes g*GROUP_SIZE .. g*GROUP_SIZE+GROUP_SIZE-1
        for s in range(g * GROUP_SIZE, (g + 1) * GROUP_SIZE):
            idx_str = f"{s:02d}"
            body = IMG_DIR / f"scene_{idx_str}_body.png"
            close = IMG_DIR / f"scene_{idx_str}_close.png"
            doctor = IMG_DIR / f"scene_{idx_str}_doctor.png"
            voice = VOICE_DIR / f"voice_{idx_str}.wav"
            voice_dur = probe_duration(voice) if exists(voice) else 3.0
            # split voice into three parts
            part = max(0.5, voice_dur / 3.0)
            parts = [part, part, max(0.5, voice_dur - 2 * part)]
            for img, dur in [(body, parts[0]), (close, parts[1]), (doctor, parts[2])]:
                if exists(img):
                    cmd += ["-loop", "1", "-t", str(dur), "-i", str(img)]
                    video_inputs.append(img)
                else:
                    cmd += ["-f", "lavfi", "-t", str(dur), "-i", "color=c=black:s=1280x720"]
                    video_inputs.append(None)
                v_durations.append(dur)
            # add voice audio input (single file duration voice_dur)
            if exists(voice):
                cmd += ["-i", str(voice)]
                audio_inputs.append(voice)
            else:
                cmd += ["-f", "lavfi", "-t", str(sum(parts)), "-i", "anullsrc=r=44100:cl=stereo"]
                audio_inputs.append(None)

    # Now we have all inputs appended. Build filter_complex
    Nvid = len(video_inputs)
    Naud = len(audio_inputs)
    v_filters = []
    for i, dur in enumerate(v_durations):
        # input video index i corresponds to input stream [i:v]
        base = f"[{i}:v]scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1"
        # apply fades for first two inputs
        if i == 0:
            in_d = LOGO_IN
            out_d = LOGO_OUT
            out_st = max(0, dur - out_d)
            base += f",fade=t=in:st=0:d={in_d},fade=t=out:st={out_st}:d={out_d}"
        elif i == 1:
            in_d = TITLE_IN
            out_d = TITLE_OUT
            out_st = max(0, dur - out_d)
            base += f",fade=t=in:st=0:d={in_d},fade=t=out:st={out_st}:d={out_d}"
        # ensure yuv420p
        base += ",format=yuv420p"
        v_filters.append(base + f"[v{i}]")
    v_filter_complex = ";".join(v_filters) + ";" + "".join(f"[v{i}]" for i in range(Nvid)) + f"concat=n={Nvid}:v=1:a=0[vout]"

    # Audio preparation: we have audio_inputs in order matching video sequence where audio segments correspond to subject cards and scene voices
    a_filters = []
    for j in range(Naud):
        a_filters.append(f"[{Nvid + j}:a]aresample=44100,aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo[a{j}]")
    a_concat_inputs = "".join(f"[a{j}]" for j in range(Naud))
    a_filter_complex = ";".join(a_filters) + ";" + f"{a_concat_inputs}concat=n={Naud}:v=0:a=1[aout]"

    filter_complex = v_filter_complex + ";" + a_filter_complex

    # assemble final cmd
    cmd += ["-filter_complex", filter_complex, "-map", "[vout]", "-map", "[aout]",
            "-c:v", "libx264", "-preset", "veryslow", "-crf", "16",
            "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
            "-r", "24", OUT_MP4]

    print('Running ffmpeg single-pass assembly...')
    subprocess.run(cmd, check=True)
    print(f'Created {OUT_MP4}')

    # Convert to MPEG-2
    subprocess.run([
        "ffmpeg", "-y", "-i", OUT_MP4,
        "-c:v", "mpeg2video", "-q:v", "2", "-b:v", "15000k", "-maxrate", "20000k", "-bufsize", "10000k",
        "-c:a", "mp2", "-b:a", "384k", "-ar", "44100", "-ac", "2",
        "-r", "24", OUT_MPG
    ], check=True)
    print(f'Created {OUT_MPG}')


if __name__ == '__main__':
    build()
