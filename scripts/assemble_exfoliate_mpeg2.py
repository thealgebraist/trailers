#!/usr/bin/env python3
"""
Assemble a high-quality MP4 and MPEG-2 (MPG) video from assets_exfoliate_positive.
Flow:
 - chimp/logo (fade in 2s, hold 2s, fade out 4s)
 - title card (fade in 2s, hold 2s, fade out 4s)
 - For each subject group (indices 15,31,47,63):
     - show subject card for 10s while playing subject_{idx}_elevator.wav
     - then show the next 16 scenes (triples: body, close, doctor) in order with the scene voiceover
       each scene uses the voice duration split across the three images (equal thirds)

Produces: out_high.mp4 (H.264 high quality) and out_high.mpg (MPEG-2)

Note: requires ffmpeg and ffprobe on PATH.
"""

import os
import subprocess
import sys
from pathlib import Path
import math

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets_exfoliate_positive"
CARD_DIR = ASSETS / "cards"
IMG_DIR = ASSETS / "images"
VOICE_DIR = ASSETS / "voice"
MUSIC_DIR = ASSETS / "voice"  # subject elevator WAVs are stored in voice dir as subject_XX_elevator.wav
TMP_DIR = ROOT / "build_video_tmp"
TMP_DIR.mkdir(exist_ok=True)

OUTPUT_MP4 = "exfoliate_high.mp4"
OUTPUT_MPG = "exfoliate_high.mpg"

SUBJECT_INDICES = [15, 31, 47, 63]
GROUP_SIZE = 16

# utilities

def exists_and_nonzero(p: Path) -> bool:
    return p and p.exists() and p.stat().st_size > 0


def probe_duration(path: Path) -> float:
    try:
        out = subprocess.run([
            'ffprobe', '-v', 'error', '-select_streams', 'a:0',
            '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', str(path)
        ], capture_output=True, text=True, check=True)
        return float(out.stdout.strip())
    except Exception:
        return 0.0


def make_video_from_image(image: Path, duration: float, outpath: Path):
    # create a short video from an image at 1280x720, padded to aspect, with no audio
    if not exists_and_nonzero(image):
        # create a black canvas
        canvas = TMP_DIR / "black_1280x720.png"
        if not canvas.exists():
            subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=1280x720", str(canvas)], check=True)
        image = canvas
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1",
        "-i", str(image),
        "-t", f"{duration}",
        "-vf", "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1",
        "-c:v", "libx264", "-preset", "veryslow", "-crf", "16", "-pix_fmt", "yuv420p",
        "-an", str(outpath)
    ]
    subprocess.run(cmd, check=True)


def concat_videos(file_list_path: Path, outpath: Path):
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(file_list_path), "-c", "copy", str(outpath)]
    subprocess.run(cmd, check=True)


def mux_video_with_audio(vpath: Path, apath: Path, outpath: Path, loop_audio=False, target_dur: float=None):
    # If loop_audio, stream_loop -1
    cmd = ["ffmpeg", "-y"]
    if apath and loop_audio:
        cmd += ["-stream_loop", "-1"]
    if vpath:
        cmd += ["-i", str(vpath)]
    if apath:
        cmd += ["-i", str(apath)]
    else:
        # create silence input
        cmd += ["-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo"]
    # map
    cmd += [
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
        "-shortest", str(outpath)
    ]
    # if target_dur provided and audio longer, trim
    if target_dur:
        cmd = cmd[:-1] + ["-t", f"{target_dur}", str(outpath)]
    subprocess.run(cmd, check=True)


def build_sequence():
    segment_files = []
    seg_idx = 0

    # 1) Chimp logo: try to find any file with 'chimp' or 'logo' in repo assets
    possible_logos = list((ROOT / 'asset').rglob('*chimp*')) + list((ROOT).rglob('*logo*.png'))
    logo_img = None
    for p in possible_logos:
        if p.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            logo_img = p
            break
    if not logo_img:
        # fallback to first card if exists
        fallback = CARD_DIR / 'intro_doctor_card.png'
        if exists_and_nonzero(fallback):
            logo_img = fallback
    if not logo_img:
        print('No logo found; using black')
    logo_duration = 2 + 2 + 4  # fadein 2s, hold 2s, fadeout 4s => 8s
    logo_vid = TMP_DIR / f"seg_{seg_idx:04d}.mp4"
    make_video_from_image(logo_img, logo_duration, logo_vid)
    # apply fades on render step by re-encoding
    logo_vid2 = TMP_DIR / f"seg_{seg_idx:04d}_f.mp4"
    fade_out_start = logo_duration - 4
    subprocess.run([
        "ffmpeg", "-y", "-i", str(logo_vid),
        "-vf", f"fade=t=in:st=0:d=2,fade=t=out:st={fade_out_start}:d=4,format=yuv420p",
        "-c:v", "libx264", "-preset", "veryslow", "-crf", "16", "-an", str(logo_vid2)
    ], check=True)
    segment_files.append(logo_vid2)
    seg_idx += 1

    # 2) Title card: pick a title file or use first card
    title_img = CARD_DIR / 'scene_00_card.png'
    if not exists_and_nonzero(title_img):
        # pick any card
        cards = sorted(CARD_DIR.glob('*.png')) if CARD_DIR.exists() else []
        title_img = cards[0] if cards else None
    title_duration = 2 + 2 + 4
    title_vid = TMP_DIR / f"seg_{seg_idx:04d}.mp4"
    make_video_from_image(title_img, title_duration, title_vid)
    title_vid2 = TMP_DIR / f"seg_{seg_idx:04d}_f.mp4"
    fade_out_start = title_duration - 4
    subprocess.run([
        "ffmpeg", "-y", "-i", str(title_vid),
        "-vf", f"fade=t=in:st=0:d=2,fade=t=out:st={fade_out_start}:d=4,format=yuv420p",
        "-c:v", "libx264", "-preset", "veryslow", "-crf", "16", "-an", str(title_vid2)
    ], check=True)
    segment_files.append(title_vid2)
    seg_idx += 1

    # Now for each subject group
    for g, subj_idx in enumerate(SUBJECT_INDICES):
        # subject card
        subj_card = CARD_DIR / f"subject_{subj_idx:02d}_card.png"
        subj_music = VOICE_DIR / f"subject_{subj_idx:02d}_elevator.wav"
        subj_vid_tmp = TMP_DIR / f"seg_{seg_idx:04d}_card.mp4"
        make_video_from_image(subj_card, 10.0, subj_vid_tmp)
        subj_vid = TMP_DIR / f"seg_{seg_idx:04d}_card_audio.mp4"
        # attach music, loop if needed
        if exists_and_nonzero(subj_music):
            cmd = ["ffmpeg", "-y", "-stream_loop", "-1", "-i", str(subj_music), "-i", str(subj_vid_tmp),
                   "-t", "10", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", str(subj_vid)]
            subprocess.run(cmd, check=True)
        else:
            # no music, just copy
            subprocess.run(["ffmpeg", "-y", "-i", str(subj_vid_tmp), "-c", "copy", str(subj_vid)], check=True)
        segment_files.append(subj_vid)
        seg_idx += 1

        # scenes for this group: compute start and end
        start_scene = g * GROUP_SIZE
        end_scene = start_scene + GROUP_SIZE
        for scene_id in range(start_scene, end_scene):
            idx_str = f"{scene_id:02d}"
            body_img = IMG_DIR / f"scene_{idx_str}_body.png"
            close_img = IMG_DIR / f"scene_{idx_str}_close.png"
            doctor_img = IMG_DIR / f"scene_{idx_str}_doctor.png"
            voice = VOICE_DIR / f"voice_{idx_str}.wav"
            voice_dur = probe_duration(voice) if exists_and_nonzero(voice) else 3.0
            # split into three equal parts, min 0.5s
            part = max(0.5, voice_dur / 3.0)
            parts = [part, part, max(0.5, voice_dur - 2 * part)]
            # create three image videos
            seg_paths = []
            for i, (img, pdur) in enumerate(zip([body_img, close_img, doctor_img], parts)):
                seg_img = TMP_DIR / f"seg_{seg_idx:04d}_{i}.mp4"
                make_video_from_image(img, pdur, seg_img)
                seg_paths.append(seg_img)
                seg_idx += 1
            # concat the three image-only segments into one scene video
            listfile = TMP_DIR / f"list_scene_{scene_id:03d}.txt"
            with open(listfile, 'w') as lf:
                for p in seg_paths:
                    lf.write(f"file '{p.resolve()}'\n")
            scene_vid = TMP_DIR / f"scene_{scene_id:03d}_video.mp4"
            concat_videos(listfile, scene_vid)
            # mux with voice
            scene_with_audio = TMP_DIR / f"scene_{scene_id:03d}_with_audio.mp4"
            if exists_and_nonzero(voice):
                mux_video_with_audio(scene_vid, voice, scene_with_audio, loop_audio=False, target_dur=voice_dur)
            else:
                # no voice, leave silent video
                subprocess.run(["ffmpeg", "-y", "-i", str(scene_vid), "-c", "copy", str(scene_with_audio)], check=True)
            segment_files.append(scene_with_audio)

    # Final concat of segment_files
    final_list = TMP_DIR / "final_list.txt"
    with open(final_list, 'w') as fl:
        for p in segment_files:
            fl.write(f"file '{p.resolve()}'\n")

    # create MP4
    concat_videos(final_list, Path(OUTPUT_MP4))

    # convert to MPEG-2 MPG at high quality
    subprocess.run([
        "ffmpeg", "-y", "-i", OUTPUT_MP4,
        "-c:v", "mpeg2video", "-q:v", "2", "-b:v", "15000k", "-maxrate", "20000k", "-bufsize", "10000k",
        "-c:a", "mp2", "-b:a", "384k", "-ar", "44100", "-ac", "2",
        "-r", "24", OUTPUT_MPG
    ], check=True)

    print(f"Created {OUTPUT_MP4} and {OUTPUT_MPG}. Temporary files in {TMP_DIR}")


if __name__ == '__main__':
    build_sequence()
