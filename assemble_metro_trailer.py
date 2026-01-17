import subprocess
import os
import sys
import wave
import contextlib
import argparse

# Scene definitions
SCENES = [
    "01_entrance",
    "02_face_scan",
    "03_finger_scan",
    "04_smell_detector",
    "05_torso_slime",
    "06_tongue_print",
    "07_retina_drill",
    "08_ear_wax_sampler",
    "09_hair_count",
    "10_sweat_analysis",
    "11_bone_crusher",
    "12_spirit_photo",
    "13_karma_scale",
    "14_dream_extract",
    "15_memory_wipe",
    "16_genetic_sieve",
    "17_final_stamp",
    "18_nail_pull",
    "19_skin_swatch",
    "20_tooth_scan",
    "21_pulse_monitor",
    "22_tear_collector",
    "23_brain_map",
    "24_shadow_audit",
    "25_breath_tax",
    "26_thought_police",
    "27_loyalty_check",
    "28_identity_shredder",
    "29_platform_edge",
    "30_empty_carriage",
    "31_train_interior",
    "32_title_card",
]

# Mapping Scene Index -> [VO indices]
# Total VOs: 0-35
SCENE_VO_MAP = {
    0: [0, 1, 2],  # Entrance: Welcome...
    1: [3],  # Face scan
    2: [4],
    3: [5],
    4: [6],
    5: [7],
    6: [8],
    7: [9],
    8: [10],
    9: [11],
    10: [12],
    11: [13],
    12: [14],
    13: [15],
    14: [16],
    15: [17],
    16: [18],
    17: [19],
    18: [20],
    19: [21],
    20: [22],
    21: [23],
    22: [24],
    23: [25],
    24: [26],
    25: [27],
    26: [28],
    27: [29],
    28: [30],  # Platform edge
    29: [31],  # Empty carriage
    30: [32],  # Train interior
    31: [33, 34, 35],  # Title card
}


def get_wav_duration(path):
    if not os.path.exists(path):
        return 0.0
    try:
        with contextlib.closing(wave.open(path, "r")) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            return frames / float(rate)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return 0.0


def assemble_metro(assets_dir, output_file):
    print(f"--- Assembling Metro Trailer (Dynamic Sync) ---")

    cmd = ["ffmpeg", "-y"]
    input_idx = 0
    filter_complex = ""

    scene_v_labels = []
    scene_a_labels = []

    # Map inputs and build per-scene filters
    for i, s_id in enumerate(SCENES):
        # 1. Determine Duration
        vo_indices = SCENE_VO_MAP.get(i, [])
        vo_files = [f"{assets_dir}/voice/vo_{vi:03d}.wav" for vi in vo_indices]

        total_vo_dur = sum(get_wav_duration(f) for f in vo_files)
        # Min duration for visual pacing
        duration = max(total_vo_dur + 0.5, 4.0)

        # 2. Add Inputs

        # Image Input (Single Frame for zoompan)
        img_path = f"{assets_dir}/images/{s_id}.png"
        has_img = os.path.exists(img_path)
        if has_img:
            cmd += ["-i", img_path]
        else:
            print(f"Warning: Missing image {img_path}, using placeholder")
            # Single frame black placeholder
            cmd += ["-f", "lavfi", "-i", "color=c=black:s=1280x720:d=0.04"]
        img_idx = input_idx
        input_idx += 1

        # SFX Input
        sfx_path = f"{assets_dir}/sfx/{s_id}.wav"
        has_sfx = os.path.exists(sfx_path)
        if has_sfx:
            cmd += ["-stream_loop", "-1", "-t", str(duration), "-i", sfx_path]
        else:
            cmd += [
                "-f",
                "lavfi",
                "-t",
                str(duration),
                "-i",
                "anullsrc=r=44100:cl=stereo",
            ]
        sfx_idx = input_idx
        input_idx += 1

        # VO Inputs
        current_vo_indices = []
        for vp in vo_files:
            if os.path.exists(vp):
                cmd += ["-i", vp]
                current_vo_indices.append(input_idx)
                input_idx += 1
            else:
                print(f"Warning: Missing VO {vp}")

        # 3. Filter Graph Construction

        # Video: Zoom slower, scale
        # Use d=duration*25 frames. Input is single frame.
        filter_complex += f"[{img_idx}:v]scale=4000:-1,zoompan=z='min(zoom+0.0005,1.5)':d={int(duration * 25)}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1280x720,setsar=1[v{i}];"
        scene_v_labels.append(f"[v{i}]")

        # Audio
        if current_vo_indices:
            # Concat VOs
            vo_inputs = "".join([f"[{idx}:a]" for idx in current_vo_indices])
            filter_complex += (
                f"{vo_inputs}concat=n={len(current_vo_indices)}:v=0:a=1[vo_raw_{i}];"
            )
            # Ensure stereo using aformat
            filter_complex += (
                f"[vo_raw_{i}]aformat=channel_layouts=stereo[vo_stereo_{i}];"
            )

            # Mix with SFX
            filter_complex += f"[{sfx_idx}:a]volume=0.3[sfx_vol_{i}];"
            filter_complex += f"[sfx_vol_{i}][vo_stereo_{i}]amix=inputs=2:duration=first:dropout_transition=0[a_scene_{i}];"
        else:
            # Just SFX
            filter_complex += f"[{sfx_idx}:a]volume=0.3[a_scene_{i}];"

        scene_a_labels.append(f"[a_scene_{i}]")

    # 4. Global Concat
    # Interleave [v0][a0][v1][a1]...
    concat_inputs = ""
    for v_lab, a_lab in zip(scene_v_labels, scene_a_labels):
        concat_inputs += f"{v_lab}{a_lab}"
    n_scenes = len(SCENES)

    filter_complex += f"{concat_inputs}concat=n={n_scenes}:v=1:a=1[main_v][main_a_raw];"

    # 5. Add Music
    music_path = f"{assets_dir}/music/metro_theme.wav"
    if os.path.exists(music_path):
        cmd += ["-stream_loop", "-1", "-i", music_path]  # Loop music
        music_idx = input_idx
        input_idx += 1

        # Mix Music with Main Audio
        filter_complex += f"[{music_idx}:a]volume=0.4[music_vol];"
        # We want the music to continue for the duration of main_a_raw
        filter_complex += f"[main_a_raw][music_vol]amix=inputs=2:duration=first:dropout_transition=0,volume=1.5[main_a];"
    else:
        print("Warning: Missing music track")
        filter_complex += f"[main_a_raw]volume=1.0[main_a];"

    # Map Output
    cmd += ["-filter_complex", filter_complex]
    cmd += ["-map", "[main_v]", "-map", "[main_a]"]

    # Encoding Options
    if output_file.endswith(".mpg") or output_file.endswith(".mpeg"):
        # MPEG2
        print("Using MPEG2 Encoding...")
        cmd += [
            "-c:v",
            "mpeg2video",
            "-pix_fmt",
            "yuv420p",
            "-q:v",
            "4",
            "-c:a",
            "mp2",
            "-b:a",
            "160k",
            "-ar",
            "48000",
        ]
    else:
        # H.264
        cmd += [
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-b:a",
            "320k",
        ]

    cmd += [output_file]

    print("--- Executing FFMPEG Assembly ---")

    with open("ffmpeg_log.txt", "w") as f:
        try:
            subprocess.run(cmd, check=True, stderr=f, stdout=f)
            print(f"--- Created {output_file} ---")
        except subprocess.CalledProcessError as e:
            print(f"--- FFMPEG Failed! See ffmpeg_log.txt ---")
            raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--assets", type=str, default="assets_metro", help="Path to assets directory"
    )
    parser.add_argument(
        "--output", type=str, default="metro_trailer.mp4", help="Output filename"
    )
    args = parser.parse_args()

    assemble_metro(args.assets, args.output)
