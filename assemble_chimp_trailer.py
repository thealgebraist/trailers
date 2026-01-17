def assemble_chimp(assets_dir, output_file):
    # This function should assemble the chimp trailer using the provided assets and output file.
    # The implementation can call the same logic as in the __main__ block or be filled in as needed.
    # For now, call the main logic:
    import subprocess
    import os
    PROJECT_NAME = "chimp"
    TOTAL_DURATION = 120 # Seconds
    SCENES = [
        "01_chimp_map", "02_chimp_packing", "03_chimp_station", "04_chimp_train_window",
        "05_chimp_penguin", "06_train_bridge", "07_fruit_city", "08_golden_banana",
        "09_chimp_running", "10_chimp_reaching", "11_chimp_guard", "12_chimp_distraction",
        "13_chimp_sneaking", "14_chimp_grab", "15_chimp_escape", "16_chimp_chase",
        "17_chimp_glider", "18_chimp_waterfall", "19_chimp_cave", "20_chimp_altar",
        "21_chimp_transformation", "22_chimp_portal", "23_chimp_step_in", "24_chimp_paradise",
        "25_chimp_friends", "26_chimp_celebration", "27_chimp_nap", "28_chimp_dream",
        "29_chimp_sunset", "30_chimp_slippery", "31_chimp_wink", "32_title_card"
    ]
    scene_duration = TOTAL_DURATION / len(SCENES)
    cmd = ["ffmpeg", "-y"]
    for s_id in SCENES:
        img_path = f"{assets_dir}/images/{s_id}.png"
        if os.path.exists(img_path):
            cmd += ["-loop", "1", "-t", str(scene_duration), "-i", img_path]
        else:
            cmd += ["-f", "lavfi", "-t", str(scene_duration), "-i", "color=c=black:s=1280x720:r=25"]
    for s_id in SCENES:
        sfx_path = f"{assets_dir}/sfx/{s_id}.wav"
        if os.path.exists(sfx_path):
            cmd += ["-stream_loop", "-1", "-t", str(scene_duration), "-i", sfx_path]
        else:
            cmd += ["-f", "lavfi", "-t", str(scene_duration), "-i", "anullsrc=r=44100:cl=stereo"]
    vo_path = f"{assets_dir}/voice/voiceover_full.wav"
    if os.path.exists(vo_path):
        cmd += ["-i", vo_path]
    else:
        cmd += ["-f", "lavfi", "-t", str(TOTAL_DURATION), "-i", "anullsrc=r=44100:cl=stereo"]
    music_path = f"{assets_dir}/music/chimp_theme.wav"
    if os.path.exists(music_path):
        cmd += ["-i", music_path]
    else:
        cmd += ["-f", "lavfi", "-t", str(TOTAL_DURATION), "-i", "anullsrc=r=44100:cl=stereo"]
    filter_complex = ""
    for i in range(len(SCENES)):
        filter_complex += f"[{i}:v]scale=8000:-1,zoompan=z='min(zoom+0.001,1.5)':d={int(scene_duration*25)}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1280x720[v{i}];"
    v_concat = "".join([f"[v{i}]" for i in range(len(SCENES))])
    filter_complex += f"{v_concat}concat=n={len(SCENES)}:v=1:a=0[vout];"
    sfx_mixed_outputs = ""
    for i in range(len(SCENES)):
        input_idx = len(SCENES) + i
        delay_ms = int(i * scene_duration * 1000)
        filter_complex += f"[{input_idx}:a]adelay={delay_ms}|{delay_ms}[sfx{i}];"
        sfx_mixed_outputs += f"[sfx{i}]"
    filter_complex += f"{sfx_mixed_outputs}amix=inputs={len(SCENES)}:dropout_transition=0[sfx_all];"
    vo_idx = len(SCENES) * 2
    music_idx = vo_idx + 1
    filter_complex += f"[sfx_all]volume=0.4[sfx_final];"
    filter_complex += f"[{vo_idx}:a]volume=1.8[vo_final];"
    filter_complex += f"[{music_idx}:a]volume=0.5[music_final];"
    filter_complex += f"[sfx_final][vo_final][music_final]amix=inputs=3:duration=first:dropout_transition=0[aout]"
    cmd += ["-filter_complex", filter_complex]
    cmd += ["-map", "[vout]", "-map", "[aout]"]
    cmd += [
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        "-c:a", "aac", "-b:a", "320k",
        "-t", str(TOTAL_DURATION),
        output_file
    ]
    print("--- Executing FFMPEG Assembly ---")
    subprocess.run(cmd, check=True)
    print(f"--- Created {output_file} ---")
import subprocess
import os
import sys
from vidlib import assemble

# --- Configuration ---
PROJECT_NAME = "chimp"
ASSETS_DIR = f"assets_{PROJECT_NAME}"
TOTAL_DURATION = 120 # Seconds

SCENES = [
    "01_chimp_map", "02_chimp_packing", "03_chimp_station", "04_chimp_train_window",
    "05_chimp_penguin", "06_train_bridge", "07_fruit_city", "08_golden_banana",
    "09_chimp_running", "10_chimp_reaching", "11_chimp_guard", "12_chimp_distraction",
    "13_chimp_sneaking", "14_chimp_grab", "15_chimp_escape", "16_chimp_chase",
    "17_chimp_glider", "18_chimp_waterfall", "19_chimp_cave", "20_chimp_altar",
    "21_chimp_transformation", "22_chimp_portal", "23_chimp_step_in", "24_chimp_paradise",
    "25_chimp_friends", "26_chimp_celebration", "27_chimp_nap", "28_chimp_dream",
    "29_chimp_sunset", "30_chimp_slippery", "31_chimp_wink", "32_title_card"
]

if __name__ == "__main__":
    assets = sys.argv[1] if len(sys.argv) > 1 else ASSETS_DIR
    out = sys.argv[2] if len(sys.argv) > 2 else f"{PROJECT_NAME}_trailer.mp4"
    assemble.assemble_chimp(assets, out)