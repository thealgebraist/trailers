import subprocess
import os
import sys

# --- Configuration ---
PROJECT_NAME = "dalek"
ASSETS_DIR = f"assets_{PROJECT_NAME}"
TOTAL_DURATION = 120 # Seconds

SCENES = [
    "01_skaro_landscape", "02_dalek_factory_closeup", "03_horseshoe_closeup", "04_golden_eye",
    "05_kansas_farmhouse", "06_baby_dalek", "07_dalek_pie", "08_dalek_fishing",
    "09_dalek_tractor", "10_red_schoolhouse", "11_class_photo", "12_town_citizens",
    "13_hide_and_seek", "14_dalek_prom", "15_skaro_return", "16_quivering_eye",
    "17_supreme_dalek_pie", "18_dalek_hugging", "19_country_road", "20_title_card",
    "21_birthday_cake"
]

def assemble_dalek(assets_dir, output_file):
    print(f"--- Assembling Dalek Trailer: 'A Dalek Comes Home' ---")
    
    scene_duration = TOTAL_DURATION / len(SCENES)
    
    cmd = ["ffmpeg", "-y"]
    
    # 1. Visual Inputs
    for s_id in SCENES:
        img_path = f"{assets_dir}/images/{s_id}.png"
        if not os.path.exists(img_path):
            img_path = f"{assets_dir}/images/{s_id}.bmp"
            
        if os.path.exists(img_path):
            cmd += ["-loop", "1", "-t", str(scene_duration), "-i", img_path]
        else:
            cmd += ["-f", "lavfi", "-t", str(scene_duration), "-i", "color=c=black:s=1280x720:r=25"]

    # 2. SFX Inputs
    for s_id in SCENES:
        sfx_path = f"{assets_dir}/sfx/{s_id}.wav"
        if os.path.exists(sfx_path):
             cmd += ["-stream_loop", "-1", "-t", str(scene_duration), "-i", sfx_path]
        else:
             cmd += ["-f", "lavfi", "-t", str(scene_duration), "-i", "anullsrc=r=44100:cl=stereo"]

    # 3. Voiceover
    vo_path = f"{assets_dir}/voice/voiceover_full.mp3"
    if not os.path.exists(vo_path):
        vo_path = f"{assets_dir}/voice/voiceover_full.wav"
        
    if os.path.exists(vo_path):
        cmd += ["-i", vo_path]
    else:
        cmd += ["-f", "lavfi", "-t", str(TOTAL_DURATION), "-i", "anullsrc=r=44100:cl=stereo"]
    
    # 4. Music
    music_path = f"{assets_dir}/music/dalek_theme.wav"
    if os.path.exists(music_path):
        cmd += ["-i", music_path]
    else:
        cmd += ["-f", "lavfi", "-t", str(TOTAL_DURATION), "-i", "anullsrc=r=44100:cl=stereo"]

    # --- Filter Complex ---
    filter_complex = ""
    
    # Visuals: Zoompan
    for i in range(len(SCENES)):
        filter_complex += f"[{i}:v]scale=8000:-1,zoompan=z='min(zoom+0.001,1.5)':d={int(scene_duration*25)}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1280x720[v{i}];"
        
    v_concat = "".join([f"[v{i}]" for i in range(len(SCENES))])
    filter_complex += f"{v_concat}concat=n={len(SCENES)}:v=1:a=0[vout];"
    
    # Audio Mixing
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

if __name__ == "__main__":
    assets = sys.argv[1] if len(sys.argv) > 1 else ASSETS_DIR
    out = sys.argv[2] if len(sys.argv) > 2 else f"{PROJECT_NAME}_trailer.mp4"
    assemble_dalek(assets, out)
