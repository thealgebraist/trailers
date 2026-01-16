import subprocess
import os
import sys

# --- Configuration ---
PROJECT_NAME = "sloppy"
ASSETS_DIR = f"assets_{PROJECT_NAME}"
TOTAL_DURATION = 120 # Seconds

SCENES = [
    "01_melting_clock_tower", "02_statue_extra_limbs", "03_classic_portrait_smear", "04_landscape_floating_rocks",
    "05_horse_too_many_legs", "06_tea_party_faceless", "07_library_infinite_books", "08_cat_spaghetti_fur",
    "09_dog_bird_hybrid", "10_vintage_car_square_wheels", "11_ballroom_dancers_merged", "12_flower_teeth",
    "13_mountain_made_of_flesh", "14_river_of_hair", "15_cloud_screaming", "16_tree_with_eyes",
    "17_dinner_plate_eating_itself", "18_hands_holding_hands_fractal", "19_mirror_reflection_wrong", "20_stairs_to_nowhere",
    "21_bicycle_made_of_meat", "22_building_breathing", "23_street_lamp_bending", "24_shadow_detached",
    "25_bird_metal_wings", "26_fish_walking", "27_chair_sitting_on_chair", "28_piano_melting_keys",
    "29_violin_made_of_water", "30_moon_cracked_egg", "31_sun_dripping", "32_forest_upside_down"
]

def assemble_sloppy(assets_dir, output_file):
    print(f"--- Assembling Sloppy Trailer: 'The Sloppy Era' ---")
    
    scene_duration = TOTAL_DURATION / len(SCENES)
    
    cmd = ["ffmpeg", "-y"]
    
    # 1. Visual Inputs
    for s_id in SCENES:
        img_path = f"{assets_dir}/images/{s_id}.png"
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
    vo_path = f"{assets_dir}/voice/voiceover_full.wav"
    if os.path.exists(vo_path):
        cmd += ["-i", vo_path]
    else:
        cmd += ["-f", "lavfi", "-t", str(TOTAL_DURATION), "-i", "anullsrc=r=44100:cl=stereo"]
    
    # 4. Music
    music_path = f"{assets_dir}/music/sloppy_theme.wav"
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
    assemble_sloppy(assets, out)