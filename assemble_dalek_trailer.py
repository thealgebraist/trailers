import subprocess
import os
import sys

# --- Configuration ---
PROJECT_NAME = "dalek"
ASSETS_DIR = f"assets_{PROJECT_NAME}"
TOTAL_DURATION = 120 # Seconds

def assemble_dalek(assets_dir, output_file):
    print(f"--- Assembling Dalek trailer (dynamic detection) ---")
    
    # 1. Detect available scenes
    all_scenes = [
        "01_dalek_city_destroyed", "02_dalek_laser_firing", "03_doctor_who_running", "04_tardis_materializing",
        "05_dalek_supreme_closeup", "06_kaled_mutant_reveal", "07_skaro_desert_ruins", "08_dalek_invasion_earth",
        "09_exterminate_text_red", "10_dalek_saucer_flying", "11_doctor_companion_flee", "12_cybermen_vs_daleks",
        "13_UNIT_soldiers_battle", "14_dalek_emperor_vision", "15_time_war_explosion", "16_dalek_army_advancing",
        "17_earth_in_ruins", "18_doctor_determined_face", "19_tardis_dematerializing", "20_dalek_shadow_large_city"
    ]
    
    scenes = []
    # Check for specific scene image files
    for s in all_scenes:
        if os.path.exists(f"{assets_dir}/images/{s}.png") or os.path.exists(f"{assets_dir}/images/{s}.bmp") or os.path.exists(f"{assets_dir}/videos/{s}.mp4"):
            scenes.append(s)

    if not scenes:
        print("No assets found to assemble for Dalek.")
        return

    scene_duration = 3.0 # Each scene will last 3 seconds
    music_themes = ["theme_dark_ominous"] # Assuming a suitable dark theme music

    # 2. Build FFmpeg Command
    cmd = ["ffmpeg", "-y"]
    
    # Visual Inputs (Images or Videos)
    for s_id in scenes:
        v_path = f"{assets_dir}/videos/{s_id}.mp4"
        if os.path.exists(v_path):
            cmd += ["-stream_loop", "-1", "-t", str(scene_duration), "-i", v_path]
        else:
            i_path = f"{assets_dir}/images/{s_id}.png"
            if not os.path.exists(i_path):
                i_path = f"{assets_dir}/images/{s_id}.bmp"
            cmd += ["-loop", "1", "-t", str(scene_duration), "-i", i_path]
        
    # Audio: Voiceover
    vo_path = f"{assets_dir}/voice/voiceover_full.wav"
    if os.path.exists(vo_path):
        cmd += ["-i", vo_path]
    else:
        # Fallback to silence if full VO missing
        cmd += ["-f", "lavfi", "-t", str(len(scenes) * scene_duration), "-i", "anullsrc=r=44100:cl=stereo"]

    # Audio: Music
    for m_id in music_themes:
        m_path = f"{assets_dir}/music/{m_id}.wav"
        if os.path.exists(m_path):
            cmd += ["-stream_loop", "-1", "-i", m_path]
        else:
            cmd += ["-f", "lavfi", "-t", str(TOTAL_DURATION), "-i", "anullsrc=r=44100:cl=stereo"]

    # 3. Filter Complex
    n_video_inputs = len(scenes)
    v_filters = ""
    for i in range(n_video_inputs):
        v_filters += f"[{i}:v]scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1[v{i}];"
    v_concat = "".join([f"[v{i}]" for i in range(n_video_inputs)])
    v_filters += f"{v_concat}concat=n={n_video_inputs}:v=1:a=0[vout];"
    
    # Audio mixing
    # Voiceover will be at index n_video_inputs
    # Music will be at index n_video_inputs + 1
    audio_prep = f"[{n_video_inputs}:a]volume=1.5[vo];" # Adjust volume for voiceover
    audio_prep += f"[{n_video_inputs+1}:a]volume=0.6[bgm];" # Adjust volume for background music
    audio_prep += f"[vo][bgm]amix=inputs=2:duration=first:normalize=0[aout]"
    
    cmd += ["-filter_complex", v_filters + audio_prep]
    cmd += ["-map", "[vout]", "-map", "[aout]"]
    
    # High Quality Output
    cmd += [
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        "-c:a", "aac", "-b:a", "384k",
        output_file
    ]

    print("--- Executing Dalek Trailer Assembly ---")
    subprocess.run(cmd, check=True)
    print(f"--- Successfully created {output_file} ---")

if __name__ == "__main__":
    assets = sys.argv[1] if len(sys.argv) > 1 else ASSETS_DIR
    out = sys.argv[2] if len(sys.argv) > 2 else f"{PROJECT_NAME}_trailer.mp4"
    assemble_dalek(assets, out)
