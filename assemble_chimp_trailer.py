import subprocess
import os
import sys

def assemble_chimp(assets_dir, output_file):
    print(f"--- Assembling Chimp trailer ---")
    
    scenes = [
        "01_chimp_map", "02_chimp_packing", "03_chimp_station", "04_chimp_train_window",
        "05_chimp_penguin", "06_train_bridge", "07_fruit_city", "08_golden_banana",
        "09_chimp_running", "10_chimp_reaching", "11_title_card", "12_chimp_slippery"
    ]
    
    available_scenes = []
    for s in scenes:
        i_path = f"{assets_dir}/images/{s}.png"
        if os.path.exists(i_path):
            available_scenes.append(s)

    if not available_scenes:
        print("No assets found to assemble.")
        return

    # Total duration of VO is usually around 20-30s.
    # 12 scenes at 2.5s each = 30s.
    scene_duration = 2.5

    # 2. Build FFmpeg Command
    cmd = ["ffmpeg", "-y"]
    
    for s_id in available_scenes:
        i_path = f"{assets_dir}/images/{s_id}.png"
        cmd += ["-loop", "1", "-t", str(scene_duration), "-i", i_path]
        
    # Audio: Voiceover
    vo_path = f"{assets_dir}/voice/voiceover_full.wav"
    if os.path.exists(vo_path):
        cmd += ["-i", vo_path]
    else:
        cmd += ["-f", "lavfi", "-t", str(len(available_scenes) * scene_duration), "-i", "anullsrc=r=44100:cl=stereo"]

    # Audio: Music
    m_path = f"{assets_dir}/music/theme_main.wav"
    if os.path.exists(m_path):
        cmd += ["-stream_loop", "-1", "-i", m_path]
    else:
        cmd += ["-f", "lavfi", "-t", "30", "-i", "anullsrc=r=44100:cl=stereo"]

    # 3. Filter Complex
    n = len(available_scenes)
    v_filters = ""
    for i in range(n):
        v_filters += f"[{i}:v]scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1[v{i}];"
    v_concat = "".join([f"[v{i}]" for i in range(n)])
    v_filters += f"{v_concat}concat=n={n}:v=1:a=0[vout];"
    
    # Audio mixing
    # Index n is Voiceover, Index n+1 is Music
    audio_prep = f"[{n}:a]volume=1.5[vo];"
    audio_prep += f"[{n+1}:a]volume=0.4[bgm];"
    audio_prep += f"[vo][bgm]amix=inputs=2:duration=first:normalize=0[aout]"
    
    cmd += ["-filter_complex", v_filters + audio_prep]
    cmd += ["-map", "[vout]", "-map", "[aout]"]
    
    # Output Settings
    cmd += [
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        "-t", str(n * scene_duration), # Ensure output length matches scenes
        "-c:a", "aac", "-b:a", "384k",
        output_file
    ]

    print("--- Executing Assembly ---")
    subprocess.run(cmd, check=True)
    print(f"--- Successfully created {output_file} ---")

if __name__ == "__main__":
    assets = sys.argv[1] if len(sys.argv) > 1 else "assets_chimp"
    out = sys.argv[2] if len(sys.argv) > 2 else "chimp_trailer.mp4"
    assemble_chimp(assets, out)
