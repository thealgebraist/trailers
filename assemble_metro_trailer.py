import subprocess
import os
import sys

# Scene definitions must match the generator
SCENES = [
    "01_entrance", "02_face_scan", "03_finger_scan", "04_smell_detector", 
    "05_torso_slime", "06_tongue_print", "07_retina_drill", "08_ear_wax_sampler", 
    "09_hair_count", "10_sweat_analysis", "11_bone_crusher", "12_spirit_photo", 
    "13_karma_scale", "14_dream_extract", "15_memory_wipe", "16_genetic_sieve", 
    "17_final_stamp", "18_nail_pull", "19_skin_swatch", "20_tooth_scan",
    "21_pulse_monitor", "22_tear_collector", "23_brain_map", "24_shadow_audit",
    "25_breath_tax", "26_thought_police", "27_loyalty_check", "28_identity_shredder",
    "29_platform_edge", "30_empty_carriage", "31_train_interior", "32_title_card"
]

def assemble_metro(assets_dir, output_file):
    print(f"--- Assembling Metro Trailer ---")
    
    # Total duration 240s. 32 Scenes.
    # 240 / 32 = 7.5 seconds per scene.
    scene_duration = 7.5
    
    cmd = ["ffmpeg", "-y"]
    
    # --- 1. Inputs ---
    
    # Video Inputs (Images) [0:v] to [31:v]
    for s_id in SCENES:
        img_path = f"{assets_dir}/images/{s_id}.png"
        if not os.path.exists(img_path):
            print(f"Warning: Missing image {img_path}")
        
        # Loop image for duration
        cmd += ["-loop", "1", "-t", str(scene_duration), "-i", img_path]

    # SFX Inputs [32:a] to [63:a]
    for s_id in SCENES:
        sfx_path = f"{assets_dir}/sfx/{s_id}.wav"
        if os.path.exists(sfx_path):
             cmd += ["-stream_loop", "-1", "-t", str(scene_duration), "-i", sfx_path]
        else:
             cmd += ["-f", "lavfi", "-t", str(scene_duration), "-i", "anullsrc=r=44100:cl=stereo"]

    # Voiceover [64:a]
    vo_path = f"{assets_dir}/voice/voiceover_full.wav"
    cmd += ["-i", vo_path]
    
    # Music [65:a]
    music_path = f"{assets_dir}/music/metro_theme.wav"
    cmd += ["-i", music_path]

    # --- 2. Filter Complex ---
    
    filter_complex = ""
    
    # Visuals: Scale, Zoom effect
    for i in range(len(SCENES)):
        filter_complex += f"[{i}:v]scale=8000:-1,zoompan=z='min(zoom+0.001,1.5)':d={int(scene_duration*25)}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1280x720[v{i}];"
        
    # Concat visuals
    v_concat = "".join([f"[v{i}]" for i in range(len(SCENES))])
    filter_complex += f"{v_concat}concat=n={len(SCENES)}:v=1:a=0[vout];"
    
    # Audio Mixing
    sfx_mixed_outputs = ""
    for i in range(len(SCENES)):
        input_idx = len(SCENES) + i
        delay_ms = int(i * scene_duration * 1000)
        filter_complex += f"[{input_idx}:a]adelay={delay_ms}|{delay_ms}[sfx{i}];"
        sfx_mixed_outputs += f"[sfx{i}]"
        
    # Mix all SFX into one track
    filter_complex += f"{sfx_mixed_outputs}amix=inputs={len(SCENES)}:dropout_transition=0[sfx_all];"
    
    # Final Mix: SFX + VO + Music
    vo_idx = len(SCENES) * 2
    music_idx = vo_idx + 1
    
    # Volume adjustments
    filter_complex += f"[sfx_all]volume=0.3[sfx_final];"
    filter_complex += f"[{vo_idx}:a]volume=1.5[vo_final];"
    filter_complex += f"[{music_idx}:a]volume=0.5[music_final];"
    
    filter_complex += f"[sfx_final][vo_final][music_final]amix=inputs=3:duration=first:dropout_transition=0[aout]"

    cmd += ["-filter_complex", filter_complex]
    cmd += ["-map", "[vout]", "-map", "[aout]"]
    
    # Output settings
    cmd += [
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        "-c:a", "aac", "-b:a", "320k",
        "-t", "240", # Hard limit 240s
        output_file
    ]

    print("--- Executing FFMPEG Assembly ---")
    subprocess.run(cmd, check=True)
    print(f"--- Created {output_file} ---")

if __name__ == "__main__":
    assets = "assets_metro"
    out = "metro_trailer.mp4"
    assemble_metro(assets, out)