import subprocess
import os
import sys

def assemble_sloppy(assets_dir, output_file):
    print(f"--- Assembling Sloppy trailer (dynamic detection) ---")
    
    # 1. Detect available scenes
    all_scenes = [
        "01_melting_clock_tower", "02_statue_extra_limbs", "03_classic_portrait_smear", "04_landscape_floating_rocks",
        "05_horse_too_many_legs", "06_tea_party_faceless", "07_library_infinite_books", "08_cat_spaghetti_fur",
        "09_dog_bird_hybrid", "10_vintage_car_square_wheels", "11_ballroom_dancers_merged", "12_flower_teeth",
        "13_mountain_made_of_flesh", "14_river_of_hair", "15_cloud_screaming", "16_tree_with_eyes",
        "17_dinner_plate_eating_itself", "18_hands_holding_hands_fractal", "19_mirror_reflection_wrong", "20_stairs_to_nowhere",
        "21_bicycle_made_of_meat", "22_building_breathing", "23_street_lamp_bending", "24_shadow_detached",
        "25_bird_metal_wings", "26_fish_walking", "27_chair_sitting_on_chair", "28_piano_melting_keys",
        "29_violin_made_of_water", "30_moon_cracked_egg", "31_sun_dripping", "32_forest_upside_down",
        "33_train_track_knot", "34_ship_sailing_on_grass", "35_plane_flapping_wings", "36_bridge_made_of_light",
        "37_statue_weeping_oil", "38_painting_bleeding_color", "39_book_reading_reader", "40_pen_writing_fire",
        "41_cup_spilling_upwards", "42_candle_burning_ice", "43_fire_freezing", "44_snow_burning",
        "45_rain_falling_sideways", "46_lightning_striking_itself", "47_thunder_visible", "48_sound_shattering_glass",
        "49_glass_bending", "50_stone_floating", "51_feather_heavy", "52_gold_rusting",
        "53_silver_rotting", "54_diamond_melting", "55_ruby_evaporating", "56_sapphire_growing_hair",
        "57_emerald_singing", "58_pearl_watching", "59_opal_screaming", "60_onyx_dreaming",
        "61_agate_laughing", "62_topaz_crying", "63_garnet_bleeding", "64_title_card_sloppy"
    ]
    
    scenes = []
    # If using C++ numeric naming
    if os.path.exists(f"{assets_dir}/images/01_scene.bmp"):
        for i in range(1, 33):
            scenes.append(f"{i:02d}_scene")
    else:
        # Otherwise use descriptive names
        for s in all_scenes:
            if os.path.exists(f"{assets_dir}/images/{s}.png") or os.path.exists(f"{assets_dir}/images/{s}.bmp") or os.path.exists(f"{assets_dir}/videos/{s}.mp4"):
                scenes.append(s)

    if not scenes:
        print("No assets found to assemble.")
        return

    scene_duration = 3.0
    music_themes = ["theme_dark"]

    # 2. Build FFmpeg Command
    cmd = ["ffmpeg", "-y"]
    
    # Visual Inputs (Images or Videos)
    # Prefer .mp4 from /videos, fallback to .png from /images
    for s_id in scenes:
        v_path = f"{assets_dir}/videos/{s_id}.mp4"
        if not os.path.exists(v_path):
            v_path = f"{assets_dir}/videos/{s_id}_dream.mp4"
        
        if os.path.exists(v_path):
            cmd += ["-stream_loop", "-1", "-t", str(scene_duration), "-i", v_path]
        else:
            i_path = f"{assets_dir}/images/{s_id}.png"
            if not os.path.exists(i_path):
                i_path = f"{assets_dir}/images/{s_id}.bmp"
            if not os.path.exists(i_path):
                i_path = f"{assets_dir}/images/{s_id}_dream.png"
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
            cmd += ["-f", "lavfi", "-t", "120", "-i", "anullsrc=r=44100:cl=stereo"]

    # 3. Filter Complex
    n = len(scenes)
    v_filters = ""
    for i in range(n):
        v_filters += f"[{i}:v]scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1[v{i}];"
    v_concat = "".join([f"[v{i}]" for i in range(n)])
    v_filters += f"{v_concat}concat=n={n}:v=1:a=0[vout];"
    
    # Audio mixing
    # Index n is Voiceover, Index n+1 is Music
    audio_prep = f"[{n}:a]volume=1.2[vo];"
    audio_prep += f"[{n+1}:a]volume=0.4[bgm];"
    audio_prep += f"[vo][bgm]amix=inputs=2:duration=first:normalize=0[aout]"
    
    cmd += ["-filter_complex", v_filters + audio_prep]
    cmd += ["-map", "[vout]", "-map", "[aout]"]
    
    # High Quality Output
    cmd += [
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        "-c:a", "aac", "-b:a", "384k",
        output_file
    ]

    print("--- Executing Assembly ---")
    subprocess.run(cmd, check=True)
    print(f"--- Successfully created {output_file} ---")

if __name__ == "__main__":
    assets = sys.argv[1] if len(sys.argv) > 1 else "assets_sloppy"
    out = sys.argv[2] if len(sys.argv) > 2 else "sloppy_trailer.mp4"
    assemble_sloppy(assets, out)
