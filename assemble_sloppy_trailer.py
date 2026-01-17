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

        # Rewritten to use vidlib
        from vidlib import assemble

        if __name__ == "__main__":
            import sys
            assets = sys.argv[1] if len(sys.argv) > 1 else "assets_sloppy"
            out = sys.argv[2] if len(sys.argv) > 2 else "sloppy_trailer.mp4"
            assemble.assemble_sloppy(assets, out)