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