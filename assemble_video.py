import PIL.Image
if not hasattr(PIL.Image, "ANTIALIAS"): PIL.Image.ANTIALIAS = PIL.Image.LANCZOS
import subprocess
import os
import sys

def assemble(assets_dir, output_file, project_type="dalek"):
    print(f"--- Assembling {project_type} trailer ---")
    
    # 1. Define Scene Lists
    if project_type == "wait":
        scenes = [
            "01_kitchen_wide", "02_the_kettle_still", "03_hand_reaching", "04_filling_water",
            "05_closing_lid", "06_pushing_button", "07_the_wait_begins", "08_clock_close",
            "09_bubbles_forming", "10_steam_rising", "11_face_extreme_close", "12_kettle_shaking",
            "13_steam_cloud", "14_orchestra_peak_1", "15_the_almost_click", "16_the_fakeout",
            "17_the_quiet_return", "18_empty_cup", "19_arthur_sigh_wait", "20_re_pushing_button",
            "21_red_glow", "22_violent_boil", "23_steam_jet", "24_arthur_screaming",
            "25_kettle_levitating", "26_the_boiling_point", "27_final_crescendo", "28_the_pop",
            "29_title_card_kettle", "30_slogan_kettle", "31_coming_soon_kettle", "32_post_credits_kettle"
        ]
        scene_duration = 7.5 
        music_themes = ["theme_build_1", "theme_silence_bridge", "theme_build_2"]
    else:
        if project_type == "dalek":
            scenes = [
                "01_skaro_landscape", "02_dalek_army", "03_dalek_close_eye", "04_rusty_lonely",
                "05_memory_glitch", "06_cornfield_reveal", "07_farmhouse_exterior", "08_baby_dalek_basket",
                "09_nana_pop_holding", "10_baking_pie", "11_fishing_trip", "12_tractor_ride",
                "13_school_exterior", "14_classroom_learning", "15_school_friends", "16_class_photo",
                "17_reading_book", "18_fishing_success", "19_prom_night", "20_first_crush",
                "21_sky_darkens", "22_rusty_looks_up", "23_leaving_home", "24_back_on_skaro",
                "25_supreme_dalek", "26_pie_reveal", "27_dalek_confusion", "28_pie_in_face",
                "29_escape", "30_reunion", "31_title_card", "32_post_credits"
            ]
            music_themes = ["theme_dark", "theme_acoustic", "theme_epic"]
        elif project_type == "snakes":
            scenes = [
                "01_titanic_dock", "02_luxury_interior", "03_captain_pride", "04_cargo_hold",
                "05_crate_shake", "06_iceberg_lookout", "07_iceberg_impact", "08_crates_break",
                "09_snakes_hallway", "10_tea_time_terror", "11_ballroom_chaos", "12_draw_me",
                "13_samuel_captain", "14_snakes_on_bow", "15_smokestack_wrap", "16_flooded_stairs",
                "17_lifeboat_full", "18_propeller_guy", "19_door_scene", "20_violin_band",
                "21_underwater_swim", "22_hero_shot", "23_ship_snap", "24_car_scene",
                "25_survivors", "26_old_rose", "27_celine_dion_snake", "28_snake_plane",
                "29_iceberg_face", "30_title_card", "31_slogan", "32_post_credits"
            ]
            music_themes = ["theme_suspense", "theme_disaster", "theme_romantic_snake"]
        elif project_type == "luftwaffe":
            scenes = [
                "01_dogfight", "02_crash_landing", "03_waking_up", "04_reveal_ape", "05_title_splash",
                "06_berlin_jungle", "07_marching_chimps", "08_gorilla_guards", "09_hitler_kong", "10_rick_defiant",
                "11_dirty_ape_line", "12_human_pets", "13_resistance_meeting", "14_secret_weapon_plan", "15_banana_rocket",
                "16_poop_grenade", "17_dogfight_monkeys", "18_tank_battle", "19_bananas", "20_bunker_assault",
                "21_interrogation", "22_rocket_launch", "23_plane_wing_fight", "24_romantic_kiss", "25_statue_of_liberty",
                "26_rick_screaming", "27_monkey_laugh", "28_hero_walk", "29_title_card_2", "30_coming_soon",
                "31_final_joke", "32_post_credits"
            ]
            music_themes = ["theme_march", "theme_action", "theme_absurd"]
        elif project_type == "romcom":
            scenes = [
                "01_industrial_north", "02_dalek_flat_cap", "03_factory_line", "04_the_meeting",
                "05_shared_tea", "06_stern_father", "07_father_shouting", "08_back_alley",
                "09_tap_dancing_reveal", "10_gene_kelly_homage", "11_technicolor_shift", "12_dance_on_stairs",
                "13_tuxedo_dalek", "14_chorus_line", "15_plunger_flute", "16_back_to_reality",
                "17_the_breakup", "18_rusty_drinking", "19_talent_poster", "20_entering_hall",
                "21_taking_stage", "22_first_tap", "23_music_explodes", "24_crowd_cheering",
                "25_betty_joins", "26_the_lift", "27_award_ceremony", "28_walking_home",
                "29_title_card_romcom", "30_critics_quotes", "31_slogan_romcom", "32_post_credits_romcom"
            ]
            music_themes = ["theme_gritty", "theme_tap", "theme_romantic_finale"]
        elif project_type == "titanic2":
            scenes = [
                "01_deep_ocean", "02_billionaire_reveal", "03_raising_the_ship", "04_dry_dock_restoration",
                "05_titanic_2_full", "06_grand_staircase_modern", "07_jack_clone", "08_rose_clone",
                "09_north_atlantic_night", "10_iceberg_waiting", "11_iceberg_tracking", "12_radar_warning",
                "13_captain_panic", "14_the_hit", "15_water_everywhere", "16_ship_tilting",
                "17_jet_ski_escape", "18_billionaire_regret", "19_iceberg_transformation", "20_underwater_fight",
                "21_door_scene_2", "22_propeller_guy_2", "23_orchestra_rock", "24_ship_snapping",
                "25_lifeboat_selfie", "26_iceberg_salute", "27_rescue_ship_iceberg", "28_jack_rose_underwater",
                "29_title_card_t2", "30_coming_july", "31_iceberg_name", "32_post_credits_t2"
            ]
            music_themes = ["theme_epic_salvage", "theme_horror_ice", "theme_action_resink"]
        elif project_type == "boring":
            scenes = [
                "01_the_wall", "02_dripping_tap", "03_arthur_staring", "04_the_patch_macro",
                "05_beige_curtains", "06_grey_porridge", "07_the_radiator", "08_arthur_watch",
                "09_dust_mote", "10_peeling_wallpaper", "11_arthur_sigh", "12_the_window_rain",
                "13_chipped_mug", "14_dead_fly", "15_the_ceiling_light", "16_arthur_sitting",
                "17_the_patch_grows", "18_close_up_eye", "19_empty_bookshelf", "20_shadow_movement",
                "21_arthur_standing", "22_touching_the_wall", "23_the_contact", "24_reaction_shot",
                "25_the_kettle", "26_staring_again", "27_fading_out", "28_title_card_boring",
                "29_slogan_boring", "30_coming_whenever", "31_critic_quote", "32_post_credits_boring"
            ]
            music_themes = ["theme_hum", "theme_clock", "theme_depressing_cello"]
        else:
            print("Unknown project type.")
            return
        scene_duration = 4.0

    # 2. Build FFmpeg Command
    cmd = ["ffmpeg", "-y"]
    
    # Image Inputs (0 to n-1)
    for s_id in scenes:
        path = f"{assets_dir}/images/{s_id}.png"
        cmd += ["-loop", "1", "-t", str(scene_duration), "-i", path]
        
    # VO Inputs (n to 2n-1)
    for s_id in scenes:
        path = f"{assets_dir}/voice/{s_id}.wav"
        if os.path.exists(path) and os.path.getsize(path) > 0:
            cmd += ["-i", path]
        else:
            cmd += ["-f", "lavfi", "-t", str(scene_duration), "-i", "anullsrc=r=44100:cl=stereo"]
            
    # SFX Inputs (2n to 3n-1)
    for s_id in scenes:
        path = f"{assets_dir}/sfx/{s_id}.wav"
        if os.path.exists(path) and os.path.getsize(path) > 0:
            cmd += ["-i", path]
        else:
            cmd += ["-f", "lavfi", "-t", str(scene_duration), "-i", "anullsrc=r=44100:cl=stereo"]

    # Music Inputs (3n to 3n + len(music)-1)
    for m_id in music_themes:
        path = f"{assets_dir}/music/{m_id}.wav"
        if os.path.exists(path) and os.path.getsize(path) > 0:
            cmd += ["-i", path]
        else:
            cmd += ["-f", "lavfi", "-t", "120", "-i", "anullsrc=r=44100:cl=stereo"]

    # 3. Filter Complex
    n = len(scenes)
    v_filters = ""
    for i in range(n):
        v_filters += f"[{i}:v]scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1[v{i}];"
    v_concat = "".join([f"[v{i}]" for i in range(n)])
    v_filters += f"{v_concat}concat=n={n}:v=1:a=0[vout];"
    
    # Prepare Audio Streams
    # Normalize all to 44.1k Stereo
    audio_prep = ""
    vo_start = n
    sfx_start = n * 2
    music_start = n * 3
    
    # VO Concatenation
    for i in range(n):
        audio_prep += f"[{vo_start + i}:a]aresample=44100,aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo[vo{i}];"
    vo_concat = "".join([f"[vo{i}]" for i in range(n)])
    audio_prep += f"{vo_concat}concat=n={n}:v=0:a=1[vo_full];"
    
    # SFX Concatenation
    for i in range(n):
        audio_prep += f"[{sfx_start + i}:a]aresample=44100,aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo[sfx{i}];"
    sfx_concat = "".join([f"[sfx{i}]" for i in range(n)])
    audio_prep += f"{sfx_concat}concat=n={n}:v=0:a=1[sfx_full];"
    
    # Music Mixing
    m_inputs = "".join([f"[{music_start + i}:a]aresample=44100,aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo," for i in range(len(music_themes))])
    # Remove trailing comma and add amix
    m_prep = ""
    for i in range(len(music_themes)):
        m_prep += f"[{music_start + i}:a]aresample=44100,aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo[m{i}];"
    m_inputs = "".join([f"[m{i}]" for i in range(len(music_themes))])
    audio_prep += m_prep + f"{m_inputs}amix=inputs={len(music_themes)}:duration=first:normalize=0[music_mix];"
    
    # Final Layering with explicit volumes
    # vo_full (1.0) + sfx_full (0.8) + music_mix (0.3)
    audio_prep += "[vo_full]volume=1.0[vov];"
    audio_prep += "[sfx_full]volume=0.8[sfxv];"
    audio_prep += "[music_mix]volume=0.3[musicv];"
    audio_prep += "[vov][sfxv][musicv]amix=inputs=3:duration=first:normalize=0[aout]"
    
    cmd += ["-filter_complex", v_filters + audio_prep]
    cmd += ["-map", "[vout]", "-map", "[aout]"]
    
    # High Bitrate Settings
    cmd += [
        "-c:v", "mpeg2video", "-q:v", "2", "-b:v", "15000k", 
        "-maxrate", "20000k", "-bufsize", "10000k",
        "-c:a", "aac", "-b:a", "384k", "-ar", "44100", "-ac", "2",
        "-r", "24", output_file
    ]

    print("--- Executing FFmpeg ---")
    # print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"--- Successfully created {output_file} ---")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python assemble_video.py <assets_dir> <output_file> <project_type>")
    else:
        assemble(sys.argv[1], sys.argv[2], sys.argv[3])