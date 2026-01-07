pub struct ProjectConfig {
    pub name: String,
    pub scene_duration: f32,
    pub scenes: Vec<&'static str>,
    pub music_themes: Vec<&'static str>,
}

pub fn get_config(project: &str) -> ProjectConfig {
    match project {
        "dalek" => ProjectConfig {
            name: "dalek".to_string(),
            scene_duration: 3.75,
            scenes: vec!["01_skaro_landscape", "02_dalek_army", "03_dalek_close_eye", "04_rusty_lonely", "05_memory_glitch", "06_cornfield_reveal", "07_farmhouse_exterior", "08_baby_dalek_basket", "09_nana_pop_holding", "10_baking_pie", "11_fishing_trip", "12_tractor_ride", "13_school_exterior", "14_classroom_learning", "15_school_friends", "16_class_photo", "17_reading_book", "18_fishing_success", "19_prom_night", "20_first_crush", "21_sky_darkens", "22_rusty_looks_up", "23_leaving_home", "24_back_on_skaro", "25_supreme_dalek", "26_pie_reveal", "27_dalek_confusion", "28_pie_in_face", "29_escape", "30_reunion", "31_title_card", "32_post_credits"],
            music_themes: vec!["theme_dark"],
        },
        "snakes" => ProjectConfig {
            name: "snakes".to_string(),
            scene_duration: 3.75,
            scenes: vec!["01_titanic_dock", "02_luxury_interior", "03_captain_pride", "04_cargo_hold", "05_crate_shake", "06_iceberg_lookout", "07_iceberg_impact", "08_crates_break", "09_snakes_hallway", "10_tea_time_terror", "11_ballroom_chaos", "12_draw_me", "13_samuel_captain", "14_snakes_on_bow", "15_smokestack_wrap", "16_flooded_stairs", "17_lifeboat_full", "18_propeller_guy", "19_door_scene", "20_violin_band", "21_underwater_swim", "22_hero_shot", "23_ship_snap", "24_car_scene", "25_survivors", "26_old_rose", "27_celine_dion_snake", "28_snake_plane", "29_iceberg_face", "30_title_card", "31_slogan", "32_post_credits"],
            music_themes: vec!["theme_disaster"],
        },
        "luftwaffe" => ProjectConfig {
            name: "luftwaffe".to_string(),
            scene_duration: 3.75,
            scenes: vec!["01_dogfight", "02_crash_landing", "03_waking_up", "04_reveal_ape", "05_title_splash", "06_berlin_jungle", "07_marching_chimps", "08_gorilla_guards", "09_hitler_kong", "10_rick_defiant", "11_dirty_ape_line", "12_human_pets", "13_resistance_meeting", "14_secret_weapon_plan", "15_banana_rocket", "16_poop_grenade", "17_dogfight_monkeys", "18_tank_battle", "19_bananas", "20_bunker_assault", "21_interrogation", "22_rocket_launch", "23_plane_wing_fight", "24_romantic_kiss", "25_statue_of_liberty", "26_rick_screaming", "27_monkey_laugh", "28_hero_walk", "29_title_card_2", "30_coming_soon", "31_final_joke", "32_post_credits"],
            music_themes: vec!["theme_march"],
        },
        "romcom" => ProjectConfig {
            name: "romcom".to_string(),
            scene_duration: 3.75,
            scenes: vec!["01_industrial_north", "02_dalek_flat_cap", "03_factory_line", "04_the_meeting", "05_shared_tea", "06_stern_father", "07_father_shouting", "08_back_alley", "09_tap_dancing_reveal", "10_gene_kelly_homage", "11_technicolor_shift", "12_dance_on_stairs", "13_tuxedo_dalek", "14_chorus_line", "15_plunger_flute", "16_back_to_reality", "17_the_breakup", "18_rusty_drinking", "19_talent_poster", "20_entering_hall", "21_taking_stage", "22_first_tap", "23_music_explodes", "24_crowd_cheering", "25_betty_joins", "26_the_lift", "27_award_ceremony", "28_walking_home", "29_title_card_romcom", "30_critics_quotes", "31_slogan_romcom", "32_post_credits_romcom"],
            music_themes: vec!["theme_dark"],
        },
        "titanic2" => ProjectConfig {
            name: "titanic2".to_string(),
            scene_duration: 3.75,
            scenes: vec!["01_deep_ocean", "02_billionaire_reveal", "03_raising_the_ship", "04_dry_dock_restoration", "05_titanic_2_full", "06_grand_staircase_modern", "07_jack_clone", "08_rose_clone", "09_north_atlantic_night", "10_iceberg_waiting", "11_iceberg_tracking", "12_radar_warning", "13_captain_panic", "14_the_hit", "15_water_everywhere", "16_ship_tilting", "17_jet_ski_escape", "18_billionaire_regret", "19_iceberg_transformation", "20_underwater_fight", "21_door_scene_2", "22_propeller_guy_2", "23_orchestra_rock", "24_ship_snapping", "25_lifeboat_selfie", "26_iceberg_salute", "27_rescue_ship_iceberg", "28_jack_rose_underwater", "29_title_card_t2", "30_coming_july", "31_iceberg_name", "32_post_credits_t2"],
            music_themes: vec!["theme_dark"],
        },
        "boring" => ProjectConfig {
            name: "boring".to_string(),
            scene_duration: 3.75,
            scenes: vec!["01_the_wall", "02_dripping_tap", "03_arthur_staring", "04_the_patch_macro", "05_beige_curtains", "06_grey_porridge", "07_the_radiator", "08_arthur_watch", "09_dust_mote", "10_peeling_wallpaper", "11_arthur_sigh", "12_the_window_rain", "13_chipped_mug", "14_dead_fly", "15_the_ceiling_light", "16_arthur_sitting", "17_the_patch_grows", "18_close_up_eye", "19_empty_bookshelf", "20_shadow_movement", "21_arthur_standing", "22_touching_the_wall", "23_the_contact", "24_reaction_shot", "25_the_kettle", "26_staring_again", "27_fading_out", "28_title_card_boring", "29_slogan_boring", "30_coming_whenever", "31_critic_quote", "32_post_credits_boring"],
            music_themes: vec!["theme_dark"],
        },
        "wait" => ProjectConfig {
            name: "wait".to_string(),
            scene_duration: 3.75,
            scenes: vec!["01_kitchen_wide", "02_the_kettle_still", "03_hand_reaching", "04_filling_water", "05_closing_lid", "06_pushing_button", "07_the_wait_begins", "08_clock_close", "09_bubbles_forming", "10_steam_rising", "11_face_extreme_close", "12_kettle_shaking", "13_steam_cloud", "14_orchestra_peak_1", "15_the_almost_click", "16_the_fakeout", "17_the_quiet_return", "18_empty_cup", "19_arthur_sigh_wait", "20_re_pushing_button", "21_red_glow", "22_violent_boil", "23_steam_jet", "24_arthur_screaming", "25_kettle_levitating", "26_the_boiling_point", "27_final_crescendo", "28_the_pop", "29_title_card_kettle", "30_slogan_kettle", "31_coming_soon_kettle", "32_post_credits_kettle"],
            music_themes: vec!["theme_build_1", "theme_build_2", "theme_silence_bridge"],
        },
        _ => ProjectConfig {
            name: project.to_string(),
            scene_duration: 30.0,
            scenes: vec!["clip_0", "clip_1", "clip_2", "clip_3"],
            music_themes: vec!["theme_dark"],
        },
    }
}
