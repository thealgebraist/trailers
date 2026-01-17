# generate_horror_everyday.py
# Generates a 4-minute horror short video with jump scares about everyday things
# Uses FLUX.1-schnell, 64 images, 64 steps each

from vidlib import assets, assemble
import argparse
import os

SCENES = [
    ("01_alarm_clock", "A close-up of a blaring alarm clock in a dark bedroom, red digits glowing ominously, horror lighting, 8k, cinematic, unsettling atmosphere", "distorted alarm sound, sudden loud ringing, eerie undertone"),
    ("02_brushing_teeth", "A person brushing their teeth, but the mirror reflection moves differently, horror lighting, 8k, cinematic, uncanny valley", "creaking mirror, distorted brushing sound, sudden glass shatter"),
    ("03_commute_train", "A crowded train carriage, faces blurred and staring, flickering lights, horror lighting, 8k, cinematic, claustrophobic", "train rumble, sudden screech, whispering voices"),
    ("04_office_desk", "A person at an office desk, papers move on their own, shadowy figures in the background, horror lighting, 8k, cinematic", "rustling papers, sudden loud bang, eerie whispers"),
    ("05_elevator", "Inside an elevator, the floor indicator glitches, doors open to darkness, horror lighting, 8k, cinematic", "elevator ding, sudden silence, distorted scream"),
    ("06_grocery_store", "A grocery aisle, products rearrange themselves, faces on packaging stare, horror lighting, 8k, cinematic", "shopping cart squeak, sudden crash, whispering voices"),
    ("07_crosswalk", "A crosswalk at night, traffic lights flicker, shadowy figures cross, horror lighting, 8k, cinematic", "traffic hum, sudden horn, footsteps echo"),
    ("08_cafe", "A cafe, coffee cup spills blood, barista's face distorts, horror lighting, 8k, cinematic", "coffee machine hiss, sudden scream, distorted laughter"),
    ("09_staircase", "A staircase, steps stretch endlessly, hands reach from the darkness, horror lighting, 8k, cinematic", "footsteps, sudden thud, whispering voices"),
    ("10_bathroom", "A bathroom, faucet drips black liquid, reflection smiles back, horror lighting, 8k, cinematic", "dripping water, sudden crash, eerie giggle"),
    ("11_kitchen", "A kitchen, knives float in the air, fridge opens to darkness, horror lighting, 8k, cinematic", "knife clatter, sudden bang, distorted voices"),
    ("12_living_room", "A living room, TV static shows ghostly faces, furniture moves, horror lighting, 8k, cinematic", "TV static, sudden loud noise, whispering"),
    ("13_bedroom", "A bedroom, closet door creaks open, eyes peer out, horror lighting, 8k, cinematic", "creaking door, sudden scream, breathing sound"),
    ("14_garage", "A garage, car headlights flicker, shadow moves behind, horror lighting, 8k, cinematic", "engine rev, sudden crash, whispering"),
    ("15_mailbox", "A mailbox, letters spill out covered in blood, horror lighting, 8k, cinematic", "mail slot clang, sudden scream, eerie whisper"),
    ("16_park", "A park, playground swings move on their own, shadowy children laugh, horror lighting, 8k, cinematic", "swing creak, sudden giggle, distorted laughter"),
    ("17_supermarket_checkout", "A checkout lane, cashier's face melts, money turns to insects, horror lighting, 8k, cinematic", "beep, sudden buzz, whispering"),
    ("18_subway", "A subway platform, train arrives empty, doors open to darkness, horror lighting, 8k, cinematic", "train screech, sudden silence, eerie voices"),
    ("19_restaurant", "A restaurant, food writhes on the plate, waiter's eyes bleed, horror lighting, 8k, cinematic", "fork clatter, sudden scream, whispering"),
    ("20_hospital", "A hospital room, monitors flatline, shadowy figure stands over bed, horror lighting, 8k, cinematic", "monitor beep, sudden silence, distorted voice"),
    ("21_school", "A classroom, chalkboard writes itself, students vanish, horror lighting, 8k, cinematic", "chalk screech, sudden bang, whispering"),
    ("22_library", "A library, books fly off shelves, librarian's face distorts, horror lighting, 8k, cinematic", "book thud, sudden scream, whispering"),
    ("23_gym", "A gym, weights levitate, mirrors crack, horror lighting, 8k, cinematic", "weight clank, sudden crash, eerie voices"),
    ("24_pool", "A swimming pool, water turns black, hands reach up, horror lighting, 8k, cinematic", "splash, sudden scream, whispering"),
    ("25_bus_stop", "A bus stop, bus arrives with no driver, passengers are shadows, horror lighting, 8k, cinematic", "bus engine, sudden silence, whispering"),
    ("26_movie_theater", "A movie theater, screen shows real-life horrors, audience vanishes, horror lighting, 8k, cinematic", "projector hum, sudden scream, whispering"),
    ("27_pet_store", "A pet store, animals speak in human voices, cages rattle, horror lighting, 8k, cinematic", "animal noises, sudden scream, whispering"),
    ("28_bank", "A bank, money turns to ashes, teller's face distorts, horror lighting, 8k, cinematic", "coin clink, sudden bang, whispering"),
    ("29_gas_station", "A gas station, pumps leak blood, attendant vanishes, horror lighting, 8k, cinematic", "pump hiss, sudden scream, whispering"),
    ("30_highway", "A highway, cars drive themselves, passengers are skeletons, horror lighting, 8k, cinematic", "car engine, sudden crash, whispering"),
    ("31_mall", "A mall, mannequins move, shoppers vanish, horror lighting, 8k, cinematic", "footsteps, sudden scream, whispering"),
    ("32_attic", "An attic, boxes open themselves, shadowy figures crawl out, horror lighting, 8k, cinematic", "box thud, sudden scream, whispering"),
    ("33_basement", "A basement, walls bleed, stairs collapse, horror lighting, 8k, cinematic", "dripping, sudden crash, whispering"),
    ("34_rooftop", "A rooftop, wind howls, shadow jumps off, horror lighting, 8k, cinematic", "wind, sudden scream, whispering"),
    ("35_garden", "A garden, plants strangle each other, gardener's face distorts, horror lighting, 8k, cinematic", "rustling leaves, sudden scream, whispering"),
    ("36_laundry_room", "A laundry room, washing machine spins endlessly, clothes turn to hands, horror lighting, 8k, cinematic", "machine hum, sudden scream, whispering"),
    ("37_balcony", "A balcony, railing bends, shadow falls, horror lighting, 8k, cinematic", "metal creak, sudden scream, whispering"),
    ("38_hallway", "A hallway, doors slam shut, lights flicker, horror lighting, 8k, cinematic", "door slam, sudden scream, whispering"),
    ("39_front_door", "A front door, knocks echo, handle turns by itself, horror lighting, 8k, cinematic", "knocking, sudden scream, whispering"),
    ("40_backyard", "A backyard, swing set moves, shadowy figure stands, horror lighting, 8k, cinematic", "swing creak, sudden scream, whispering"),
    ("41_driveway", "A driveway, car doors open and close, shadow moves, horror lighting, 8k, cinematic", "car door slam, sudden scream, whispering"),
    ("42_street", "A street, lamplights flicker, shadows chase, horror lighting, 8k, cinematic", "lamp buzz, sudden scream, whispering"),
    ("43_bridge", "A bridge, water below turns red, shadow jumps, horror lighting, 8k, cinematic", "water splash, sudden scream, whispering"),
    ("44_tunnel", "A tunnel, walls close in, shadowy hands reach, horror lighting, 8k, cinematic", "echo, sudden scream, whispering"),
    ("45_park_bench", "A park bench, shadow sits, birds fly away, horror lighting, 8k, cinematic", "bird flapping, sudden scream, whispering"),
    ("46_phone_booth", "A phone booth, phone rings, voice whispers, horror lighting, 8k, cinematic", "phone ring, sudden scream, whispering"),
    ("47_post_office", "A post office, letters fly, clerk vanishes, horror lighting, 8k, cinematic", "letter flutter, sudden scream, whispering"),
    ("48_bar", "A bar, drinks spill blood, bartender's face distorts, horror lighting, 8k, cinematic", "glass clink, sudden scream, whispering"),
    ("49_hotel_room", "A hotel room, bed levitates, shadow stands at window, horror lighting, 8k, cinematic", "bed creak, sudden scream, whispering"),
    ("50_waiting_room", "A waiting room, clock spins backwards, people vanish, horror lighting, 8k, cinematic", "clock tick, sudden scream, whispering"),
    ("51_pharmacy", "A pharmacy, pills crawl, pharmacist's face distorts, horror lighting, 8k, cinematic", "pill rattle, sudden scream, whispering"),
    ("52_hardware_store", "A hardware store, tools fly, clerk vanishes, horror lighting, 8k, cinematic", "tool clang, sudden scream, whispering"),
    ("53_bookstore", "A bookstore, books bleed, cashier's face distorts, horror lighting, 8k, cinematic", "book thud, sudden scream, whispering"),
    ("54_cemetery", "A cemetery, graves open, shadows crawl out, horror lighting, 8k, cinematic", "earth rumble, sudden scream, whispering"),
    ("55_church", "A church, pews move, priest's face distorts, horror lighting, 8k, cinematic", "organ hum, sudden scream, whispering"),
    ("56_fire_station", "A fire station, hoses leak blood, firefighter vanishes, horror lighting, 8k, cinematic", "hose hiss, sudden scream, whispering"),
    ("57_police_station", "A police station, handcuffs float, officer's face distorts, horror lighting, 8k, cinematic", "handcuff clink, sudden scream, whispering"),
    ("58_airport", "An airport, planes fly backwards, passengers vanish, horror lighting, 8k, cinematic", "plane engine, sudden scream, whispering"),
    ("59_train_station", "A train station, trains arrive empty, shadowy figures board, horror lighting, 8k, cinematic", "train screech, sudden scream, whispering"),
    ("60_factory", "A factory, machines move on their own, workers vanish, horror lighting, 8k, cinematic", "machine hum, sudden scream, whispering"),
    ("61_construction_site", "A construction site, cranes move by themselves, shadow falls, horror lighting, 8k, cinematic", "crane creak, sudden scream, whispering"),
    ("62_playground", "A playground, swings move, children vanish, horror lighting, 8k, cinematic", "swing creak, sudden scream, whispering"),
    ("63_gas_meter", "A gas meter, numbers spin wildly, shadowy hands reach, horror lighting, 8k, cinematic", "meter tick, sudden scream, whispering"),
    ("64_laundry_basket", "A laundry basket, clothes move, shadow crawls out, horror lighting, 8k, cinematic", "cloth rustle, sudden scream, whispering"),
]

ASSETS_DIR = "assets_horror_everyday"
MODEL_ID = "black-forest-labs/FLUX.1-schnell"
STEPS = 64
GUIDANCE = 0.0
QUANT = "4bit"

class Args:
    def __init__(self):
        self.model = MODEL_ID
        self.flux2 = None
        self.steps = STEPS
        self.guidance = GUIDANCE
        self.quant = QUANT
        self.offload = False
        self.scalenorm = False
        self.assets_dir = ASSETS_DIR

if __name__ == "__main__":
    os.makedirs(f"{ASSETS_DIR}/images", exist_ok=True)
    os.makedirs(f"{ASSETS_DIR}/sfx", exist_ok=True)
    args = Args()
    # Generate images
    for idx, (scene_id, prompt, sfx_prompt) in enumerate(SCENES):
        args.scene_id = scene_id
        args.prompt = prompt
        args.sfx_prompt = sfx_prompt
        assets.metro_generate_images(args)
        assets.metro_generate_sfx(args)
    # Optionally, generate music and voiceover using horror prompts
    # Assemble video
    output_file = "horror_everyday_short.mp4"
    assemble.assemble_metro(ASSETS_DIR, output_file)
    print(f"Created horror short: {output_file}")
