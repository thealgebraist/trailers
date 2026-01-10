import torch
import scipy.io.wavfile
import os
import gc
import subprocess
import re
import numpy as np
from pathlib import Path
from dalek.core import get_device, flush, ensure_dir
from transformers import AutoProcessor, BarkModel, MusicgenForConditionalGeneration

# --- Configuration ---
OUTPUT_DIR = "assets_soviet_alf"
ensure_dir(f"{OUTPUT_DIR}/images")
ensure_dir(f"{OUTPUT_DIR}/voice")
ensure_dir(f"{OUTPUT_DIR}/sfx")
ensure_dir(f"{OUTPUT_DIR}/music")

DEVICE = get_device()
print(f"Using device: {DEVICE}")

# --- Movie Definition ---
TITLE = "ALFONS (АЛЬФ)"
# Using Bark's [laughter] and [clears throat] tags
VO_SCENES = [
    "From the cold depths of the cosmos to the warm embrace of the Motherland.",
    "Meet Comrade Alfons, fuzzy, hungry, and an exemplary citizen.",
    "He loves the Party, he loves the people, and he loves the neighbor's cat with sour cream.",
    "Approved by the Central Committee, a sitcom for every communal kitchen.",
    "ALFONS: coming soon to a collective farm near you.",
    "A cosmonaut fox once traded his tail for a radio; Alfons barters jokes instead.",
    "In a snowbound village, a girl shares her last log with Alfons, the lost alien guest.",
    "A soldier returns from the front with only a harmonica; Alfons brings laughter.",
    "An engineer rebuilds a tractor with spoons; Alfons fixes spirits with tea.",
    "A librarian hides banned fairy tales; Alfons reads them aloud at dusk.",
    "Grandpa weaves birch bark shoes; Alfons stitches hope into each pair.",
    "A postman delivers letters by sled; Alfons adds jokes to the envelopes.",
    "An orphaned bear cub is raised by miners; Alfons tells it bedtime tales.",
    "A river barge crew sings shanties; Alfons harmonizes in metallic baritone.",
    "A young chemist invents raspberry ink; Alfons doodles heroes on ration cards.",
    "A train stoker shares sunflower seeds; Alfons repays with cat-sitting favors.",
    "A radio play about the firebird crackles; Alfons improvises the squawks.",
    "A painter swaps brushes for coal dust; Alfons hangs the portrait in the canteen.",
    "A kolkhoz cook stretches soup for thirty; Alfons brings extra potatoes.",
    "A tram driver stops to rescue a kitten; Alfons salutes with his antenna.",
    "A night watchman hears footsteps of legends; Alfons whispers the punchlines.",
    "A factory choir rehearses 'Polyushko-polye'; Alfons conducts with a ladle.",
    "A seamstress sews stars on jackets; Alfons adds hidden pockets for candy.",
    "Two chess masters argue theory; Alfons checkmates with a samovar gambit.",
    "A geologist finds amber; Alfons pockets a tiny bee-shaped fossil.",
    "A sailor trades a compass for a poem; Alfons frames it above the stove.",
    "A nurse hums lullabies in the ward; Alfons syncs beeps to the melody.",
    "A border guard befriends a migratory crane; Alfons teaches it to dance.",
    "A meteorologist predicts clear skies; Alfons forecasts showers of laughter.",
    "A shoemaker repairs valenki by candlelight; Alfons polishes the samovar.",
    "A poet recites under a bare bulb; Alfons claps in perfect rhythm.",
    "A blacksmith reforges a bell; Alfons hears echoes of ancient folktales.",
    "A beekeeper gifts honeycomb; Alfons trades stories of distant moons.",
    "A streetcar clatters over cobbles; Alfons waves to bundled passengers.",
    "A film projectionist threads celluloid; Alfons splices in comic frames.",
    "A tailor measures silence; Alfons stitches laughter between the seams.",
    "A clockmaker resets town square bells; Alfons sets hearts to beat in time.",
    "A shepherd names each sheep after stars; Alfons hums them lullabies.",
    "A fisherman mends nets at dawn; Alfons counts ripples like constellations.",
    "A schoolteacher chalks Cyrillic letters; Alfons draws little rockets beside them.",
    "A glassblower shapes snowflake ornaments; Alfons fogs the pane in wonder.",
    "A tram bell rings at curfew; Alfons sneaks extra stories past the hour.",
    "A carpenter carves nesting dolls; Alfons hides a joke in each hollow.",
    "A radio operator hears distant Morse; Alfons taps back a friendly knock.",
    "A candlemaker guards every wick; Alfons shields each flame with his paw.",
    "A baker braids black bread; Alfons sprinkles memories like caraway.",
    "A cobbler hums tango in the stairwell; Alfons waltzes with the mop.",
    "A typist pounds samizdat pages; Alfons fans the ink to dry faster.",
    "A stone mason restores a cracked star; Alfons buffs it to a shine.",
    "A doctor boils instruments; Alfons counts to sixty in soft Russian.",
    "A trolley line hums in winter fog; Alfons draws hearts on frosted glass.",
    "A puppeteer lifts marionettes; Alfons tugs invisible strings of hope.",
    "A farmhand whistles for geese; Alfons joins with off-beat bleeps.",
    "A lighthouse keeper trims the flame; Alfons signals ships with jokes.",
    "A post office cat naps on parcels; Alfons guards its dreams from mice.",
    "A violinist rehearses Shostakovich; Alfons adds percussion with spoons.",
    "A miner pins medals to his coat; Alfons salutes with solemn pride.",
    "A milkmaid balances pails at dawn; Alfons trails with a lantern.",
    "A bookbinder glues frayed spines; Alfons presses flowers between pages.",
    "A radio choir sings folk rounds; Alfons harmonizes in static.",
    "A metallurgist pours glowing steel; Alfons sketches dragons in the sparks.",
    "A janitor sweeps marble halls; Alfons dusts off forgotten legends.",
    "A child trades a marble for a story; Alfons tells sixty-four in return."
]

IMAGE_PROMPTS = [
    "Title card 'АЛЬФ' featuring Alfons silhouette, bold 1950s Soviet typography, black and white, heavy film grain, scratches, vintage cinema style.",
    "Alfons in a communal kitchen pouring tea from a brass samovar, 1950s Soviet photo, black and white, high grain.",
    "Alfons saluting a red flag in snowy Red Square, long shadow, vintage film still, monochrome.",
    "Alfons peeking from behind heavy curtains at a fat cat on a radiator, cramped Soviet apartment, B&W, grainy.",
    "Alfons at a family dinner with stern parents, wooden laughter, sitcom lighting, black and white.",
    "Close-up of Alfons laughing wide, vignetting and film scratches, 1950s lens.",
    "Alfons trading a harmonica with a soldier at a village well, snow dust, monochrome realism.",
    "Alfons sharing a log by a brick stove with a bundled village girl, candlelit shadows, B&W.",
    "Alfons riding on a tractor rebuilt from scrap spoons, kolkhoz yard, high contrast film.",
    "Alfons reading banned fairy tales in a dim library corridor, flashlight beam, grainy photo.",
    "Alfons helping grandpa weave birch bark shoes, sawdust floating, black and white.",
    "Alfons stuffing jokes into letters at a snowy post station, sled outside, monochrome.",
    "Miners cradle a bear cub while Alfons holds a lantern underground, dusty film look.",
    "Barge crew singing on the Volga, Alfons keeps tempo with a tin cup, misty monochrome.",
    "Alfons doodling heroes with raspberry ink on ration cards, small desk lamp glow, B&W.",
    "Train stoker sharing sunflower seeds with Alfons beside the locomotive, steam and grain.",
    "Radio play studio with Alfons voicing the firebird squawk, ribbon mics, black and white.",
    "Canteen wall with a coal-dust portrait of a worker painted by Alfons, vintage patina.",
    "Soup line in a kolkhoz kitchen, Alfons sneaks extra potatoes, overhead harsh light.",
    "Tram driver rescuing a kitten, Alfons salutes with antenna, snow swirling, monochrome.",
    "Night watchman and Alfons hearing folklore footsteps in a factory hall, single bulb light.",
    "Factory choir rehearsing, Alfons conducts with a soup ladle, grainy stage photo.",
    "Seamstress sewing stars on jackets, Alfons adds hidden pockets, close-up hands, B&W.",
    "Chess masters in a smoky club, Alfons offers a samovar gambit, cinematic framing.",
    "Geologist shows amber to Alfons by campfire, pine silhouettes, monochrome.",
    "Sailor trades compass for a poem, Alfons frames it on galley wall, soft focus.",
    "Hospital ward at night, nurse hums, Alfons syncs monitors to melody, film grain.",
    "Border guard and crane silhouetted at dawn, Alfons leads a dance step, B&W.",
    "Meteorologist in a wooden tower, Alfons forecasts with chalk clouds, vintage photo.",
    "Cobbling valenki by candlelight, Alfons polishes a samovar, warm monochrome.",
    "Poet reciting under bare bulb, Alfons claps rhythmically, smoky room, B&W.",
    "Blacksmith reforging a bell, Alfons hears echoes, sparks captured in film grain.",
    "Beekeeper gifting honeycomb to Alfons, birch trees behind, soft monochrome.",
    "Streetcar clattering over cobbles, Alfons waving to passengers, winter haze.",
    "Projectionist threading film, Alfons splicing comic frames, projector beam glow.",
    "Tailor measuring silence, Alfons stitching laughter into seams, close-up needle.",
    "Clockmaker resetting town square bells, Alfons aligns gears, snowy plaza, B&W.",
    "Shepherd under stars, Alfons hums lullabies to sheep, lantern light.",
    "Fisherman mending nets at dawn, Alfons counts ripples, river fog, monochrome.",
    "Schoolteacher chalking Cyrillic, Alfons draws rockets beside letters, dusty light.",
    "Glassblower shaping snowflake ornaments, Alfons fogs the pane, high grain.",
    "Tram bell at curfew, Alfons sneaks stories past the hour, empty street, B&W.",
    "Carpenter carving nesting dolls, Alfons hides jokes in each hollow, workshop grain.",
    "Radio operator hearing distant Morse, Alfons taps a reply, dim green glow.",
    "Candlemaker guarding wicks, Alfons shields small flames, close-up hands, monochrome.",
    "Baker braiding black bread, Alfons sprinkles caraway, oven glow, B&W.",
    "Cobbler humming tango, Alfons waltzes with a mop in stairwell shadows.",
    "Typist pounding samizdat pages, Alfons fans drying ink, cramped room.",
    "Stone mason restoring cracked star, Alfons buffs it to shine, scaffolding, B&W.",
    "Doctor boiling instruments, Alfons counts seconds, enamel basin reflections.",
    "Trolley line humming in fog, Alfons draws hearts on frosted glass, streetlamp glow.",
    "Puppeteer lifting marionettes, Alfons tugs invisible strings, backstage dust.",
    "Farmhand whistling for geese, Alfons answers with off-beat bleeps, muddy path.",
    "Lighthouse keeper trimming flame, Alfons signals distant ship, storm spray, monochrome.",
    "Post office cat napping on parcels, Alfons guards its dreams, soft focus.",
    "Violinist rehearsing Shostakovich, Alfons taps spoons for percussion, parlor light.",
    "Miner pinning medals to coat, Alfons salutes solemnly, coal dust air, B&W.",
    "Milkmaid balancing pails at dawn, Alfons trails with lantern, breath clouds.",
    "Bookbinder gluing frayed spines, Alfons presses flowers between pages, close-up.",
    "Radio choir singing folk rounds, Alfons harmonizes in static, control room glow.",
    "Metallurgist pouring glowing steel, Alfons sketches dragons in sparks, high contrast.",
    "Janitor sweeping marble halls, Alfons dusts off forgotten legends, echoing steps.",
    "Child trading a marble for a story, Alfons kneels to tell sixty-four tales, soft grain."
]

SD_CLI = Path("/opt/homebrew/bin/sd-cli")
SD_MODEL = Path("/Users/anders/models/sd/stable-diffusion-v1-5-Q8_0.gguf")

def generate_voice_and_laughs():
    print('--- Generating Bark VO with Laughs ---')
    try:
        processor = AutoProcessor.from_pretrained("suno/bark")
        model = BarkModel.from_pretrained("suno/bark", torch_dtype=torch.float32).to(DEVICE)
        voice_preset = "v2/en_speaker_9"
        
        for i, text in enumerate(VO_SCENES):
            out_file = f"{OUTPUT_DIR}/voice/voice_{i:02d}.wav"
            if os.path.exists(out_file): continue
            
            print(f'Generating scene {i:02d}: "{text}"')
            inputs = processor(text, voice_preset=voice_preset, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                audio_array = model.generate(**inputs, min_eos_p=0.05).cpu().numpy().squeeze()
            
            sample_rate = model.generation_config.sample_rate
            scipy.io.wavfile.write(out_file, sample_rate, audio_array)
            apply_vintage_effect(out_file)
            
        del model, processor; flush()
    except Exception as e:
        print(f'Bark generation failed: {e}')

def apply_vintage_effect(file_path):
    try:
        temp = file_path.replace('.wav', '_tmp.wav')
        af = "highpass=f=300,lowpass=f=3000,compand=0.3|0.3:1|1:-90/-60|-60/-40|-40/-30|-20/-20:6:0:-90:0.2"
        subprocess.run(["ffmpeg", "-y", "-i", file_path, "-af", af, temp], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.replace(temp, file_path)
    except: pass

def generate_images():
    print('--- Generating Grainy B&W Images via sd-cli (SD 1.5 GGUF) ---')
    if not SD_CLI.exists():
        print(f"sd-cli not found at {SD_CLI}")
        return
    if not SD_MODEL.exists():
        print(f"Model not found at {SD_MODEL}")
        return

    for i, prompt in enumerate(IMAGE_PROMPTS):
        out_file = f"{OUTPUT_DIR}/images/scene_{i:02d}.png"
        if os.path.exists(out_file): 
            continue
        print(f'Generating image {i:02d}...')
        cmd = [
            str(SD_CLI),
            "-m", str(SD_MODEL),
            "-p", prompt,
            "-o", out_file,
            "--width", "512",
            "--height", "512",
            "--steps", "32",
            "--cfg-scale", "5",
            "--seed", str(42 + i),
            "-t", "-1",
            "--vae-tiling",
            "--clip-on-cpu",
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Image generation failed for index {i}: {e}")

def generate_music():
    print('--- Generating Soviet Sitcom Music (MusicGen) ---')
    try:
        model_id = "facebook/musicgen-small"
        processor = AutoProcessor.from_pretrained(model_id)
        model = MusicgenForConditionalGeneration.from_pretrained(model_id).to(DEVICE)
        
        prompt = (
            "1950s Soviet television theme, ALF main melody played on a detuned vintage upright piano "
            "and a wobbly, drifting vacuum-tube synthesizer. Style is Socialist Realism orchestral pop "
            "meets early electronic experimentalism. Out-of-tune brass section, heavy tape flutter, "
            "7.5 IPS reel-to-reel saturation, distorted Soviet radio broadcast quality. Low-fidelity, "
            "hauntingly nostalgic, slightly discordant harmonies, monophonic recording, rhythmic gallop "
            "with a 'ticking' mechanical percussion."
        )
        out_file = f"{OUTPUT_DIR}/music/theme.wav"
        if not os.path.exists(out_file):
            print("Generating theme with MusicGen...")
            inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                audio_values = model.generate(**inputs, max_new_tokens=1000)
            
            audio_data = audio_values[0, 0].cpu().numpy()
            sample_rate = model.config.audio_encoder.sampling_rate
            scipy.io.wavfile.write(out_file, sample_rate, audio_data)
            apply_vintage_effect(out_file)
        del model, processor; flush()
    except Exception as e:
        print(f'Music generation failed: {e}')

if __name__ == '__main__':
    generate_images()
    generate_music()
    generate_voice_and_laughs()
