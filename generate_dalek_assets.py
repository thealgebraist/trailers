import torch
import os
import sys
import numpy as np
import scipy.io.wavfile as wavfile
import requests
from diffusers import DiffusionPipeline, StableAudioPipeline
from transformers import pipeline
from PIL import Image

# --- Configuration ---
PROJECT_NAME = "dalek"
ASSETS_DIR = f"assets_{PROJECT_NAME}"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# ElevenLabs Config
try:
    with open("eleven_key.txt", "r") as f:
        ELEVEN_API_KEY = f.read().strip()
except:
    ELEVEN_API_KEY = None

# Scene Definitions (Prompts & SFX Prompts)
SCENES = [
    ("01_skaro_landscape", "Wide Shot: A bleak, metallic alien landscape Skaro, jagged towers of steel, grey skies, smoke. Thousands of Daleks moving in formation, cinematic, 8k", "dark brooding metallic throbbing drone industrial wind"),
    ("02_dalek_factory_closeup", "Close Up: A single Dalek with a small scratch on its casing staring blankly at a desolate robotic factory, gritty, high detail", "marching mechanical sounds distant laser fire metallic clanking"),
    ("03_horseshoe_closeup", "Extreme close up: A Dalek eye-stalk looking at a piece of rusted scrap metal that looks like a horseshoe, shallow depth of field", "high pitched digital glitch sound resonant piano note"),
    ("04_golden_eye", "Close up: A Dalek eye-piece flickering and glowing with a warm golden light, magical atmosphere", "wooden screen door creaking open rustic sound"),
    ("05_kansas_farmhouse", "Cinematic Shot: 1950s Kodachrome style. A golden wheat field in rural Kansas, a quaint farmhouse under a blue sky, nostalgic", "birds chirping wind in wheat cow mooing distance"),
    ("06_baby_dalek", "Mid Shot: An elderly human couple in overalls and an apron, lovingly holding a baby-sized Dalek casing wrapped in a knit blanket, heart-warming", "gentle acoustic guitar fingerpicking birds"),
    ("07_dalek_pie", "Small Dalek helping an old woman bake a pie, a kitchen whisk is taped to its plunger arm, messy flour on counter", "laughter of elderly people kitchen sounds bubbling"),
    ("08_dalek_fishing", "Old man teaching a Dalek to fish in a creek, a fishing rod is taped to its laser arm, sunny day", "splashing water river ambience nature sounds"),
    ("09_dalek_tractor", "A Dalek sitting proudly on a red farm tractor in a field, cinematic lighting", "tractor engine chugging diesel idle"),
    ("10_red_schoolhouse", "Wide Shot: A tiny red one-room schoolhouse in a rural setting, idyllic", "school bell ringing children shouting happy"),
    ("11_class_photo", "Group Shot: 5 children posing for a class photo. 4 human kids and one Dalek wearing a propeller beanie hat, 1950s style", "children cheering happy atmosphere"),
    ("12_town_citizens", "Montage of town citizens: A baker holding a baguette, a teacher at a chalkboard, and a cop tipping his hat, smiling", "bicycle bell ringing small town ambience"),
    ("13_hide_and_seek", "Action Shot: A Dalek playing hide and seek, poorly hiding behind a thin tree in a park", "playful footsteps giggling Dalek gliding on gravel"),
    ("14_dalek_prom", "A Dalek at a high school prom, a floral corsage taped to its dome, sitting next to a punch bowl, disco lights", "muffled 1950s prom music chatter"),
    ("15_skaro_return", "Dramatic: A Dalek back on the bleak Skaro landscape, surrounded by dozens of angry, dark Daleks, high contrast", "ominous silence weapons powering up hum"),
    ("16_quivering_eye", "Extreme close up: A Dalek eye-stalk quivering, showing emotion, blue light flickering", "high pitched mechanical whine electrical static"),
    ("17_supreme_dalek_pie", "Key Scene: A Dalek facing a giant Supreme Dalek. The small Dalek extends its plunger, holding a warm apple pie with steam rising", "tense silence steam hiss"),
    ("18_dalek_hugging", "The Dalek giving a clumsy hug to a town policeman, heartwarming moment", "triumphant orchestral swell"),
    ("19_country_road", "The Dalek gliding fast down a country road with 4 happy children riding on its casing, explosion of colorful confetti", "joyful shouting wind rushing celebration"),
    ("20_title_card", "Cinematic title card: 'A DALEK COMES HOME' in bold metallic font, subtitle 'EXTERMINATE THE LONELINESS'", "deep cinematic bass boom hawk screech"),
    ("21_birthday_cake", "A Dalek trying to blow out birthday candles on a cake, its laser beam accidentally fires and explodes the cake into crumbs", "laser blast sound explosion muffled oops"),
]

VO_SCRIPT = """
In a universe... of infinite hate... where emotion is a crime... and compassion is deleted...
One soldier... is about to remember... where he really came from.
He wasn't born in a factory. He was found... in a cornfield.
Raised by Nana and Pop-Pop. They didn't see a killing machine. They saw... their little Rusty.
In a town with one baker... one teacher... one cop... and a graduating class of five... he was just one of the guys.
They taught him to read. They taught him to fish. And they taught him the most dangerous weapon of all...
Love.
He's going home. And he's bringing dessert.
This summer...
A Dalek Comes Home.
"""

def generate_images():
    print(f"--- Generating {len(SCENES)} FLUX.2-dev Images ---")
    pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.2-dev", torch_dtype=DTYPE).to(DEVICE)
    os.makedirs(f"{ASSETS_DIR}/images", exist_ok=True)
    for s_id, prompt, _ in SCENES:
        out_path = f"{ASSETS_DIR}/images/{s_id}.png"
        if not os.path.exists(out_path):
            print(f"Generating: {s_id}")
            image = pipe(prompt, num_inference_steps=50, guidance_scale=3.5, width=1280, height=720).images[0]
            image.save(out_path)
    del pipe
    torch.cuda.empty_cache()

def generate_sfx():
    print(f"--- Generating SFX with Stable Audio Open ---")
    pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16).to(DEVICE)
    os.makedirs(f"{ASSETS_DIR}/sfx", exist_ok=True)
    for s_id, _, sfx_prompt in SCENES:
        out_path = f"{ASSETS_DIR}/sfx/{s_id}.wav"
        if not os.path.exists(out_path):
            print(f"Generating SFX for: {s_id}")
            audio = pipe(sfx_prompt, num_inference_steps=100, audio_length_in_s=10.0).audios[0]
            audio_np = audio.T.cpu().numpy()
            wavfile.write(out_path, 44100, (audio_np * 32767).astype(np.int16))
    del pipe
    torch.cuda.empty_cache()

def generate_voiceover():
    print(f"--- Generating Voiceover with ElevenLabs ---")
    os.makedirs(f"{ASSETS_DIR}/voice", exist_ok=True)
    out_path = f"{ASSETS_DIR}/voice/voiceover_full.wav"
    if os.path.exists(out_path) or not ELEVEN_API_KEY:
        if not ELEVEN_API_KEY: print("No ElevenLabs key found, skipping VO.")
        return

    # Voice ID for a deep trailer voice (e.g., 'George' or similar)
    # Using a common one or searching for 'Don'
    VOICE_ID = "pNInz6obpg8nEByWQX7d" # Adam - deep and versatile

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVEN_API_KEY
    }
    data = {
        "text": VO_SCRIPT,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.8
        }
    }

    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        with open(out_path.replace(".wav", ".mp3"), "wb") as f:
            f.write(response.content)
        # Convert mp3 to wav if needed, but ffmpeg handles both.
        print(f"VO saved to {out_path.replace('.wav', '.mp3')}")
    else:
        print(f"ElevenLabs Error: {response.text}")

def generate_music():
    print(f"--- Generating Music with MusicGen-Large ---")
    os.makedirs(f"{ASSETS_DIR}/music", exist_ok=True)
    out_path = f"{ASSETS_DIR}/music/dalek_theme.wav"
    if os.path.exists(out_path): return

    synthesiser = pipeline("text-to-audio", "facebook/musicgen-large", device=DEVICE)
    # Montage of music styles as described
    prompts = [
        "dark industrial synth drone, metallic, sci-fi horror",
        "warm acoustic Americana guitar, rural, nostalgic",
        "soaring orchestral emotional crescendo, cinematic"
    ]
    
    clips = []
    sr = 32000
    for i, p in enumerate(prompts):
        print(f"Generating music part {i+1}...")
        output = synthesiser(p, forward_params={"max_new_tokens": 1500})
        clips.append(output["audio"][0].flatten())
        sr = output["sampling_rate"]
        
    combined = np.concatenate(clips, axis=0)
    wavfile.write(out_path, sr, (combined * 32767).astype(np.int16))
    del synthesiser
    torch.cuda.empty_cache()

if __name__ == "__main__":
    generate_images()
    generate_sfx()
    generate_voiceover()
    generate_music()
