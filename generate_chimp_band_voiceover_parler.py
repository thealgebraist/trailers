import torch
import numpy as np
import soundfile as sf
import os
import gc
import random
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration

# --- Configuration ---
OUTPUT_DIR = "assets_chimp_band_64"
os.makedirs(f"{OUTPUT_DIR}/voice", exist_ok=True)

if torch.backends.mps.is_available(): DEVICE = "mps"
elif torch.cuda.is_available(): DEVICE = "cuda"
else: DEVICE = "cpu"

print(f"Using device: {DEVICE}")

# Mirror the instrument list from generate_chimp_band_images.py
RIDICULOUS_INSTRUMENTS = [
    "banjo", "cowbell", "bass guitar", "kazoo", "tuba", 
    "accordion", "triangle", "keytar", "theremin", "electric violin",
    "slide whistle", "didgeridoo"
]

# Reconstruct the 64 scenes using a fixed seed to mirror image generation
def get_mirrored_scenes(seed=42):
    random.seed(seed)
    scenes = []
    for i in range(64):
        roll = random.random()
        if roll < 0.4:
            instruments = ["bongos"]
        elif roll < 0.7:
            instruments = ["bongos", random.choice(RIDICULOUS_INSTRUMENTS)]
        else:
            instruments = ["bongos"] + random.sample(RIDICULOUS_INSTRUMENTS, 2)
        scenes.append(instruments)
    return scenes

SCENE_DATA = get_mirrored_scenes(42)

# Narrative Personas for the 16 Variations
PERSONAS = [
    "The Enthusiast", "The Absurdist", "The Wildlife Documentarian", "The Hype Man",
    "The Confused Spectator", "The Musical Critic", "The Zen Master", "The Storyteller",
    "The Excited Child", "The Beatnik Poet", "The Sports Commentator", "The Space Explorer",
    "The Time Traveler", "The Detective", "The Professor", "The Dreamer"
]

def generate_narrative(persona, scenes_subset, part_name):
    """Generates a long-form prompt for Parler based on persona and actual instruments."""
    # List unique instruments in this subset (excluding bongos which are everywhere)
    instruments = []
    for s in scenes_subset:
        for inst in s:
            if inst != "bongos" and inst not in instruments:
                instruments.append(inst)
    
    inst_str = ", ".join(instruments[:-1]) + " and " + instruments[-1] if len(instruments) > 1 else (instruments[0] if instruments else "just bongos")
    
    prompts = {
        "The Enthusiast": f"Oh wow! Look at these happy chimps! In this {part_name}, they are rocking the {inst_str}. The energy is incredible! The bongo rhythm is tight and the {instruments[0] if instruments else 'drums'} sounds amazing. I've never seen such joy in a primate band!",
        "The Absurdist": f"Reality is breaking. We have primates playing {inst_str}. Why a {instruments[-1] if instruments else 'bongo'}? Because the universe is a joke and these chimps are the punchline. This {part_name} is a fever dream of kazoos and chaos.",
        "The Wildlife Documentarian": f"Observe the primate in its new habitat. Here in {part_name}, the group has adopted the {inst_str}. The dominant male leads on the bongos, while the others experiment with the {instruments[0] if instruments else 'percussion'}. Truly fascinating social behavior.",
        "The Hype Man": f"Yo! Check the flow! Chimp Band is in the building for {part_name}! We got the {inst_str} dropping heat! That bongo beat is fire and the happy chimps are taking over the world! Let's go!",
        "The Confused Spectator": f"I... I don't understand what I'm looking at. Is that a chimp with a {instruments[0] if instruments else 'bongo'}? And now they're playing {inst_str}? In this {part_name}, my brain is just melting. They look so happy, but why the tuba?",
        "The Musical Critic": f"The tonal quality of this {part_name} is... experimental. The juxtaposition of bongos with {inst_str} creates a post-modern jungle soundscape. The {instruments[-1] if instruments else 'chimp'} shows surprisingly good technique.",
        "The Zen Master": f"Breathe. Listen to the {part_name}. The bongos are the heartbeat. The {inst_str} are the thoughts passing through the mind. The happy chimps are one with the rhythm of the universe.",
        "The Storyteller": f"Once upon a time, in a clearing filled with white light, the chimps found the {inst_str}. In this {part_name}, they began a song that would echo through the ages, led by the rhythmic call of the ancient bongos.",
        "The Excited Child": f"Look, look! The monkeys are playing music! They have a {inst_str}! Yay! The happy chimps are jumping and drumming on the bongos! This {part_name} is the best concert ever!",
        "The Beatnik Poet": f"Bongo drums... tapping... snapping... the {inst_str} crying out in the white void of {part_name}. Primate souls, neon jungle, happy chimps finding the soul of the rhythm. Dig it.",
        "The Sports Commentator": f"And they're off! A strong start in {part_name} with the bongos taking the lead! Oh! A spectacular move with the {instruments[0] if instruments else 'kazoo'}! The {inst_str} are working together perfectly! What a performance!",
        "The Space Explorer": f"Captain's log. We've discovered a planet of musical primates. In this {part_name}, they are broadcasting signals using {inst_str} and rhythmic bongo pulses. The chimps appear peaceful and highly evolved.",
        "The Time Traveler": f"I've seen the future, and it sounds like this {part_name}. A fusion of bongos and {inst_str}. In the year 3000, these happy chimps are the only ones who still know how to party.",
        "The Detective": f"The case of the Musical Monkeys. The evidence in {part_name} is clear: {inst_str} found at the scene. The bongos provide the heartbeat of the mystery. The suspects? Very happy chimps.",
        "The Professor": f"Welcome to Ethnomusicology 101. Today in {part_name} we examine the 'Bongo Frenzy'. Note the use of {inst_str}. The happy chimps demonstrate a complex understanding of rhythmic syncopation.",
        "The Dreamer": f"I had a dream about a white room where chimps played {inst_str}. In this {part_name}, that dream has come true. The bongos are calling us all to join the happy chimps in their dance."
    }
    
    base_text = prompts.get(persona, prompts["The Enthusiast"])
    # Extend the text to ensure it reaches ~150 words to fill ~60s
    filler = " The rhythm carries us deeper into the jungle of sound. Every beat of the bongos is a testament to the primal joy of creation. The chimps don't need a map; they have the melody. They don't need a reason; they have the rhythm. Onward we go, through the kazoos and the tubas, towards the heart of the frenzy. Happy chimps, happy world, happy music."
    return (base_text + filler * 3)[:800] # Cap to sensible length

def generate_parler_voiceover_variations():
    print(f"--- Generating 16 Variations of 180s Voiceover (Mirrored to Image Prompts) ---")
    repo_id = "parler-tts/parler-tts-mini-v1"
    model = ParlerTTSForConditionalGeneration.from_pretrained(repo_id).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    sample_rate = model.config.sampling_rate

    description = ("A male speaker with a deep, calm voice delivers his words "
                   "extremely slowly with long pauses, in a very quiet room "
                   "with very clear audio quality.")

    # Split 64 scenes into three parts (~21 scenes each for 60s)
    parts = [SCENE_DATA[0:21], SCENE_DATA[21:42], SCENE_DATA[42:64]]
    part_names = ["first movement", "middle section", "grand finale"]

    try:
        for v_idx, persona in enumerate(PERSONAS):
            var_name = f"v{v_idx+1}_{persona.lower().replace(' ', '_')}"
            print(f"\n--- Variation {v_idx+1}/16: {persona} ---")
            var_dir = f"{OUTPUT_DIR}/voice/{var_name}"
            os.makedirs(var_dir, exist_ok=True)
            
            full_audio_segments = []
            for p_idx, (subset, p_name) in enumerate(zip(parts, part_names)):
                out_file = f"{var_dir}/part_{p_idx+1}.wav"
                if os.path.exists(out_file):
                    audio_data, _ = sf.read(out_file)
                    full_audio_segments.append(audio_data)
                    continue

                text = generate_narrative(persona, subset, p_name)
                print(f"Generating {var_name} part {p_idx+1}/3...")
                input_ids = tokenizer(description, return_tensors="pt").input_ids.to(DEVICE)
                prompt_input_ids = tokenizer(text, return_tensors="pt").input_ids.to(DEVICE)
                generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
                audio_arr = generation.cpu().numpy().squeeze()
                sf.write(out_file, audio_arr, sample_rate)
                full_audio_segments.append(audio_arr)
                flush()

            master_path = f"{var_dir}/full_voiceover.wav"
            combined = np.concatenate(full_audio_segments)
            sf.write(master_path, combined, sample_rate)
            print(f"Created master: {master_path}")

    except Exception as e: print(f"Generation failed: {e}")
    finally: del model; flush()

if __name__ == "__main__":
    generate_parler_voiceover_variations()


if __name__ == "__main__":
    generate_parler_voiceover_variations()
