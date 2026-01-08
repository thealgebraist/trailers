import torch
import numpy as np
import os
import random
from dalek.core import get_device, flush, ensure_dir
from dalek.audio import load_parler, generate_parler_audio, save_audio

# --- Configuration ---
OUTPUT_DIR = "assets_chimp_band_64"
ensure_dir(f"{OUTPUT_DIR}/voice")

DEVICE = get_device()
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
    filler = " The rhythm carries us deeper into the jungle of sound. Every beat of the bongos is a testament to the primal joy of creation. The chimps don't need a map; they have the melody. They don't need a reason; they have the rhythm. Onward we go, through the kazoos and the tubas, towards the heart of the frenzy. Happy chimps, happy world, happy music."
    return (base_text + filler * 3)[:800]

def generate_parler_voiceover_variations():
    print(f"--- Generating 16 Variations of 180s Voiceover (Mirrored to Image Prompts) ---")
    model, tokenizer = load_parler()
    sample_rate = model.config.sampling_rate

    description = ("A male speaker with a deep, calm voice delivers his words "