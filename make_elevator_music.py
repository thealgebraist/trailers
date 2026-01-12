import torch
import scipy.io.wavfile
import numpy as np
from transformers import AutoProcessor, MusicgenForConditionalGeneration, MusicgenConfig
import os
import gc

def generate_extreme_slop_music():
    device = "cpu"
    print(f"Using device: {device}")

    model_id = "facebook/musicgen-small"
    print(f"Loading MusicGen model {model_id}...")
    
    MusicgenForConditionalGeneration.config_class = MusicgenConfig
    
    processor = AutoProcessor.from_pretrained(model_id)
    model = MusicgenForConditionalGeneration.from_pretrained(model_id, dtype=torch.float32)
    model.to(device)

    sample_rate = model.config.audio_encoder.sampling_rate

    # Prompts designed for maximum dissonance and rhythmic chaos
    prompts = [
        "Extremely out of tune bossa nova, drums out of sync, clumsy rhythm, broken instruments",
        "High pitched staccato voices singing off-key, random pitch shifts, annoying melody",
        "Dissonant elevator music, out of tune piano, drums skipping beats, chaotic",
        "Atonal smooth jazz, screeching saxophone out of tune, erratic drum machine",
        "Low pitch distorted voices grunting out of tune, staccato, rhythmic mess",
        "Broken synthesizer playing random out of tune notes, no clear rhythm, frustrating",
        "Clumsy amateur band playing out of sync, detuned guitars, stumbling drums",
        "Ear-piercing high pitch staccato synth, random tempo changes, extremely annoying"
    ]

    print(f"Generating 8 tracks of chaotic slop on CPU...")

    for i, prompt in enumerate(prompts):
        filename = f"elevator_{i}.wav"
        # Overwrite previous ones to get the new extreme versions
        print(f"Generating {filename}: '{prompt}'")
        
        inputs = processor(
            text=[prompt],
            padding=True,
            return_tensors="pt",
        ).to(device)
        
        with torch.no_grad():
             # Increasing guidance_scale for more "slop" adherence
             audio_values = model.generate(**inputs, max_new_tokens=400, do_sample=True, guidance_scale=6.0) 
        
        audio_array = audio_values[0].cpu().numpy().squeeze()
        
        scipy.io.wavfile.write(filename, sample_rate, audio_array)
        print(f"Saved {filename}")
        
        del audio_values
        gc.collect()

if __name__ == "__main__":
    generate_extreme_slop_music()
