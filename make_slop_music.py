import torch
import scipy.io.wavfile
import numpy as np
from transformers import AutoProcessor, MusicgenForConditionalGeneration, MusicgenConfig
import os

def generate_and_mix():
    # Setup device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Load MusicGen
    model_id = "facebook/musicgen-small"
    print(f"Loading MusicGen model {model_id}...")
    # Monkeypatch to fix transformers bug
    MusicgenForConditionalGeneration.config_class = MusicgenConfig
    
    processor = AutoProcessor.from_pretrained(model_id)
    model = MusicgenForConditionalGeneration.from_pretrained(model_id)
    model.to(device)

    # Parameters
    sample_rate = model.config.audio_encoder.sampling_rate
    instruments = {
        "drums": "A loud, clumsy, out-of-sync drum beat, lo-fi",
        "bass": "A very loud distorted slap bass groove, out of tune",
        "guitar": "A scratchy out-of-tune electric guitar riff, heavy metal slop",
        "voice": "A person singing 'all work and no play' very badly and off-key"
    }
    
    all_samples = {name: [] for name in instruments}

    # Generate 8 samples for each instrument
    for name, prompt in instruments.items():
        print(f"Generating 8 samples for {name}...")
        for i in range(8):
            inputs = processor(
                text=[prompt],
                padding=True,
                return_tensors="pt",
            ).to(device)
            
            # Generate 3 seconds for each sample (to keep it fast)
            with torch.no_grad():
                 audio_values = model.generate(**inputs, max_new_tokens=256, do_sample=True) 
            
            audio_array = audio_values[0].cpu().numpy().squeeze()
            
            # Save individual sample
            filename = f"{name}_{i}.wav"
            scipy.io.wavfile.write(filename, sample_rate, audio_array)
            print(f"Saved {filename}")
            
            all_samples[name].append(audio_array)

    # Create a "bad midi song"
    target_length = 15 * sample_rate
    song = np.zeros(target_length)

    print("Composing bad slop song from 32 samples...")
    
    # Randomly scatter the samples
    for name, samples in all_samples.items():
        weight = {"drums": 0.4, "bass": 0.5, "guitar": 0.3, "voice": 0.7}[name]
        for sample in samples:
            # Place each sample at a random position
            start = np.random.randint(0, target_length - len(sample))
            end = start + len(sample)
            song[start:end] += sample * weight

    # Normalize
    max_val = np.max(np.abs(song))
    if max_val > 0:
        song = song / max_val
        
    scipy.io.wavfile.write("slop_song.wav", sample_rate, song.astype(np.float32))
    print("Saved slop_song.wav")

if __name__ == "__main__":
    generate_and_mix()
