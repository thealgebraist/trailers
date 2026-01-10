#!/usr/bin/env python3
"""
Generate speech samples using Parler TTS Tiny v1
Counts from 1 to 50 with voice synthesis
"""
import torch
import torchaudio
import numpy as np
from scipy.io import wavfile
import os

def generate_counting_tts(start=1, end=50, speaker_description="A clear female voice speaks naturally"):
    """
    Use Parler TTS to generate counting samples
    """
    print("Loading Parler TTS Tiny v1...")
    
    try:
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer
    except ImportError:
        print("Installing parler-tts...")
        os.system("pip3 install git+https://github.com/huggingface/parler-tts.git --break-system-packages -q")
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer
    
    model_name = "parler-tts/parler-tts-tiny-v1"
    
    # Load model and tokenizer
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Generate samples
    audio_samples = []
    sample_rate = model.config.sampling_rate
    
    print(f"\nGenerating speech for numbers {start} to {end}...")
    print(f"Speaker: {speaker_description}\n")
    
    for i in range(start, end + 1):
        text = str(i)
        print(f"Generating: {i}", end="\r")
        
        # Tokenize
        input_ids = tokenizer(speaker_description, return_tensors="pt").input_ids.to(device)
        prompt_input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        
        # Generate
        with torch.no_grad():
            generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        
        audio = generation.cpu().numpy().squeeze()
        audio_samples.append((i, audio))
    
    print(f"\n✓ Generated {len(audio_samples)} speech samples")
    
    return audio_samples, sample_rate

def save_individual_samples(audio_samples, sample_rate, output_dir="counting_samples"):
    """
    Save each number as individual WAV file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving individual samples to {output_dir}/...")
    for number, audio in audio_samples:
        filename = f"{output_dir}/number_{number:02d}.wav"
        # Normalize audio
        audio_normalized = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio
        wavfile.write(filename, sample_rate, (audio_normalized * 32767).astype(np.int16))
    
    print(f"✓ Saved {len(audio_samples)} files")

def save_combined_sample(audio_samples, sample_rate, output_path="counting_combined.wav", silence_duration=0.3):
    """
    Combine all numbers into one audio file with pauses
    """
    silence_samples = int(sample_rate * silence_duration)
    silence = np.zeros(silence_samples)
    
    combined = []
    for i, (number, audio) in enumerate(audio_samples):
        # Normalize
        audio_normalized = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio
        combined.append(audio_normalized)
        
        # Add silence between numbers (but not after last one)
        if i < len(audio_samples) - 1:
            combined.append(silence)
    
    # Concatenate all
    final_audio = np.concatenate(combined)
    
    # Save
    wavfile.write(output_path, sample_rate, (final_audio * 32767).astype(np.int16))
    print(f"\n✓ Saved combined audio: {output_path}")
    print(f"  Duration: {len(final_audio)/sample_rate:.2f}s")

def export_for_cpp(audio_samples, sample_rate):
    """
    Export data for C++ inference
    """
    print("\n--- Exporting for C++ ---")
    
    # Save first few samples as numpy arrays
    for i, (number, audio) in enumerate(audio_samples[:5]):
        np.save(f'parler_sample_{number}.npy', audio)
    
    # Save sample rate and metadata
    with open('parler_metadata.txt', 'w') as f:
        f.write(f"sample_rate: {sample_rate}\n")
        f.write(f"num_samples: {len(audio_samples)}\n")
        f.write(f"model: parler-tts/parler-tts-tiny-v1\n")
    
    print(f"✓ Saved sample data for C++ (first 5 numbers)")
    print(f"✓ Saved parler_metadata.txt")

if __name__ == "__main__":
    print("=== Parler TTS Counting Generator ===\n")
    
    # Generate counting from 1 to 50
    speaker = "A clear, natural female voice"
    audio_samples, sample_rate = generate_counting_tts(1, 50, speaker)
    
    # Save individual files
    save_individual_samples(audio_samples, sample_rate)
    
    # Save combined file
    save_combined_sample(audio_samples, sample_rate)
    
    # Export for C++
    export_for_cpp(audio_samples, sample_rate)
    
    print("\n=== Generation Complete ===")
    print("\nGenerated files:")
    print("  - counting_samples/number_01.wav ... number_50.wav")
    print("  - counting_combined.wav")
    print("  - parler_sample_*.npy (for C++ inference)")
    print("  - parler_metadata.txt")
    
    # Play a sample
    print("\nPlaying sample: '1, 2, 3, 4, 5'...")
    import subprocess
    subprocess.run(["afplay", "counting_combined.wav"], timeout=5)
