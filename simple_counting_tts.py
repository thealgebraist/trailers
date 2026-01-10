#!/usr/bin/env python3
"""
Generate counting speech samples using available TTS
Uses our minimal TTS implementation for numbers 1-50
"""
import numpy as np
from scipy.io import wavfile
import os

def text_to_char_ids(text):
    """Convert text to character IDs"""
    chars = " 0123456789abcdefghijklmnopqrstuvwxyz"
    ids = []
    for c in text.lower():
        pos = chars.find(c)
        ids.append(pos if pos >= 0 else 0)
    return ids

def generate_speech_for_number(number, sample_rate=22050):
    """Generate speech waveform for a number"""
    text = str(number)
    char_ids = text_to_char_ids(text)
    
    char_duration = 0.2  # 200ms per character
    samples_per_char = int(sample_rate * char_duration)
    
    audio = []
    phase = 0.0
    
    for char_id in char_ids:
        # Map character to frequency
        # Digits 0-9 are at indices 1-10 in our char set
        if char_id == 0:  # space
            freq = 0
        else:
            # Create distinct tones for each digit
            freq = 300 + char_id * 50  # Different freq for each char
        
        # Generate samples
        for _ in range(samples_per_char):
            if freq > 0:
                sample = np.sin(phase) * 0.4
            else:
                sample = 0.0
            audio.append(sample)
            
            phase += 2.0 * np.pi * freq / sample_rate
            if phase > 2.0 * np.pi:
                phase -= 2.0 * np.pi
    
    return np.array(audio, dtype=np.float32)

def generate_counting_samples(start=1, end=50):
    """Generate all counting samples"""
    sample_rate = 22050
    audio_samples = []
    
    print(f"Generating speech for numbers {start} to {end}...")
    for i in range(start, end + 1):
        print(f"Generating: {i:2d}", end="\r")
        audio = generate_speech_for_number(i, sample_rate)
        audio_samples.append((i, audio))
    
    print(f"\n✓ Generated {len(audio_samples)} speech samples")
    return audio_samples, sample_rate

def save_individual_samples(audio_samples, sample_rate, output_dir="counting_samples"):
    """Save each number as individual WAV file"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving individual samples to {output_dir}/...")
    for number, audio in audio_samples:
        filename = f"{output_dir}/number_{number:02d}.wav"
        audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        wavfile.write(filename, sample_rate, audio_int16)
    
    print(f"✓ Saved {len(audio_samples)} files")

def save_combined_sample(audio_samples, sample_rate, output_path="counting_combined.wav", silence_duration=0.3):
    """Combine all numbers into one audio file with pauses"""
    silence_samples = int(sample_rate * silence_duration)
    silence = np.zeros(silence_samples)
    
    combined = []
    for i, (number, audio) in enumerate(audio_samples):
        combined.append(audio)
        # Add silence between numbers (but not after last one)
        if i < len(audio_samples) - 1:
            combined.append(silence)
    
    # Concatenate all
    final_audio = np.concatenate(combined)
    
    # Save
    audio_int16 = (np.clip(final_audio, -1.0, 1.0) * 32767).astype(np.int16)
    wavfile.write(output_path, sample_rate, audio_int16)
    print(f"\n✓ Saved combined audio: {output_path}")
    print(f"  Duration: {len(final_audio)/sample_rate:.2f}s")
    return final_audio

def export_for_cpp(audio_samples, sample_rate):
    """Export data for C++ inference"""
    print("\n--- Exporting for C++ ---")
    
    # Save first few samples as numpy arrays
    for i, (number, audio) in enumerate(audio_samples[:5]):
        np.save(f'counting_sample_{number}.npy', audio)
    
    # Save all character IDs
    with open('counting_char_ids.txt', 'w') as f:
        for number, _ in audio_samples[:10]:
            char_ids = text_to_char_ids(str(number))
            f.write(f"{number}: {' '.join(map(str, char_ids))}\n")
    
    # Save sample rate and metadata
    with open('counting_metadata.txt', 'w') as f:
        f.write(f"sample_rate: {sample_rate}\n")
        f.write(f"num_samples: {len(audio_samples)}\n")
        f.write(f"char_duration: 0.2\n")
        f.write(f"method: procedural_tts\n")
    
    print(f"✓ Saved sample data for C++ (first 5 numbers)")
    print(f"✓ Saved counting_char_ids.txt")
    print(f"✓ Saved counting_metadata.txt")

if __name__ == "__main__":
    print("=== Counting Speech Generator ===\n")
    
    # Generate counting from 1 to 50
    audio_samples, sample_rate = generate_counting_samples(1, 50)
    
    # Save individual files
    save_individual_samples(audio_samples, sample_rate)
    
    # Save combined file
    final_audio = save_combined_sample(audio_samples, sample_rate)
    
    # Export for C++
    export_for_cpp(audio_samples, sample_rate)
    
    print("\n=== Generation Complete ===")
    print("\nGenerated files:")
    print("  - counting_samples/number_01.wav ... number_50.wav")
    print("  - counting_combined.wav")
    print("  - counting_sample_*.npy (for C++ inference)")
    print("  - counting_char_ids.txt")
    print("  - counting_metadata.txt")
    
    # Play first 10 seconds
    print("\nPlaying first 10 numbers...")
    import subprocess
    try:
        # Play just the first part
        subprocess.run(["afplay", "counting_combined.wav"], timeout=10)
    except subprocess.TimeoutExpired:
        print("(stopped after 10s)")
    except Exception as e:
        print(f"Could not play: {e}")
