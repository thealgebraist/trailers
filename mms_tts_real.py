#!/usr/bin/env python3
"""
Generate speech using Facebook MMS TTS (English)
Real neural TTS - no fallback code
"""
import torch
import torchaudio
import numpy as np
from scipy.io import wavfile

def generate_mms_tts(text, output_path="mms_output.wav"):
    """
    Generate speech using facebook/mms-tts-eng
    """
    print(f"Loading facebook/mms-tts-eng model...")
    
    from transformers import VitsModel, AutoTokenizer
    
    model = VitsModel.from_pretrained("facebook/mms-tts-eng")
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
    
    print(f"\nGenerating speech for: '{text}'")
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt")
    
    # Generate
    with torch.no_grad():
        output = model(**inputs).waveform
    
    # Convert to numpy
    audio = output.squeeze().cpu().numpy()
    sample_rate = model.config.sampling_rate
    
    # Save
    audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    wavfile.write(output_path, sample_rate, audio_int16)
    
    print(f"\n✓ Saved: {output_path}")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Duration: {len(audio)/sample_rate:.2f}s")
    print(f"  Samples: {len(audio)}")
    
    return audio, sample_rate

def generate_counting(start=1, end=10):
    """Generate counting samples"""
    import os
    os.makedirs("mms_counting", exist_ok=True)
    
    from transformers import VitsModel, AutoTokenizer
    
    print(f"\nLoading model...")
    model = VitsModel.from_pretrained("facebook/mms-tts-eng")
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
    sample_rate = model.config.sampling_rate
    
    print(f"Generating numbers {start} to {end}...")
    
    all_audio = []
    silence = np.zeros(int(sample_rate * 0.3))  # 300ms silence
    
    for i in range(start, end + 1):
        text = str(i)
        print(f"  {i}", end="\r")
        
        inputs = tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            output = model(**inputs).waveform
        
        audio = output.squeeze().cpu().numpy()
        
        # Save individual
        audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        wavfile.write(f"mms_counting/number_{i:02d}.wav", sample_rate, audio_int16)
        
        # Add to combined
        all_audio.append(audio)
        if i < end:
            all_audio.append(silence)
    
    # Save combined
    combined = np.concatenate(all_audio)
    combined_int16 = (np.clip(combined, -1.0, 1.0) * 32767).astype(np.int16)
    wavfile.write("mms_counting_combined.wav", sample_rate, combined_int16)
    
    print(f"\n✓ Generated {end-start+1} number samples")
    print(f"✓ Saved combined: mms_counting_combined.wav ({len(combined)/sample_rate:.2f}s)")
    
    return combined, sample_rate

if __name__ == "__main__":
    print("=== Facebook MMS TTS (English) ===\n")
    
    # Test sentence
    text = "Hello world. This is a test of the MMS text to speech system."
    generate_mms_tts(text, "mms_test_output.wav")
    
    # Generate counting 1-50
    print("\n" + "="*50)
    combined_audio, sr = generate_counting(1, 50)
    
    print("\n=== Complete ===")
    print("\nGenerated files:")
    print("  - mms_test_output.wav")
    print("  - mms_counting/number_01.wav ... number_50.wav")
    print("  - mms_counting_combined.wav")
    
    # Play test
    print("\nPlaying test output...")
    import subprocess
    subprocess.run(["afplay", "mms_test_output.wav"])
