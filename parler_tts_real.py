#!/usr/bin/env python3
"""
Generate counting speech samples using Parler TTS
Real implementation - no fallback
"""
import torch
import soundfile as sf
import numpy as np
import os

def install_parler_tts():
    """Install parler-tts with Python 3.14 compatibility"""
    print("Installing parler-tts with Python 3.14 forward compatibility...")
    # Use forward compatibility flag for Python 3.14
    os.system("PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 pip3 install git+https://github.com/huggingface/parler-tts.git --break-system-packages -q")
    os.system("pip3 install soundfile --break-system-packages -q")

def generate_counting_with_parler(start=1, end=50):
    """Use real Parler TTS model"""
    try:
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer
    except ImportError:
        install_parler_tts()
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer
    
    print("Loading Parler TTS Tiny v1...")
    model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-tiny-v1")
    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-tiny-v1")
    
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")
    
    description = "A clear female voice speaks the numbers"
    
    audio_samples = []
    sample_rate = model.config.sampling_rate
    
    print(f"\nGenerating speech for numbers {start} to {end}...")
    
    for num in range(start, end + 1):
        text = str(num)
        print(f"Generating: {num}", end="\r")
        
        input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
        prompt_input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        
        with torch.no_grad():
            generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        
        audio = generation.cpu().numpy().squeeze()
        audio_samples.append((num, audio))
    
    print(f"\n✓ Generated {len(audio_samples)} speech samples")
    return audio_samples, sample_rate

def save_samples(audio_samples, sample_rate):
    """Save individual and combined samples"""
    os.makedirs("counting_samples", exist_ok=True)
    
    # Save individual
    print(f"\nSaving individual samples...")
    for num, audio in audio_samples:
        sf.write(f"counting_samples/number_{num:02d}.wav", audio, sample_rate)
    print(f"✓ Saved {len(audio_samples)} files")
    
    # Save combined
    silence = np.zeros(int(sample_rate * 0.3))
    combined = []
    for i, (num, audio) in enumerate(audio_samples):
        combined.append(audio)
        if i < len(audio_samples) - 1:
            combined.append(silence)
    
    final_audio = np.concatenate(combined)
    sf.write("counting_combined.wav", final_audio, sample_rate)
    print(f"✓ Saved counting_combined.wav ({len(final_audio)/sample_rate:.2f}s)")

if __name__ == "__main__":
    print("=== Real Parler TTS Counting Generator ===\n")
    
    audio_samples, sample_rate = generate_counting_with_parler(1, 50)
    save_samples(audio_samples, sample_rate)
    
    print("\n=== Complete ===")
    print("Generated: counting_samples/number_01.wav ... number_50.wav")
    print("Generated: counting_combined.wav")
    
    # Play sample
    os.system("afplay counting_combined.wav &")
