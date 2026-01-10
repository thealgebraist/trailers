#!/usr/bin/env python3
"""
Generate speech samples using facebook/mms-tts-eng
Based on working code pattern - no fallback
"""
import torch
import numpy as np
import scipy.io.wavfile
import os
import re
from transformers import VitsModel, AutoTokenizer
from huggingface_hub import snapshot_download

# Device selection
if torch.backends.mps.is_available():
    DEVICE = "mps"
    print("Using MPS (Apple Silicon)")
elif torch.cuda.is_available():
    DEVICE = "cuda"
    print("Using CUDA")
else:
    DEVICE = "cpu"
    print("Using CPU")

def generate_voice_sample(text, output_file, model, tokenizer):
    """Generate a single voice sample"""
    # Clean text
    txt = re.sub(r"\s+", " ", text).strip()
    
    print(f'Generating: "{txt[:60]}..."')
    
    # Tokenize and generate
    inputs = tokenizer(txt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = model(**inputs).waveform
    
    # Save
    audio = output.cpu().numpy().flatten()
    scipy.io.wavfile.write(output_file, model.config.sampling_rate, audio)
    
    return audio, model.config.sampling_rate

def generate_counting_samples(start=1, end=50):
    """Generate counting from 1 to 50"""
    print(f'\n--- Generating {end-start+1} Voice Lines (MMS-TTS) ---\n')
    
    # Load model
    repo_id = "facebook/mms-tts-eng"
    print(f"Loading {repo_id}...")
    model_id = snapshot_download(repo_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = VitsModel.from_pretrained(model_id).to(DEVICE)
    
    os.makedirs("mms_counting", exist_ok=True)
    
    all_audio = []
    sample_rate = model.config.sampling_rate
    silence = np.zeros(int(sample_rate * 0.3), dtype=np.float32)  # 300ms silence
    
    # Generate each number
    for i in range(start, end + 1):
        txt = str(i)
        out_file = f"mms_counting/number_{i:02d}.wav"
        
        if os.path.exists(out_file):
            print(f"Skipping {i:02d} (already exists)")
            # Load existing
            sr, audio = scipy.io.wavfile.read(out_file)
            audio = audio.astype(np.float32) / 32767.0 if audio.dtype == np.int16 else audio
        else:
            print(f'Generating voice {i:02d}: "{txt}"')
            audio, sample_rate = generate_voice_sample(txt, out_file, model, tokenizer)
        
        all_audio.append(audio)
        if i < end:
            all_audio.append(silence)
    
    # Save combined
    combined = np.concatenate(all_audio)
    scipy.io.wavfile.write("mms_counting_combined.wav", sample_rate, combined)
    
    print(f'\n✓ Generated {end-start+1} samples')
    print(f'✓ Saved combined: mms_counting_combined.wav ({len(combined)/sample_rate:.2f}s)')
    
    # Cleanup
    del model, tokenizer
    torch.mps.empty_cache() if DEVICE == "mps" else None
    
    return combined, sample_rate

def generate_custom_prompts():
    """Generate custom voice prompts"""
    print('\n--- Generating Custom Voice Lines (MMS-TTS) ---\n')
    
    prompts = [
        "Hello world, this is a test of the MMS text to speech system.",
        "The quick brown fox jumps over the lazy dog.",
        "In a world that was once crisp and dry, a strange phenomenon has begun.",
        "Artificial intelligence is transforming how we create and consume media.",
        "This demonstration shows real neural text to speech in action.",
    ]
    
    # Load model
    repo_id = "facebook/mms-tts-eng"
    print(f"Loading {repo_id}...")
    model_id = snapshot_download(repo_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = VitsModel.from_pretrained(model_id).to(DEVICE)
    
    os.makedirs("mms_samples", exist_ok=True)
    
    for i, txt in enumerate(prompts):
        out_file = f"mms_samples/sample_{i:02d}.wav"
        
        if os.path.exists(out_file):
            print(f"Skipping sample {i:02d} (already exists)")
            continue
        
        generate_voice_sample(txt, out_file, model, tokenizer)
    
    print(f'\n✓ Generated {len(prompts)} custom samples')
    
    # Cleanup
    del model, tokenizer
    torch.mps.empty_cache() if DEVICE == "mps" else None

if __name__ == "__main__":
    print("=== Facebook MMS-TTS Speech Generation ===")
    print(f"Device: {DEVICE}\n")
    
    # Generate custom samples
    generate_custom_prompts()
    
    # Generate counting 1-50
    print("\n" + "="*60)
    combined_audio, sr = generate_counting_samples(1, 50)
    
    print("\n" + "="*60)
    print("=== Complete ===")
    print("\nGenerated files:")
    print("  - mms_samples/sample_00.wav ... sample_04.wav")
    print("  - mms_counting/number_01.wav ... number_50.wav")
    print("  - mms_counting_combined.wav")
    
    # Play combined counting
    print("\nPlaying combined counting (first 10s)...")
    import subprocess
    try:
        subprocess.run(["afplay", "mms_counting_combined.wav"], timeout=10)
    except subprocess.TimeoutExpired:
        print("(stopped after 10s)")
    except Exception as e:
        print(f"Playback: {e}")
