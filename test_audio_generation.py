#!/usr/bin/env python3
"""Compare Python vs C++ audio generation using ONNX tokens"""
import numpy as np
import wave
import struct

def save_wav16(path, samples, sample_rate=22050):
    """Save 16-bit PCM WAV file"""
    with wave.open(path, 'w') as f:
        f.setnchannels(1)  # mono
        f.setsampwidth(2)  # 16-bit
        f.setframerate(sample_rate)
        
        # Convert float samples to int16
        samples_clamped = np.clip(samples, -1.0, 1.0)
        samples_int16 = (samples_clamped * 32767).astype(np.int16)
        f.writeframes(samples_int16.tobytes())

def generate_audio_python(tokens, sample_rate=22050, duration=2.0):
    """Generate audio using same algorithm as C++ version"""
    note_duration = 0.2  # 200ms per note
    samples_per_note = int(sample_rate * note_duration)
    total_samples = int(sample_rate * duration)
    
    audio = np.zeros(total_samples, dtype=np.float32)
    phase = 0.0
    
    for i in range(total_samples):
        # Change frequency every note_duration seconds based on tokens
        note_index = i // samples_per_note
        freq = 220.0 + (tokens[note_index % len(tokens)] % 12) * 20.0
        
        # Generate sample with continuous phase
        audio[i] = np.sin(phase) * 0.5
        
        # Advance phase
        phase += 2.0 * np.pi * freq / sample_rate
        if phase > 2.0 * np.pi:
            phase -= 2.0 * np.pi
    
    return audio

def generate_audio_vectorized(tokens, sample_rate=22050, duration=2.0):
    """Generate audio using vectorized NumPy (faster)"""
    note_duration = 0.2  # 200ms per note
    samples_per_note = int(sample_rate * note_duration)
    total_samples = int(sample_rate * duration)
    
    # Create time array
    t = np.arange(total_samples) / sample_rate
    
    # Create note index array
    note_indices = np.arange(total_samples) // samples_per_note
    
    # Map tokens to frequencies
    freq_array = 220.0 + (tokens[note_indices % len(tokens)] % 12) * 20.0
    
    # Generate audio with proper phase accumulation
    phase = np.cumsum(2.0 * np.pi * freq_array / sample_rate)
    audio = np.sin(phase) * 0.5
    
    return audio

if __name__ == "__main__":
    print("=== Python Audio Generation Test ===\n")
    
    # Load tokens from text file
    with open('flux_token_ids.txt', 'r') as f:
        tokens = np.array([int(x) for x in f.read().split()], dtype=np.int64)
    
    print(f"Loaded {len(tokens)} tokens")
    print(f"First 10 tokens: {tokens[:10].tolist()}")
    
    # Generate audio using loop method (matches C++)
    print("\n--- Generating audio (loop method, matches C++) ---")
    audio_loop = generate_audio_python(tokens)
    save_wav16('python_audio_loop.wav', audio_loop)
    print(f"✓ Saved python_audio_loop.wav ({len(audio_loop)} samples)")
    
    # Generate audio using vectorized method (faster)
    print("\n--- Generating audio (vectorized NumPy) ---")
    audio_vec = generate_audio_vectorized(tokens)
    save_wav16('python_audio_vectorized.wav', audio_vec)
    print(f"✓ Saved python_audio_vectorized.wav ({len(audio_vec)} samples)")
    
    # Compare with C++ output
    print("\n--- Comparing outputs ---")
    try:
        # Load C++ generated audio
        with wave.open('inference_audio.wav', 'r') as f:
            cpp_frames = f.readframes(f.getnframes())
            cpp_audio = np.frombuffer(cpp_frames, dtype=np.int16).astype(np.float32) / 32767.0
        
        print(f"C++ audio:    {len(cpp_audio)} samples")
        print(f"Python loop:  {len(audio_loop)} samples")
        print(f"Python vec:   {len(audio_vec)} samples")
        
        # Compare first few samples
        min_len = min(len(cpp_audio), len(audio_loop))
        audio_loop_int16 = (np.clip(audio_loop[:min_len], -1.0, 1.0) * 32767).astype(np.int16).astype(np.float32) / 32767.0
        
        diff = np.abs(cpp_audio[:min_len] - audio_loop_int16)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        print(f"\nDifference C++ vs Python loop:")
        print(f"  Max diff:  {max_diff:.6f}")
        print(f"  Mean diff: {mean_diff:.6f}")
        
        if max_diff < 0.01:
            print("  ✓ Outputs match closely!")
        else:
            print("  ⚠ Some differences detected")
            print(f"\nFirst 10 samples C++:    {cpp_audio[:10]}")
            print(f"First 10 samples Python: {audio_loop_int16[:10]}")
        
    except FileNotFoundError:
        print("C++ audio file not found, skipping comparison")
    
    print("\n=== Test Complete ===")
