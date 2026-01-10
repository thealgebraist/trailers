#!/usr/bin/env python3
"""Visualize and compare C++ vs Python audio waveforms"""
import numpy as np
import wave
import matplotlib.pyplot as plt

def load_wav(path):
    """Load WAV file and return as float array"""
    with wave.open(path, 'r') as f:
        frames = f.readframes(f.getnframes())
        sample_rate = f.getframerate()
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
    return audio, sample_rate

# Load all three versions
cpp_audio, sr_cpp = load_wav('inference_audio.wav')
py_loop, sr_py_loop = load_wav('python_audio_loop.wav')
py_vec, sr_py_vec = load_wav('python_audio_vectorized.wav')

print(f"Sample rates: C++={sr_cpp}, Py-loop={sr_py_loop}, Py-vec={sr_py_vec}")

# Create visualization
fig, axes = plt.subplots(4, 1, figsize=(12, 10))

# Plot first 0.5 seconds
samples_to_plot = int(0.5 * sr_cpp)
time = np.arange(samples_to_plot) / sr_cpp

axes[0].plot(time, cpp_audio[:samples_to_plot], label='C++ (inference_audio.wav)', alpha=0.8)
axes[0].set_title('C++ Generated Audio (First 0.5s)')
axes[0].set_ylabel('Amplitude')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].plot(time, py_loop[:samples_to_plot], label='Python Loop', alpha=0.8, color='orange')
axes[1].set_title('Python Loop Generated Audio (First 0.5s)')
axes[1].set_ylabel('Amplitude')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

axes[2].plot(time, py_vec[:samples_to_plot], label='Python Vectorized', alpha=0.8, color='green')
axes[2].set_title('Python Vectorized Audio (First 0.5s)')
axes[2].set_ylabel('Amplitude')
axes[2].grid(True, alpha=0.3)
axes[2].legend()

# Plot difference
diff = cpp_audio[:samples_to_plot] - py_loop[:samples_to_plot]
axes[3].plot(time, diff, label='C++ - Python Loop', alpha=0.8, color='red')
axes[3].set_title('Difference (C++ - Python)')
axes[3].set_xlabel('Time (seconds)')
axes[3].set_ylabel('Amplitude')
axes[3].grid(True, alpha=0.3)
axes[3].legend()

plt.tight_layout()
plt.savefig('audio_comparison.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved audio_comparison.png")

# Statistical comparison
print("\n=== Statistical Analysis ===")
print(f"C++ Audio:        min={np.min(cpp_audio):.4f}, max={np.max(cpp_audio):.4f}, mean={np.mean(cpp_audio):.6f}")
print(f"Python Loop:      min={np.min(py_loop):.4f}, max={np.max(py_loop):.4f}, mean={np.mean(py_loop):.6f}")
print(f"Python Vectorized: min={np.min(py_vec):.4f}, max={np.max(py_vec):.4f}, mean={np.mean(py_vec):.6f}")

print(f"\nDifference C++ vs Python Loop:")
print(f"  Max absolute diff: {np.max(np.abs(cpp_audio - py_loop)):.6f}")
print(f"  Mean absolute diff: {np.mean(np.abs(cpp_audio - py_loop)):.6f}")
print(f"  RMS diff: {np.sqrt(np.mean((cpp_audio - py_loop)**2)):.6f}")

print(f"\nDifference Python Loop vs Vectorized:")
print(f"  Max absolute diff: {np.max(np.abs(py_loop - py_vec)):.6f}")
print(f"  Mean absolute diff: {np.mean(np.abs(py_loop - py_vec)):.6f}")

# Frequency analysis on a small segment
from scipy import signal
print("\n=== Frequency Content (First 0.2s) ===")
segment_len = int(0.2 * sr_cpp)
freqs_cpp, psd_cpp = signal.welch(cpp_audio[:segment_len], sr_cpp, nperseg=1024)
freqs_py, psd_py = signal.welch(py_loop[:segment_len], sr_py_loop, nperseg=1024)

# Find dominant frequencies
cpp_peak_idx = np.argmax(psd_cpp)
py_peak_idx = np.argmax(psd_py)
print(f"C++ dominant frequency: {freqs_cpp[cpp_peak_idx]:.2f} Hz")
print(f"Python dominant frequency: {freqs_py[py_peak_idx]:.2f} Hz")

print("\n✓ Analysis complete!")
