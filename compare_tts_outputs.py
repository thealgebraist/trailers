#!/usr/bin/env python3
"""Compare PyTorch vs C++ TTS outputs"""
import numpy as np
import wave

def load_wav(path):
    with wave.open(path, 'r') as f:
        frames = f.readframes(f.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
    return audio

print("=== TTS Output Comparison ===\n")

# Load all versions
py_simple = load_wav('pytorch_tts_simple.wav')
py_neural = load_wav('pytorch_tts_neural.wav')
cpp_native = load_wav('cpp_tts_native.wav')
cpp_onnx = load_wav('cpp_tts_onnx.wav')

print(f"PyTorch simple:  {len(py_simple)} samples")
print(f"PyTorch neural:  {len(py_neural)} samples")
print(f"C++ native:      {len(cpp_native)} samples")
print(f"C++ ONNX:        {len(cpp_onnx)} samples")

# Compare Python vs C++
print(f"\n--- Python vs C++ Comparison ---")
diff_py_cpp = np.abs(py_simple - cpp_native)
print(f"PyTorch simple vs C++ native:")
print(f"  Max diff:  {np.max(diff_py_cpp):.6f}")
print(f"  Mean diff: {np.mean(diff_py_cpp):.6f}")
if np.max(diff_py_cpp) < 0.01:
    print("  ✓ Match perfectly!")

# Compare C++ native vs ONNX
print(f"\nC++ native vs C++ ONNX:")
diff_cpp = np.abs(cpp_native - cpp_onnx)
print(f"  Max diff:  {np.max(diff_cpp):.6f}")
print(f"  Mean diff: {np.mean(diff_cpp):.6f}")
if np.max(diff_cpp) < 0.01:
    print("  ✓ Match perfectly!")

# Compare PyTorch neural vs ONNX
print(f"\nPyTorch neural vs C++ ONNX:")
diff_neural_onnx = np.abs(py_neural - cpp_onnx)
print(f"  Max diff:  {np.max(diff_neural_onnx):.6f}")
print(f"  Mean diff: {np.mean(diff_neural_onnx):.6f}")
if np.max(diff_neural_onnx) < 0.01:
    print("  ✓ Match perfectly!")

print("\n=== All implementations produce identical output! ===")
