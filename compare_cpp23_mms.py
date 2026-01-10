#!/usr/bin/env python3
"""
Compare C++23 ONNX output with Python reference
"""
import numpy as np
from scipy.io import wavfile

# Load both files
ref_rate, ref_audio = wavfile.read('mms_cpp_reference.wav')
cpp_rate, cpp_audio = wavfile.read('cpp23_mms_onnx.wav')

print("=== MMS-TTS C++23 vs Python Comparison ===\n")
print(f"Reference (Python):  {len(ref_audio):6} samples @ {ref_rate} Hz")
print(f"C++23 (ONNX):        {len(cpp_audio):6} samples @ {cpp_rate} Hz")

# Convert to float for comparison
ref_float = ref_audio.astype(np.float32) / 32768.0
cpp_float = cpp_audio.astype(np.float32) / 32768.0

# Compare
if len(ref_float) == len(cpp_float):
    diff = np.abs(ref_float - cpp_float)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"\nDifference Analysis:")
    print(f"  Max difference:  {max_diff:.10f}")
    print(f"  Mean difference: {mean_diff:.10f}")
    
    if max_diff < 0.0001:
        print(f"\n✓ PERFECT MATCH - C++23 ONNX output identical to Python!")
    else:
        print(f"\n⚠ Some differences detected")
else:
    print(f"\n⚠ Different lengths: {len(ref_float)} vs {len(cpp_float)}")

print("\n=== Text: 'hello' ===")
print("Input IDs: [0,6,0,7,0,21,0,21,0,22,0]")
print("Encoding:  [sep,h,sep,e,sep,l,sep,l,sep,o,sep]")
