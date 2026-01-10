#!/usr/bin/env python3
"""
Export MMS-TTS to TorchScript using scripting (not tracing) for LibTorch
"""
import torch
import numpy as np
from transformers import VitsModel, AutoTokenizer
from huggingface_hub import snapshot_download
import scipy.io.wavfile

print("=== MMS-TTS TorchScript Export (Scripting Mode) ===\n")

# Load model
print("Loading facebook/mms-tts-eng...")
repo_id = "facebook/mms-tts-eng"
model_id = snapshot_download(repo_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = VitsModel.from_pretrained(model_id)
model.eval()

# Test
test_text = "world"
print(f"Test text: '{test_text}'")

inputs = tokenizer(test_text, return_tensors="pt")
input_ids = inputs["input_ids"]

print(f"Input IDs: {input_ids.tolist()}")

# Generate reference
with torch.no_grad():
    output = model(**inputs).waveform

audio = output.squeeze().cpu().numpy()
scipy.io.wavfile.write("mms_world_reference.wav", model.config.sampling_rate, audio)
print(f"✓ Reference: mms_world_reference.wav ({len(audio)} samples)")

# Save input
np.savetxt('mms_world_input_ids.txt', input_ids.numpy().flatten(), fmt='%d')
print(f"✓ Input IDs: mms_world_input_ids.txt")

# Try scripting (instead of tracing)
print("\nAttempting torch.jit.script export...")
try:
    # Script the model
    scripted = torch.jit.script(model)
    scripted.save("mms_tts_scripted.pt")
    print("✓ Scripted: mms_tts_scripted.pt")
    
    # Verify
    loaded = torch.jit.load("mms_tts_scripted.pt")
    loaded.eval()
    
    with torch.no_grad():
        test_out = loaded(input_ids).waveform
    
    test_audio = test_out.squeeze().cpu().numpy()
    scipy.io.wavfile.write("mms_world_scripted.wav", model.config.sampling_rate, test_audio)
    print(f"✓ Verified: mms_world_scripted.wav ({len(test_audio)} samples)")
    
except Exception as e:
    print(f"✗ Scripting failed: {e}")
    print("\nVITS model is too complex for automatic TorchScript conversion.")
    print("Would require manual model refactoring for LibTorch.")

print("\n=== Export Complete ===")
