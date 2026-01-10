import torch
from diffusers import StableDiffusionPipeline
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import os

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

def test_tiny_sd():
    path = "/Users/anders/.cache/huggingface/hub/segmind_tiny-sd_model"
    print(f"Testing Tiny-SD from {path}...")
    try:
        pipe = StableDiffusionPipeline.from_pretrained(path, torch_dtype=torch.float16, local_files_only=True).to(DEVICE)
        print("Tiny-SD loaded successfully!")
        return True
    except Exception as e:
        print(f"Tiny-SD load failed: {e}")
        return False

def test_musicgen():
    print("Testing MusicGen...")
    try:
        model_id = "facebook/musicgen-small"
        processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
        model = MusicgenForConditionalGeneration.from_pretrained(model_id, local_files_only=True).to(DEVICE)
        print("MusicGen loaded successfully!")
        
        inputs = processor(text=["1950s Soviet music"], padding=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            audio_values = model.generate(**inputs, max_new_tokens=10)
        print("MusicGen generation successful!")
        return True
    except Exception as e:
        print(f"MusicGen failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_tiny_sd()
    test_musicgen()
