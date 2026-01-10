import torch
from transformers import AutoProcessor, BarkModel
import sys

# Increase recursion depth just in case, though it shouldn't be needed for normal operation
sys.setrecursionlimit(2000)

DEVICE = "cpu" # Test on CPU first to rule out MPS issues
print(f"Testing Bark on {DEVICE}...")

try:
    model_id = "suno/bark"
    # Using specific revisions or tags sometimes helps if the main one is broken
    processor = AutoProcessor.from_pretrained(model_id)
    print("Processor loaded.")
    model = BarkModel.from_pretrained(model_id, torch_dtype=torch.float32).to(DEVICE)
    print("Model loaded.")
    
    inputs = processor("hello", return_tensors="pt").to(DEVICE)
    print("Inputs prepared.")
    with torch.no_grad():
        output = model.generate(**inputs, min_eos_p=0.05)
    print("Generation successful!")
except Exception as e:
    print(f"Failed: {e}")
    import traceback
    traceback.print_exc()
