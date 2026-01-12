import os
import psutil
import torch
import gc
from transformers import AutoProcessor, MusicgenForConditionalGeneration, MusicgenConfig

def print_memory(step):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"[{step}] RSS Memory: {mem_info.rss / 1024**3:.2f} GB")

def profile():
    print_memory("Start")
    
    device = "cpu"
    print(f"Using device: {device}")
    
    # Load Model
    print("Loading model...")
    model_id = "facebook/musicgen-small"
    MusicgenForConditionalGeneration.config_class = MusicgenConfig
    processor = AutoProcessor.from_pretrained(model_id)
    model = MusicgenForConditionalGeneration.from_pretrained(model_id, dtype=torch.float32)
    model.to(device)
    
    print_memory("Model Loaded")
    
    # Generate
    print("Generating...")
    inputs = processor(
        text=["Test prompt for memory profiling"],
        padding=True,
        return_tensors="pt",
    ).to(device)
    
    with torch.no_grad():
        # Generate short clip
        audio_values = model.generate(**inputs, max_new_tokens=256, do_sample=True)
        
    print_memory("After Generation")
    
    del audio_values
    del inputs
    gc.collect()
    
    print_memory("After GC")

if __name__ == "__main__":
    profile()
