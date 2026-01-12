import time
import torch
from diffusers import FluxFillPipeline
from optimum.quanto import freeze, qfloat8, quantize
import logging
import threading
import psutil
import subprocess
import csv
import datetime
import os
import random
from PIL import Image

# Suppress logging
logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

def get_random_slop_prompt():
    nouns = [
        "artist", "baker", "cat", "dog", "elephant", "farmer", "ghost", "hunter",
        "insect", "joker", "king", "lion", "monster", "nurse", "ocean", "pilot",
        "queen", "robot", "soldier", "teacher", "universe", "vampire", "wizard", "xenomorph",
        "yacht", "zombie", "apple", "bridge", "cloud", "dream", "engine", "forest",
        "garden", "house", "island", "jungle", "kite", "lake", "mountain", "night",
        "orange", "planet", "quilt", "river", "star", "tree", "umbrella", "valley",
        "whale", "xylophone", "yeti", "zebra", "angel", "demon", "dragon", "fairy",
        "goblin", "hero", "imp", "jewel", "knight", "lemma", "machine", "ninja"
    ]
    adjectives = [
        "angry", "brave", "calm", "dark", "eager", "fast", "giant", "happy",
        "icy", "jolly", "kind", "lazy", "mad", "nervous", "old", "proud",
        "quiet", "red", "sad", "tall", "ugly", "vast", "wild", "xenophobic",
        "young", "zany", "bright", "cold", "deep", "empty", "fierce", "green",
        "hot", "infinite", "jagged", "keen", "loud", "mysterious", "noisy", "odd",
        "pale", "quick", "rough", "sharp", "tiny", "unique", "vivid", "warm",
        "yellow", "zealous", "ancient", "blind", "cruel", "dead", "evil", "fragile",
        "grim", "hollow", "iron", "jade", "lost", "metal", "neon", "obsidian"
    ]
    verbs = [
        "attacks", "bites", "calls", "dreams", "eats", "fights", "grabs", "hunts",
        "ignores", "jumps", "kicks", "loves", "makes", "needs", "opens", "pulls",
        "questions", "runs", "sings", "takes", "uses", "visits", "walks", "x-rays",
        "yells", "zaps", "admires", "breaks", "builds", "chases", "destroys", "enters",
        "finds", "guards", "helps", "invites", "joins", "kills", "leaves", "meets",
        "notices", "observes", "paints", "quits", "reads", "sees", "touches", "understands",
        "values", "wants", "yearns", "zooms", "analyzes", "burns", "cooks", "drinks",
        "explores", "fears", "grows", "hates", "imagines", "judges", "knows", "lifts"
    ]
    
    # Structure: The [adj] [noun] [verb] [adj] [noun].
    adj1 = random.choice(adjectives)
    noun1 = random.choice(nouns)
    verb = random.choice(verbs)
    adj2 = random.choice(adjectives)
    noun2 = random.choice(nouns)
    
    sentence = f"The {adj1} {noun1} {verb} the {adj2} {noun2}, highly detailed, surreal, 8k, cinematic lighting"
    return sentence

class ResourceMonitor:
    def __init__(self, log_file="benchmark_log.txt", interval=1.0):
        self.log_file = log_file
        self.interval = interval
        self.monitoring = False
        self.thread = None
        self.headers_written = False

    def get_gpu_stats(self):
        try:
            result = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                encoding="utf-8"
            )
            util, mem = result.strip().split(", ")
            return float(util), float(mem)
        except:
            if torch.cuda.is_available():
                mem = torch.cuda.memory_allocated() / (1024 * 1024)
                return 0.0, mem
            return 0.0, 0.0

    def _monitor_loop(self):
        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            if not self.headers_written:
                writer.writerow(["Timestamp", "CPU_Percent", "RAM_Used_MB", "GPU_Util_Percent", "GPU_Mem_MB"])
                self.headers_written = True

            while self.monitoring:
                cpu_pct = psutil.cpu_percent(interval=None)
                ram_used = psutil.virtual_memory().used / (1024 * 1024)
                gpu_util, gpu_mem = self.get_gpu_stats()
                
                writer.writerow([
                    datetime.datetime.now().isoformat(),
                    f"{cpu_pct:.1f}",
                    f"{ram_used:.1f}",
                    f"{gpu_util:.1f}",
                    f"{gpu_mem:.1f}"
                ])
                f.flush()
                time.sleep(self.interval)

    def start(self):
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.monitoring = False
        if self.thread:
            self.thread.join()

def benchmark():
    output_dir = "benchmark_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    log_file = "benchmark_log.txt"
    if os.path.exists(log_file):
        os.remove(log_file)
    
    print(f"Logging metrics to {log_file}")
    print(f"Saving images to {output_dir}")
    monitor = ResourceMonitor(log_file)
    
    # We will load the base model and then quantize it
    model_id = "black-forest-labs/FLUX.1-Fill-dev"

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16 # Load in bfloat16 first, then quantize
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.bfloat16
    else:
        device = "cpu"
        dtype = torch.float32

    print(f"Device: {device}")
    print(f"Loading {model_id} in {dtype}...")
    
    # Load on CPU first to avoid OOM during loading/quantization
    pipe = FluxFillPipeline.from_pretrained(
        model_id, 
        torch_dtype=dtype,
        use_safetensors=True
    )
    
    print("Quantizing transformer to FP8 (qfloat8) using optimum-quanto...")
    # Quantize the transformer
    quantize(pipe.transformer, weights=qfloat8)
    freeze(pipe.transformer)
    
    if device == "cuda":
        print("Enabling model CPU offload for memory efficiency...")
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)
        if device == "mps":
            pipe.enable_attention_slicing()

    # Dummy images for Fill pipeline
    size = (512, 512)
    dummy_image = Image.new("RGB", size, (255, 255, 255))
    dummy_mask = Image.new("L", size, 255) # Fill everything

    # Warmup
    print("Warming up...")
    pipe(
        prompt="warmup",
        image=dummy_image,
        mask_image=dummy_mask,
        num_inference_steps=1
    ).images[0]

    print("Starting benchmark (1 minute)...")
    monitor.start()
    
    start_time = time.time()
    count = 0
    
    while time.time() - start_time < 60:
        prompt = get_random_slop_prompt()
        
        cpu_usage = psutil.cpu_percent()
        gpu_util, gpu_mem = monitor.get_gpu_stats()
        print(f"Generating image {count+1} | CPU: {cpu_usage}% | GPU: {gpu_util}% | VRAM: {gpu_mem}MB")
        
        # Iterations fixed at 32
        image = pipe(
            prompt=prompt,
            image=dummy_image,
            mask_image=dummy_mask,
            num_inference_steps=32
        ).images[0]
        
        # Save image
        random_suffix = random.randint(1000, 9999)
        image_path = os.path.join(output_dir, f"slop_{{count}}_{random_suffix}.png")
        image.save(image_path)
        
        count += 1
        elapsed = time.time() - start_time
        print(f"Generated and saved {count} images in {elapsed:.2f}s")

    monitor.stop()
    
    print(f"\nBenchmark Complete.")
    print(f"Total images generated and saved in 60 seconds: {count}")
    print(f"Average time per image: {60/count if count > 0 else 0:.2f}s")

if __name__ == "__main__":
    benchmark()
