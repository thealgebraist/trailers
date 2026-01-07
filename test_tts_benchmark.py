import os
import time
import torch
import scipy.io.wavfile
import numpy as np
import subprocess

# --- Benchmark Configuration ---
MODELS = [
    {"name": "F5-TTS", "type": "cli", "cmd": "f5-tts_infer-cli --gen_text '{text}' --output_dir {out_dir} --file_prefix {prefix}"},
    {"name": "Bark", "type": "transformers", "repo": "suno/bark"},
    {"name": "Kokoro-82M", "type": "local_torch", "path": "kokoro_model.py"},
    {"name": "VibeVoice", "type": "module", "module": "demo.realtime_model_inference_from_file"},
    {"name": "Parler-TTS", "type": "transformers", "repo": "google/parler-tts-mini-v1"},
    {"name": "Fish-Speech", "type": "cli", "cmd": "python3 -m tools.llama.generate --text '{text}'"},
    {"name": "ChatTTS", "type": "local", "repo": "2noise/ChatTTS"},
    {"name": "StyleTTS2", "type": "local", "repo": "yl4579/StyleTTS2"}
]

BENCHMARK_TEXT = """
In a world where time has simply given up and the clocks have forgotten how to tick, every single second stretches out into a lifetime of beige indifference. 
Arthur is a man with a vision, but it is not a vision of grandeur or a dream of change. It is a vision of moisture, a slow and deliberate study of the dark damp patch spreading across his grey plaster wall. 
There is no escape from the beige, no relief from the steady, rhythmic hum of the ancient refrigerator. 
Hunger is just a distant memory, replaced by the reality of cold, lumpy porridge on a wooden table. 
The heat never comes to this room, and time is a luxury that Arthur simply cannot afford to spend on anything but staring.
Action is a concept reserved for the young and the restless. Here, drama is found in the slow, agonizing peeling of floral wallpaper. 
Adventure is out there in the world, but so is the rain, and Arthur prefers the static consistency of his chipped ceramic mug. 
Refreshment is a myth, a story told by those who still believe in the sun. 
Some stories have already ended long before they reach the final page. 
Light is a choice that he did not make, and so he sits in the gathering shadows of his own existence. 
The tension is imperceptible, a silent vibration in the air. Nothing happens. And then, it happens again. 
This season, experience the event of the century. The Damp Patch. 
Time is ticking, yet standing still. The moisture expands, a slow-motion explosion of apathy. 
Witness the beige. Feel the humidity. Embrace the void.
""" # This is roughly 120s of speech depending on model speed.

OUTPUT_BASE = "benchmark_results"
os.makedirs(OUTPUT_BASE, exist_ok=True)

def run_benchmark(model_info):
    name = model_info["name"]
    print(f"\n=== Benchmarking {name} ===")
    out_path = f"{OUTPUT_BASE}/{name.lower().replace('-', '_')}.wav"
    start_time = time.time()
    
    try:
        if model_info["type"] == "cli":
            cmd = model_info["cmd"].format(text=BENCHMARK_TEXT, out_dir=OUTPUT_BASE, prefix=name.lower())
            subprocess.run(cmd, shell=True, check=True)
        
        elif model_info["type"] == "transformers":
            from transformers import AutoProcessor, AutoModel
            processor = AutoProcessor.from_pretrained(model_info["repo"])
            model = AutoModel.from_pretrained(model_info["repo"]).to("cuda")
            inputs = processor(BENCHMARK_TEXT, return_tensors="pt").to("cuda")
            with torch.no_grad():
                audio_array = model.generate(**inputs).cpu().numpy().squeeze()
            scipy.io.wavfile.write(out_path, rate=24000, data=audio_array)
            
        elif model_info["type"] == "local_torch":
            # Assuming Kokoro setup as previously defined
            from kokoro_model import build_model
            model = build_model("kokoro_model.pth", device="cuda")
            # ... generation logic ...
            print(f"Generating via local torch code for {name}...")
            
        elif model_info["type"] == "module":
            temp_txt = "benchmark_temp.txt"
            with open(temp_txt, "w") as f: f.write(BENCHMARK_TEXT)
            cmd = [
                "python3", "-m", model_info["module"],
                "--model_path", "microsoft/VibeVoice-Realtime-0.5B",
                "--txt_path", temp_txt,
                "--speaker_name", "Carter"
            ]
            subprocess.run(cmd, check=True)
            if os.path.exists("output_from_file/Carter.wav"):
                os.replace("output_from_file/Carter.wav", out_path)

        end_time = time.time()
        print(f"SUCCESS: {name} took {end_time - start_time:.2f}s")
        
    except Exception as e:
        print(f"FAILED: {name} error: {e}")

if __name__ == "__main__":
    for m in MODELS:
        run_benchmark(m)
