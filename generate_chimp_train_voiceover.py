import torch
import scipy.io.wavfile
import os
import gc
import subprocess
import re
import numpy as np
from transformers import AutoProcessor, BarkModel

# --- Configuration ---
OUTPUT_DIR = "assets_chimp_train"
os.makedirs(f"{OUTPUT_DIR}/voice", exist_ok=True)

if torch.cuda.is_available(): DEVICE = "cuda"
else: DEVICE = "cpu"

VO_SCRIPTS = [
    "Charlie the chimp sits in his cozy jungle hut, dreaming of a glowing golden banana.",
    "He can almost taste the sweetness as he imagines the perfect fruit.",
    "Today is the day; he packs his small bag and prepares for a grand journey.",
    "Charlie arrives at the jungle train station, where the steam engine huffs and puffs.",
    "He stands on the platform, his ticket held tightly in his furry hand.",
    "The whistle blows, and Charlie knows his adventure is finally beginning.",
    "He climbs aboard the wooden carriage and finds a comfortable seat.",
    "The train starts to move, clicking and clacking along the iron rails.",
    "Charlie sits quietly, watching the jungle landscape begin to move.",
    "He presses his face against the cool glass of the window, mesmerized.",
    "Tall trees and rushing rivers blur into a beautiful green streak.",
    "The rhythm of the train lulls him into a peaceful, expectant state.",
    "Finally, the train slows down as it pulls into the distant station.",
    "Charlie hops off the train, looking around at the exciting new place.",
    "He knows the great banana market is just through the nearby woods.",
    "He enters the deep, lush forest, where sunlight filters through the canopy.",
    "Every rustle in the leaves makes him think he is getting closer.",
    "He walks with a steady pace, driven by the thought of that golden banana.",
    "At last, he reaches the bustling banana market, run by friendly chimps.",
    "He searches through the stalls until he sees it: the perfect golden banana.",
    "He holds the banana high, his heart filled with pure, simple joy.",
    "As evening falls, the forest turns into a landscape of deep blues and shadows.",
    "Charlie walks back through the trees, the jungle alive with night sounds.",
    "The moon rises high, lighting his path as he carries his treasure.",
    "He reaches the station at night, the platform quiet under the glowing lamps.",
    "He waits for the late-night train, his golden banana tucked safely away.",
    "The distant light of the locomotive appears, cutting through the darkness.",
    "Back on the train, Charlie watches the moonlight reflect off the trees.",
    "The carriage is dim and peaceful as the train carries him back home.",
    "He is tired but happy, resting his head against the wooden seat.",
    "At last, Charlie is back in his own jungle bed, the journey complete.",
    "He falls asleep with a smile, dreaming of his next big adventure."
]

# Total duration 120s for 32 images = 3.75s per segment
TOTAL_DURATION = 120.0
SEGMENT_DURATION = TOTAL_DURATION / len(VO_SCRIPTS)

def flush():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def apply_audio_enhancement(file_path):
    """Applies high-quality voice processing using ffmpeg."""
    temp_path = file_path.replace(".wav", "_enhanced.wav")
    filter_complex = "bass=g=3,acompressor=threshold=-12dB:ratio=3:makeup=4dB,loudnorm"
    cmd = ["ffmpeg", "-y", "-i", file_path, "-af", filter_complex, temp_path]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.replace(temp_path, file_path)
    except Exception as e:
        print(f"Enhancement failed for {file_path}: {e}")

def generate_voiceover():
    print(f"--- Generating 120s High Quality Voiceover (Bark) ---")
    
    # PyTorch 2.6+ Workaround for Bark weights
    try:
        import importlib
        import warnings
        np_mod = importlib.import_module('numpy')
        if hasattr(torch.serialization, 'add_safe_globals'):
            safe_types = []
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for prefix in ['_core', 'core']:
                    if hasattr(np_mod, prefix):
                        mod = getattr(np_mod, prefix)
                        if hasattr(mod, 'multiarray') and hasattr(mod.multiarray, 'scalar'):
                            safe_types.append(mod.multiarray.scalar)
            torch.serialization.add_safe_globals(list(set(safe_types)))
    except Exception as e:
        print(f"Safe globals patch failed: {e}")

    try:
        processor = AutoProcessor.from_pretrained("suno/bark")
        model = BarkModel.from_pretrained("suno/bark", dtype=torch.float32).to(DEVICE)
        sample_rate = model.generation_config.sample_rate
        voice_preset = "v2/en_speaker_6"
        
        full_audio_segments = []
        
        for i, txt in enumerate(VO_SCRIPTS):
            idx = i + 1
            out_file = f"{OUTPUT_DIR}/voice/voice_{idx:02d}.wav"
            print(f"Generating segment {idx}/32: {txt[:50]}...")
            
            inputs = processor(txt, voice_preset=voice_preset, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                audio_array = model.generate(**inputs, attention_mask=inputs.get("attention_mask"), min_eos_p=0.05).cpu().numpy().squeeze()
            
            # Ensure precise timing (3.75s)
            target_samples = int(SEGMENT_DURATION * sample_rate)
            if len(audio_array) < target_samples:
                audio_array = np.pad(audio_array, (0, target_samples - len(audio_array)))
            else:
                audio_array = audio_array[:target_samples]
            
            scipy.io.wavfile.write(out_file, sample_rate, audio_array)
            apply_audio_enhancement(out_file)
            
            # Re-read for concatenation
            _, enhanced_data = scipy.io.wavfile.read(out_file)
            full_audio_segments.append(enhanced_data)
            
        print("Creating master 120s voiceover file...")
        master_path = f"{OUTPUT_DIR}/voice/voiceover_full_120s.wav"
        combined = np.concatenate(full_audio_segments)
        scipy.io.wavfile.write(master_path, sample_rate, combined)
        
        print(f"Success! Master file: {master_path}")
        del model; del processor; flush()

    except Exception as e:
        print(f"Voiceover generation failed: {e}")

if __name__ == "__main__":
    generate_voiceover()