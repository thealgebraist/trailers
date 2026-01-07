import torch
import numpy as np
import soundfile as sf
import os
import gc
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration

# --- Configuration ---
OUTPUT_DIR = "assets_chimp_train"
os.makedirs(f"{OUTPUT_DIR}/voice", exist_ok=True)

# Select device: mps for macOS, cuda for NVIDIA, else cpu
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")

# Three 60s sample prompts from the original script
VO_SCRIPTS = [
    # Segment 1: The Start and the Journey
    "Charlie the chimp sits in his cozy jungle hut, dreaming of a glowing golden banana. He can almost taste the sweetness as he imagines the perfect fruit. Today is the day; he packs his small bag and prepares for a grand journey. Charlie arrives at the jungle train station, where the steam engine huffs and puffs. He stands on the platform, his ticket held tightly in his furry hand. The whistle blows, and Charlie knows his adventure is finally beginning. He climbs aboard the wooden carriage and finds a comfortable seat. The train starts to move, clicking and clacking along the iron rails. Charlie sits quietly, watching the jungle landscape begin to move. He presses his face against the cool glass of the window, mesmerized.",
    
    # Segment 2: The Arrival and the Market
    "Tall trees and rushing rivers blur into a beautiful green streak. The rhythm of the train lulls him into a peaceful, expectant state. Finally, the train slows down as it pulls into the distant station. Charlie hops off the train, looking around at the exciting new place. He knows the great banana market is just through the nearby woods. He enters the deep, lush forest, where sunlight filters through the canopy. Every rustle in the leaves makes him think he is getting closer. He walks with a steady pace, driven by the thought of that golden banana. At last, he reaches the bustling banana market, run by friendly chimps. He searches through the stalls until he sees it: the perfect golden banana.",
    
    # Segment 3: The Return and Home
    "He holds the banana high, his heart filled with pure, simple joy. As evening falls, the forest turns into a landscape of deep blues and shadows. Charlie walks back through the trees, the jungle alive with night sounds. The moon rises high, lighting his path as he carries his treasure. He reaches the station at night, the platform quiet under the glowing lamps. He waits for the late-night train, his golden banana tucked safely away. The distant light of the locomotive appears, cutting through the darkness. Back on the train, Charlie watches the moonlight reflect off the trees. The carriage is dim and peaceful as the train carries him back home. He is tired but happy, resting his head against the wooden seat. At last, Charlie is back in his own jungle bed, the journey complete. He falls asleep with a smile, dreaming of his next big adventure."
]

def flush():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        # torch.mps.empty_cache() is available in newer torch versions
        try:
            torch.mps.empty_cache()
        except:
            pass

def generate_parler_voiceover():
    print(f"--- Generating 180s High Quality Voiceover (Parler-TTS) ---")
    
    # Load model and tokenizer
    repo_id = "parler-tts/parler-tts-mini-v1"
    print(f"Loading Parler-TTS model: {repo_id}...")
    model = ParlerTTSForConditionalGeneration.from_pretrained(repo_id).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    
    sample_rate = model.config.sampling_rate
    target_sec = 60
    target_samples = target_sec * sample_rate

    # Description that explicitly asks for very slow, clear speech
    description = (
        "A male speaker with a deep, calm voice delivers his words "
        "extremely slowly with long pauses, in a very quiet room "
        "with very clear audio quality."
    )

    full_audio_segments = []

    try:
        for i, text in enumerate(VO_SCRIPTS):
            idx = i + 1
            out_file = f"{OUTPUT_DIR}/voice/voice_long_{idx}.wav"
            print(f"Generating segment {idx}/3: {text[:50]}...")

            # Prepare inputs
            input_ids = tokenizer(description, return_tensors="pt").input_ids.to(DEVICE)
            prompt_input_ids = tokenizer(text, return_tensors="pt").input_ids.to(DEVICE)

            # Generate audio
            generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
            audio_arr = generation.cpu().numpy().squeeze()

            # If the sentence is still too short for 60s, pad it with silence
            if len(audio_arr) < target_samples:
                print(f"Padding segment {idx} with silence to reach {target_sec}s")
                silence = np.zeros(target_samples - len(audio_arr))
                full_audio = np.concatenate([audio_arr, silence])
            else:
                full_audio = audio_arr[:target_samples]

            sf.write(out_file, full_audio, sample_rate)
            print(f"Generated {out_file}: {len(full_audio)/sample_rate:.2f} seconds")
            full_audio_segments.append(full_audio)
            flush()

        print("Creating master voiceover file...")
        master_path = f"{OUTPUT_DIR}/voice/voiceover_full_parler.wav"
        combined = np.concatenate(full_audio_segments)
        sf.write(master_path, combined, sample_rate)
        
        print(f"Success! Master file: {master_path}")

    except Exception as e:
        print(f"Voiceover generation failed: {e}")
    finally:
        del model
        flush()

if __name__ == "__main__":
    generate_parler_voiceover()
