import torch
import numpy as np
import os
from dalek.core import get_device, flush, ensure_dir
from dalek.audio import load_parler, generate_parler_audio, save_audio

# --- Configuration ---
OUTPUT_DIR = "assets_chimp_train"
ensure_dir(f"{OUTPUT_DIR}/voice")

DEVICE = get_device()
print(f"Using device: {DEVICE}")

VO_SCRIPTS = [
    "Charlie the chimp sits in his cozy jungle hut, dreaming of a glowing golden banana. He can almost taste the sweetness as he imagines the perfect fruit. Today is the day; he packs his small bag and prepares for a grand journey. Charlie arrives at the jungle train station, where the steam engine huffs and puffs. He stands on the platform, his ticket held tightly in his furry hand. The whistle blows, and Charlie knows his adventure is finally beginning. He climbs aboard the wooden carriage and finds a comfortable seat. The train starts to move, clicking and clacking along the iron rails. Charlie sits quietly, watching the jungle landscape begin to move. He presses his face against the cool glass of the window, mesmerized.",
    "Tall trees and rushing rivers blur into a beautiful green streak. The rhythm of the train lulls him into a peaceful, expectant state. Finally, the train slows down as it pulls into the distant station. Charlie hops off the train, looking around at the exciting new place. He knows the great banana market is just through the nearby woods. He enters the deep, lush forest, where sunlight filters through the canopy. Every rustle in the leaves makes him think he is getting closer. He walks with a steady pace, driven by the thought of that golden banana. At last, he reaches the bustling banana market, run by friendly chimps. He searches through the stalls until he sees it: the perfect golden banana.",
    "He holds the banana high, his heart filled with pure, simple joy. As evening falls, the forest turns into a landscape of deep blues and shadows. Charlie walks back through the trees, the jungle alive with night sounds. The moon rises high, lighting his path as he carries his treasure. He reaches the station at night, the platform quiet under the glowing lamps. He waits for the late-night train, his golden banana tucked safely away. The distant light of the locomotive appears, cutting through the darkness. Back on the train, Charlie watches the moonlight reflect off the trees. The carriage is dim and peaceful as the train carries him back home. He is tired but happy, resting his head against the wooden seat. At last, Charlie is back in his own jungle bed, the journey complete. He falls asleep with a smile, dreaming of his next big adventure."
]

def generate_parler_voiceover():
    print(f"--- Generating 180s High Quality Voiceover (Parler-TTS) ---")
    
    model, tokenizer = load_parler()
    sample_rate = model.config.sampling_rate
    target_sec = 60
    target_samples = target_sec * sample_rate

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

            audio_arr = generate_parler_audio(model, tokenizer, text, description, DEVICE)

            if len(audio_arr) < target_samples:
                print(f"Padding segment {idx} with silence to reach {target_sec}s")
                silence = np.zeros(target_samples - len(audio_arr))
                full_audio = np.concatenate([audio_arr, silence])
            else:
                full_audio = audio_arr[:target_samples]

            save_audio(out_file, full_audio, sample_rate)
            print(f"Generated {out_file}: {len(full_audio)/sample_rate:.2f} seconds")
            full_audio_segments.append(full_audio)
            flush()

        print("Creating master voiceover file...")
        master_path = f"{OUTPUT_DIR}/voice/voiceover_full_parler.wav"
        combined = np.concatenate(full_audio_segments)
        save_audio(master_path, combined, sample_rate)
        
        print(f"Success! Master file: {master_path}")

    except Exception as e:
        print(f"Voiceover generation failed: {e}")
    finally:
        del model
        flush()

if __name__ == "__main__":
    generate_parler_voiceover()