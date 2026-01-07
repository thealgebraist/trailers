import torch
import scipy.io.wavfile
import os
import gc
import numpy as np
import subprocess

# --- Configuration ---
OUTPUT_DIR = "assets_chimp_train"
os.makedirs(f"{OUTPUT_DIR}/voice", exist_ok=True)

VO_SCRIPTS = [
    "Charlie the chimp wakes up in his jungle home, dreaming of the perfect banana.",
    "He finds a golden train ticket under a leaf and knows today is special.",
    "Charlie waves goodbye to his monkey friends and sets off on his adventure.",
    "He arrives at the bustling jungle train station, eyes wide with excitement.",
    "The train pulls in, steam hissing, and Charlie hops aboard with a big grin.",
    "He finds a window seat and presses his face to the glass, watching the trees blur by.",
    "A friendly toucan conductor checks Charlie’s ticket and tips his hat.",
    "The train rattles over a river, where crocodiles wave from the water below.",
    "Charlie shares a snack with a shy lemur sitting beside him.",
    "The train enters a dark tunnel, and everyone holds their breath in the shadows.",
    "Out of the tunnel, sunlight floods the carriage and Charlie laughs with joy.",
    "A family of parrots sings a song, filling the train with cheerful music.",
    "Charlie sketches a banana in his notebook, imagining its sweet taste.",
    "The train stops at a mountain station, and snow monkeys throw snowballs at the windows.",
    "Charlie helps a lost baby elephant find her seat.",
    "The train zooms past fields of wildflowers, colors swirling outside.",
    "A wise old gorilla tells Charlie stories of legendary bananas.",
    "Charlie spots a distant city and wonders if the best bananas are there.",
    "The train slows as it nears Banana Market Station, excitement building.",
    "Vendors wave bunches of bananas as the train comes to a stop.",
    "Charlie leaps off, heart pounding, and races to the biggest fruit stand.",
    "He inspects every banana, searching for the perfect one.",
    "At last, he finds a huge, golden banana shining in the sunlight.",
    "Charlie trades his ticket for the banana and hugs it close.",
    "He takes a big bite, savoring the sweet, creamy flavor.",
    "Other animals gather around, and Charlie shares his prize with new friends.",
    "The sun sets as Charlie sits on the station bench, happy and full.",
    "He waves goodbye to the market and boards the train home, banana in paw.",
    "Charlie dreams of new adventures as the train chugs into the night.",
    "Back in his jungle bed, Charlie smiles, knowing dreams can come true.",
    "The stars twinkle above, and the jungle is peaceful once more.",
    "Charlie’s train adventure is a story he’ll never forget."
]

# Total duration 120s for 32 images = 3.75s per segment
TOTAL_DURATION = 120.0
SEGMENT_DURATION = TOTAL_DURATION / len(VO_SCRIPTS)

def apply_audio_enhancement(file_path):
    """Applies high-quality voice processing using ffmpeg."""
    temp_path = file_path.replace(".wav", "_enhanced.wav")
    # Compression, normalization and slight warmth
    filter_complex = "bass=g=3,acompressor=threshold=-12dB:ratio=3:makeup=4dB,loudnorm"
    cmd = ["ffmpeg", "-y", "-i", file_path, "-af", filter_complex, temp_path]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.replace(temp_path, file_path)
    except Exception as e:
        print(f"Enhancement failed for {file_path}: {e}")

def generate_voiceover():
    print(f"--- Generating 120s High Quality Voiceover (Fish Speech V1.5) ---")
    
    try:
        from fish_audio_sdk import Session, TTS
        session = Session() # Assumes FISH_AUDIO_API_KEY is in environment
        tts = TTS(session)
        
        full_audio_segments = []
        sample_rate = 44100 # Default output target
        
        for i, txt in enumerate(VO_SCRIPTS):
            idx = i + 1
            out_file = f"{OUTPUT_DIR}/voice/voice_{idx:02d}.wav"
            print(f"Generating segment {idx}/32: {txt[:50]}...")
            
            # Generate speech
            # reference_id can be swapped for a specific high-quality narrator voice
            audio_bytes = tts.tts(text=txt, reference_id="default")
            
            # Write temporary segment
            with open(out_file, "wb") as f:
                f.write(audio_bytes)
            
            # Load back to process duration
            sr, data = scipy.io.wavfile.read(out_file)
            sample_rate = sr
            
            # Convert to float32 for processing
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            
            # Pad or trim to exactly 3.75 seconds to ensure 120s total
            target_samples = int(SEGMENT_DURATION * sample_rate)
            if len(data) < target_samples:
                # Add silence at the end
                data = np.pad(data, (0, target_samples - len(data)))
            else:
                # Trim (unlikely for a single sentence but safe)
                data = data[:target_samples]
            
            # Save the perfectly timed segment
            scipy.io.wavfile.write(out_file, sample_rate, (data * 32767).astype(np.int16))
            
            # Apply enhancement
            apply_audio_enhancement(out_file)
            
            # Re-read enhanced for concatenation
            _, enhanced_data = scipy.io.wavfile.read(out_file)
            full_audio_segments.append(enhanced_data)
            
        # Concatenate all into the 120s master file
        print("Creating master 120s voiceover file...")
        master_path = f"{OUTPUT_DIR}/voice/voiceover_full_120s.wav"
        combined = np.concatenate(full_audio_segments)
        scipy.io.wavfile.write(master_path, sample_rate, combined)
        
        print(f"Success! Master file: {master_path}")
        print(f"Individual segments stored in {OUTPUT_DIR}/voice/")

    except ImportError:
        print("Error: 'fish-audio-sdk' not found.")
    except Exception as e:
        print(f"Voiceover generation failed: {e}")

if __name__ == "__main__":
    generate_voiceover()
