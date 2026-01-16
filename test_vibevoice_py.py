import os
import subprocess
import time

os.environ["HF_HOME"] = "/Users/anders/.cache/huggingface"

# Text for one 30s segment
base_text = """
The horizon was a jagged line of slate and silver, where the sea met the sky in a cold, indifferent embrace. 
Arthur stood on the pier, his coat collar turned up against the biting wind, watching the last of the fishing boats retreat into the gathering fog. 
"""

print("--- Starting Iterative VibeVoice Generation (8 x 30s segments) ---")
print("Model: microsoft/VibeVoice-Realtime-0.5B")

output_files = []

for i in range(8):
    print(f"Generating segment {i+1}/8...")
    file_prefix = f"vibevoice_segment_{i}"
    cmd = [
        "./venv/bin/python3", "-m", "mlx_audio.tts.generate",
        "--model", "microsoft/VibeVoice-Realtime-0.5B",
        "--text", base_text,
        "--ref_audio", "voices/af_heart_ref.wav",
        "--file_prefix", file_prefix,
        "--max_tokens", "2048",
        "--ddpm_steps", "30",
        "--verbose",
        "--join_audio"
    ]
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(f"  [{i+1}] {line}", end="")
        process.wait()
        
        output_name = f"{file_prefix}.wav"
        if os.path.exists(output_name):
            output_files.append(output_name)
        else:
            print(f"  [Error] Segment {i+1} output not found.")
            
    except Exception as e:
        print(f"  [Error] Segment {i+1} failed: {e}")

# Note: Combining multiple WAVs in pure Python without extra libraries is tricky, 
# but since the goal is to provide the sample on the desktop, I'll move the first one 
# and notify the user about the segments.
if output_files:
    # Use ffmpeg to join them into the final file
    print("\nJoining segments using ffmpeg...")
    concat_list = "concat_list.txt"
    with open(concat_list, "w") as f:
        for f_name in output_files:
            f.write(f"file '{f_name}'\n")
            
    final_output = os.path.expanduser("~/Desktop/vibevoice_female_en_240s.wav")
    join_cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list, "-c", "copy", final_output]
    
    try:
        subprocess.run(join_cmd, check=True)
        print(f"Success! Full sample at: {final_output}")
        # Cleanup
        for f_name in output_files: os.remove(f_name)
        os.remove(concat_list)
    except Exception as e:
        print(f"Failed to join files: {e}")