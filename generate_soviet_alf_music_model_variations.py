import subprocess
from pathlib import Path

import torch
import scipy.io.wavfile as wavfile
from transformers import pipeline

from dalek.core import get_device, ensure_dir


MODEL_ID = "facebook/musicgen-small"
OUT_DIR = Path("assets_soviet_alf/music/variations_model")
BASE_LEN_SEC = 120

PROMPTS = [
    "Soviet sitcom opening for Alfons, detuned upright piano, tape flutter, monophonic, smoky studio.",
    "Alfons theme on wheezing accordion and creaking balalaika, reel-to-reel wow and flutter.",
    "Marching-band Alfons motif with dusty brass, warbly tube preamps, black-and-white broadcast tone.",
    "Alfons lullaby on worn celesta and radio hiss, nostalgic and wobbly, mono 1950s room.",
    "Playful Alfons chase cue with plucky strings, off-kilter percussion, tape saturation.",
    "Slow Alfons reprise on bowed saw and accordion, fluttering reel noise, melancholic yet warm.",
    "Factory-floor Alfons rhythm with clanking pipes, muted horns, and soft chorus hum.",
    "Alfons bedtime story underscored by glass harmonica, faint vinyl crackle, gentle tempo.",
    "Cabaret Alfons vamp with honky-tonk piano, brushed kit, fluttering air-raid siren motif.",
    "Finale Alfons credits with whistled melody, accordion drone, reel-to-reel saturation, mono.",
]


def loop_to_length(src: Path, dst: Path, seconds: int):
    cmd = [
        "ffmpeg",
        "-y",
        "-stream_loop",
        "-1",
        "-i",
        str(src),
        "-t",
        str(seconds),
        "-c",
        "copy",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def main():
    ensure_dir(OUT_DIR)
    device = 0 if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else -1
    print(f"Loading MusicGen pipeline: {MODEL_ID} on device={device}")
    pipe = pipeline(
        task="text-to-audio",
        model=MODEL_ID,
        device=device,
        trust_remote_code=True,
    )

    for idx, prompt in enumerate(PROMPTS):
        base_wav = OUT_DIR / f"tmp_base_{idx:02d}.wav"
        final_wav = OUT_DIR / f"alfons_var_{idx:02d}.wav"
        if final_wav.exists():
            continue

        print(f"[{idx+1}/{len(PROMPTS)}] Generating base clip for: {prompt}")
        result = pipe(
            prompt,
            forward_params={
                "do_sample": True,
                "guidance_scale": 3.0,
                "max_new_tokens": 512,
            },
        )
        audio_arr = result["audio"]
        sr = result["sampling_rate"]
        wavfile.write(base_wav, sr, audio_arr.astype("float32"))

        print(f"  Looping/padding to {BASE_LEN_SEC}s -> {final_wav.name}")
        loop_to_length(base_wav, final_wav, BASE_LEN_SEC)
        base_wav.unlink(missing_ok=True)
    print(f"Done. Variations saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
