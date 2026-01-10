import os
from pathlib import Path
import torch
import imageio

try:
    from diffusers import DiffusionPipeline
except ImportError as exc:
    raise SystemExit("Please install diffusers>=0.24.0 to run this script") from exc


SD_VIDEO_MODEL = "ali-vilab/modelscope-damo-text-to-video-synthesis"
OUTPUT_DIR = Path("assets_soviet_alf/video_clips")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Reuse a concise subset of the 64 Soviet Alf prompts (all include Alfons)
PROMPTS = [
    "Alfons in a communal kitchen pouring tea from a brass samovar, 1950s Soviet photo, black and white, high grain.",
    "Alfons saluting a red flag in snowy Red Square, long shadow, vintage film still, monochrome.",
    "Alfons peeking from behind heavy curtains at a fat cat on a radiator, cramped Soviet apartment, grainy B&W film.",
    "Alfons at a family dinner with stern parents, wooden laughter, sitcom lighting, black and white.",
    "Alfons riding on a tractor rebuilt from scrap spoons, kolkhoz yard, high contrast film.",
    "Alfons reading banned fairy tales in a dim library corridor, flashlight beam, grainy photo.",
    "Alfons conducting a factory choir with a soup ladle, stage spotlights, monochrome grain.",
    "Alfons kneels to tell sixty-four tales to a child in a winter courtyard, soft grain, B&W."
]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = DiffusionPipeline.from_pretrained(
        SD_VIDEO_MODEL,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    for idx, prompt in enumerate(PROMPTS):
        out_mp4 = OUTPUT_DIR / f"soviet_alf_clip_{idx:02d}.mp4"
        if out_mp4.exists():
            continue
        print(f"[{idx+1}/{len(PROMPTS)}] Generating clip: {prompt}")
        result = pipe(prompt, num_inference_steps=25)
        frames = result.frames[0]  # list of PIL images
        imageio.mimwrite(out_mp4, frames, fps=8)
        print(f"Saved {out_mp4}")


if __name__ == "__main__":
    main()
