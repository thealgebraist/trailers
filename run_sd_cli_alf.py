#!/usr/bin/env python3
import subprocess
from pathlib import Path

SD_CLI = "/opt/homebrew/bin/sd-cli"
MODEL = "/Users/anders/models/sd/stable-diffusion-v1-5-Q8_0.gguf"
OUTDIR = Path("assets_soviet_alf/images")
OUTDIR.mkdir(parents=True, exist_ok=True)

PROMPTS = [
    "Title card 'АЛЬФ' in bold 1950s Soviet typography, black and white, heavy film grain, scratches, vintage cinema style.",
    "A small, hairy alien with a long nose sitting in a 1950s Soviet communal kitchen, wearing a ushanka, drinking tea from a glass with a metal holder, black and white, grainy, high contrast.",
    "Comrade Alfons saluting a red flag in a snowy Moscow street, vintage 1950s photo, black and white, grainy, blurry edges.",
    "Alfons peering mischievously from behind a heavy curtain at a fat cat sitting on a radiator, 1950s Soviet apartment, black and white, grainy film still.",
    "A stern Soviet family (father in suit, mother in floral dress) laughing woodenly at a hairy alien at their dinner table, 1950s sitcom aesthetic, black and white, grainy.",
    "Close up of the alien Alfons laughing with his mouth wide open, 1950s film quality, black and white, heavy grain, vignetting."
]

def run():
    for i, prompt in enumerate(PROMPTS):
        out = OUTDIR / f"scene_{i:02d}.png"
        if out.exists():
            print(f"Skipping existing: {out}")
            continue

        print(f"Generating {out} ...")
        cmd = [
            SD_CLI,
            '-m', MODEL,
            '-p', prompt,
            '-o', str(out),
            '--width', '512',
            '--height', '512',
            '--steps', '32',
            '--cfg-scale', '7.5',
            '--seed', str(42 + i),
            '--vae-tiling',
            '--clip-on-cpu'
        ]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"sd-cli failed for prompt {i}: {e}")
            break

if __name__ == '__main__':
    run()
