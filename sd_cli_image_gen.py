import subprocess
import shlex
from pathlib import Path


SD_CLI = Path("/tmp/stable-diffusion.cpp/build/bin/sd-cli")
MODEL = Path("/Users/anders/models/sd/stable-diffusion-v1-5-Q8_0.gguf")


def run_sd_cli(prompt: str, output: str, width: int = 512, height: int = 512, steps: int = 20, cfg: float = 7.0, seed: int = 42):
    if not SD_CLI.exists():
        raise SystemExit(f"sd-cli not found at {SD_CLI}")
    if not MODEL.exists():
        raise SystemExit(f"GGUF model not found at {MODEL}")
    cmd = [
        str(SD_CLI),
        "-m", str(MODEL),
        "-p", prompt,
        "-o", output,
        "--width", str(width),
        "--height", str(height),
        "--steps", str(steps),
        "--cfg-scale", str(cfg),
        "--seed", str(seed),
        "-t", "-1",
        "--vae-tiling",
        "--clip-on-cpu",
    ]
    print("Running:", " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True)


def main():
    jobs = [
        ("a fluffy orange cat sitting on a wooden desk, photorealistic, high quality", "py_sd_cat.png", 512, 512, 20, 7.0, 42),
        ("an astronaut riding a horse on mars, digital art, trending on artstation", "py_sd_astronaut.png", 512, 512, 20, 7.5, 123),
        ("a chrome dalek robot in a retro 1960s living room, vintage photograph", "py_sd_dalek.png", 512, 512, 25, 8.0, 999),
    ]
    for prompt, out, w, h, steps, cfg, seed in jobs:
        run_sd_cli(prompt, out, w, h, steps, cfg, seed)


if __name__ == "__main__":
    main()
