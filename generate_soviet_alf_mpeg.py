import shutil
import subprocess
import tempfile
from pathlib import Path


IMAGES_DIR = Path("assets_soviet_alf/images")
OUTPUT_PATH = Path("assets_soviet_alf/soviet_alf_interpolated.mpg")


def main():
    if not IMAGES_DIR.exists():
        raise SystemExit(f"Image folder not found: {IMAGES_DIR}")

    images = sorted([p for p in IMAGES_DIR.glob("*.png")])
    if not images:
        raise SystemExit(f"No PNGs found in {IMAGES_DIR}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        for idx, img in enumerate(images):
            target = tmpdir_path / f"{idx:04d}.png"
            shutil.copy(img, target)

        # Base framerate 1 fps, interpolate to 9 fps (adds 8 frames between originals)
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            "1",
            "-i",
            str(tmpdir_path / "%04d.png"),
            "-vf",
            "minterpolate=fps=9:mi_mode=mci:mc_mode=aobmc:vsbmc=1,format=yuv420p",
            "-c:v",
            "mpeg2video",
            "-q:v",
            "2",
            str(OUTPUT_PATH),
        ]
        subprocess.run(cmd, check=True)

    print(f"Interpolated video written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
