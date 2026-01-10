import subprocess
from pathlib import Path


BASE = Path("assets_soviet_alf/music/theme.wav")
OUT_DIR = Path("assets_soviet_alf/music/variations")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 10 subtle tempo/EQ/FX variations, each trimmed to 120s
TEMPOS = [0.96 + 0.008 * i for i in range(10)]  # 0.96 .. 1.032


def build_filter(idx: int, tempo: float) -> str:
    parts = [f"atempo={tempo:.3f}"]
    if idx % 3 == 1:
        parts.append("aecho=0.8:0.7:60:0.4")
    elif idx % 3 == 2:
        parts.append("aphaser=0.6:0.66:3:0.6:2:0.5")
    parts.append("highpass=f=180")
    parts.append("lowpass=f=6200")
    return ",".join(parts)


def main():
    if not BASE.exists():
        raise SystemExit(f"Base music not found: {BASE}")

    for idx, tempo in enumerate(TEMPOS):
        out = OUT_DIR / f"theme_var_{idx:02d}.wav"
        if out.exists():
            continue
        af = build_filter(idx, tempo)
        cmd = [
            "ffmpeg",
            "-y",
            "-stream_loop",
            "-1",  # loop input enough times
            "-i",
            str(BASE),
            "-t",
            "120",
            "-af",
            af,
            "-ar",
            "32000",
            "-ac",
            "1",
            str(out),
        ]
        print(f"[{idx+1}/10] Writing {out.name} (tempo {tempo:.3f})")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
