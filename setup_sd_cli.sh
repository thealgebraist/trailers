#!/usr/bin/env bash
set -euo pipefail

# Build stable-diffusion.cpp sd-cli and fetch the SD 1.5 GGUF model.
# Model:  ~/models/sd/stable-diffusion-v1-5-Q8_0.gguf
# Binary: /tmp/stable-diffusion.cpp/build/bin/sd-cli

MODEL_DIR="${HOME}/models/sd"
MODEL_FILE="${MODEL_DIR}/stable-diffusion-v1-5-Q8_0.gguf"
REPO_DIR="/tmp/stable-diffusion.cpp"
MODEL_REPO="second-state/stable-diffusion-v1-5-GGUF"
MODEL_NAME="stable-diffusion-v1-5-Q8_0.gguf"

need_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "Missing required command: $1" >&2; exit 1; }; }

need_cmd git
need_cmd cmake
need_cmd python3

echo "==> Ensuring model directory: ${MODEL_DIR}"
mkdir -p "${MODEL_DIR}"

if [[ ! -f "${MODEL_FILE}" ]]; then
  echo "==> Downloading GGUF model via huggingface_hub: ${MODEL_NAME}"
  python3 - <<'PY'
import sys, subprocess, os
from pathlib import Path

def ensure_hf():
    try:
        import huggingface_hub  # noqa: F401
        return True
    except Exception:
        return False

if not ensure_hf():
    # Attempt user install to avoid breaking system Python
    print("huggingface_hub missing; installing with --user ...", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "huggingface_hub"])
    if not ensure_hf():
        sys.exit("Failed to install huggingface_hub")

from huggingface_hub import hf_hub_download

MODEL_REPO = "second-state/stable-diffusion-v1-5-GGUF"
MODEL_NAME = "stable-diffusion-v1-5-Q8_0.gguf"
MODEL_DIR = Path.home() / "models" / "sd"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
dest = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_NAME, local_dir=MODEL_DIR, local_dir_use_symlinks=False)
print(f"Downloaded to {dest}")
PY
else
  echo "==> Model already present at ${MODEL_FILE}"
fi

if [[ ! -x "${REPO_DIR}/build/bin/sd-cli" ]]; then
  echo "==> Cloning stable-diffusion.cpp"
  rm -rf "${REPO_DIR}"
  git clone --depth=1 --recurse-submodules https://github.com/leejet/stable-diffusion.cpp.git "${REPO_DIR}"
  cd "${REPO_DIR}"
  echo "==> Building sd-cli"
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)"
else
  echo "==> sd-cli already built at ${REPO_DIR}/build/bin/sd-cli"
fi

echo "==> Done. Binary: ${REPO_DIR}/build/bin/sd-cli"
echo "          Model: ${MODEL_FILE}"
