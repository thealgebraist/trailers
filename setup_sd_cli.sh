#!/usr/bin/env bash
set -euo pipefail

# Build stable-diffusion.cpp sd-cli and fetch the SD 1.5 GGUF model.
# Model goes to ~/models/sd/stable-diffusion-v1-5-Q8_0.gguf
# Binary goes to /tmp/stable-diffusion.cpp/build/bin/sd-cli

MODEL_DIR="${HOME}/models/sd"
MODEL_FILE="${MODEL_DIR}/stable-diffusion-v1-5-Q8_0.gguf"
REPO_DIR="/tmp/stable-diffusion.cpp"
MODEL_REPO="second-state/stable-diffusion-v1-5-GGUF"
MODEL_NAME="stable-diffusion-v1-5-Q8_0.gguf"

echo "==> Ensuring model directory: ${MODEL_DIR}"
mkdir -p "${MODEL_DIR}"

if [[ ! -f "${MODEL_FILE}" ]]; then
  echo "==> Downloading GGUF model via huggingface_hub: ${MODEL_NAME}"
  python3 - <<'PY'
import sys
from pathlib import Path
try:
    from huggingface_hub import hf_hub_download
except Exception as exc:
    sys.exit(f"Install huggingface_hub first (pip install huggingface_hub): {exc}")

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
  git clone --depth=1 https://github.com/leejet/stable-diffusion.cpp.git "${REPO_DIR}"
  cd "${REPO_DIR}"
  echo "==> Building sd-cli"
  cmake -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)"
else
  echo "==> sd-cli already built at ${REPO_DIR}/build/bin/sd-cli"
fi

echo "==> Done. Binary: ${REPO_DIR}/build/bin/sd-cli"
echo "          Model: ${MODEL_FILE}"
