#!/usr/bin/env bash
set -euo pipefail

# Minimal runner: fetch models, pick prebuilt binary (cuda/cpu/mps) and run asset generator
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN_DIR="$ROOT/bin"
MODELS_DIR="$ROOT/models"

# Detect backend
BACKEND="cpu"
if command -v nvidia-smi >/dev/null 2>&1; then
  BACKEND="cuda"
elif [[ "$(uname)" == "Darwin" ]]; then
  BACKEND="mps"
fi

BINARY_NAME="generate_trailer_assets_full_${BACKEND}"
BINARY_PATH="$BIN_DIR/$BINARY_NAME"

echo "Selected backend: $BACKEND"

# Ensure models are present
if [ ! -d "$MODELS_DIR" ]; then
  mkdir -p "$MODELS_DIR"
fi

# Fetch small test models (tiny-sd, musicgen-small, mms-tts-eng)
if [ ! -d "$MODELS_DIR/tiny-sd_model" ] || [ ! -d "$MODELS_DIR/musicgen_small_model" ] || [ ! -d "$MODELS_DIR/mms_tts_model" ]; then
  echo "Fetching models into $MODELS_DIR..."
  "$ROOT/minimal/fetch_models.sh" "$MODELS_DIR"
fi

# Attempt to download prebuilt binaries
bash "$(dirname "$0")/get_prebuilt.sh" || true
# Check binary
if [ ! -x "$BINARY_PATH" ]; then
  echo "Precompiled binary not found: $BINARY_PATH"
  echo "You can either build it locally (see README) or place a precompiled binary named $BINARY_NAME in $BIN_DIR"
  echo "Attempting to build using system ONNX Runtime and clang++..."
  if command -v make >/dev/null 2>&1; then
    echo "Running make build-cpp"
    make build-cpp || true
  fi
  if [ ! -x "$BINARY_PATH" ]; then
    echo "Binary still not available. Exiting." >&2
    exit 1
  fi
fi

# Run binary with model paths
echo "Running $BINARY_PATH"
"$BINARY_PATH" "$MODELS_DIR/tiny-sd_model" "$MODELS_DIR/musicgen_small_model" "$MODELS_DIR/mms_tts_model"
