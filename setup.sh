#!/usr/bin/env bash
set -euo pipefail

echo "Setup script: install deps and prepare environment"
UNAME=$(uname)
if [[ "$UNAME" == "Darwin" ]]; then
  echo "Detected macOS"
  which brew >/dev/null 2>&1 || (echo "Homebrew not found; please install Homebrew" && exit 1)
  which cmake >/dev/null 2>&1 || (echo "Installing cmake via brew" && brew install cmake)
  # prefer system onnxruntime if installed by brew
  if [ -d "/opt/homebrew/Cellar/onnxruntime" ]; then
    echo "System onnxruntime detected at /opt/homebrew/Cellar/onnxruntime"
    echo "Skipping pip onnxruntime install; will use system onnxruntime for C++ linking"
  fi
elif [[ "$UNAME" == "Linux" ]]; then
  echo "Detected Linux"
  if ! which cmake >/dev/null 2>&1; then
    echo "Installing cmake via apt-get"
    sudo apt-get update
    sudo apt-get install -y cmake
  fi
else
  echo "Unknown OS: $UNAME"
fi

# Create and activate venv
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# Try to install Python packages; onnx/onnxruntime may fail to build on some platforms
echo "Installing Python requirements (may skip onnx/onnxruntime if build fails)"
if ! pip install -r requirements.txt; then
  echo "pip install -r requirements.txt failed; attempting to install core packages without onnx/onnxruntime"
  pip install --no-deps -r requirements.txt || true
  echo "Attempting onnx install"
  if ! pip install onnx==1.16.0; then
    echo "onnx pip install failed; you can use system onnx/onnxruntime for C++"
  fi
  echo "Attempting onnxruntime install"
  if ! pip install onnxruntime==1.18.0; then
    echo "onnxruntime pip install failed; use system onnxruntime for C++"
  fi
fi

echo "Setup complete. Activate the venv with: source .venv/bin/activate" > /dev/stderr
chmod +x setup.sh
