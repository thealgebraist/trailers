#!/usr/bin/env bash
set -euo pipefail

echo "Attempting to build and install onnx from source into venv"
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel cmake

OS=$(uname)
if [[ "$OS" == "Darwin" ]]; then
  which brew >/dev/null 2>&1 || { echo "Homebrew not found; please install Homebrew to install build deps."; exit 1; }
  brew install protobuf || true
elif [[ "$OS" == "Linux" ]]; then
  sudo apt-get update && sudo apt-get install -y build-essential protobuf-compiler libprotobuf-dev cmake || true
fi

# Try building onnx from source
pip download onnx --no-binary=onnx -d /tmp/onnx_src || true
PKG=$(ls /tmp/onnx_src | grep onnx | head -n1)
if [ -z "$PKG" ]; then
  echo "Failed to download onnx source package"; exit 1
fi
cd /tmp/onnx_src
tar xvf "$PKG" || true
DIR=$(tar -tf "$PKG" | head -n1 | cut -f1 -d"/")
cd "$DIR"
python -m pip install --upgrade pip
python -m pip wheel . -w /tmp/onnx_wheel || { echo "Wheel build failed"; exit 1; }
WHEEL=$(ls /tmp/onnx_wheel | head -n1)
if [ -z "$WHEEL" ]; then
  echo "Wheel not produced"; exit 1
fi
python -m pip install /tmp/onnx_wheel/$WHEEL

echo "onnx built and installed from source"
