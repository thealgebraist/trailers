#!/usr/bin/env bash
set -euo pipefail

# Create and activate a venv isolated to the container run
python3 -m venv .container_venv
. .container_venv/bin/activate
pip install -U pip setuptools wheel

# Install a minimal, compatible set of packages for exports/tokenizers
# Pin protobuf to avoid conflicts and install numpy/transformers/diffusers
pip install "protobuf>=4.25.0,<5.0.0" numpy==1.26.4 transformers==4.40.2 diffusers==0.16.1 scipy==1.13.0 onnx==1.16.0 onnxruntime==1.18.0 || true

echo "container venv ready: $(python -c 'import sys,platform; print(platform.python_version())')"
