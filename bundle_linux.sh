#!/bin/bash
set -e

DIST_DIR="vidgen_linux_dist"
LIB_DIR="$DIST_DIR/libs"

echo "--- Creating Portable Linux C++ Bundle ---"
mkdir -p "$LIB_DIR"

# 1. Copy binary
cp v_cpp "$DIST_DIR/"

# 2. Identify and copy dependencies
TORCH_LIB_DIR=$(./venv/bin/python3 -c "import torch; import os; print(os.path.dirname(torch.__file__) + '/lib')")
ONNX_LIB=$(find /usr/local /usr/lib -name "libonnxruntime.so" | head -n 1)

echo "Copying Torch libs..."
cp "$TORCH_LIB_DIR"/lib*.so* "$LIB_DIR/" 2>/dev/null || true

if [ -n "$ONNX_LIB" ]; then
    echo "Copying ONNX Runtime from $ONNX_LIB..."
    cp "$ONNX_LIB" "$LIB_DIR/"
fi

# 3. Fix RPATHs using patchelf
echo "Fixing RPATHs using patchelf..."
if command -v patchelf >/dev/null; then
    patchelf --set-rpath '$ORIGIN/libs' "$DIST_DIR/v_cpp"
else
    echo "Warning: patchelf not found. You may need to run: sudo apt install patchelf"
fi

echo "--- Linux Bundle Complete in $DIST_DIR ---"
