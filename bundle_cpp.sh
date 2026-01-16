#!/bin/bash
set -e

DIST_DIR="vidgen_cpp_dist"
LIB_DIR="$DIST_DIR/libs"

echo "--- Creating Portable C++ Bundle (Mac) ---"
mkdir -p "$LIB_DIR"

# 1. Copy binary
cp v_cpp "$DIST_DIR/"

# 2. Identify and copy dependencies
TORCH_LIB_DIR=$(./venv/bin/python3 -c "import torch; import os; print(os.path.dirname(torch.__file__) + '/lib')")
ONNX_LIB="/opt/homebrew/opt/onnxruntime/lib/libonnxruntime.dylib"

echo "Copying Torch libs..."
cp "$TORCH_LIB_DIR/libtorch.dylib" "$LIB_DIR/"
cp "$TORCH_LIB_DIR/libtorch_cpu.dylib" "$LIB_DIR/"
cp "$TORCH_LIB_DIR/libc10.dylib" "$LIB_DIR/"

echo "Copying ONNX Runtime..."
cp "$ONNX_LIB" "$LIB_DIR/"

# 3. Fix RPATHs
echo "Fixing library paths in binary..."
install_name_tool -change "$ONNX_LIB" "@executable_path/libs/libonnxruntime.dylib" "$DIST_DIR/v_cpp"
# Use -add_rpath only if it doesn't exist, or just use -delete_rpath first to be safe
install_name_tool -delete_rpath "/Users/anders/projects/vid/venv/lib/python3.14/site-packages/torch/lib" "$DIST_DIR/v_cpp" 2>/dev/null || true
install_name_tool -add_rpath "@executable_path/libs" "$DIST_DIR/v_cpp" 2>/dev/null || true

echo "--- Bundle Complete in $DIST_DIR ---"
