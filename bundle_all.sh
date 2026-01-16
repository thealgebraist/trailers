#!/bin/bash
set -e

PACKAGE_NAME="vidgen_master_bundle"
DIST_DIR="$PACKAGE_NAME"
LIB_DIR="$DIST_DIR/libs"
MODEL_DIR="$DIST_DIR/models"

echo "--- Creating Master Self-Contained Bundle ---"
mkdir -p "$LIB_DIR"
mkdir -p "$MODEL_DIR"

# 1. Collect Binaries
echo "Collecting binaries..."
# C++
cp cpp/v_cpp_static "$DIST_DIR/v_cpp"
# Zig
cp zig/zig-out/bin/zig-sloppy "$DIST_DIR/"
# Rust (Assuming built)
if [ -f rust/target/release/vidgen ]; then
    cp rust/target/release/vidgen "$DIST_DIR/"
fi

# 2. Collect Shared Libraries (Mac)
echo "Collecting shared libraries..."
TORCH_LIB_DIR=$(./venv/bin/python3 -c "import torch; import os; print(os.path.dirname(torch.__file__) + '/lib')")
ONNX_LIB="/opt/homebrew/opt/onnxruntime/lib/libonnxruntime.dylib"

cp "$TORCH_LIB_DIR"/lib*.dylib "$LIB_DIR/"
cp "$ONNX_LIB" "$LIB_DIR/"

# 3. Patch Binaries for Portability
echo "Patching binary RPATHs..."
# Patch C++
install_name_tool -change "$ONNX_LIB" "@executable_path/libs/libonnxruntime.dylib" "$DIST_DIR/v_cpp"
install_name_tool -add_rpath "@executable_path/libs" "$DIST_DIR/v_cpp" 2>/dev/null || true

# Patch Zig
install_name_tool -change "$ONNX_LIB" "@executable_path/libs/libonnxruntime.dylib" "$DIST_DIR/zig-sloppy"
install_name_tool -add_rpath "@executable_path/libs" "$DIST_DIR/zig-sloppy" 2>/dev/null || true

# 4. Collect Essential Models (Harness)
echo "Bundling model paths (Configuring for local lookup)..."
# We create a simple config or env script inside the bundle
cat <<EOF > "$DIST_DIR/env.sh"
export HF_HOME="./models"
export LD_LIBRARY_PATH="./libs:\$LD_LIBRARY_PATH"
export DYLD_LIBRARY_PATH="./libs:\$DYLD_LIBRARY_PATH"
echo "Vidgen Master Bundle Ready."
EOF

chmod +x "$DIST_DIR/env.sh"

echo "--- Bundle Complete: $DIST_DIR ---"
echo "Move this folder anywhere and 'source env.sh' to run."
