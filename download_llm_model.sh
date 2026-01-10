#!/bin/bash
# Download TinyLlama model properly (smallest version)

cd "$(dirname "$0")/models"

echo "Downloading TinyLlama-1.1B-Chat Q2_K (smallest quantization, ~460MB)..."
echo "This will take a few minutes on a ~400KB/s connection"
echo ""

curl -L --continue-at - -o tinyllama-1.1b-chat-v1.0.Q2_K.gguf \
  "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q2_K.gguf?download=true"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Download complete!"
    ls -lh tinyllama-1.1b-chat-v1.0.Q2_K.gguf
else
    echo "✗ Download failed or interrupted"
    exit 1
fi
