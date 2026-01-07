#!/usr/bin/env bash
set -euo pipefail

CACHE_DIR="${HF_CACHE:-$HOME/.cache/huggingface/hub}"
DEST_DIR="$(cd "$(dirname "$0")/.." && pwd)/models"
mkdir -p "$DEST_DIR"

declare -A MAP
MAP[segmind/tiny-sd]="models--segmind--tiny-sd"
MAP[facebook/musicgen-small]="models--facebook--musicgen-small"
MAP[facebook/mms-tts-eng]="models--facebook--mms-tts-eng"

for key in "${!MAP[@]}"; do
  cache_sub="${MAP[$key]}"
  src="$CACHE_DIR/$cache_sub"
  dest="$DEST_DIR/$(echo $key | tr '/' '_')_model"
  if [ -d "$src" ]; then
    echo "Copying $src -> $dest"
    rm -rf "$dest"
    mkdir -p "$dest"
    if command -v rsync >/dev/null 2>&1; then
      # use portable rsync flags
      rsync -a --progress "$src/" "$dest/"
    else
      cp -a "$src/"* "$dest/" || true
    fi
  else
    echo "Warning: cached model path not found: $src"
  fi
done

echo "Copy complete. Models available at:"
ls -la "$DEST_DIR" || true
