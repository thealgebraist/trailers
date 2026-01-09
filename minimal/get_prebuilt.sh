#!/usr/bin/env bash
set -euo pipefail

# Attempt to download prebuilt binaries from PREBUILT_BASE_URL if set.
BASE_URL=${PREBUILT_BASE_URL:-""}
BIN_DIR="$(cd "$(dirname "$0")/.." && pwd)/bin"
mkdir -p "$BIN_DIR"
if [ -z "$BASE_URL" ]; then
  echo "PREBUILT_BASE_URL not set; skipping download"
  exit 0
fi
for backend in cpu cuda mps; do
  name="generate_trailer_assets_full_${backend}"
  url="$BASE_URL/$name"
  out="$BIN_DIR/$name"
  if [ ! -x "$out" ]; then
    echo "Downloading $url -> $out"
    if command -v curl >/dev/null 2>&1; then
      curl -L -o "$out" "$url" || { echo "Failed to download $url"; rm -f "$out"; }
    elif command -v wget >/dev/null 2>&1; then
      wget -O "$out" "$url" || { echo "Failed to download $url"; rm -f "$out"; }
    fi
    chmod +x "$out" || true
  fi
done

echo "get_prebuilt finished"
