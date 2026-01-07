#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DIR="$ROOT/models"

echo "Fixing model layouts in $MODELS_DIR"
for d in "$MODELS_DIR"/*; do
  [ -d "$d" ] || continue
  echo "Processing $d"
  # If top-level has snapshots/<snapid>/, move contents up
  if [ -d "$d/snapshots" ]; then
    # find the latest snapshot dir
    snap=$(ls -1 "$d/snapshots" | head -n 1 || true)
    if [ -n "$snap" ] && [ -d "$d/snapshots/$snap" ]; then
      echo "Found snapshot $snap, flattening..."
      # copy contents to $d
      cp -a "$d/snapshots/$snap/"* "$d/" || true
      # optional: remove snapshots dir
      rm -rf "$d/snapshots"
    fi
  fi
  # Also, if there's a subdir like models--owner--repo, try to flatten
  sub=$(find "$d" -maxdepth 1 -type d -name 'models--*' -print -quit || true)
  if [ -n "$sub" ]; then
    echo "Found nested $sub, flattening"
    cp -a "$sub/"* "$d/" || true
    rm -rf "$sub"
  fi
done

echo "Layout fix complete. Listing models/ contents:"
ls -la "$MODELS_DIR" || true
