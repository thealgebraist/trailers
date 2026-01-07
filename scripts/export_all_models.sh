#!/usr/bin/env bash
set -euo pipefail

OUT=${1:-exports}
mkdir -p "$OUT"

python3 scripts/export_bark_model.py --model-dir models/suno_bark_model --out-dir "$OUT" || true
python3 scripts/export_musicgen_model.py --model-dir models/facebook_musicgen-small_model --out-dir "$OUT" || true
python3 scripts/export_flux_model.py --model-dir models/segmind_tiny-sd_model --out-dir "$OUT" || true

echo "Exports attempted; inspect $OUT for produced artifacts"