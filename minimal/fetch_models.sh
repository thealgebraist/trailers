#!/usr/bin/env bash
set -euo pipefail

DEST=${1:-$(pwd)/models}
mkdir -p "$DEST"
cd "$DEST"
python3 - <<'PY'
from huggingface_hub import snapshot_download
models = {
    'tiny-sd':'segmind/tiny-sd',
    'musicgen_small':'facebook/musicgen-small',
    'mms_tts':'facebook/mms-tts-eng'
}
for name, repo in models.items():
    print(f"Downloading {repo} into {name}_model...")
    try:
        path = snapshot_download(repo_id=repo, cache_dir=None, local_dir=f"{name}_model", local_dir_use_symlinks=False)
        print(f"Saved {repo} -> {path}")
    except Exception as e:
        print(f"Failed to download {repo}: {e}")
PY
