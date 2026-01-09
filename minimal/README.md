Minimal renderer package

Purpose:
- Fetch small models (tiny-sd, musicgen-small, mms-tts-eng)
- Use a precompiled C++ binary (CUDA/CPU/MPS) to render images, voice, and music

Layout:
- bin/                      -> place precompiled binaries here
  - generate_trailer_assets_full_cuda
  - generate_trailer_assets_full_cpu
  - generate_trailer_assets_full_mps
- minimal/
  - run.sh                 -> orchestrates fetching models and running binary
  - fetch_models.sh        -> downloads small models via huggingface_hub
  - Makefile               -> convenience wrapper

Usage:
1. Ensure you have a precompiled binary in bin/ matching your backend (or build it locally using Makefile at repo root).
2. From the repo root:
   cd minimal && ./run.sh

Notes:
- The script will try to build the C++ binary via `make build-cpp` if a precompiled binary is not found.
- Adjust model choices in fetch_models.sh if you prefer different light-weight models.
- For production, provide properly compiled binaries for each platform (CUDA / CPU / MPS).
