Exporting models to TorchScript and ONNX

This repository includes helper scripts to attempt exporting three models (Bark TTS, MusicGen/LDM2, and FLUX/tiny-sd) to TorchScript and ONNX.

Scripts:
- scripts/export_bark_model.py --model-dir <path> --out-dir <outdir>
- scripts/export_musicgen_model.py --model-dir <path> --out-dir <outdir>
- scripts/export_flux_model.py --model-dir <path> --out-dir <outdir>
- scripts/export_all_models.sh <outdir>

Notes and requirements:
- Exports are heavy and require a Python environment with torch, diffusers, transformers, and adequate disk/RAM.
- Exporting diffusion models (UNet/VAE) and MusicGen requires model-specific handling; the provided scripts are best-effort scaffolds.
- Prefer running these on a machine with CUDA and matching PyTorch+CUDA build to avoid CPU-only slowness.

Run example:
  python3 -m venv .venv
  . .venv/bin/activate
  pip install torch diffusers transformers scipy soundfile
  bash scripts/export_all_models.sh exports

The scripts will attempt to save files under the `exports/` directory.
