# Dalek Comes Home - Model & Asset Brain

This document consolidates the current state of knowledge regarding AI models, image generation, voiceovers, and audio assets within the project.

## 🎙️ Voiceover & TTS (Text-to-Speech)

### Current Standard: Parler-TTS
For long-form, atmospheric voiceovers, we utilize **Parler-TTS** due to its description-based control system.

- **Model:** `parler-tts/parler-tts-mini-v1`.
- **Primary Script:** `generate_chimp_train_voiceover_parler.py`.
- **Key Technique:** 
    - **Pacing Control:** We use phrases like *"speaks very slowly"* and *"long pauses"* in the description prompt to naturally extend speech duration.
    - **Padding:** Since Parler may not reach a full 60s per sentence, we use `numpy` to pad the generated arrays with silence to hit exact timing requirements.
- **Environment:** Requires **Python 3.13** (Python 3.14 is currently incompatible with `tokenizers` build requirements).
- **Previous Tooling:** ChatTTS (implemented in `generate_chimp_train_voiceover.py`) was used but lacks the fine-grained pacing control found in Parler.

## 🖼️ Image Generation

### Workflow: SDXL (Stable Diffusion XL)
We use SDXL for both rapid prototyping and high-fidelity branding assets.

- **Model:** `stabilityai/stable-diffusion-xl-base-1.0`.
- **Fast Mode (Lightning):** Uses `ByteDance/SDXL-Lightning` (distilled weights) with 8 steps and 0 guidance for rapid scene generation (`generate_chimp_train_images.py`).
- **High-Quality Mode:** Uses the base model with **128 iterations** and a guidance scale of 7.5 for critical assets like title cards and studio logos (`generate_chimp_train_branding.py`).
- **Hardware Optimization:** 
    - **MPS (Metal Performance Shaders):** Optimized for macOS GPU execution.
    - **Attention Slicing:** Enabled for high-iteration counts to manage memory on Mac systems.
    - **VAE Upcasting:** To avoid precision issues and follow `diffusers` deprecation warnings, always upcast the VAE to `float32` via `pipe.vae.to(torch.float32)` even when the rest of the pipeline uses `float16`.

### Branding Assets
- **Title Card:** "CHIMP BANANA TRAIN" (Embossed gold on dark wood).
- **Studio Logo:** "UNIVERSAL CHIMP STUDIOS" (Golden chimp silhouette).
- **Styling:** "Minimalist style, cinematic lighting, no humans, only animals, 8k photorealistic."

## 🎵 Audio, SFX & Music

### Assets Structure
- **Music/SFX:** Stored in `assets_chimp_train/music` and `assets_chimp_train/sfx`.
- **Model Reference:** `facebook_musicgen-small_model` is identified in the directory structure for generating background tracks.

## 🛠️ Technical Infrastructure

### Execution Environment
- **Python Version:** 3.13 is the stable target for modern transformer models in this repo.
- **Virtual Env:** `.venv_parler` for the Parler-specific stack.
- **Hardware Acceleration:** Defaulting to `mps` on Darwin (macOS) and `cuda` on Linux.

### Model Formats & Conversion
- **GGUF Support:** The project contains logic for GGUF conversion, handling model architectures, tensor mapping, and vocabulary (GPT-2 BPE, SentencePiece).
- **SafeTensors:** Preferred format for loading UNet and VAE weights.

---
*Last Updated: January 8, 2026*
