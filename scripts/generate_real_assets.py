#!/usr/bin/env python3
import os
import sys
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
OUT_BASE = ROOT / "assets_real"
OUT_BASE.mkdir(exist_ok=True)

# local model dirs
TINY_SD = MODELS / "segmind_tiny-sd_model"
MUSICGEN = MODELS / "facebook_musicgen-small_model"
MMS_TTS = MODELS / "facebook_mms-tts-eng_model"
SD_LEGACY = ROOT.joinpath('.cache','huggingface','hub','models--sd-legacy--stable-diffusion-v1-5') if False else None

print('Models:')
print(' TINY_SD:', TINY_SD.exists())
print(' MUSICGEN:', MUSICGEN.exists())
print(' MMS_TTS:', MMS_TTS.exists())

# Load movie definitions from generate_absurd_trailers if available
try:
    from generate_absurd_trailers import MOVIES
except Exception:
    MOVIES = [
        {
            'id':'test', 'title':'Test Trailer', 'vo':'This is a short test voiceover.',
            'scenes':["Photorealistic cinematic shot of a living room, 8k.", "Photorealistic close-up of a banana, 8k."]
        }
    ]

# Image generation using diffusers (tiny-sd should be a diffusers compatible checkpoint)
try:
    from diffusers import DiffusionPipeline
    import torch
    has_diffusers = True
except Exception as e:
    print('diffusers not available:', e)
    has_diffusers = False

# TTS using transformers pipeline
try:
    from transformers import pipeline
    has_transformers = True
except Exception as e:
    print('transformers not available:', e)
    has_transformers = False

for movie in MOVIES:
    out_dir = OUT_BASE / f"assets_{movie['id']}"
    imgs_dir = out_dir / 'images'
    voice_dir = out_dir / 'voice'
    music_dir = out_dir / 'music'
    imgs_dir.mkdir(parents=True, exist_ok=True)
    voice_dir.mkdir(parents=True, exist_ok=True)
    music_dir.mkdir(parents=True, exist_ok=True)

    # Images
    if has_diffusers and TINY_SD.exists():
        try:
            print('Loading tiny-sd pipeline from', TINY_SD)
            pipe = DiffusionPipeline.from_pretrained(str(TINY_SD))
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            pipe.to(device)
            for i, prompt in enumerate(movie.get('scenes', [])):
                out_path = imgs_dir / f'clip_{i}.png'
                if out_path.exists():
                    print('Skipping existing', out_path)
                    continue
                print('Generating image for:', prompt)
                image = pipe(prompt=prompt, guidance_scale=7.5, num_inference_steps=20).images[0]
                image.save(out_path)
                print('Saved', out_path)
            del pipe
        except Exception as e:
            print('Image generation failed:', e)
    else:
        print('Skipping image generation; diffusers or tiny-sd missing')

    # Voiceover
    full_text = movie.get('vo', '')
    if has_transformers and MMS_TTS.exists():
        try:
            print('Using transformers TTS pipeline with', MMS_TTS)
            tts = pipeline('text-to-speech', model=str(MMS_TTS))
            out_wav = voice_dir / 'voiceover_full.wav'
            if not out_wav.exists():
                print('Generating TTS (may be slow)')
                audio = tts(full_text)
                # pipeline returns dict with 'audio' as array or bytes depending on model
                if isinstance(audio, dict) and 'audio' in audio:
                    data = audio['audio']
                    # write bytes
                    if isinstance(data, (bytes, bytearray)):
                        with open(out_wav, 'wb') as f:
                            f.write(data)
                    else:
                        # assume numpy array float32
                        import soundfile as sf
                        sf.write(str(out_wav), data, 22050)
                else:
                    # fallback: try writing audio['wav']
                    print('Unexpected TTS output:', type(audio))
            else:
                print('Skipping existing TTS', out_wav)
        except Exception as e:
            print('TTS generation failed:', e)
    else:
        print('Skipping TTS; transformers or MMS_TTS model missing')

    # Music (musicgen) - best-effort using transformers or skip
    try:
        # try to use musicgen via transformers if available
        from transformers import AutoProcessor, AutoModel
        if MUSICGEN.exists():
            print('Musicgen model detected; attempting stub generation')
            # MusicGen integration is non-trivial; skipping heavy inference here
            print('Skipping musicgen heavy inference.');
        else:
            print('Musicgen model not found; skipping music')
    except Exception as e:
        print('Music generation unavailable:', e)

print('Done. Assets in', OUT_BASE)
