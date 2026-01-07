#!/usr/bin/env python3
"""Generate stub assets (image and WAVs) from token id text files without downloading models."""
import os
import math
import wave
import struct

ROOT = os.path.dirname(os.path.dirname(__file__))
BARK_TOK = os.path.join(ROOT, 'bark_token_ids.txt')
LDM2_TOK = os.path.join(ROOT, 'ldm2_token_ids.txt')
FLUX_TOK = os.path.join(ROOT, 'flux_token_ids.txt')
OUT_DIR = os.path.join(ROOT, 'assets_stub')
os.makedirs(OUT_DIR, exist_ok=True)

def read_tokens(path):
    try:
        with open(path, 'r') as f:
            return [int(x) for x in f.read().split() if x.strip()]
    except Exception:
        return []

bark = read_tokens(BARK_TOK)
ldm2 = read_tokens(LDM2_TOK)
flux = read_tokens(FLUX_TOK)

print('Token counts -> bark:', len(bark), 'ldm2:', len(ldm2), 'flux:', len(flux))

# Simple WAV generator
def write_sine_wav(path, freq=440.0, duration=3.0, sr=22050, amp=0.5):
    n = int(duration * sr)
    with wave.open(path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        for i in range(n):
            t = i / sr
            v = amp * math.sin(2 * math.pi * freq * t)
            iv = int(max(-32767, min(32767, int(v * 32767))))
            wf.writeframes(struct.pack('<h', iv))

# Generate bark voice stub
if bark:
    dur = max(2.0, min(8.0, len(bark) * 0.2))
    freq = 220.0 + (sum(bark) % 2000) / (len(bark) + 1)
else:
    dur = 3.0
    freq = 440.0
bark_out = os.path.join(OUT_DIR, 'bark_voice.wav')
print('Writing', bark_out, 'freq', freq, 'dur', dur)
write_sine_wav(bark_out, freq=freq, duration=dur, sr=22050, amp=0.5)

# Generate ldm2 music stub
if ldm2:
    dur2 = max(4.0, min(12.0, len(ldm2) * 0.3))
    freq2 = 110.0 + (sum(ldm2) % 800) / (len(ldm2) + 1)
else:
    dur2 = 6.0
    freq2 = 330.0
music_out = os.path.join(OUT_DIR, 'ldm2_music.wav')
print('Writing', music_out, 'freq', freq2, 'dur', dur2)
write_sine_wav(music_out, freq=freq2, duration=dur2, sr=44100, amp=0.4)

# Generate flux image stub (PPM)
def write_ppm(path, side, rgb_data):
    with open(path, 'wb') as f:
        f.write(f'P6\n{side} {side}\n255\n'.encode('ascii'))
        f.write(bytes(rgb_data))

if flux:
    side = max(64, min(256, len(flux) * 8))
    side = int(min(side, 256))
else:
    side = 128
print('Writing image side', side)
# create simple gradient
rgb = bytearray()
for y in range(side):
    for x in range(side):
        r = int(255 * x / max(1, side-1))
        g = int(255 * y / max(1, side-1))
        b = int(255 * ((x+y)/(2*max(1, side-1))))
        rgb.extend([r,g,b])
img_out = os.path.join(OUT_DIR, 'flux_image.ppm')
write_ppm(img_out, side, rgb)

print('Stub assets written to', OUT_DIR)
