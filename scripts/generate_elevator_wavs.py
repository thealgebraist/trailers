#!/usr/bin/env python3
"""
Generate four 10s elevator WAVs for subject cards.
Preferred: Suno (torch). Fallbacks: music21+FluidSynth, simple synth.
Writes files to assets_exfoliate_positive/subject_{idx:02d}_elevator.wav for indices [15,31,47,63].
"""
from pathlib import Path
import numpy as np
import scipy.io.wavfile

OUT_DIR = Path("assets_exfoliate_positive")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUBJECT_INDICES = [15, 31, 47, 63]
DEFAULT_SR = 32000

def generate_elevator_music(outfile: Path, duration: float = 10.0, sr: int = 22050):
    import math
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # chord progression: two slow triads alternating
    chords = [
        [261.63, 329.63, 392.00],  # C major
        [220.00, 277.18, 329.63],  # A minor-ish
    ]
    audio = np.zeros_like(t)
    seg = duration / len(chords)
    for i, chord in enumerate(chords):
        start = int(i * seg * sr)
        end = int((i + 1) * seg * sr)
        for f in chord:
            # gentle sine with slow envelope
            env = np.linspace(0, 1, end - start)
            audio[start:end] += 0.2 * env * np.sin(2 * math.pi * f * t[start:end])
    # gentle low-pass by simple smoothing (moving average)
    kernel = np.ones(5) / 5.0
    audio = np.convolve(audio, kernel, mode='same')
    audio = audio / np.max(np.abs(audio) + 1e-9)
    audio_int16 = (audio * 16000).astype(np.int16)
    scipy.io.wavfile.write(outfile, sr, audio_int16)


def generate_elevator_music_music21(outfile: Path, duration: float = 10.0) -> bool:
    try:
        import tempfile, os, glob, subprocess, shutil
        from music21 import stream, chord, tempo, instrument, midi
        s = stream.Stream()
        s.append(tempo.MetronomeMark(number=60))
        s.append(instrument.ElectricPiano())
        chord_notes = [["C4","E4","G4"],["A3","C4","E4"],["F3","A3","C4"],["G3","B3","D4"]]
        seg = max(1, int(duration / len(chord_notes)))
        for cn in chord_notes:
            c = chord.Chord(cn)
            c.quarterLength = seg * 1.0
            s.append(c)
        with tempfile.TemporaryDirectory() as td:
            midi_path = os.path.join(td, 'elevator.mid')
            mf = midi.translate.streamToMidiFile(s)
            mf.open(midi_path, 'wb')
            mf.write()
            mf.close()
            sf2 = os.environ.get('MUSIC_SF2')
            candidates = [sf2] if sf2 else []
            candidates += [
                '/usr/share/sounds/sf2/FluidR3_GM.sf2',
                '/usr/local/share/sounds/sf2/FluidR3_GM.sf2',
                '/Library/Audio/Sounds/Banks/FluidR3_GM.sf2'
            ]
            candidates += glob.glob('/usr/share/sounds/**/*.sf2', recursive=True)
            found = None
            for c in candidates:
                if c and os.path.exists(c):
                    found = c
                    break
            if not found:
                print('FluidSynth soundfont not found; music21->FluidSynth rendering unavailable')
                return False
            wav_tmp = os.path.join(td, 'elevator.wav')
            cmd = ['fluidsynth', '-ni', found, midi_path, '-F', wav_tmp, '-r', '22050']
            res = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode != 0 or not os.path.exists(wav_tmp):
                print(f'FluidSynth render failed: {res.stderr.decode()[:200]}')
                return False
            shutil.move(wav_tmp, str(outfile))
        return True
    except Exception as e:
        print(f"music21+FluidSynth generation failed: {e}")
        return False


def generate_elevator_music_model(outfile: Path, duration: float = 10.0) -> bool:
    """Prefer Suno (torch) for generation, fall back to MusicGen, then music21, then synth."""
    # Try Suno
    try:
        import soundfile as sf
        # Try to use facebook/MusicGen via a local MusicGen wrapper if available
        try:
            from musicgen import MusicGen
            model = MusicGen.from_pretrained('facebook/musicgen-small')
            prompt = 'boring elevator music, mellow piano, soft pad, slow tempo, unobtrusive background music'
            wav = model.generate(prompt, duration=duration)
            if isinstance(wav, tuple):
                wav, sr = wav
            else:
                sr = getattr(model, 'sample_rate', 32000)
            sf.write(outfile, wav, sr)
            return True
        except Exception:
            # try an alternate import path
            from transformers import AutoModel
            # If transformers path works, user should implement specific loader; raise to fallback
            raise
    except Exception as e:
        print(f"MusicGen (facebook) generation failed: {e}")
    # Fallback: music21+FluidSynth
    if generate_elevator_music_music21(outfile, duration=duration):
        return True
    # Last resort: synth
    generate_elevator_music(outfile, duration=duration)
    return True


if __name__ == '__main__':
    for i, sidx in enumerate(SUBJECT_INDICES, start=1):
        out = OUT_DIR / f"subject_{sidx:02d}_elevator.wav"
        print(f"Generating elevator music for subject index {sidx} -> {out}")
        # try Suno first (generate_elevator_music_model prefers Suno)
        if generate_elevator_music_model(out, duration=10.0):
            continue
        # fallbacks handled inside generate_elevator_music_model
        # ensure at least synth exists
        if not out.exists():
            generate_elevator_music(out, duration=10.0)
