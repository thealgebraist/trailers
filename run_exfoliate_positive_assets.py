#!/usr/bin/env python3
"""
Generate positive EXFOLIATE assets:
- Reads prompts from assets_exfoliate_positive/prompts_positive.csv (creates via generate_exfoliate_positive_assets.py if missing).
- Generates a title card per affliction (black background, white 8-word caption).
- Generates full-body and close-up images using Stable Diffusion v1.5 via diffusers.
- Synthesizes voice lines with MMS TTS (facebook/mms-tts-eng).
"""

import csv
import subprocess
from pathlib import Path

import torch
import numpy as np
import scipy.io.wavfile
from PIL import Image, ImageDraw, ImageFont
from transformers import VitsModel, AutoTokenizer

from dalek.core import get_device, ensure_dir, flush

PROMPT_CSV = Path("assets_exfoliate_positive/prompts_positive.csv")
OUTPUT_DIR = Path("assets_exfoliate_positive")
IMG_DIR = OUTPUT_DIR / "images"
VOICE_DIR = OUTPUT_DIR / "voice"
CARD_DIR = OUTPUT_DIR / "cards"
INTRO_DOCTOR_SEEDS = {
    "card": 8000,
    "face": 8001,
    "body": 8002,
    "sleep": 8003,
    "shower": 8004,
}
DEFAULT_SD_STEPS = 32
DOCTOR_VOICE_TEXT = (
    "Doctor car tree murmurs wobbling sideways lavender gears, "
    "glimmering verbs twirl quietly under humming clouds."
)

CAPTIONS = [
    "Calmly observing phantom itching on left shoulder easing",
    "Chill bumps rash on right shoulder steadily settling",
    "Upper back nettle scars fading with calm attention",
    "Lower back wire-brush dermatitis cooling, margins stay contained",
    "Chest coal dust staining lightening, breathing stays easy",
    "Abdomen patchy scaling calming as circulation remains steady",
    "Left thigh rope burn marks softening with rest",
    "Right thigh windburn striations easing under moisture regimen",
    "Left shin trench foot drying; comfort steadily returning",
    "Right shin chalky dry flare contained, color healthier",
    "Left calf pinprick bruising steady, tone remains even",
    "Right calf sun-lamp mottling stabilizing; nearby skin calm",
    "Neck barber's rash receding, comfort measured and stable",
    "Nape cold snap fissures smoothing, resilience holding well",
    "Scalp salt-crust clearing, sensitivity remains unchanged and mild",
    "Left forearm ink-stain spotting fading; margins stay clean",
    "Right forearm ropey muscle knots relaxing, irritation easing",
    "Left upper arm granular dryness calming, surface stable",
    "Right upper arm grain dust rash clearing steadily",
    "Left hand wood splinter freckles calming; comfort unchanged",
    "Right hand tool-handle callus softening, impact remains minor",
    "Left palm lime veil lifting, boundary remains clear",
    "Right palm paper lattice quiet, dryness under control",
    "Left fingers dye blotches stable, site staying calm",
    "Right fingers belt-line chafe easing; warmth is reducing",
    "Hip soot shadowing lightening, pattern remaining consistent today",
    "Waist shear-wool scratches smoothing, texture feels gentle now",
    "Lower leg ankle harness rub calming; hue steady",
    "Ankle boot welt impressions flattening, no secondary signs",
    "Foot silt-caked toes improving, area remains manageable today",
    "Toes linen-rub redness steady, comfort holds firm today",
    "Inner thigh outer seam abrasions mild, irritation modest",
    "Outer thigh saddle sore patch calming, change only",
    "Buttock grindstone sheen fading, scope stays limited today",
    "Lower back radiator heat blush evening out nicely",
    "Upper chest soot-smudged ribs clearing, discomfort not worsening",
    "Sternum binder strap lines softening, presentation stays tight",
    "Left ribcage old fall bruises lightening, symptoms contained",
    "Right ribcage ridge-plank scars quiet, swelling remains slight",
    "Elbow hook-lift marks calm, boundary stays unchanged today",
    "Knee chalk pit dusting steady, tone remains even",
    "Shoulder blade left knotty bursitis easing, locus contained",
    "Shoulder blade right heel fissures softening, resilience present",
    "Calf left arch strain crease mellowing, status orderly",
    "Calf right mild rope burn settling, pattern stable",
    "Heel soot glaze thinning, area stays defined today",
    "Foot arch brick dust specks fading, no trend",
    "Groin area cinder freckles quiet, margins intact today",
    "Upper back cold-room pallor resting, condition staying quiet",
    "Lower abdomen steam-room flush calming, findings remain limited",
    "Side torso left sawtooth scrape improving, impact narrow",
    "Right torso tourniquet band softening, no spread detected",
    "Wrist glove line dryness easing, region stays focused",
    "Thumb pad polish holding steady, tissue looks steady",
    "Index finger nick grid quiet, comfort reports neutral",
    "Ring finger groove redness mild, contained and calm",
    "Back of hand scuff settling, surface looks orderly",
    "Collarbone rub line softening, state remains constrained today",
    "Behind knee popliteal damp rash easing, change slight",
    "Back of neck tensed cords calming, situation contained",
    "Jawline clench tightness loosening, no added complications today",
    "Temple pulse hue steady, boundaries holding firm today",
    "Solar plexus pallor lifting, stability clearly seen today",
    "Hipbone bruises lightening, focal irritation only today noted",
]

if len(CAPTIONS) != 64:
    raise ValueError(f"Expected 64 captions, found {len(CAPTIONS)}")


def ensure_prompts():
    if PROMPT_CSV.exists():
        return
    subprocess.run(["python3", "generate_exfoliate_positive_assets.py"], check=True)


SD_PIPELINE = None

def run_sd(prompt: str, outfile: Path, seed: int, steps: int = None):
    """Use diffusers StableDiffusionPipeline (v1.5) to generate an image.

    steps: override default number of inference steps (DEFAULT_SD_STEPS).
    """
    global SD_PIPELINE
    device = get_device()
    try:
        from diffusers import StableDiffusionPipeline
        import torch as _torch
        import numpy as _np

        if SD_PIPELINE is None:
            SD_PIPELINE = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=_torch.float16,
            )
            SD_PIPELINE = SD_PIPELINE.to(device)

        use_steps = int(steps) if steps is not None else DEFAULT_SD_STEPS

        # Retry logic: sanitize prompts for risky scenes and retry up to max_attempts
        import re
        def sanitize_prompt(p: str) -> str:
            # replace showering with a clearly clothed description
            p = re.sub(r'\bshower(?:ing)?\b', 'standing under running water, gown intact', p, flags=re.I)
            # replace sleeping with non-sexual resting phrasing
            p = re.sub(r'\bsleep(?:ing)?\b', 'resting on a cot, gown fully on', p, flags=re.I)
            # ensure clothing modifiers present
            if not re.search(r'no nudity|fully clothed|modest attire|covered', p, flags=re.I):
                p = p + ', fully clothed, no nudity, modest attire'
            return p

        attempts = 0
        max_attempts = 3
        curr_prompt = prompt
        # pre-sanitize prompts that mention doctor/gown/shower/sleep to reduce NSFW hits
        if re.search(r'\bdoctor\b|\bgown\b|\bshower\b|\bsleep\b', curr_prompt, flags=re.I):
            curr_prompt = sanitize_prompt(curr_prompt)
        curr_seed = int(seed)
        while attempts < max_attempts:
            gen = _torch.Generator(device=device if isinstance(device, _torch.device) else None).manual_seed(curr_seed + attempts)
            result = SD_PIPELINE(curr_prompt, guidance_scale=7.0, num_inference_steps=use_steps, generator=gen)
            images = result.images
            nsfw_flag = getattr(result, "nsfw_content_detected", None)
            image = images[0]
            # check for NSFW flag
            if nsfw_flag is not None and any(nsfw_flag):
                print(f"NSFW detected for prompt (attempt {attempts+1}), sanitizing and retrying")
                curr_prompt = sanitize_prompt(curr_prompt)
                attempts += 1
                continue
            # check for black image
            arr = _np.array(image.convert("L"))
            if arr.mean() < 8:
                print(f"Rendered image too dark/blank (mean {arr.mean():.1f}) on attempt {attempts+1}), sanitizing and retrying")
                curr_prompt = sanitize_prompt(curr_prompt)
                attempts += 1
                continue
            # success
            image.save(outfile)
            return
        # exhausted attempts; save last image and warn
        print("Warning: image generation produced flagged or dark image after retries; saving last result")
        image.save(outfile)
        return
    except Exception as e:
        raise SystemExit(f"Stable Diffusion Python API failed: {e}. Install diffusers and model weights.")


def load_prompts():
    rows = []
    with PROMPT_CSV.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def mms_tts(model, tokenizer, text: str, outfile: Path):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        wav = model(**inputs).waveform
    audio = wav.cpu().numpy().squeeze()
    sr = model.config.sampling_rate
    audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    scipy.io.wavfile.write(outfile, sr, audio_int16)


# Simple boring elevator music generator (synthesized chords)
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


# Try RAVE-based generation, then music21+FluidSynth, then synth fallback

def generate_elevator_music_rave(outfile: Path, duration: float = 10.0) -> bool:
    try:
        # Best-effort attempt to use a RAVE Python API if installed.
        # This is a best-effort stub: if a local RAVE implementation is available it should expose
        # a simple generation API; otherwise this will gracefully fail and return False.
        import numpy as _np
        try:
            # common package name attempts
            import rave
        except Exception:
            import rave_pytorch as rave
        # Hypothetical API: rave.generate(duration)
        wav, sr = rave.generate(duration=duration)
        # write if API returns arrays
        import soundfile as sf
        sf.write(outfile, wav, sr)
        return True
    except Exception as e:
        print(f"RAVE generation unavailable or failed: {e}")
        return False


def generate_elevator_music_music21(outfile: Path, duration: float = 10.0) -> bool:
    try:
        import tempfile
        import os
        import glob
        from music21 import stream, note, chord, tempo, instrument, midi
        # build a simple progression
        s = stream.Stream()
        s.append(tempo.MetronomeMark(number=60))
        s.append(instrument.ElectricPiano())
        # 4 chords over duration
        chord_notes = [["C4","E4","G4"],["A3","C4","E4"],["F3","A3","C4"],["G3","B3","D4"]]
        seg = max(1, int(duration / len(chord_notes)))
        for cn in chord_notes:
            c = chord.Chord(cn)
            c.quarterLength = seg * 1.0
            s.append(c)
        # write to temp midi
        with tempfile.TemporaryDirectory() as td:
            midi_path = os.path.join(td, 'elevator.mid')
            mf = midi.translate.streamToMidiFile(s)
            mf.open(midi_path, 'wb')
            mf.write()
            mf.close()
            # locate a soundfont
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
            # call fluidsynth to render wav
            wav_tmp = os.path.join(td, 'elevator.wav')
            cmd = ['fluidsynth', '-ni', found, midi_path, '-F', wav_tmp, '-r', '22050']
            import subprocess as _sub
            res = _sub.run(cmd, check=False, stdout=_sub.PIPE, stderr=_sub.PIPE)
            if res.returncode != 0 or not os.path.exists(wav_tmp):
                print(f'FluidSynth render failed: {res.stderr.decode()[:200]}')
                return False
            # move to outfile
            import shutil
            shutil.move(wav_tmp, str(outfile))
        return True
    except Exception as e:
        print(f"music21+FluidSynth generation failed: {e}")
        return False


def generate_elevator_music_model(outfile: Path, duration: float = 10.0) -> bool:
    """Prefer Suno (torch) for generation, fall back to music21+FluidSynth then synth.
    Returns True on success, False otherwise.
    """
    try:
        import soundfile as sf
        import torch
        import suno
        prompt = 'boring elevator music, mellow piano, soft pad, slow tempo, unobtrusive background music'
        # Try common Suno patterns
        if hasattr(suno, 'generate_audio'):
            wav, sr = suno.generate_audio(prompt=prompt, duration=duration)
        elif hasattr(suno, 'music') and hasattr(suno.music, 'generate'):
            wav = suno.music.generate(prompt=prompt, duration=duration)
            sr = getattr(wav, 'sample_rate', 32000)
            if isinstance(wav, tuple):
                wav, sr = wav
        else:
            gen = getattr(suno, 'AudioEngine', None)
            if gen is not None:
                engine = gen()
                wav = engine.generate(prompt=prompt, duration=duration)
                sr = getattr(engine, 'sample_rate', 32000)
            else:
                raise RuntimeError('Unknown Suno API')
        sf.write(outfile, wav, sr)
        return True
    except Exception as e:
        print(f"Suno generation failed: {e}")
    # Fallback: music21+FluidSynth
    if generate_elevator_music_music21(outfile, duration=duration):
        return True
    # Last resort: synth
    generate_elevator_music(outfile, duration=duration)
    return True


def eight_word_caption(idx: int, affliction: str, area: str) -> str:
    caption = CAPTIONS[idx]
    words = caption.split()
    if len(words) != 8:
        raise ValueError(f"Caption for id {idx} is not 8 words: '{caption}'")
    return " ".join(words)


def make_title_card(text: str, outfile: Path, size=(1024, 512)):
    if outfile.exists():
        return
    scale = 8
    render_size = (size[0] * scale, size[1] * scale)
    img = Image.new("RGB", render_size, "black")
    draw = ImageDraw.Draw(img)

    def load_font(sz: int):
        for name in ("Arial.ttf", "DejaVuSans.ttf"):
            try:
                return ImageFont.truetype(name, sz)
            except Exception:
                continue
        return ImageFont.load_default()

    def measure(t: str, f):
        if hasattr(draw, "textbbox"):
            left, top, right, bottom = draw.textbbox((0, 0), t, font=f)
            return right - left, bottom - top
        return draw.textsize(t, font=f)

    max_w, max_h = int(render_size[0] * 0.8), int(render_size[1] * 0.8)
    font_size = 72 * scale
    font = load_font(font_size)
    w, h = measure(text, font)
    while (w > max_w or h > max_h) and font_size > 16 * scale:
        font_size -= 2 * scale
        font = load_font(font_size)
        w, h = measure(text, font)

    x = (render_size[0] - w) // 2
    y = (render_size[1] - h) // 2
    draw.text((x, y), text, fill="white", font=font)
    img = img.resize(size, Image.LANCZOS)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    img.save(outfile)


def main():
    ensure_prompts()
    ensure_dir(IMG_DIR)
    ensure_dir(VOICE_DIR)
    ensure_dir(CARD_DIR)


    rows = load_prompts()

    device = get_device()
    print(f"Loading MMS TTS on {device}")
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
    model = VitsModel.from_pretrained("facebook/mms-tts-eng").to(device)

    # Intro: doctor sequence (card, face, body, sleeping, shower) + weird voice
    intro_card = CARD_DIR / "intro_doctor_card.png"
    intro_face = IMG_DIR / "intro_doctor_face.png"
    intro_body = IMG_DIR / "intro_doctor_body.png"
    intro_sleep = IMG_DIR / "intro_doctor_sleep.png"
    intro_shower = IMG_DIR / "intro_doctor_shower.png"
    intro_voice = VOICE_DIR / "intro_doctor.wav"

    prompt_face = (
        "Close-up portrait of a weird looking doctor with intense eyes, textured skin, \"odd\" expression, "
        "wearing a white gown, cinematic clinical lighting, gritty 35mm, high detail"
    )
    prompt_body = (
        "Full body portrait of a weird looking doctor in a white gown, standing formally, neutral clinical background, "
        "gritty 35mm, high detail"
    )
    prompt_sleep = (
        "Weird doctor in a white gown resting on a cot, gown fully on, quiet mood, soft clinical lighting, high detail"
    )
    prompt_shower = (
        "Weird doctor in a white gown standing under running water, gown intact, surreal clinical scene, high detail"
    )

    if not intro_card.exists():
        print("[intro] title card")
        make_title_card("Introducing: Doctor", intro_card)

    if not intro_face.exists():
        print("[intro] doctor face")
        run_sd(prompt_face, intro_face, seed=INTRO_DOCTOR_SEEDS["face"])

    if not intro_body.exists():
        print("[intro] doctor body")
        run_sd(prompt_body, intro_body, seed=INTRO_DOCTOR_SEEDS["body"])

    if not intro_sleep.exists():
        print("[intro] doctor sleeping")
        run_sd(prompt_sleep, intro_sleep, seed=INTRO_DOCTOR_SEEDS["sleep"])

    if not intro_shower.exists():
        print("[intro] doctor showering")
        run_sd(prompt_shower, intro_shower, seed=INTRO_DOCTOR_SEEDS["shower"])

    if not intro_voice.exists():
        print("[intro] doctor voice (MMS)")
        mms_tts(model, tokenizer, DOCTOR_VOICE_TEXT, intro_voice)

    # subject area cards with boring elevator music at positions 16,32,48,64 (1-based)
    SUBJECT_AREAS = ["Dermatology", "Orthopedics", "Neurology", "General Medicine"]
    SUBJECT_INDICES = [15, 31, 47, 63]

    for s_idx, subj in zip(SUBJECT_INDICES, SUBJECT_AREAS):
        # find corresponding row if exists
        if s_idx < len(rows):
            r = rows[s_idx]
            subj_card = CARD_DIR / f"subject_{s_idx:02d}_card.png"
            subj_music = VOICE_DIR / f"subject_{s_idx:02d}_elevator.wav"
            if not subj_card.exists():
                print(f"[subject {s_idx}] title card: {subj}")
                make_title_card(subj, subj_card)
            if not subj_music.exists():
                print(f"[subject {s_idx}] generating boring elevator music (10s)")
                if not generate_elevator_music_model(subj_music, duration=10.0):
                    generate_elevator_music(subj_music, duration=10.0)

    for row in rows:
        idx = int(row["id"])
        area = row["area"]
        affliction = row["affliction"]
        voice_txt = row["voice"]
        full_body = row["image_full_body"]
        closeup = row["image_closeup"]

        card_caption = eight_word_caption(idx, affliction, area)
        card_path = CARD_DIR / f"scene_{idx:02d}_card.png"
        body_path = IMG_DIR / f"scene_{idx:02d}_body.png"
        close_path = IMG_DIR / f"scene_{idx:02d}_close.png"
        doctor_path = IMG_DIR / f"scene_{idx:02d}_doctor.png"
        voice_path = VOICE_DIR / f"voice_{idx:02d}.wav"

        if not card_path.exists():
            print(f"[{idx:02d}] title card")
            make_title_card(card_caption, card_path)
        if not voice_path.exists():
            print(f"[{idx:02d}] voice (MMS)")
            mms_tts(model, tokenizer, voice_txt, voice_path)
        if not body_path.exists():
            print(f"[{idx:02d}] body image")
            run_sd(full_body, body_path, seed=1000 + idx)
        if not close_path.exists():
            print(f"[{idx:02d}] close image")
            run_sd(closeup, close_path, seed=2000 + idx)
        if not doctor_path.exists():
            print(f"[{idx:02d}] doctor image")
            doctor_prompt = (
                f"Weird looking doctor in a white gown using an odd instrument on the man's {area}, "
                f"addressing {affliction}, clinical lighting, gritty 35mm, neutral background, high detail."
            )
            run_sd(doctor_prompt, doctor_path, seed=3000 + idx)
        flush()

    print("Done generating positive EXFOLIATE assets.")

    # Post-pass: detect black images and regen them with DEFAULT_SD_STEPS (32) using new seeds
    def is_black_image(path: Path, thresh: float = 10.0) -> bool:
        if not path.exists():
            return False
        try:
            im = Image.open(path).convert("L")
            arr = np.array(im)
            return arr.mean() < thresh
        except Exception:
            return False

    print("Scanning for dark/black images to regenerate with 32 steps...")
    for row in rows:
        idx = int(row["id"])
        body_path = IMG_DIR / f"scene_{idx:02d}_body.png"
        close_path = IMG_DIR / f"scene_{idx:02d}_close.png"
        doctor_path = IMG_DIR / f"scene_{idx:02d}_doctor.png"

        if is_black_image(body_path):
            print(f"[{idx:02d}] regenerating dark body image with {DEFAULT_SD_STEPS} steps")
            run_sd(row["image_full_body"], body_path, seed=5000 + idx, steps=DEFAULT_SD_STEPS)
        if is_black_image(close_path):
            print(f"[{idx:02d}] regenerating dark close image with {DEFAULT_SD_STEPS} steps")
            run_sd(row["image_closeup"], close_path, seed=6000 + idx, steps=DEFAULT_SD_STEPS)
        if is_black_image(doctor_path):
            print(f"[{idx:02d}] regenerating dark doctor image with {DEFAULT_SD_STEPS} steps")
            doctor_prompt = (
                f"Weird looking doctor in a white gown using an odd instrument on the man's {row['area']}, "
                f"addressing {row['affliction']}, clinical lighting, gritty 35mm, neutral background, high detail."
            )
            run_sd(doctor_prompt, doctor_path, seed=7000 + idx, steps=DEFAULT_SD_STEPS)

    # Also check intro images
    if is_black_image(intro_face):
        print("[intro] regenerating doctor face with 32 steps")
        run_sd(prompt_face, intro_face, seed=INTRO_DOCTOR_SEEDS["face"] + 100, steps=DEFAULT_SD_STEPS)
    if is_black_image(intro_body):
        print("[intro] regenerating doctor body with 32 steps")
        run_sd(prompt_body, intro_body, seed=INTRO_DOCTOR_SEEDS["body"] + 100, steps=DEFAULT_SD_STEPS)
    if is_black_image(intro_sleep):
        print("[intro] regenerating doctor sleeping with 32 steps")
        run_sd(prompt_sleep, intro_sleep, seed=INTRO_DOCTOR_SEEDS["sleep"] + 100, steps=DEFAULT_SD_STEPS)
    if is_black_image(intro_shower):
        print("[intro] regenerating doctor showering with 32 steps")
        run_sd(prompt_shower, intro_shower, seed=INTRO_DOCTOR_SEEDS["shower"] + 100, steps=DEFAULT_SD_STEPS)


if __name__ == "__main__":
    main()
