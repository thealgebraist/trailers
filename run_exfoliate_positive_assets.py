#!/usr/bin/env python3
"""
Generate positive EXFOLIATE assets:
- Reads prompts from assets_exfoliate_positive/prompts_positive.csv (creates via generate_exfoliate_positive_assets.py if missing).
- Generates a title card per affliction (black background, white 8-word caption).
- Generates full-body and close-up images with sd-cli (SD 1.5 GGUF).
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

SD_CLI = Path("/tmp/stable-diffusion.cpp/build/bin/sd-cli")
SD_MODEL = Path("/Users/anders/models/sd/stable-diffusion-v1-5-Q8_0.gguf")
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

def run_sd(prompt: str, outfile: Path, seed: int):
    """Try to use diffusers StableDiffusionPipeline (v1.5); fall back to sd-cli if unavailable."""
    global SD_PIPELINE
    device = get_device()
    try:
        # lazy import to avoid heavy dependency unless used
        from diffusers import StableDiffusionPipeline
        import torch as _torch

        if SD_PIPELINE is None:
            SD_PIPELINE = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=_torch.float16,
            )
            SD_PIPELINE = SD_PIPELINE.to(device)

        gen = _torch.Generator(device=device if isinstance(device, (str, _torch.device)) else device).manual_seed(int(seed))
        result = SD_PIPELINE(prompt, guidance_scale=7.0, num_inference_steps=22, generator=gen)
        image = result.images[0]
        image.save(outfile)
        return
    except Exception as e:
        print(f"Python SD API failed ({e}), falling back to sd-cli")

    # fallback to sd-cli binary
    cmd = [
        str(SD_CLI),
        "-m", str(SD_MODEL),
        "-p", prompt,
        "-o", str(outfile),
        "--width", "768",
        "--height", "768",
        "--steps", "22",
        "--cfg-scale", "7",
        "--seed", str(seed),
        "-t", "-1",
        "--vae-tiling",
        "--clip-on-cpu",
    ]
    subprocess.run(cmd, check=True)


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

    if not SD_CLI.exists():
        raise SystemExit(f"sd-cli not found at {SD_CLI}")
    if not SD_MODEL.exists():
        raise SystemExit(f"GGUF model not found at {SD_MODEL}")

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

    if not intro_card.exists():
        print("[intro] title card")
        make_title_card("Introducing: Doctor", intro_card)

    if not intro_face.exists():
        print("[intro] doctor face")
        prompt_face = (
            "Close-up portrait of a weird looking doctor with intense eyes, textured skin, \"odd\" expression, "
            "wearing a white gown, cinematic clinical lighting, gritty 35mm, high detail"
        )
        run_sd(prompt_face, intro_face, seed=INTRO_DOCTOR_SEEDS["face"])

    if not intro_body.exists():
        print("[intro] doctor body")
        prompt_body = (
            "Full body portrait of a weird looking doctor in a white gown, standing formally, neutral clinical background, "
            "gritty 35mm, high detail"
        )
        run_sd(prompt_body, intro_body, seed=INTRO_DOCTOR_SEEDS["body"])

    if not intro_sleep.exists():
        print("[intro] doctor sleeping")
        prompt_sleep = (
            "Weird doctor in a white gown sleeping on a cot, gown rumpled, quiet mood, soft clinical lighting, high detail"
        )
        run_sd(prompt_sleep, intro_sleep, seed=INTRO_DOCTOR_SEEDS["sleep"])

    if not intro_shower.exists():
        print("[intro] doctor showering")
        prompt_shower = (
            "Weird doctor in a white gown showering, water through the gown, surreal clinical scene, high detail"
        )
        run_sd(prompt_shower, intro_shower, seed=INTRO_DOCTOR_SEEDS["shower"])

    if not intro_voice.exists():
        print("[intro] doctor voice (MMS)")
        mms_tts(model, tokenizer, DOCTOR_VOICE_TEXT, intro_voice)

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


if __name__ == "__main__":
    main()
