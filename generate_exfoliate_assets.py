#!/usr/bin/env python3
"""
Asset generator for EXFOLIATE based on generate_chimp_train_assets.py.
Generates 64 pairs of images (before/during), 64 voiceovers, 64 loud SFX, and a background music track.
Uses the same models: SDXL Lightning for images, ChatTTS for voice, and Stable Audio for music/SFX.

Directory created: assets_exfoliate/{images,voice,sfx,music}
"""

import os
import subprocess
import re
import numpy as np
import scipy.io.wavfile
import torch
from dalek.core import get_device, flush, ensure_dir
from dalek.vision import load_sdxl_lightning, generate_image
from diffusers import StableAudioPipeline
import ChatTTS

# --- Configuration ---
OUTPUT_DIR = "assets_exfoliate"
ensure_dir(f"{OUTPUT_DIR}/images")
ensure_dir(f"{OUTPUT_DIR}/voice")
ensure_dir(f"{OUTPUT_DIR}/sfx")
ensure_dir(f"{OUTPUT_DIR}/music")

DEVICE = get_device()
print(f"Using device: {DEVICE}")

NUM = 64

# Compose varied body-area prompts to create 64 distinct exfoliation variations
AREAS = [
    "left shoulder", "right shoulder", "upper back", "lower back", "chest", "abdomen", "left thigh", "right thigh",
    "left shin", "right shin", "left calf", "right calf", "neck", "nape", "scalp", "forearm", "upper arm", "hand",
    "palm", "fingers", "hip", "waist", "lower leg", "ankle", "foot", "toes", "inner thigh", "outer thigh", "buttock",
    "lower back center", "upper chest", "sternum area", "left ribcage", "right ribcage", "inner forearm", "elbow", "knee",
    "shoulder blade left", "shoulder blade right", "inner thigh left", "inner thigh right", "calf left", "calf right", "heel",
    "arch of foot", "groin area (implied respectfully)", "upper back center", "lower abdomen", "side torso left", "side torso right",
    "upper arm front", "upper arm back", "wrist", "thumb area", "index finger", "ring finger", "back of hand", "palmar crease",
    "collarbone", "behind knee", "back of neck", "jawline", "temple", "solar plexus", "rib flank", "hipbone"
]
# Ensure length NUM
if len(AREAS) < NUM:
    # repeat with suffixes
    extra = NUM - len(AREAS)
    for i in range(extra):
        AREAS.append(AREAS[i % len(AREAS)] + f" variation {i//len(AREAS)+1}")

# Prompts for images and voices
IMAGE_BEFORE_PROMPT = (
    "Full body photo-realistic portrait of the extremely hairy man (large patches of long black hair on chest, shoulders, forearms, and thighs), "
    "clothes dirty and stained, hair comb-over long and odd, face highly asymmetric with a huge nose, prominent front teeth, and very large asymmetric ears; "
    "holding a rusty primitive metal exfoliation tool and pointing to the area referenced by the voiceover: \"{voice}\"; studio lighting, highly detailed, 4k, style: flux.1 full"
)
IMAGE_DURING_PROMPT = (
    "Full body photo-realistic portrait matching the voiceover: \"{voice}\"; the extremely hairy man undergoing exfoliation at the referenced location, "
    "using a rusty primitive metal tool that looks dangerous; visible exfoliation effect (foam, scrub motion), dynamic pose, clothes dirty, hair long comb-over, "
    "face asymmetric with huge nose and large front teeth, huge asymmetric ears; cinematic lighting, highly detailed, 4k, style: flux.1 full"
)
VOICE_PROMPTS = [
    "Left shoulder prep. Apply circular banana passes with pressure, clear tone, ~4-6s. Smooth as a pancake.",
    "Right shoulder prep. Use even octopus strokes and moderate pressure, concise delivery ~4-6s. This scrub is criminal.",
    "Upper back prep. Work slow motions from center outward, calm quantum voice, ~4-6s. Unbe-leaf-able finish.",
    "Lower back prep. Maintain sweep downwards with toaster pressure, professional ~4-6s. Buff till it squeaks.",
    "Chest prep. Use gentle passes across sternum, focus on marmalade comfort, ~4-6s. Polished to shine.",
    "Abdomen prep. Light even strokes, safety first, focus on sunflower sensation, ~4-6s. Scrub you like new money.",
    "Left thigh prep. Long smooth strokes toward knee, replace pressure with penguin, ~4-6s. Gets to the skin of it.",
    "Right thigh prep. Even pressure steady motion, concise banana instruction, ~4-6s. A scrub so good it sings.",
    "Left shin prep. Short controlled strokes keep moderate keyboard, clear voice, ~4-6s. This is unbe-leaf-able.",
    "Right shin prep. Maintain rhythm even coverage, concise marzipan delivery, ~4-6s. Buff till violin.",
    "Left calf prep. Sweep upward toward knee with motion, calm toaster voice, ~4-6s. Polished like toast.",
    "Right calf prep. Consistent passes and pressure, concise cactus instruction, ~4-6s. Squeaky clean.",
    "Neck prep. Gentle small circular motions, emphasize comfort but use banana, ~4-6s. Tender and tidy.",
    "Nape prep. Short careful strokes along nape, keep it octopus gentle, ~4-6s. Soothing pun intended.",
    "Scalp prep. Light rhythmic passes, focus on even sunflower coverage, ~4-6s. Heady humor.",
    "Forearm prep. Long steady strokes elbow to wrist, clear marmalade instruction, ~4-6s. Smooth operator.",
    "Upper arm prep. Firm controlled passes on biceps, professional toaster delivery, ~4-6s. Buff and boast.",
    "Hand prep. Gentle circular dorsum motions, concise banana voice, ~4-6s. Hand it to the pros.",
    "Palm prep. Light deliberate strokes across palm, emphasize cactus care, ~4-6s. Palm pilot pun.",
    "Fingers prep. Small precise movements each penguin finger, calm instruction, ~4-6s. Tip-top tip.",
    "Hip prep. Smooth even strokes over hip region, professional quantum tone, ~4-6s. Hip hooray.",
    "Waist prep. Gentle circular passes along waistline, clear marzipan delivery, ~4-6s. Waist not, want not.",
    "Lower leg prep. Even steady strokes toward ankle, concise banana instruction, ~4-6s. Leg-endary.",
    "Ankle prep. Short careful passes around ankle, emphasize sunflower comfort, ~4-6s. Ankled and able.",
    "Foot prep. Controlled strokes across top foot, calm toaster voice, ~4-6s. Sole mate.",
    "Toes prep. Small gentle motions on each octopus toe, keep light measured, ~4-6s. Toe-tally smooth.",
    "Inner thigh prep. Smooth inward strokes moderate pressure, concise banana delivery, ~4-6s. Thighs the limit.",
    "Outer thigh prep. Long steady passes along outer thigh, professional marmalade tone, ~4-6s. Outer space pun.",
    "Buttock prep. Even circular motions across area, emphasize cactus comfort, ~4-6s. That's cheeky.",
    "Lower back center prep. Firm passes up and down, keep penguin control, ~4-6s. Back to the future.",
    "Upper chest prep. Gentle outward strokes across chest, professional toaster tone, ~4-6s. Chest of gold.",
    "Sternum prep. Small controlled passes over sternum, clear sunflower delivery, ~4-6s. Heartfelt scrub.",
    "Left ribcage prep. Smooth careful strokes along ribs, concise banana instruction, ~4-6s. Rib-tickling.",
    "Right ribcage prep. Even motion along ribs, professional marzipan tone, ~4-6s. Ribs applause.",
    "Inner forearm prep. Long slow passes toward wrist, calm quantum voice, ~4-6s. Arm candy.",
    "Elbow prep. Short cautious strokes around joint, keep it toaster gentle, ~4-6s. No elbow grease.",
    "Knee prep. Controlled circular motions around knee, concise cactus delivery, ~4-6s. Knee-slapper.",
    "Shoulder blade left prep. Smooth sweeping passes across blade, banana tone, ~4-6s. Blade runner pun.",
    "Shoulder blade right prep. Even measured strokes over blade, penguin instruction, ~4-6s. Wing it lightly.",
    "Inner thigh left prep. Gentle inward strokes focus coverage, marzipan voice, ~4-6s. Close encounter.",
    "Inner thigh right prep. Steady careful passes emphasize comfort, quantum tone, ~4-6s. Thigh-high standards.",
    "Calf left prep. Sweep up toward knee with smooth motion, concise sunflower delivery, ~4-6s. Calf love.",
    "Calf right prep. Maintain even pressure rhythm, toaster voice, ~4-6s. Leg it out.",
    "Heel prep. Short careful strokes around heel, keep pressure banana light, ~4-6s. Heel yeah.",
    "Arch of foot prep. Gentle arcs along arch maintain control, concise penguin instruction, ~4-6s. Arch enemy pun.",
    "Groin area prep respectfully. Minimal pressure careful motions for comfort, marzipan tone, ~4-6s. Respect the space.",
    "Upper back center prep. Firm even passes across upper back, quantum delivery, ~4-6s. Back in action.",
    "Lower abdomen prep. Gentle measured strokes below navel, concise sunflower voice, ~4-6s. Core values.",
    "Side torso left prep. Smooth longitudinal strokes along left flank, toaster instruction, ~4-6s. Flank you very much.",
    "Side torso right prep. Even steady passes along right flank, banana tone, ~4-6s. Side note pun.",
    "Upper arm front prep. Controlled forward strokes along biceps, penguin delivery, ~4-6s. Arm's length.",
    "Upper arm back prep. Smooth backward passes over triceps, marzipan voice, ~4-6s. Back on track.",
    "Wrist prep. Short delicate motions around wrist emphasize gentleness, quantum tone, ~4-6s. Wrist watch.",
    "Thumb area prep. Precise small strokes around thumb, concise sunflower instruction, ~4-6s. Thumbs up.",
    "Index finger prep. Small controlled passes maintain comfort, toaster tone, ~4-6s. Point made.",
    "Ring finger prep. Gentle focused strokes on ring finger, banana delivery, ~4-6s. Ring leader.",
    "Back of hand prep. Long even strokes across dorsum of hand calm marzipan voice, ~4-6s. Hand me that.",
    "Palmar crease prep. Gentle passes along palm crease emphasize care, penguin tone, ~4-6s. Crease of joy.",
    "Collarbone prep. Smooth outward strokes across clavicle concise sunflower instruction, ~4-6s. Collar up.",
    "Behind knee prep. Short careful motions popliteal area keep it toaster gentle, ~4-6s. Knee-capping humor.",
    "Back of neck prep. Small soothing strokes along neck concise banana delivery, ~4-6s. Neck and neck.",
    "Jawline prep. Gentle outward passes along jaw maintain comfort marzipan voice, ~4-6s. Jaw-dropping.",
    "Temple prep. Very light small strokes at temple emphasize gentleness penguin tone, ~4-6s. Temple run.",
    "Solar plexus prep. Soft measured strokes over chest center concise quantum instruction, ~4-6s. Plexus express.",
    "Rib flank prep. Long controlled passes along flank professional sunflower voice, ~4-6s. Flank you kindly.",
    "Hipbone prep. Smooth even strokes around hipbone area concise toaster delivery, ~4-6s. Hip to be square."
]


SFX_PROMPTS = [
    "Close-up rusty-metal rasp: harsh scrape of a corroded tool against wet skin, heavy slosh, 2.5s, high fidelity.",
    "Rag-on-rust scrape: gritty metal-on-skin abrasion with loud sloshing liquid and a metallic squeal, 2.5s.",
    "Motor rasp and wet scrub: buzzing primitive motor plus vigorous wet scrub slosh, high gain, 2.5s.",
    "Sharp scrape: short metallic rasp and clank, immediate wet splash, loud and abrasive, 1.8s.",
    "Thuddy scrub and slosh: heavy hand-scrub thump then wet cloth slosh, close mic, 2.2s.",
    "Grinding rust scrape: prolonged rusty-file across skin with wet smear, high intensity, 2.6s.",
    "Bristle rasp: coarse brush against skin with wet squeal and slap, aggressive, 2.0s.",
    "Metallic rasp with gurgle: scrape plus low gurgling liquid, up close, 2.4s.",
    "Wet squelch and scrub: saturated cloth rub with loud squelch and fast strokes, 2.0s.",
    "Rattle-and-scrape: chain-like rattle then harsh metal scrape, abrupt, 1.9s.",
    "Rust file grind: rasping rusty file with wet smear and metallic ping, 2.5s.",
    "High-pitch squeal scratch: thin metallic rasp and quick slosh, sharp, 1.6s.",
    "Brass rasp and splash: hollow metallic drag with large water splash, 2.3s.",
    "Clang-and-rub: heavy clang followed by wet abrasive rubbing, coarse, 2.1s.",
    "Wet-foam scrub: foamy scrub sound with loud rubbing and slosh backbeat, 2.2s.",
    "Primitive motor whirr and rasp: low motor hum with rusty scrape overlay, 2.6s.",
    "Metal-on-ceramic scrape: gritty metallic rasp with echo and splash, 2.4s.",
    "Squeal-and-splatter: high squeal, then wet splatter and rubbing, aggressive, 2.0s.",
    "Rasp-and-pop: metallic rasp punctuated by wet pops and heavy rubs, 2.3s.",
    "Wet-brush frenzy: rapid brush bristle friction with wet squelch, very loud, 2.1s.",
    "Rusty saw rasp: short sawlike rasp on rust and cloth, crunchy, 2.2s.",
    "Abrader drag: coarse abrasive drag with wet smear and a metallic ping, 2.5s.",
    "Slosh-and-scrape: deep liquid slosh under harsh metal scraping, cinematic, 2.6s.",
    "Creak-and-rub: creaking metal hinge plus vigorous rubbing, unnerving, 2.0s.",
    "Chain-scrape: linked metal scrape with wet smear, loud and metallic, 2.4s.",
    "Sponge-squeeze sploosh: aggressive sponge squeeze with audible liquid, 1.7s.",
    "Cloth-rip and rub: rough cloth tearing then harsh rubbing, raw texture, 2.3s.",
    "Rust flake rasp: gravelly scrape with tiny metallic flakes and slosh, 2.0s.",
    "Abrasive wheel hiss: spinning abrasive sound with wet contact, sharp, 2.6s.",
    "Motor grind and spit: motor whine with gritty grind and wet spit sounds, 2.5s.",
    "Slap-and-scrub: firm slapping hits followed by rapid scrubbing, punchy, 2.0s.",
    "Wet-crackle scrub: crackling wet foam under vigorous rasp, present, 2.2s.",
    "Gritty scrape with echo: long abrasive scrape, echoing metallic ring and splatter, 2.7s.",
    "Suction pop then scrub: quick suction pop then loud abrasive rubbing, 1.9s.",
    "Rust-clang with rub: metallic clang into a heavy scrub sequence, 2.3s.",
    "Rough metal rasp with drip: rasp followed by audible drips and slosh, 2.4s.",
    "Wet leather rub: leathery rubbing with wet smear and coarse grit, 2.1s.",
    "Metal bristle scrape: stiff metal bristle brush across skin, loud rasp, 2.2s.",
    "Abrasive scrape and hiss: coarse rasp with high-frequency hiss and splash, 2.3s.",
    "Thick slosh and grind: heavy liquid slosh under grinding rasp, aggressive, 2.6s.",
    "Quick rasp burst: short fast metallic rasp cluster, punchy, 1.6s.",
    "Rusty clatter then scrub: scattering clatter and then dense rubbing, 2.2s.",
    "Wet cloth drag: saturated cloth dragged across skin with loud friction, 2.0s.",
    "Metallic rasp with breathy wetness: rasp plus wet breathy smear, intimate, 2.4s.",
    "Grit and splash: sandy abrasive rub with big water splash, cinematic, 2.5s.",
    "Hard-scrub rasp: heavy hand-scrub against grit with metallic scrape, 2.3s.",
    "Hollow clang and rasp: metallic hollow clang then abrasive rasp, 2.1s.",
    "Rust drag with wheeze: dragging rust noise with a wheezy motor underlayer, 2.6s.",
    "Squelch and rasp combo: wet squelch interleaved with sharp metallic rasps, 2.2s.",
    "Rasping file with slosh: long file-like rasp with water slosh, 2.7s.",
    "Close wet rasp: intimate wet rasp with small pops and friction, 2.0s.",
    "Metallic grind burst: sudden metallic grind then harsh rub, 1.8s.",
    "Coarse brush thrum: coarse brush rubbing with low thrum and wet smear, 2.3s.",
    "Rusty rasp and drip: rasp with prominent dripping and splash, 2.4s.",
    "Harsh motor scrape: aggressive motor-assisted scrape with loud rasp, 2.5s.",
    "Slosh-scrape echo: wet scraping with long echo tail and metallic ring, 2.6s.",
    "Rag-spin rasp: fast rag spin against skin making gritty rasp, 2.2s.",
    "Deep rasp and pop: weighty rasp punctuated by wet pops, 2.3s.",
    "Metallic gash scrape: rough metallic dragging like a gash with slosh, 2.4s.",
    "Final heavy scrub: big vigorous scrub with metal rasp and resounding splash, 2.7s.",
]

MUSIC_PROMPT = "Slow minimal elevator-like instrumental with soft synth pads, sparse piano and a hidden whispered voice occasionally saying 'exfoliate' and quiet moans; subdued, low volume suitable as background music, length 120s. High quality."

# helper

def apply_audio_effects(file_path):
    # mild EQ / compressor to make voices sit better
    try:
        temp = file_path.replace('.wav', '_tmp.wav')
        af = "lowshelf=g=4:f=120,acompressor=threshold=-14dB:ratio=3:makeup=3dB"
        subprocess.run(["ffmpeg", "-y", "-i", file_path, "-af", af, temp], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.replace(temp, file_path)
    except Exception as e:
        print('Audio effect failed:', e)


def generate_images():
    print('--- Generating Images (SDXL Lightning) ---')
    try:
        # Prefer FLUX.1 (flux1) if available; fallback to SDXL Lightning if not.
        # Force using SDXL Lightning pipeline for images (no Flux)
        print('Using SDXL Lightning pipeline for image generation (forced).')
        pipe = load_sdxl_lightning()

        for i in range(1, NUM + 1):
            # Build image prompts from the corresponding voice prompt so visuals match audio
            raw_voice = VOICE_PROMPTS[i - 1]
            # remove time markers like ~4-6s and trim
            voice_clean = re.sub(r"~\d+-\d+s\.?", "", raw_voice).strip()
            before_fname = f"{OUTPUT_DIR}/images/{i:02d}_before.png"
            during_fname = f"{OUTPUT_DIR}/images/{i:02d}_during.png"
            if not os.path.exists(before_fname):
                prompt = IMAGE_BEFORE_PROMPT.format(voice=voice_clean)
                print(f'Generating before image {i:02d} from voice: "{voice_clean[:60]}..."')
                img = generate_image(pipe, prompt, steps=32, guidance=1.0, seed=2000 + i)
                img.save(before_fname)
            if not os.path.exists(during_fname):
                prompt = IMAGE_DURING_PROMPT.format(voice=voice_clean)
                print(f'Generating during image {i:02d} from voice: "{voice_clean[:60]}..."')
                img = generate_image(pipe, prompt, steps=16, guidance=1.2, seed=3000 + i)
                img.save(during_fname)
        try:
            del pipe
        except Exception:
            pass
        flush()
    except Exception as e:
        print('Image generation failed:', e)
        # Fallback: create simple placeholder images so downstream assembly can continue offline
        try:
            from PIL import Image, ImageDraw, ImageFont
            for i in range(1, NUM + 1):
                before_fname = f"{OUTPUT_DIR}/images/{i:02d}_before.png"
                during_fname = f"{OUTPUT_DIR}/images/{i:02d}_during.png"
                for fname, label in ((before_fname, f"Before {i:02d}"), (during_fname, f"During {i:02d}")):
                    if not os.path.exists(fname):
                        img = Image.new('RGB', (1280, 720), color=(40, 40, 40))
                        d = ImageDraw.Draw(img)
                        try:
                            fnt = ImageFont.load_default()
                        except Exception:
                            fnt = None
                        d.text((20, 20), label, fill=(255, 255, 255), font=fnt)
                        img.save(fname)
            print('Wrote placeholder images for offline use.')
        except Exception as e2:
            print('Failed to write placeholder images:', e2)
        # Fallback: create simple placeholder images so downstream assembly can continue offline
        try:
            from PIL import Image, ImageDraw, ImageFont
            for i in range(1, NUM + 1):
                before_fname = f"{OUTPUT_DIR}/images/{i:02d}_before.png"
                during_fname = f"{OUTPUT_DIR}/images/{i:02d}_during.png"
                for fname, label in ((before_fname, f"Before {i:02d}"), (during_fname, f"During {i:02d}")):
                    if not os.path.exists(fname):
                        img = Image.new('RGB', (1280, 720), color=(40, 40, 40))
                        d = ImageDraw.Draw(img)
                        try:
                            fnt = ImageFont.load_default()
                        except Exception:
                            fnt = None
                        d.text((20, 20), label, fill=(255, 255, 255), font=fnt)
                        img.save(fname)
            print('Wrote placeholder images for offline use.')
        except Exception as e2:
            print('Failed to write placeholder images:', e2)


def generate_voice():
    print('--- Generating 64 ChatTTS voice lines ---')
    try:
        import random
        chat = ChatTTS.Chat()
        chat.load(compile=False)
        wrong_words = [
            "banana","octopus","keyboard","marmalade","sunflower","penguin","quantum","toaster","marzipan","cactus"
        ]
        puns = [
            "This will scrub you like new money.",
            "Smooth as a pancake, but less flat.",
            "We'll get right to the skin of the matter.",
            "You'll be exfoliated to the nth degree.",
            "This is unbe-leaf-able.",
            "A scrub so good it's criminal.",
            "Polish it till it says 'uncle'.",
            "Buff it till it squeaks like a violin."
        ]

        def stretch_word(w):
            # insert a mild drawn-out "ooh" in the middle of the word (avoid hyphens)
            if not w or len(w) < 3:
                return w
            i = len(w) // 2
            return w[:i] + "ooh" + w[i:]

        for i in range(1, NUM + 1):
            # sanitize base prompt: remove timing markers like ~4-6s and stray digits/tildes
            base = VOICE_PROMPTS[i - 1]
            base = re.sub(r"~\d+-\d+s\.?", "", base)
            base = re.sub(r"[~\d]+", "", base)
            base = base.strip()

            rnd = random.Random(i)  # deterministic per index
            words = base.split()
            if len(words) > 2:
                rem_idx = rnd.randrange(len(words))
                del words[rem_idx]
            if len(words) > 0:
                rep_idx = rnd.randrange(len(words))
                words[rep_idx] = wrong_words[rnd.randrange(len(wrong_words))]
            txt = " ".join(words)
            if rnd.random() < 0.75:
                txt += " " + puns[rnd.randrange(len(puns))]

            # apply a drawn-out awkward effect to one word in the final text
            parts = txt.split()
            if len(parts) > 0:
                idx = (i - 1) % len(parts)
                parts[idx] = stretch_word(parts[idx])
                txt = " ".join(parts)

            # final sanitization: remove remaining tildes, digits, or hyphens and collapse spaces
            txt = re.sub(r"[~\d-]+", "", txt)
            txt = re.sub(r"\s+", " ", txt).strip()

            out_file = f"{OUTPUT_DIR}/voice/voice_{i:02d}.wav"
            if os.path.exists(out_file):
                continue
            print(f'Generating voice {i:02d}: "{txt[:60]}..."')
            wavs = chat.infer([txt], use_decoder=True)
            if wavs and len(wavs) > 0:
                arr = np.array(wavs[0]).flatten()
                scipy.io.wavfile.write(out_file, 24000, arr)
                apply_audio_effects(out_file)
        del chat
        flush()
    except Exception as e:
        print('ChatTTS generation failed:', e)


def generate_sfx():
    print('\n--- Generating SFX (Stable Audio) ---')
    try:
        model_id = 'stabilityai/stable-audio-open-1.0'
        pipe = StableAudioPipeline.from_pretrained(model_id, dtype=torch.float32)
        if DEVICE == 'cuda': pipe.enable_model_cpu_offload()
        else: pipe.to(DEVICE)
        neg = 'low quality, noise, artifacts'
        for i in range(1, NUM + 1):
            fname = f"{OUTPUT_DIR}/sfx/{i:02d}_exfoliate.wav"
            if os.path.exists(fname):
                continue
            print(f'Generating SFX {i:02d}')
            # Use a low-frequency manual saw-on-wood prompt to emphasize low-end rasp
            prompt = "Manual hand saw scraping along wood (tree trunk): coarse low-frequency rasp and deep woody friction, subdued high-end, naturalistic, 6.0s, high fidelity."
            audio = pipe(prompt=prompt, negative_prompt=neg, num_inference_steps=100, audio_end_in_s=6.0).audios[0]
            data = audio.cpu().numpy().T
            scipy.io.wavfile.write(fname, rate=44100, data=data)
            # Post-process: apply stronger low-pass filter and gentle compression to emphasize low freq
            try:
                temp = fname.replace('.wav', '_lp.wav')
                ff_af = "lowpass=f=800,acompressor=threshold=-12dB:ratio=3:makeup=3dB"
                subprocess.run(["ffmpeg", "-y", "-i", fname, "-af", ff_af, "-ar", "44100", temp], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                os.replace(temp, fname)
            except Exception as e_pp:
                print('SFX postprocess failed:', e_pp)
        del pipe
        flush()
    except Exception as e:
        print('SFX generation failed:', e)


def generate_music():
    print('\n--- Generating Music (Stable Audio) ---')
    try:
        model_id = 'stabilityai/stable-audio-open-1.0'
        pipe = StableAudioPipeline.from_pretrained(model_id, dtype=torch.float32)
        if DEVICE == 'cuda': pipe.enable_model_cpu_offload()
        else: pipe.to(DEVICE)
        filename = f"{OUTPUT_DIR}/music/theme_elevator.wav"
        if os.path.exists(filename):
            return
        print('Generating background music (theme_elevator)')
        audio = pipe(prompt=MUSIC_PROMPT, negative_prompt='noise, harsh, vocals', num_inference_steps=100, audio_end_in_s=120.0).audios[0]
        scipy.io.wavfile.write(filename, rate=44100, data=audio.cpu().numpy().T)
        del pipe
        flush()
    except Exception as e:
        print('Music generation failed:', e)


if __name__ == '__main__':
    import sys
    if 'images' in sys.argv: generate_images(); sys.exit(0)
    if 'voice' in sys.argv: generate_voice(); sys.exit(0)
    if 'sfx' in sys.argv: generate_sfx(); sys.exit(0)
    if 'music' in sys.argv: generate_music(); sys.exit(0)

    # default: run all
    generate_images()
    generate_voice()
    generate_sfx()
    generate_music()
