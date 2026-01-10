#!/usr/bin/env python3
"""
Asset generator for EXFOLIATE based on generate_chimp_train_assets.py.
Generates 66 scenes: 
- 00: Chimpo-Studios Logo
- 01: Title Card "EXFOLIATE"
- 02-65: 64 pairs of images (before/during) + voiceovers + loud SFX.
Plus a background music track.
"""

import os
import subprocess
import re
import shutil
import numpy as np
import scipy.io.wavfile
import torch
from dalek.core import get_device, flush, ensure_dir
from dalek.vision import generate_image
from transformers import VitsModel, AutoTokenizer
from diffusers import StableDiffusionPipeline, AudioLDMPipeline
from huggingface_hub import snapshot_download
from PIL import Image, ImageDraw, ImageFont

# --- Configuration ---
OUTPUT_DIR = "assets_exfoliate"
ensure_dir(f"{OUTPUT_DIR}/images")
ensure_dir(f"{OUTPUT_DIR}/voice")
ensure_dir(f"{OUTPUT_DIR}/sfx")
ensure_dir(f"{OUTPUT_DIR}/music")

DEVICE = get_device()
print(f"Using device: {DEVICE}")

# 64 positions + logo + title
NUM_POSITIONS = 64
TOTAL_SCENES = NUM_POSITIONS + 2

# Compose varied body-area prompts to create 64 distinct exfoliation variations
AREAS = [
    "left shoulder", "right shoulder", "upper back", "lower back", "chest", "abdomen", "left thigh", "right thigh",
    "left shin", "right shin", "left calf", "right calf", "neck", "nape", "scalp", "forearm", "upper arm", "hand",
    "palm", "fingers", "hip", "waist", "lower leg", "ankle", "foot", "toes", "inner thigh", "outer thigh", "buttock",
    "lower back center", "upper chest", "sternum area", "left ribcage", "right ribcage", "inner forearm", "elbow", "knee",
    "shoulder blade left", "shoulder blade right", "inner thigh left", "inner thigh right", "calf left", "calf right", "heel",
    "arch of foot", "groin area", "upper back center", "lower abdomen", "side torso left", "side torso right",
    "upper arm front", "upper arm back", "wrist", "thumb area", "index finger", "ring finger", "back of hand", "palmar crease",
    "collarbone", "behind knee", "back of neck", "jawline", "temple", "solar plexus", "rib flank", "hipbone"
]

# Prompts for images and voices
IMAGE_BEFORE_PROMPT = (
    "Full body photo of a very hairy asymmetric man, huge nose, long comb-over, "
    "holding a rusty metal tool, pointing to {area}, studio lighting, 4k, detailed"
)
IMAGE_DURING_PROMPT = (
    "Full body photo of a very hairy asymmetric man, huge nose, long comb-over, "
    "exfoliating {area} with a rusty metal tool, foam, scrub motion, cinematic lighting, 4k, detailed"
)

# Professional Stern Doctor Voice Prompts (starting from index 2)
DOCTOR_PROMPTS = [
    "Targeting left shoulder. Epidermal resurfacing initiated. Apply the abrasive tool with consistent downward force. Protocol must be followed strictly.",
    "Moving to right shoulder. Prepare the site for deep tissue exfoliation. Ensure all follicular obstructions are removed immediately. Maintain steady pressure.",
    "Upper back region identified. Initiate systematic circular passes. The patient's excessive hair density requires increased scrape depth. Proceed with caution.",
    "Lower back maintenance. Execute firm longitudinal strokes. Surface contaminants must be eliminated. Do not allow the tool to slip.",
    "Chest area protocol. This is a sensitive zone but requires rigorous mechanical debridement. Monitor dermal redness throughout the process.",
    "Abdominal region. Apply the metal scraper with deliberate, rhythmic motions. Ensure even coverage from the sternum to the waistline.",
    "Left thigh procedure. Long, sweeping passes are necessary to clear the thick hair patches. Maintain a 45-degree angle with the tool.",
    "Right thigh procedure. Identical parameters as the previous limb. Precision is paramount. Do not rush the abrasion cycle.",
    "Left shin identified. Short, controlled strokes along the bone. The abrasive sound must remain consistent. Watch for micro-abrasions.",
    "Right shin identified. Repeat the debridement process. Surface smoothing is the objective. Any unevenness in the scrape is unacceptable.",
    "Left calf area. Maintain firm pressure against the muscle tissue. Systematic upward strokes only. Efficiency is expected.",
    "Right calf area. Concluding the lower limb sequence. Ensure the transition between areas is seamless. Clean the tool after this pass.",
    "Neck region preparation. High precision required. Use the tip of the exfoliation tool for localized debris removal. Steady hands only.",
    "Nape of the neck. This area often harbors hidden follicles. Scrape with intent. The patient must remain perfectly still.",
    "Scalp treatment. Abrasive friction must be maximized despite the hair density. This is a deep-clean operation. No exceptions.",
    "Left forearm sequence. Epidermal layers are thinner here. Adjust pressure but maintain the abrasive effect. Surface must be cleared.",
    "Upper left arm. Focus on the triceps region. Deep-seated dirt requires a more aggressive rasp. Do not hesitate.",
    "Left hand dorsum. Precise movements around the knuckles. Every millimeter must be treated. Professional standards apply.",
    "Left palm. Remove calloused layers with the serrated edge. This is a functional restoration, not a cosmetic one.",
    "Left fingers. Individual attention to each digit. Small, sharp strokes. Ensure complete follicular extraction.",
    "Hip region. Wide, circular passes. The abrasive tool must remain in constant contact with the skin. Monitor the sound of the scrape.",
    "Waistline. Maintain a tight grip on the scraper. Follow the natural contour of the torso. Thoroughness is the only metric.",
    "Lower leg transition. Ensure the ankle joint is bypassed correctly before returning to the main calf muscle. Protocol is absolute.",
    "Ankle joint. Careful scraping around the bone. Do not compromise the depth of the exfoliation. Results must be uniform.",
    "Left foot. Extensive resurfacing needed. Use the heavy-duty rasp for the heel and mid-sole. Deep abrasion is necessary.",
    "Toes. Minute adjustments to the tool's angle. Clear all inter-digital debris. This is a critical sanitation step.",
    "Inner thigh left. Maintain professional distance while ensuring maximum abrasive efficiency. Skin must be left raw but clean.",
    "Outer thigh left. Long, powerful strokes. The tool's teeth must bite into the epidermal layer. This is for the patient's own good.",
    "Buttock region. Large surface area requires a systematic grid-like scraping pattern. Efficiency and speed are required.",
    "Lower back center. Target the spine line with vertical passes. Surface irregularities must be ground down.",
    "Upper chest center. Direct contact with the sternum area. Use short, high-pressure bursts. The sound of the tool is your guide.",
    "Sternum area. Focus on the central ridge. Abrasive depth should be maximized here. No follicular remnants allowed.",
    "Left ribcage. Careful navigation of the intercostal spaces. Maintain consistent scraping noise. Professional focus is mandatory.",
    "Right ribcage. Synchronize your movements with the patient's breathing. Do not slow down. The protocol dictates the pace.",
    "Inner forearm left. Delicate but firm. Clear the translucent hair patches. The resulting surface must be perfectly smooth.",
    "Elbow joint. Rasp the calloused skin until the pink layer is visible. This is a mandatory resurfacing step.",
    "Knee joint. Circular passes around the patella. Ensure full range of motion is cleared of debris. Precision is key.",
    "Left shoulder blade. Scrape from the medial to lateral edge. Use the full length of the metal tool. No area is to be missed.",
    "Right shoulder blade. Repeat the scapular debridement. Maintain a stern focus on the abrasive texture. This is a medical necessity.",
    "Inner thigh right. Consistent pressure is vital. Do not deviate from the marked zones. Follow the instructions to the letter.",
    "Outer thigh right. Large scale mechanical scraping. The patient's comfort is secondary to the quality of the exfoliation.",
    "Calf left medial. Focus on the inner muscle line. Scrape with upward momentum. Ensure all surface oil is removed.",
    "Calf right medial. Identical procedure. The tool must be kept at a sharp angle for maximum epidermal take-up.",
    "Heel of the foot. Apply maximum pressure. This thick tissue requires a rigorous grinding motion. Use the coarse side of the rasp.",
    "Arch of the foot. Sensitive but necessary. Maintain a steady hand. The tool must not skip over the surface.",
    "Groin area. Proceed with clinical detachment. Focus on the peripheral follicles. Abrasive standards must not drop.",
    "Upper back center. The highest density of debris is located here. Use the tool like a scalpel. Precise, deep, and final.",
    "Lower abdomen. Soft tissue requires rhythmic scraping. Do not let the skin bunch up under the metal tool.",
    "Side torso left. Long vertical passes. From the armpit to the hip. This is a comprehensive cleaning cycle.",
    "Side torso right. Maintain the same vertical intensity. The scraper must be cleared of hair every three passes.",
    "Upper arm front. Bicep region debridement. Fast, sharp strokes. We are looking for a complete textural shift.",
    "Upper arm back. Tricep region. Use the tool's edge for better penetration. The patient's skin must be resurfaced.",
    "Wrist. Small circular motions. Careful with the tendons. The metal tool must remain in control at all times.",
    "Thumb area. Detailed work on the thenar eminence. Remove all dead cells. This is a standard surgical-prep exfoliation.",
    "Index finger. From base to tip. Each stroke must be calculated. We are removing years of accumulation.",
    "Ring finger. Precision is demanded. Use the fine-grained side of the metal scraper. Do not allow for error.",
    "Back of hand right. Consistent with the left. The asymmetric man must have symmetric results. Scrape firmly.",
    "Palmar crease. Dig the tool into the lines of the hand. No debris must remain in the folds. Professional grade cleaning.",
    "Collarbone. Scrape along the bone line. The sound should be sharp and clear. This confirms the tool is working.",
    "Behind the knee. High moisture area requires more frequent tool cleaning. Maintain the rasping intensity.",
    "Back of neck lower. Focus on the transition to the shoulders. Broad, heavy strokes. The hair must go.",
    "Jawline. Scrape from the ear to the chin. Maintain a stern clinical gaze. Every follicle is a target.",
    "Temple region. Use extreme caution but do not reduce pressure. This is a targeted epidermal strike.",
    "Hipbone area. The final position. Concluding the full body exfoliation. Ensure the metal tool makes one last definitive pass."
]

VOICE_PROMPTS = ["Chimpo-Studios. A subsidiary of Universal.", "Exfoliate."] + DOCTOR_PROMPTS

SFX_PROMPTS = [
    "Film projector whirr, studio intro music flourish, cinematic.",
    "Deep echoing bass thud, mechanical click, metallic reverb."
] + [
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
    "Final heavy scrub: big vigorous scrub with metal rasp and resounding splash, 2.7s."
]

MUSIC_PROMPT = "Slow minimal elevator-like instrumental with soft synth pads, sparse piano; subdued, low volume suitable as background music, length 120s. High quality."

def apply_audio_effects(file_path):
    try:
        temp = file_path.replace('.wav', '_tmp.wav')
        af = "lowshelf=g=4:f=120,acompressor=threshold=-14dB:ratio=3:makeup=3dB"
        subprocess.run(["ffmpeg", "-y", "-i", file_path, "-af", af, temp], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.replace(temp, file_path)
    except Exception as e:
        print('Audio effect failed:', e)

def generate_intro():
    # Scene 00: Logo
    logo_src = "assets_chimp_train/images/00_studio_logo.png"
    logo_dst = f"{OUTPUT_DIR}/images/00_scene.png"
    if os.path.exists(logo_src) and not os.path.exists(logo_dst):
        shutil.copy(logo_src, logo_dst)
        print("Copied studio logo.")
    elif not os.path.exists(logo_dst):
        img = Image.new('RGB', (1280, 720), color=(0, 0, 0))
        d = ImageDraw.Draw(img)
        d.text((500, 350), "CHIMPO-STUDIOS", fill=(255, 255, 255))
        img.save(logo_dst)

    # Scene 01: Title Card
    title_dst = f"{OUTPUT_DIR}/images/01_scene.png"
    if not os.path.exists(title_dst):
        img = Image.new('RGB', (1280, 720), color=(10, 10, 10))
        d = ImageDraw.Draw(img)
        # Draw a big bold EXFOLIATE
        text = "EXFOLIATE"
        try:
            # try to find a system font
            fnt = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Black.ttf", 120)
        except:
            fnt = None
        d.text((350, 280), text, fill=(200, 200, 200), font=fnt)
        img.save(title_dst)
        print("Generated title card.")

def generate_images():
    print('--- Generating Images (Tiny SD) ---')
    generate_intro()
    try:
        repo_id = "segmind/tiny-sd"
        model_id = snapshot_download(repo_id)
        dtype = torch.float16 if DEVICE == "cuda" else torch.float32
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
        if DEVICE == "cuda":
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(DEVICE)
        if hasattr(pipe, 'safety_checker'): pipe.safety_checker = None

        for i in range(NUM_POSITIONS):
            scene_idx = i + 2
            area = AREAS[i]
            before_fname = f"{OUTPUT_DIR}/images/{scene_idx:02d}_before.png"
            during_fname = f"{OUTPUT_DIR}/images/{scene_idx:02d}_during.png"
            
            def sanitize(p): return re.sub(r'\s+', ' ', p).strip()

            if not os.path.exists(before_fname):
                prompt = IMAGE_BEFORE_PROMPT.format(area=area)
                print(f'Generating before image {scene_idx:02d} ({area})...')
                img = generate_image(pipe, sanitize(prompt), steps=25, guidance=7.5, seed=101 + i)
                img.save(before_fname)
            
            if not os.path.exists(during_fname):
                prompt = IMAGE_DURING_PROMPT.format(area=area)
                print(f'Generating during image {scene_idx:02d} ({area})...')
                img = generate_image(pipe, sanitize(prompt), steps=25, guidance=7.5, seed=201 + i)
                img.save(during_fname)
        del pipe; flush()
    except Exception as e:
        print('Image generation failed:', e)

def generate_voice():
    print('--- Generating 66 Voice Lines (MMS-TTS) ---')
    try:
        repo_id = "facebook/mms-tts-eng"
        model_id = snapshot_download(repo_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = VitsModel.from_pretrained(model_id).to(DEVICE)
        
        for i in range(TOTAL_SCENES):
            txt = VOICE_PROMPTS[i]
            txt = re.sub(r"\s+", " ", txt).strip()
            out_file = f"{OUTPUT_DIR}/voice/voice_{i:02d}.wav"
            if os.path.exists(out_file): continue
            
            print(f'Generating voice {i:02d}: "{txt[:50]}..."')
            inputs = tokenizer(txt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                output = model(**inputs).waveform
            scipy.io.wavfile.write(out_file, model.config.sampling_rate, output.cpu().numpy().flatten())
            apply_audio_effects(out_file)
        del model, tokenizer; flush()
    except Exception as e:
        print('MMS-TTS generation failed:', e)

def generate_sfx():
    print('\n--- Generating SFX (AudioLDM) ---')
    try:
        repo_id = "cvssp/audioldm-s-full-v2"
        model_id = snapshot_download(repo_id)
        print(f"Loading AudioLDM from {model_id}...")
        pipe = AudioLDMPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
        if DEVICE == "cuda":
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(DEVICE)
        
        for i in range(len(SFX_PROMPTS)):
            fname = f"{OUTPUT_DIR}/sfx/{i:02d}_exfoliate.wav"
            if os.path.exists(fname): continue
            print(f'Generating SFX {i:02d}')
            prompt = SFX_PROMPTS[i]
            audio = pipe(prompt, num_inference_steps=50, audio_length_in_s=3.0).audios[0]
            scipy.io.wavfile.write(fname, 16000, audio)
            try:
                temp = fname.replace('.wav', '_lp.wav')
                subprocess.run(["ffmpeg", "-y", "-i", fname, "-af", "lowpass=f=800", "-ar", "32000", temp], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                os.replace(temp, fname)
            except: pass
        del pipe; flush()
    except Exception as e:
        print('SFX generation failed:', e)

def generate_music():
    print('\n--- Generating Music (AudioLDM2) ---')
    try:
        repo_id = "cvssp/audioldm2-music"
        model_id = snapshot_download(repo_id)
        pipe = AudioLDM2Pipeline.from_pretrained(model_id, torch_dtype=torch.float32)
        if DEVICE == "cuda":
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(DEVICE)
        filename = f"{OUTPUT_DIR}/music/theme_elevator.wav"
        if os.path.exists(filename): return
        print('Generating background music...')
        audio = pipe(MUSIC_PROMPT, num_inference_steps=100, audio_length_in_s=15.0).audios[0]
        scipy.io.wavfile.write(filename, 44100, audio)
        del pipe; flush()
    except Exception as e:
        print('Music generation failed:', e)

if __name__ == '__main__':
    import sys
    if 'images' in sys.argv: generate_images(); sys.exit(0)
    if 'voice' in sys.argv: generate_voice(); sys.exit(0)
    if 'sfx' in sys.argv: generate_sfx(); sys.exit(0)
    if 'music' in sys.argv: generate_music(); sys.exit(0)
    generate_images()
    generate_voice()
    generate_sfx()
    generate_music()