import torch
import os
import sys
import numpy as np
import scipy.io.wavfile as wavfile
import argparse
import utils
from diffusers import DiffusionPipeline, StableAudioPipeline
from transformers import BarkModel, AutoProcessor, BitsAndBytesConfig
from PIL import Image

# --- Configuration & Defaults ---
PROJECT_NAME = "metro"
ASSETS_DIR = f"assets_{PROJECT_NAME}"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# H200 Detection for default behavior
IS_H200 = False
if DEVICE == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    if "H200" in gpu_name:
        IS_H200 = True

# Default values based on hardware
DEFAULT_MODEL = (
    "black-forest-labs/FLUX.1-dev" if IS_H200 else "black-forest-labs/FLUX.1-schnell"
)
DEFAULT_STEPS = 64 if IS_H200 else 16
DEFAULT_GUIDANCE = 3.5 if IS_H200 else 0.0
DEFAULT_QUANT = "none" if IS_H200 else "4bit"

# Scene Definitions (Prompts & SFX Prompts)
SCENES = [
    (
        "01_entrance",
        "Cinematic overground wide shot of a massive brutalist concrete metro entrance leading deep underground, surrounded by a bleak dystopian sci-fi world with flickering neon signs, fog, rain, 8k, highly detailed",
        "subway station ambience wind howling distant eerie drone",
    ),
    (
        "02_face_scan",
        "Close up grotesque biometric face scanner, red laser grid mapping a weeping human face, dystopian technology",
        "digital scanning noise textured mid-pitch beep laser hum",
    ),
    (
        "03_finger_scan",
        "Futuristic security device crushing a human finger against a glass plate, green light, macro photography",
        "mechanical servo growl textured glass squeak crunch",
    ),
    (
        "04_smell_detector",
        "Bizarre nose-shaped mechanical sensor sniffing a person's neck, medical aesthetic, sterile white",
        "sniffing sound vacuum pump sucking air",
    ),
    (
        "05_torso_slime",
        "Person pressing bare chest against a wall of gelatinous bio-luminescent blue slime, imprint visible",
        "wet squelch slime dripping sticky sound",
    ),
    (
        "06_tongue_print",
        "Metal surgical clamp holding a human tongue, scanning laser, saliva dripping, high detail",
        "wet mouth sound metallic click servo motor",
    ),
    (
        "07_retina_drill",
        "Eye scanning device that looks like a surgical drill, red laser beam pointing into pupil, extreme close up",
        "gritty mechanical drill growl textured laser zap",
    ),
    (
        "08_ear_wax_sampler",
        "Tiny robotic probe entering a human ear, futuristic macro photography, cold lighting",
        "squishy probing sound mechanical whir",
    ),
    (
        "09_hair_count",
        "Robotic tweezers plucking a single hair from a scalp, digital counter display showing numbers",
        "gritty pluck sound textured digital counter tone",
    ),
    (
        "10_sweat_analysis",
        "Person standing in a glass tube sweating profusley under heat lamps, collection drains at feet",
        "heavy breathing steam hiss dripping water",
    ),
    (
        "11_bone_crusher",
        "Hydraulic press gently compressing a human arm to measure density, medical readout, chrome metal",
        "hydraulic hiss metallic thud bone creak",
    ),
    (
        "12_spirit_photo",
        "Ectoplasmic aura camera screen, person looks like a ghost in the viewfinder, distortion, grainy",
        "static noise ghostly moan electrical crackle",
    ),
    (
        "13_karma_scale",
        "Golden mechanical scales weighing a human heart against a feather, futuristic minimalist court",
        "metallic clinking scales balancing heavy thud",
    ),
    (
        "14_dream_extract",
        "Helmet with wires sucking glowing mist from person's head, fiber optic cables, cyberpunk",
        "vacuum suction electrical humming bubbling liquid",
    ),
    (
        "15_memory_wipe",
        "Flash of white light, person looking dazed and empty, pupil dilated, bright overexposed",
        "camera flash capacitor charge textured mid-pitch ring",
    ),
    (
        "16_genetic_sieve",
        "Blood sample passing through glowing filter, DNA strands visible, microscopic view",
        "liquid pumping bubbling biological squish",
    ),
    (
        "17_final_stamp",
        "Hot branding iron stamping 'APPROVED' on a forehead, steam rising, skin texture",
        "sizzling burning sound heavy stamp thud",
    ),
    (
        "18_nail_pull",
        "Automated pliers extracting a single fingernail for mineral analysis, clinical cold lighting",
        "textured metallic snap scream muffled",
    ),
    (
        "19_skin_swatch",
        "Robotic laser cutter removing a small square of skin from a forearm, precise glowing line",
        "textured laser sizzle clinical tone",
    ),
    (
        "20_tooth_scan",
        "Mechanical mouth spreader exposing teeth, blue UV light scanning for dental records",
        "textured dental drill growl mid-frequency vibration",
    ),
    (
        "21_pulse_monitor",
        "Heavy iron collar with glowing sensors measuring heartbeat, cold metallic texture",
        "rhythmic thumping deep bass",
    ),
    (
        "22_tear_collector",
        "Glass vial catching a single tear from an eye held open by metal retractors",
        "dripping sound glass tinkling",
    ),
    (
        "23_brain_map",
        "Transparent skull cap with pulsing neon neurons, mapping brain activity in real time",
        "electrical static brainwave hum",
    ),
    (
        "24_shadow_audit",
        "Person standing against a white wall while their shadow is measured by laser sensors",
        "laser sweeping digital clicking",
    ),
    (
        "25_breath_tax",
        "Gas mask measuring oxygen consumption, digital display showing cost per breath",
        "heavy breathing mechanical hiss",
    ),
    (
        "26_thought_police",
        "Holographic screen displaying a person's private thoughts as distorted text",
        "distorted whispers data processing noise",
    ),
    (
        "27_loyalty_check",
        "Person staring into a bright hypnotic light, pupils dilating and contracting",
        "low frequency pulse hypnotic hum",
    ),
    (
        "28_identity_shredder",
        "Old paper documents being shredded by a massive industrial machine in a dark room",
        "paper shredding grinding metal",
    ),
    (
        "29_platform_edge",
        "Crowded subway platform with people standing precariously close to the edge, yellow warning strip",
        "distant train roar wind rushing",
    ),
    (
        "30_empty_carriage",
        "A single person sitting in a vast empty metro carriage, flickering fluorescent lights",
        "rhythmic train clatter flickering light buzz",
    ),
    (
        "31_train_interior",
        "Inside metro train, minimalist grey seats, sad people staring at feet, uniform grey clothing, sterile",
        "subway train interior rumble wheels on track rhythmic clacking",
    ),
    (
        "32_title_card",
        "Text 'METRO' in minimal sans-serif font, glowing white on black background, cinematic typography",
        "deep bass boom cinematic hit silence",
    ),
]

# Voiceover Script
VO_SCRIPT = """
Welcome to the Metro. 
The future of transit is secure. 
For your safety, we require a few... verifications.
Face scan. Don't blink. We need to see the fear in your eyes.
Finger scan. Press harder. Until it hurts. Good.
Olfactory analysis. You smell like anxiety. And cheap coffee.
Torso imprint. The slime is sterile. Mostly.
Tongue print. Taste the sensor. It tastes like copper. And submission.
Retina check. Keep your eye open. The laser is warm.
Auricular sampling. We are listening to your thoughts. Through your earwax.
Follicle audit. One hair. Two hair. Three. We are counting.
Sweat extraction. Perspire for the state. Your fluids are data.
Bone density verification. Just a little pressure. To ensure you are solid.
Spirit photography. Your aura is grey. How disappointing.
Karma weighing. Your sins are heavy. You will pay extra.
Dream extraction. Leave your hopes here. You won't need them.
Memory wipe. Forget why you came. Forget who you are.
Genetic sieve. You are filtered. You are processed.
Final stamp. Approved.
Nail extraction. Mineral analysis complete.
Skin swatch. DNA archived.
Dental audit. Smile for the state.
Heartbeat synchronization. Your pulse is erratic. Calm down.
Lacrimal collection. Your tears are salty. And inefficient.
Neurological mapping. We know what you are thinking.
Shadow measurement. You are slightly too large. Shrink.
Respiration tax. Every breath has a price.
Thought audit. Your ideas are... non-compliant.
Hypnotic loyalty check. You will obey. You have no choice.
Identity shredding. Your past is gone.
Platform approach. Mind the gap. Between your life and the void.
Eternal transit. The train is coming.
Sit down. Be sad.
This is the Metro.
We are going nowhere.
Fast.
"""


def generate_images(args):
    if args.flux2:
        model_id = args.flux2
        steps = args.steps if args.steps != DEFAULT_STEPS else 32
    else:
        model_id = args.model
        steps = args.steps

    guidance = args.guidance
    quant = args.quant
    offload = args.offload
    use_scalenorm = args.scalenorm

    print(
        f"--- Generating {len(SCENES)} {model_id} Images ({steps} steps) on {DEVICE} ---"
    )

    pipe_kwargs = {
        "torch_dtype": torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    }

    if quant != "none" and DEVICE == "cuda":
        bnb_4bit_compute_dtype = torch.bfloat16 if IS_H200 else torch.float16
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
        )
        pipe_kwargs["transformer_quantization_config"] = quant_config
        if not offload:
            pipe_kwargs["device_map"] = "balanced"

    is_local = os.path.isdir(model_id)
    pipe = DiffusionPipeline.from_pretrained(
        model_id, local_files_only=is_local, **pipe_kwargs
    )

    utils.remove_weight_norm(pipe)
    if use_scalenorm:
        utils.apply_stability_improvements(pipe.transformer, use_scalenorm=True)

    if offload and DEVICE == "cuda":
        pipe.enable_model_cpu_offload()
    elif quant != "none" and DEVICE == "cuda":
        print("Moving non-quantized components to GPU...")
        for name, component in pipe.components.items():
            if name != "transformer" and hasattr(component, "to"):
                component.to(DEVICE)
    else:
        pipe.to(DEVICE)

    os.makedirs(f"{ASSETS_DIR}/images", exist_ok=True)
    for s_id, prompt, _ in SCENES:
        out_path = f"{ASSETS_DIR}/images/{s_id}.png"
        if not os.path.exists(out_path):
            print(f"Generating: {s_id}")
            image = pipe(
                prompt=prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                width=1280,
                height=720,
            ).images[0]
            image.save(out_path)
    del pipe
    torch.cuda.empty_cache()


def generate_sfx(args):
    print(f"--- Generating SFX with Stable Audio Open on {DEVICE} ---")
    pipe = StableAudioPipeline.from_pretrained(
        "stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16
    ).to(DEVICE)
    utils.remove_weight_norm(pipe)
    if args.scalenorm:
        utils.apply_stability_improvements(pipe.transformer, use_scalenorm=True)

    os.makedirs(f"{ASSETS_DIR}/sfx", exist_ok=True)
    for s_id, _, sfx_prompt in SCENES:
        out_path = f"{ASSETS_DIR}/sfx/{s_id}.wav"
        if not os.path.exists(out_path):
            print(f"Generating SFX for: {s_id} -> {sfx_prompt}")
            audio = pipe(
                sfx_prompt, num_inference_steps=100, audio_end_in_s=12.0
            ).audios[0]
            audio_np = audio.T.cpu().numpy()
            wavfile.write(out_path, 44100, (audio_np * 32767).astype(np.int16))
    del pipe
    torch.cuda.empty_cache()


def generate_voiceover(args):
    print(f"--- Generating Voiceover with Bark (Intelligible TTS) on {DEVICE} ---")
    os.makedirs(f"{ASSETS_DIR}/voice", exist_ok=True)
    out_path = f"{ASSETS_DIR}/voice/voiceover_full.wav"
    if os.path.exists(out_path):
        return

    # Load model and processor directly to avoid pipeline issues with voice_preset
    processor = AutoProcessor.from_pretrained("suno/bark")
    model = BarkModel.from_pretrained("suno/bark").to(DEVICE)

    lines = [l for l in VO_SCRIPT.split("\n") if l.strip()]
    full_audio = []
    sampling_rate = 24000

    for line in lines:
        print(f"  Speaking: {line[:30]}...")
        inputs = processor(line, voice_preset=args.voice_preset).to(DEVICE)
        audio_array = model.generate(**inputs, pad_token_id=10000)
        audio_data = audio_array.cpu().numpy().squeeze()

        silence = np.zeros(int(sampling_rate * 0.8))
        full_audio.append(audio_data)
        full_audio.append(silence)

    combined = np.concatenate(full_audio)
    wavfile.write(out_path, sampling_rate, (combined * 32767).astype(np.int16))
    del model
    del processor
    torch.cuda.empty_cache()


def generate_music(args):
    print(f"--- Generating Background Music with Stable Audio on {DEVICE} ---")
    os.makedirs(f"{ASSETS_DIR}/music", exist_ok=True)
    out_path = f"{ASSETS_DIR}/music/metro_theme.wav"
    if os.path.exists(out_path):
        return

    pipe = StableAudioPipeline.from_pretrained(
        "stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16
    ).to(DEVICE)
    utils.remove_weight_norm(pipe)
    if args.scalenorm:
        utils.apply_stability_improvements(pipe.transformer, use_scalenorm=True)

    prompt = "eerie minimal synth drone, dark ambient, sci-fi horror soundtrack, slow pulsing deep bass, cinematic atmosphere, high quality"
    print("Generating eerie synth theme...")
    audio = pipe(prompt, num_inference_steps=100, audio_end_in_s=45.0).audios[0]
    audio_np = audio.T.cpu().numpy()
    wavfile.write(out_path, 44100, (audio_np * 32767).astype(np.int16))
    del pipe
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Metro Assets")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model ID")
    parser.add_argument("--flux2", type=str, help="Path to FLUX.2 model directory")
    parser.add_argument(
        "--steps", type=int, default=DEFAULT_STEPS, help="Inference steps"
    )
    parser.add_argument(
        "--guidance", type=float, default=DEFAULT_GUIDANCE, help="Guidance scale"
    )
    parser.add_argument(
        "--quant",
        type=str,
        default=DEFAULT_QUANT,
        choices=["none", "4bit", "8bit"],
        help="Quantization type",
    )
    parser.add_argument("--offload", action="store_true", help="Enable CPU offload")
    parser.add_argument(
        "--scalenorm", action="store_true", help="Use ScaleNorm improvement"
    )

    parser.add_argument(
        "--voice_preset", type=str, default="v2/en_speaker_6", help="Bark voice preset"
    )

    args = parser.parse_args()
    generate_images(args)
    generate_sfx(args)
    generate_voiceover(args)
    generate_music(args)
