import torch
import os
import sys
import numpy as np
import scipy.io.wavfile as wavfile
import argparse
import utils
from diffusers import DiffusionPipeline, StableAudioPipeline
from transformers import pipeline
from PIL import Image

# --- Configuration & Defaults ---
PROJECT_NAME = "sloppy"
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
DEFAULT_MODEL = "black-forest-labs/FLUX.1-dev" if IS_H200 else "black-forest-labs/FLUX.1-schnell"
DEFAULT_STEPS = 64 if IS_H200 else 16
DEFAULT_GUIDANCE = 3.5 if IS_H200 else 0.0
DEFAULT_QUANT = "none" if IS_H200 else "4bit"

# --- Prompts ---
SCENES = [
    ("01_melting_clock_tower", "Cinematic melting clock tower glitch art, high detail, masterpiece, 8k", "distorted mid-pitch metallic grinding textured glitch"),
    ("02_statue_extra_limbs", "Cinematic statue with extra limbs glitch art, high detail, masterpiece, 8k", "stone scraping textured mid-frequency rumble"),
    ("03_classic_portrait_smear", "Cinematic classic portrait smear glitch art, high detail, masterpiece, 8k", "wet paint squelch textured canvas ripping"),
    ("04_landscape_floating_rocks", "Cinematic landscape floating rocks glitch art, high detail, masterpiece, 8k", "low frequency humming textured rock debris"),
    ("05_horse_too_many_legs", "Cinematic horse with too many legs glitch art, high detail, masterpiece, 8k", "rhythmic galloping distortion textured horse snort"),
    ("06_tea_party_faceless", "Cinematic tea party with faceless people glitch art, high detail, masterpiece, 8k", "distorted whispers tea cup clinking textured tea pouring"),
    ("07_library_infinite_books", "Cinematic library with infinite books glitch art, high detail, masterpiece, 8k", "paper fluttering textured book shelf shifting"),
    ("08_cat_spaghetti_fur", "Cinematic cat with spaghetti fur glitch art, high detail, masterpiece, 8k", "wet pasta squelch textured cat purr distortion"),
    ("09_dog_bird_hybrid", "Cinematic dog bird hybrid glitch art, high detail, masterpiece, 8k", "distorted barking textured wings flapping"),
    ("10_vintage_car_square_wheels", "Cinematic vintage car with square wheels glitch art, high detail, masterpiece, 8k", "clunky mechanical thumping textured car engine idle"),
    ("11_ballroom_dancers_merged", "Cinematic ballroom dancers merged glitch art, high detail, masterpiece, 8k", "distorted waltz music textured fabric rustle"),
    ("12_flower_teeth", "Cinematic flower with teeth glitch art, high detail, masterpiece, 8k", "organic snapping textured plant growth"),
    ("13_mountain_made_of_flesh", "Cinematic mountain made of flesh glitch art, high detail, masterpiece, 8k", "deep biological thumping textured meat squelch"),
    ("14_river_of_hair", "Cinematic river of hair glitch art, high detail, masterpiece, 8k", "soft flowing hair textured water rushing"),
    ("15_cloud_screaming", "Cinematic cloud screaming glitch art, high detail, masterpiece, 8k", "distorted vocal moan textured wind howling"),
    ("16_tree_with_eyes", "Cinematic tree with eyes glitch art, high detail, masterpiece, 8k", "wood creaking textured blinking sound"),
    ("17_dinner_plate_eating_itself", "Cinematic dinner plate eating itself glitch art, high detail, masterpiece, 8k", "ceramic cracking textured chewing"),
    ("18_hands_holding_hands_fractal", "Cinematic hands holding hands fractal glitch art, high detail, masterpiece, 8k", "skin rubbing textured finger snapping"),
    ("19_mirror_reflection_wrong", "Cinematic mirror reflection wrong glitch art, high detail, masterpiece, 8k", "glass shimmering textured mirror cracking"),
    ("20_stairs_to_nowhere", "Cinematic stairs to nowhere glitch art, high detail, masterpiece, 8k", "echoing footsteps textured wood groan"),
    ("21_bicycle_made_of_meat", "Cinematic bicycle made of meat glitch art, high detail, masterpiece, 8k", "meaty chain rattle textured muscle fiber stretch"),
    ("22_building_breathing", "Cinematic building breathing glitch art, high detail, masterpiece, 8k", "heavy masonry breathing textured brick grinding"),
    ("23_street_lamp_bending", "Cinematic street lamp bending glitch art, high detail, masterpiece, 8k", "metallic stress groan textured bulb buzzing"),
    ("24_shadow_detached", "Cinematic shadow detached glitch art, high detail, masterpiece, 8k", "low frequency pulse textured shadow sliding"),
    ("25_bird_metal_wings", "Cinematic bird with metal wings glitch art, high detail, masterpiece, 8k", "metallic wing flapping textured avian screech"),
    ("26_fish_walking", "Cinematic fish walking glitch art, high detail, masterpiece, 8k", "wet flipper slapping textured underwater bubbles"),
    ("27_chair_sitting_on_chair", "Cinematic chair sitting on chair glitch art, high detail, masterpiece, 8k", "wooden furniture creaking textured leg dragging"),
    ("28_piano_melting_keys", "Cinematic piano with melting keys glitch art, high detail, masterpiece, 8k", "distorted piano note textured dripping plastic"),
    ("29_violin_made_of_water", "Cinematic violin made of water glitch art, high detail, masterpiece, 8k", "watery string bow textured liquid splashing"),
    ("30_moon_cracked_egg", "Cinematic moon shaped like a cracked egg glitch art, high detail, masterpiece, 8k", "celestial cracking textured yolk dripping"),
    ("31_sun_dripping", "Cinematic sun dripping glitch art, high detail, masterpiece, 8k", "solar flare sizzle textured molten liquid"),
    ("32_forest_upside_down", "Cinematic forest upside down glitch art, high detail, masterpiece, 8k", "inverted nature sounds textured leaves rustling"),
]

def generate_images(args):
    model_id = args.model
    steps = args.steps
    guidance = args.guidance
    quant = args.quant
    offload = args.offload
    use_scalenorm = args.scalenorm

    print(f"--- Generating {len(SCENES)} {model_id} Images ({steps} steps) on {DEVICE} ---")
    print(f"Quantization: {quant}, CPU Offload: {offload}, ScaleNorm: {use_scalenorm}")
    
    pipe_kwargs = {
        "torch_dtype": torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    }

    if quant != "none" and DEVICE == "cuda":
        from diffusers import PipelineQuantizationConfig
        backend = "bitsandbytes_4bit" if quant == "4bit" else "bitsandbytes_8bit"
        quant_kwargs = {"load_in_4bit": True} if quant == "4bit" else {"load_in_8bit": True}
        
        if quant == "8bit":
            pipe_kwargs["torch_dtype"] = torch.float16

        pipe_kwargs["quantization_config"] = PipelineQuantizationConfig(
            quant_backend=backend,
            quant_kwargs=quant_kwargs,
            components_to_quantize=["transformer"]
        )
    
    pipe = DiffusionPipeline.from_pretrained(model_id, **pipe_kwargs)
    
    utils.remove_weight_norm(pipe)
    if use_scalenorm:
        utils.apply_scalenorm_to_transformer(pipe.transformer)

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
            image = pipe(prompt, num_inference_steps=steps, guidance_scale=guidance, width=1280, height=720).images[0]
            image.save(out_path)
    del pipe
    torch.cuda.empty_cache()

def generate_sfx(args):
    print(f"--- Generating SFX with Stable Audio Open on {DEVICE} ---")
    pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16).to(DEVICE)
    utils.remove_weight_norm(pipe)
    if args.scalenorm:
        utils.apply_scalenorm_to_transformer(pipe.transformer)

    os.makedirs(f"{ASSETS_DIR}/sfx", exist_ok=True)
    for s_id, _, sfx_prompt in SCENES:
        out_path = f"{ASSETS_DIR}/sfx/{s_id}.wav"
        if not os.path.exists(out_path):
            print(f"Generating SFX for: {s_id} -> {sfx_prompt}")
            audio = pipe(sfx_prompt, num_inference_steps=100, audio_end_in_s=10.0).audios[0]
            audio_np = audio.T.cpu().numpy()
            wavfile.write(out_path, 44100, (audio_np * 32767).astype(np.int16)) 
    del pipe
    torch.cuda.empty_cache()

def generate_voiceover(args):
    print(f"--- Generating Voiceover with Stable Audio ---")
    os.makedirs(f"{ASSETS_DIR}/voice", exist_ok=True)
    out_path = f"{ASSETS_DIR}/voice/voiceover_full.wav"
    if os.path.exists(out_path): return

    pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16).to(DEVICE)
    utils.remove_weight_norm(pipe)
    if args.scalenorm:
        utils.apply_scalenorm_to_transformer(pipe.transformer)

    prompt = "A distorted glitchy uncanny male voiceover narration, experimental texture, spoken word, cinematic horror"
    print("Generating voiceover audio...")
    audio = pipe(prompt, num_inference_steps=100, audio_end_in_s=45.0).audios[0]
    audio_np = audio.T.cpu().numpy()
    wavfile.write(out_path, 44100, (audio_np * 32767).astype(np.int16))
    del pipe
    torch.cuda.empty_cache()

def generate_music(args):
    print(f"--- Generating Music with Stable Audio ---")
    os.makedirs(f"{ASSETS_DIR}/music", exist_ok=True)
    out_path = f"{ASSETS_DIR}/music/sloppy_theme.wav"
    if os.path.exists(out_path): return

    pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16).to(DEVICE)
    utils.remove_weight_norm(pipe)
    if args.scalenorm:
        utils.apply_scalenorm_to_transformer(pipe.transformer)

    prompt = "dark experimental industrial noise, glitchy rhythmic scraping, uncanny cinematic horror ambient, high quality"
    print("Generating music theme...")
    audio = pipe(prompt, num_inference_steps=100, audio_end_in_s=45.0).audios[0]
    audio_np = audio.T.cpu().numpy()
    wavfile.write(out_path, 44100, (audio_np * 32767).astype(np.int16))
    del pipe
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Sloppy Assets")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model ID")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Inference steps")
    parser.add_argument("--guidance", type=float, default=DEFAULT_GUIDANCE, help="Guidance scale")
    parser.add_argument("--quant", type=str, default=DEFAULT_QUANT, choices=["none", "4bit", "8bit"], help="Quantization type")
    parser.add_argument("--offload", action="store_true", help="Enable CPU offload")
    parser.add_argument("--scalenorm", action="store_true", help="Use ScaleNorm instead of LayerNorm")
    
    args = parser.parse_args()

    generate_images(args)
    generate_sfx(args)
    generate_voiceover(args)
    generate_music(args)
