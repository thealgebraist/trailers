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
PROJECT_NAME = "chimp"
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

# --- Scenes ---
SCENES = [
    ("01_chimp_map", "Close up of a cute chimpanzee wearing a tiny explorer's hat, looking intensely at a map showing a Golden Banana, cinematic lighting, 8k, pixar style", "textured mid-pitch paper rustling map unfolding"),
    ("02_chimp_packing", "A chimpanzee packing a small vintage leather suitcase with a toothbrush and a magnifying glass, cozy bedroom, cinematic lighting, 8k, pixar style", "mechanical suitcase latches clicking soft fabric movement"),
    ("03_chimp_station", "A chimpanzee standing on a train platform as a massive, steam-puffing vintage train pulls into the station, steam everywhere, 8k, pixar style", "heavy steam train chugging rhythmic metallic clanking"),
    ("04_chimp_train_window", "A chimpanzee sitting in a plush velvet train seat, looking out at passing green mountains through a window, 8k, pixar style", "gentle train interior rumble rhythmic tracks"),
    ("05_chimp_penguin", "A chimpanzee sharing a train seat with a confused penguin wearing a tuxedo, funny interaction, 8k, pixar style", "funny mid-pitch slide whistle wobble penguin squeak"),
    ("06_train_bridge", "A steam train crossing a high precarious stone bridge over a lush tropical jungle, cinematic wide shot, 8k, pixar style", "distant train whistle wind rushing jungle birds"),
    ("07_fruit_city", "A bustling futuristic city where buildings are shaped like giant fruits, pineapple towers, melon domes, 8k, pixar style", "bustling cartoon city ambience whimsical mid-pitch horns"),
    ("08_golden_banana", "A glowing golden banana resting on a red velvet cushion in a high-end shop window, magical aura, 8k, pixar style", "magical shimmering bells textured mid-pitch chime"),
    ("09_chimp_running", "A chimpanzee sprinting through a colorful city street towards a fruit boutique, motion blur, 8k, pixar style", "fast cartoon footsteps rhythmic breathing"),
    ("10_chimp_reaching", "Close up of a chimpanzee's hand reaching out to touch a glowing golden banana, 8k, pixar style", "tense mid-pitch synth swell textured heart thud"),
    ("11_chimp_guard", "A large grumpy gorilla guard wearing a suit and sunglasses, standing in front of the golden banana, 8k, pixar style", "deep gorilla grunt textured heavy footsteps"),
    ("12_chimp_distraction", "A chimpanzee throwing a handful of colorful marbles to distract the gorilla guard, marbles rolling everywhere, 8k, pixar style", "marbles clattering on floor textured glass rolling"),
    ("13_chimp_sneaking", "A chimpanzee tip-toeing past the distracted guard on a shiny marble floor, funny heartbeat rhythm, 8k, pixar style", "squeaky rubber shoe sound textured tip-toe rhythmic"),
    ("14_chimp_grab", "A chimpanzee's hand finally grasping the glowing golden banana, sparkle effects, 8k, pixar style", "magical sparkle sound textured mid-pitch shimmer"),
    ("15_chimp_escape", "A chimpanzee jumping through a fruit-shaped window with the golden banana, glass shattering into fruit slices, 8k, pixar style", "glass shattering sound textured fruit squelch"),
    ("16_chimp_chase", "A fleet of hover-fruit vehicles chasing the chimpanzee through the fruit city streets, 8k, pixar style", "hovercraft hum textured mid-pitch engine whine"),
    ("17_chimp_glider", "A chimpanzee using a giant leaf as a hang-glider, soaring over the jungle canopy, 8k, pixar style", "wind rushing sound textured jungle ambience"),
    ("18_chimp_waterfall", "A chimpanzee gliding behind a massive sparkling waterfall, rainbow in the mist, 8k, pixar style", "roaring waterfall sound textured water splash"),
    ("19_chimp_cave", "A chimpanzee entering a mysterious cave shaped like a giant mouth, glowing mushrooms inside, 8k, pixar style", "echoing cave drips textured damp atmosphere"),
    ("20_chimp_altar", "A chimpanzee placing the golden banana on an ancient stone altar in the heart of the jungle, 8k, pixar style", "stone grinding sound textured ancient mechanical click"),
    ("21_chimp_transformation", "The golden banana glowing brightly and transforming into a giant banana-shaped portal, 8k, pixar style", "pulsing magical energy textured mid-pitch hum"),
    ("22_chimp_portal", "A chimpanzee looking with wonder into the banana portal, showing a paradise of fruit trees, 8k, pixar style", "heavenly choir texture mid-pitch celestial drone"),
    ("23_chimp_step_in", "A chimpanzee stepping into the portal, half his body already inside the fruit paradise, 8k, pixar style", "electrical portal sizzle textured energy flow"),
    ("24_chimp_paradise", "A wide shot of the chimpanzee in fruit paradise, mountains of bananas, rivers of juice, 8k, pixar style", "peaceful nature sounds textured flowing juice"),
    ("25_chimp_friends", "A group of chimpanzees and penguins all eating fruit together in paradise, 8k, pixar style", "happy chimp chatter textured chewing sounds"),
    ("26_chimp_celebration", "A chimpanzee being lifted up by his friends in celebration, confetti made of fruit peels, 8k, pixar style", "joyful shouting textured party ambience"),
    ("27_chimp_nap", "A chimpanzee sleeping soundly on a bed of soft banana leaves in paradise, 8k, pixar style", "soft snoring sound textured jungle breeze"),
    ("28_chimp_dream", "A chimpanzee's dream bubble showing his next adventure: a trip to the moon made of cheese, 8k, pixar style", "dreamy synth texture mid-pitch twinkling"),
    ("29_chimp_sunset", "A chimpanzee watching a beautiful sunset over the fruit paradise, orange and purple sky, 8k, pixar style", "calm evening crickets textured soft wind"),
    ("30_chimp_slippery", "A chimpanzee trying to peel a golden banana but it's very slippery and flying out of his hands, funny expression, 8k, pixar style", "cartoon slip sound rubbery stretch textured pop"),
    ("31_chimp_wink", "Close up of the chimpanzee winking at the camera while holding a regular banana, 8k, pixar style", "cartoon wink sound textured mid-pitch ding"),
    ("32_title_card", "Movie title card 'THE BANANA QUEST' with a golden banana icon, tropical jungle background, professional typography, 8k", "triumphant orchestral hit deep cinematic bass")
]

VO_PROMPT = """
One chimp. One dream. And a ticket to the ultimate prize. 
Across the Great Divide, to the city of legends. 
He's not just hungry... he's on a mission. 
But the path is guarded. The stakes are high.
One slip could end it all.
Experience the adventure of a lifetime. 
From the streets of Fruit City to the heart of the jungle.
Witness the quest that changed everything.
The Banana Quest. 
Coming this Summer.
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
    
    is_local = os.path.isdir(model_id)
    pipe = DiffusionPipeline.from_pretrained(model_id, local_files_only=is_local, **pipe_kwargs)
    
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
    print(f"--- Generating SFX with Stable Audio Open ---")
    pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16).to(DEVICE)
    utils.remove_weight_norm(pipe)
    if args.scalenorm:
        utils.apply_scalenorm_to_transformer(pipe.transformer)

    os.makedirs(f"{ASSETS_DIR}/sfx", exist_ok=True)
    for s_id, _, sfx_prompt in SCENES:
        out_path = f"{ASSETS_DIR}/sfx/{s_id}.wav"
        if not os.path.exists(out_path):
            print(f"Generating SFX for: {s_id}")
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

    prompt = "An enthusiastic whimsical narrator voiceover, adventure story style, spoken word, cinematic"
    print("Generating voiceover audio...")
    audio = pipe(prompt, num_inference_steps=100, audio_end_in_s=45.0).audios[0]
    audio_np = audio.T.cpu().numpy()
    wavfile.write(out_path, 44100, (audio_np * 32767).astype(np.int16))
    del pipe
    torch.cuda.empty_cache()

def generate_music(args):
    print(f"--- Generating Music with Stable Audio ---")
    os.makedirs(f"{ASSETS_DIR}/music", exist_ok=True)
    out_path = f"{ASSETS_DIR}/music/chimp_theme.wav"
    if os.path.exists(out_path): return

    pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16).to(DEVICE)
    utils.remove_weight_norm(pipe)
    if args.scalenorm:
        utils.apply_scalenorm_to_transformer(pipe.transformer)

    prompt = "upbeat whimsical orchestral adventure theme, funny, lighthearted, cinematic, high quality"
    print("Generating music theme...")
    audio = pipe(prompt, num_inference_steps=100, audio_end_in_s=45.0).audios[0]
    audio_np = audio.T.cpu().numpy()
    wavfile.write(out_path, 44100, (audio_np * 32767).astype(np.int16))
    del pipe
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Chimp Assets")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model ID")
    parser.add_argument("--flux2", type=str, help="Path to FLUX.2 model directory (sets steps to 32)")
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
