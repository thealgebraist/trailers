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
PROJECT_NAME = "dalek"
ASSETS_DIR = f"assets_{PROJECT_NAME}"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# H200 Detection
IS_H200 = False
if DEVICE == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    if "H200" in gpu_name: IS_H200 = True

# Default values
DEFAULT_MODEL = "black-forest-labs/FLUX.1-dev" if IS_H200 else "black-forest-labs/FLUX.1-schnell"
DEFAULT_STEPS = 50 if IS_H200 else 16
DEFAULT_GUIDANCE = 3.5 if IS_H200 else 0.0
DEFAULT_QUANT = "none" if IS_H200 else "4bit"

# Scene Definitions
SCENES = [
    ("01_skaro_landscape", "Wide Shot: A bleak, metallic alien landscape Skaro, jagged towers of steel, grey skies, smoke. Thousands of Daleks moving in formation, cinematic, 8k", "dark brooding metallic throbbing drone industrial wind"),
    ("02_dalek_factory_closeup", "Close Up: A single Dalek with a small scratch on its casing staring blankly at a desolate robotic factory, gritty, high detail", "marching mechanical sounds distant laser fire metallic clanking"),
    ("03_horseshoe_closeup", "Extreme close up: A Dalek eye-stalk looking at a piece of rusted scrap metal that looks like a horseshoe, shallow depth of field", "textured digital glitch sound resonant piano note"),
    ("04_golden_eye", "Close up: A Dalek eye-piece flickering and glowing with a warm golden light, magical atmosphere", "wooden screen door creaking open rustic sound"),
    ("05_kansas_farmhouse", "Cinematic Shot: 1950s Kodachrome style. A golden wheat field in rural Kansas, a quaint farmhouse under a blue sky, nostalgic", "birds chirping wind in wheat cow mooing distance"),
    ("06_baby_dalek", "Mid Shot: An elderly human couple in overalls and an apron, lovingly holding a baby-sized Dalek casing wrapped in a knit blanket, heart-warming", "gentle acoustic guitar fingerpicking birds"),
    ("07_dalek_pie", "Small Dalek helping an old woman bake a pie, a kitchen whisk is taped to its plunger arm, messy flour on counter", "laughter of elderly people kitchen sounds bubbling"),
    ("08_dalek_fishing", "Old man teaching a Dalek to fish in a creek, a fishing rod is taped to its laser arm, sunny day", "splashing water river ambience nature sounds"),
    ("09_dalek_tractor", "A Dalek sitting proudly on a red farm tractor in a field, cinematic lighting", "tractor engine chugging diesel idle"),
    ("10_red_schoolhouse", "Wide Shot: A tiny red one-room schoolhouse in a rural setting, idyllic", "school bell ringing children shouting happy"),
    ("11_class_photo", "Group Shot: 5 children posing for a class photo. 4 human kids and one Dalek wearing a propeller beanie hat, 1950s style", "children cheering happy atmosphere"),
    ("12_town_citizens", "Montage of town citizens: A baker holding a baguette, a teacher at a chalkboard, and a cop tipping his hat, smiling", "bicycle bell ringing small town ambience"),
    ("13_hide_and_seek", "Action Shot: A Dalek playing hide and seek, poorly hiding behind a thin tree in a park", "playful footsteps giggling Dalek gliding on gravel"),
    ("14_dalek_prom", "A Dalek at a high school prom, a floral corsage taped to its dome, sitting next to a punch bowl, disco lights", "muffled 1950s prom music chatter"),
    ("15_rusty_reading", "A Dalek in a library, using its plunger to carefully turn the pages of a giant storybook, 8k", "paper turning sound textured library hush"),
    ("16_rusty_swinging", "A Dalek sitting on a wooden tire swing, gently swaying under a massive oak tree, sunset, 8k", "creaking wood sound textured wind in leaves"),
    ("17_rusty_bicycle", "A Dalek gliding alongside a group of children on bicycles, they are all laughing, 8k", "bicycle bell sound textured gravel crunch"),
    ("18_rusty_storm", "A Dalek standing in the rain, looking up at a dark storm cloud that looks like a Dalek saucer, 1950s style", "thunder rumble textured rain on metal"),
    ("19_rusty_goodbye", "A Dalek waving its plunger at Nana and Pop-Pop from the back of a truck, they are crying, 8k", "truck engine receding textured soft weeping"),
    ("20_skaro_return", "Dramatic: A Dalek back on the bleak Skaro landscape, surrounded by dozens of angry, dark Daleks, high contrast", "ominous silence weapons powering up hum"),
    ("21_quivering_eye", "Extreme close up: A Dalek eye-stalk quivering, showing emotion, blue light flickering", "textured mechanical growl electrical static"),
    ("22_rusty_defiant", "A single Dalek standing alone against a massive army of Daleks on Skaro, 8k, cinematic", "low frequency pulse textured army chanting"),
    ("23_supreme_dalek_pie", "Key Scene: A Dalek facing a giant Supreme Dalek. The small Dalek extends its plunger, holding a warm apple pie with steam rising", "tense silence steam hiss"),
    ("24_pie_impact", "Close up of the Supreme Dalek's eye-stalk looking with confusion at the apple pie, 8k", "digital glitch sound textured mechanical click"),
    ("25_dalek_confusion", "A group of Daleks all looking at each other, their lights flickering in confusion, 8k", "static noise textured electronic chatter"),
    ("26_rusty_hugging", "The Dalek giving a clumsy hug to a town policeman, heartwarming moment", "triumphant orchestral swell"),
    ("27_rusty_flying", "A Dalek flying through space with a giant apple pie icon painted on its casing, 8k", "space vacuum hum textured rocket fire"),
    ("28_earth_paradise", "A wide shot of Earth as a green and blue paradise, seen from space, 8k", "celestial drone textured cosmic shimmer"),
    ("29_rusty_landing", "A Dalek saucer landing in the middle of the Kansas wheat field, 1950s style", "saucer landing hum textured wind in wheat"),
    ("30_reunion", "Nana and Pop-Pop running towards the Dalek in the wheat field, arms open wide, 8k", "happy shouting textured footsteps in grass"),
    ("31_title_card", "Cinematic title card: 'A DALEK COMES HOME' in bold metallic font, subtitle 'EXTERMINATE THE LONELINESS'", "deep cinematic bass boom hawk screech"),
    ("32_birthday_cake", "A Dalek trying to blow out birthday candles on a cake, its laser beam accidentally fires and explodes the cake into crumbs", "laser blast sound explosion muffled oops"),
]

VO_SCRIPT = """
In a universe... of infinite hate... where emotion is a crime... and compassion is deleted...
One soldier... is about to remember... where he really came from.
He wasn't born in a factory. He was found... in a cornfield.
Raised by Nana and Pop-Pop. They didn't see a killing machine. They saw... their little Rusty.
They taught him to read. They taught him to fish. And they taught him the most dangerous weapon of all...
Love.
But his past is calling. Skaro demands its soldier back.
He's going home. And he's bringing dessert.
Witness the most unlikely hero in the galaxy.
This summer...
A Dalek Comes Home.
"""

def generate_images(args):
    model_id = args.flux2 if args.flux2 else args.model
    steps = args.steps if args.steps != DEFAULT_STEPS else (32 if args.flux2 else args.steps)
    
    print(f"--- Generating {len(SCENES)} {model_id} Images ({steps} steps) on {DEVICE} ---")
    pipe_kwargs = {"torch_dtype": torch.bfloat16 if DEVICE == "cuda" else torch.float32}

    if args.quant != "none" and DEVICE == "cuda":
        from diffusers import PipelineQuantizationConfig
        backend = "bitsandbytes_4bit" if args.quant == "4bit" else "bitsandbytes_8bit"
        if args.quant == "8bit": pipe_kwargs["torch_dtype"] = torch.float16
        pipe_kwargs["quantization_config"] = PipelineQuantizationConfig(
            quant_backend=backend, quant_kwargs={"load_in_4bit" if args.quant == "4bit" else "load_in_8bit": True},
            components_to_quantize=["transformer"]
        )
    
    pipe = DiffusionPipeline.from_pretrained(model_id, local_files_only=os.path.isdir(model_id), **pipe_kwargs)
    utils.remove_weight_norm(pipe)
    if args.scalenorm: utils.apply_stability_improvements(pipe.transformer, use_scalenorm=True)

    if args.offload and DEVICE == "cuda":
        pipe.enable_model_cpu_offload()
    elif args.quant != "none" and DEVICE == "cuda":
        for name, component in pipe.components.items():
            if name != "transformer" and hasattr(component, "to"):
                component.to(DEVICE)
    else:
        pipe.to(DEVICE)

    os.makedirs(f"{ASSETS_DIR}/images", exist_ok=True)
    for s_id, prompt, _ in SCENES:
        out_path = f"{ASSETS_DIR}/images/{s_id}.png"
        if not os.path.exists(out_path):
            image = pipe(prompt, num_inference_steps=steps, guidance_scale=args.guidance, width=1280, height=720).images[0]
            image.save(out_path)
    del pipe
    torch.cuda.empty_cache()

def generate_sfx(args):
    print(f"--- Generating SFX with Stable Audio on {DEVICE} ---")
    pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16).to(DEVICE)
    utils.remove_weight_norm(pipe)
    if args.scalenorm: utils.apply_stability_improvements(pipe.transformer, use_scalenorm=True)

    os.makedirs(f"{ASSETS_DIR}/sfx", exist_ok=True)
    for s_id, _, sfx_prompt in SCENES:
        out_path = f"{ASSETS_DIR}/sfx/{s_id}.wav"
        if not os.path.exists(out_path):
            audio = pipe(sfx_prompt, num_inference_steps=100, audio_end_in_s=10.0).audios[0]
            wavfile.write(out_path, 44100, (audio.T.cpu().numpy() * 32767).astype(np.int16))
    del pipe
    torch.cuda.empty_cache()

def generate_voiceover(args):
    print(f"--- Generating Voiceover with Bark (Intelligible TTS) on {DEVICE} ---")
    os.makedirs(f"{ASSETS_DIR}/voice", exist_ok=True)
    out_path = f"{ASSETS_DIR}/voice/voiceover_full.wav"
    if os.path.exists(out_path): return

    tts = pipeline("text-to-speech", model="suno/bark", device=DEVICE)
    lines = [l for l in VO_SCRIPT.split('\n') if l.strip()]
    full_audio = []
    sampling_rate = 24000
    for line in lines:
        print(f"  Speaking: {line[:30]}...")
        output = tts(line, voice_preset="v2/en_speaker_6")
        full_audio.append(output["audio"].flatten())
        full_audio.append(np.zeros(int(output["sampling_rate"] * 0.8)))
        sampling_rate = output["sampling_rate"]
        
    wavfile.write(out_path, sampling_rate, (np.concatenate(full_audio) * 32767).astype(np.int16))
    del tts
    torch.cuda.empty_cache()

def generate_music(args):
    print(f"--- Generating Music with Stable Audio on {DEVICE} ---")
    os.makedirs(f"{ASSETS_DIR}/music", exist_ok=True)
    out_path = f"{ASSETS_DIR}/music/dalek_theme.wav"
    if os.path.exists(out_path): return

    pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16).to(DEVICE)
    utils.remove_weight_norm(pipe)
    if args.scalenorm: utils.apply_stability_improvements(pipe.transformer, use_scalenorm=True)

    prompt = "dark industrial synth drone transitions to warm acoustic guitar builds to orchestral crescendo, cinematic"
    audio = pipe(prompt, num_inference_steps=100, audio_end_in_s=45.0).audios[0]
    wavfile.write(out_path, 44100, (audio.T.cpu().numpy() * 32767).astype(np.int16))
    del pipe
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Dalek Assets")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--flux2", type=str)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--guidance", type=float, default=DEFAULT_GUIDANCE)
    parser.add_argument("--quant", type=str, default=DEFAULT_QUANT, choices=["none", "4bit", "8bit"])
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--scalenorm", action="store_true")
    args = parser.parse_args()
    generate_images(args); generate_sfx(args); generate_voiceover(args); generate_music(args)