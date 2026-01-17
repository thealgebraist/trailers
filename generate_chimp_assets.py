import torch
import os
import numpy as np
import scipy.io.wavfile as wavfile
import utils
import gc
from diffusers import DiffusionPipeline, StableAudioPipeline
from transformers import BitsAndBytesConfig

def generate_images(args):
    pass  # TODO: Adapt from metro for chimp prompts

def generate_sfx(args):
    pass  # TODO: Adapt from metro for chimp SFX

def generate_voiceover(args):
    pass  # TODO: Adapt from metro for chimp voiceover

def generate_music(args):
    pass  # TODO: Adapt from metro for chimp music

# Rewritten to use vidlib
from vidlib import assets

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate Chimp Assets")
    parser.add_argument("--model", type=str)
    parser.add_argument("--flux2", type=str)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--guidance", type=float)
    parser.add_argument("--quant", type=str, choices=["none", "4bit", "8bit"])
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--scalenorm", action="store_true")
    args = parser.parse_args()

    assets.chimp_generate_images(args)
    assets.chimp_generate_sfx(args)
    assets.chimp_generate_voiceover(args)
    assets.chimp_generate_music(args)