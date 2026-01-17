
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