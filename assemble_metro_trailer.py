
# Rewritten to use vidlib
from vidlib import assemble

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--assets", type=str, default="assets_metro", help="Path to assets directory"
    )
    parser.add_argument(
        "--output", type=str, default="metro_trailer.mp4", help="Output filename"
    )
    args = parser.parse_args()

    assemble.assemble_metro(args.assets, args.output)
