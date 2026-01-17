from vidlib import assemble


if __name__ == "__main__":
    import sys
    assets = sys.argv[1] if len(sys.argv) > 1 else "assets_dalek"
    out = sys.argv[2] if len(sys.argv) > 2 else "dalek_trailer.mp4"
    assemble.assemble_dalek(assets, out)