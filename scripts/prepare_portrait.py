#!/usr/bin/env python3
"""Turn a face-centred portrait into the 512x512 RGB PNG that the reference
dumps (and later the C++ engine) take as input.

The upstream pipeline detects the face with S3FD (face_alignment), crops a
square around it and resizes to 512 with cv2.INTER_AREA. This script does not
run a face detector. It takes a centred square crop (or the square given with
--box) and resizes it with PIL's BOX filter, which is area averaging like
INTER_AREA when it reduces the image. An image that is already 512x512 is
written unchanged, only re-encoded as PNG so every reader decodes the same
pixels (JPEG decoders differ in their rounding).
"""
import argparse

from PIL import Image


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True, help="output .png")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--box", type=int, nargs=3, metavar=("X0", "Y0", "SIDE"),
                    help="square crop (pixels); default: largest centred square")
    args = ap.parse_args()

    img = Image.open(args.input).convert("RGB")
    w, h = img.size
    if args.box:
        x0, y0, side = args.box
    else:
        side = min(w, h)
        x0, y0 = (w - side) // 2, (h - side) // 2
    img = img.crop((x0, y0, x0 + side, y0 + side))
    if img.size != (args.size, args.size):
        img = img.resize((args.size, args.size), Image.BOX)
    img.save(args.output, format="PNG")
    print(f"wrote {args.output} ({args.size}x{args.size}) from {args.input} crop=({x0},{y0},{side})")


if __name__ == "__main__":
    main()
