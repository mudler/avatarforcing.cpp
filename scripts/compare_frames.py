#!/usr/bin/env python3
"""Compare two directories of frame_NNNNN.png files (the C++ CLI frames and
the reference frames of reference_stream.py) frame by frame.

    python scripts/compare_frames.py --got cpp-frames/ --ref ref-frames/ \\
        --min-ssim 0.995 [--csv ssim.csv]

Per frame: SSIM (Wang et al. 2004: Gaussian window 11, sigma 1.5, valid
region, on [0, 1] images, mean over the 3 channels; the same SSIM as
tests/test_motion_parity.cpp), PSNR in dB over all pixels and channels
(peak 255; identical frames count as 100 dB), the largest absolute pixel
difference and the mean absolute difference, both in uint8 levels. The exit status is 1
when the frame counts differ or a frame is below --min-ssim.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np


def gauss_kernel(k=11, sigma=1.5):
    x = np.arange(k) - k // 2
    g = np.exp(-(x * x) / (2 * sigma * sigma))
    return g / g.sum()


def ssim(a, b):
    """a, b: [H, W, 3] uint8."""
    from scipy.ndimage import correlate1d
    g = gauss_kernel()
    r = len(g) // 2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    total = 0.0
    for c in range(3):
        x = a[..., c].astype(np.float64) / 255.0
        y = b[..., c].astype(np.float64) / 255.0

        def filt(m):
            m = correlate1d(m, g, axis=1, mode="reflect")
            m = correlate1d(m, g, axis=0, mode="reflect")
            return m[r:-r, r:-r]   # valid region: the window lies inside the image

        mx, my = filt(x), filt(y)
        vx, vy, cxy = filt(x * x) - mx * mx, filt(y * y) - my * my, filt(x * y) - mx * my
        s = ((2 * mx * my + C1) * (2 * cxy + C2)) / ((mx * mx + my * my + C1) * (vx + vy + C2))
        total += s.mean()
    return total / 3.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--got", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--min-ssim", type=float, default=0.995)
    ap.add_argument("--csv", default=None, help="write per-frame values to this CSV file")
    args = ap.parse_args()

    from PIL import Image
    got = sorted(glob.glob(os.path.join(args.got, "frame_*.png")))
    ref = sorted(glob.glob(os.path.join(args.ref, "frame_*.png")))
    if not got or len(got) != len(ref):
        print(f"frame count mismatch: got {len(got)}, ref {len(ref)}")
        sys.exit(1)

    rows = []
    for i, (pg, pr) in enumerate(zip(got, ref)):
        a = np.asarray(Image.open(pg).convert("RGB"))
        b = np.asarray(Image.open(pr).convert("RGB"))
        if a.shape != b.shape:
            print(f"frame {i}: shape {a.shape} vs {b.shape}")
            sys.exit(1)
        d = np.abs(a.astype(np.int16) - b.astype(np.int16))
        mse = float(np.mean(d.astype(np.float64) ** 2))
        psnr = 100.0 if mse == 0 else min(100.0, 10.0 * np.log10(255.0 ** 2 / mse))
        rows.append((i, ssim(a, b), int(d.max()), float(d.mean()), psnr))

    s = np.array([r[1] for r in rows])
    p = np.array([r[4] for r in rows])
    mx = max(r[2] for r in rows)
    mean_abs = float(np.mean([r[3] for r in rows]))
    worst = int(np.argmin(s))
    n_fail = int((s < args.min_ssim).sum())
    if args.csv:
        with open(args.csv, "w") as f:
            f.write("frame,ssim,psnr_db,max_abs_diff,mean_abs_diff\n")
            for r in rows:
                f.write(f"{r[0]},{r[1]:.6f},{r[4]:.2f},{r[2]},{r[3]:.4f}\n")
    print(f"frames: {len(rows)}")
    print(f"SSIM: min {s.min():.6f} (frame {worst}), mean {s.mean():.6f}, max {s.max():.6f}")
    print(f"PSNR: min {p.min():.2f} dB (frame {int(np.argmin(p))}), mean {p.mean():.2f} dB")
    print(f"pixel diff (uint8 levels): max {mx}, mean {mean_abs:.4f}")
    print(f"gate SSIM >= {args.min_ssim}: {'PASS' if n_fail == 0 else f'FAIL ({n_fail} frames below)'}")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
