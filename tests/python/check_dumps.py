#!/usr/bin/env python3
"""Compare two reference-dump directories written by scripts/dump_reference.py
with the same arguments, and require them to be identical.

This is the determinism self-check for M0: run the dump twice (separate
processes) and compare. Every array in every .npz must match bit for bit.

    python tests/python/check_dumps.py dumps-a dumps-b

Exit codes: 0 identical, 1 different, 77 skip (a directory is missing).
"""
import os
import sys

import numpy as np


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    a_dir, b_dir = sys.argv[1:]
    if not (os.path.isdir(a_dir) and os.path.isdir(b_dir)):
        print("check_dumps: dump directory missing; skipping", file=sys.stderr)
        sys.exit(77)
    files_a = sorted(f for f in os.listdir(a_dir) if f.endswith(".npz"))
    files_b = sorted(f for f in os.listdir(b_dir) if f.endswith(".npz"))
    bad = 0
    if not files_a:
        print(f"check_dumps: no .npz files in {a_dir}")
        sys.exit(1)
    if files_a != files_b:
        print(f"file sets differ: {sorted(set(files_a) ^ set(files_b))}")
        bad += 1
    n_arrays = 0
    for f in sorted(set(files_a) & set(files_b)):
        za, zb = np.load(os.path.join(a_dir, f)), np.load(os.path.join(b_dir, f))
        if sorted(za.files) != sorted(zb.files):
            print(f"{f}: key sets differ: {sorted(set(za.files) ^ set(zb.files))}")
            bad += 1
        diffs = []
        for k in sorted(set(za.files) & set(zb.files)):
            x, y = za[k], zb[k]
            n_arrays += 1
            if x.shape != y.shape or x.dtype != y.dtype:
                diffs.append(f"{k}: shape/dtype {x.shape}/{x.dtype} vs {y.shape}/{y.dtype}")
            elif not np.array_equal(x, y, equal_nan=True):
                d = np.abs(x.astype(np.float64) - y.astype(np.float64)).max()
                diffs.append(f"{k}: max |d| = {d:.3g}")
        if diffs:
            bad += len(diffs)
            print(f"{f}: {len(diffs)} array(s) differ")
            for d in diffs[:10]:
                print(f"  {d}")
        else:
            print(f"{f}: {len(za.files)} arrays identical")
    if bad:
        print(f"check_dumps: FAIL ({bad} difference(s))")
        sys.exit(1)
    print(f"check_dumps: PASS ({n_arrays} arrays in {len(files_a)} files, bit-identical)")


if __name__ == "__main__":
    main()
