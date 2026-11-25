#!/usr/bin/env python3
"""Pack the reference dumps that the C++ parity tests need into one baseline
GGUF file, so the tests read them with ggml's own GGUF reader.

    python scripts/dumps_to_baseline.py --dumps dumps-gold --nfe 6 \
        --output flow-baseline.gguf

Contents (tensor names; GGUF stores the numpy shape reversed, so a dump array
[2, N, 512] is a ggml tensor with ne = [512, N, 2]):

  adaln.*                 every float array of adaln_nfe{N}.npz
  <tag>.*                 every float array of flow_forward_nfe{N}.npz
                          (tags primer_step0, primer_kvupdate, steady0_step0,
                          steady0_final)
  audio.<key>             stacked, proj.linear, proj.norm, features of
                          wav2vec_raw.npz (audio projection parity)

Arrays with more than 4 dimensions (the KV caches, [layer, branch, head, pos,
dim]) have their leading dimensions merged: ggml tensors have at most 4.

KV: ``baseline.nfe``, ``baseline.steady0.start_pos`` and, per tag,
``baseline.<tag>.update_kv_cache`` and ``baseline.<tag>.kv_read``.

With ``--w2v2`` the script writes the wav2vec2 baseline instead (a separate
file, so the flow test does not load it):

  raw.<key>, normalized.<key>  pcm_in, model_input, conv.0, conv.6, interp,
                               feature_projection(.norm), pos_conv,
                               hidden.0 .. hidden.12, stacked, proj.linear,
                               proj.norm, features of wav2vec_<mode>.npz
  idle.pcm, idle.features      idle_pink.npz

KV: ``baseline.<mode>.n_frames``.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

AUDIO_KEYS = ("stacked", "proj.linear", "proj.norm", "features")
W2V2_KEYS = ("pcm_in", "model_input", "conv.0", "conv.6", "interp", "feature_projection",
             "feature_projection.norm", "pos_conv") + tuple(f"hidden.{i}" for i in range(13)) + \
            ("stacked", "proj.linear", "proj.norm", "features")
TAGS = ("primer_step0", "primer_kvupdate", "steady0_step0", "steady0_final")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dumps", required=True, help="directory written by dump_reference.py")
    ap.add_argument("--nfe", type=int, default=6, help="nfe of the detail run (default 6)")
    ap.add_argument("--output", required=True, help="baseline GGUF to write")
    ap.add_argument("--w2v2", action="store_true", help="write the wav2vec2 baseline instead")
    args = ap.parse_args()

    import gguf

    def load(name):
        path = os.path.join(args.dumps, name)
        if not os.path.exists(path):
            sys.exit(f"missing dump file: {path}")
        return np.load(path)

    w = gguf.GGUFWriter(args.output, "avatarforcing-baseline")
    w.add_uint32("baseline.nfe", args.nfe)
    n = 0

    def add(name, arr):
        nonlocal n
        if len(name) > 63:
            sys.exit(f"tensor name too long for GGUF: {name}")
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        if arr.ndim > 4:
            # ggml has 4 dims: merge the leading ones (the data order is unchanged).
            arr = arr.reshape((-1,) + arr.shape[-3:])
        w.add_tensor(name, arr)
        n += 1

    if args.w2v2:
        for mode in ("raw", "normalized"):
            z = load(f"wav2vec_{mode}.npz")
            w.add_uint32(f"baseline.{mode}.n_frames", int(z["n_frames"]))
            for k in W2V2_KEYS:
                add(f"{mode}.{k}", z[k])
        z = load("idle_pink.npz")
        add("idle.pcm", z["pcm"])
        add("idle.features", z["features"])
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
        print(f"wrote {args.output}: {n} tensors")
        return

    z = load(f"adaln_nfe{args.nfe}.npz")
    for k in z.files:
        add(f"adaln.{k}", z[k])

    z = load(f"flow_forward_nfe{args.nfe}.npz")
    for k in z.files:
        a = z[k]
        if a.dtype == np.bool_:
            w.add_bool(f"baseline.{k}", bool(a))
        elif a.dtype == np.float32:
            add(k, a)
        else:
            sys.exit(f"unexpected dtype {a.dtype} for {k}")
    for tag in TAGS:
        if f"{tag}.out.v" not in z.files:
            sys.exit(f"flow_forward_nfe{args.nfe}.npz has no tag {tag}")

    z = load(f"sampler_nfe{args.nfe}.npz")
    w.add_uint32("baseline.steady0.start_pos", int(z["steady.0.start_pos"]))

    z = load("wav2vec_raw.npz")
    for k in AUDIO_KEYS:
        add(f"audio.{k}", z[k])

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.output}: {n} tensors")


if __name__ == "__main__":
    main()
