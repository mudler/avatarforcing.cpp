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
  sampler<M>.*            every float array of sampler_nfe{M}.npz, for each
                          nfe M of --sampler-nfe (sampler loop parity)

Arrays with more than 4 dimensions (the KV caches, [layer, branch, head, pos,
dim]) have their leading dimensions merged: ggml tensors have at most 4.

KV: ``baseline.nfe``, ``baseline.steady0.start_pos`` and, per tag,
``baseline.<tag>.update_kv_cache`` and ``baseline.<tag>.kv_read``. For the
sampler runs: ``baseline.sampler_nfe`` (the list of M), and per M
``baseline.sampler<M>.sigma_index`` (i32 array), ``.a_cfg_scale``,
``.steady_blocks`` and ``.steady.<b>.start_pos``.

With ``--component motion`` the file holds the motion autoencoder dumps
instead (for tests/test_motion_parity.cpp):

  enc.<key>               every float array of motion_encoder.npz, except
                          enc.convs.{0..6}, which are the same arrays as
                          feat.{6..0}
  dec.<key>               every float array of decoder_nfe{N}.npz;
                          dec.frames_u8 is stored as float32
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

AUDIO_KEYS = ("stacked", "proj.linear", "proj.norm", "features")
TAGS = ("primer_step0", "primer_kvupdate", "steady0_step0", "steady0_final")


def write_motion(args, load, add):
    z = load("motion_encoder.npz")
    n_res = sum(1 for k in z.files if k.startswith("enc.convs."))
    n_feat = sum(1 for k in z.files if k.startswith("feat."))
    dup = {f"enc.convs.{n_res - 3 - l}" for l in range(n_feat)}   # feat.l = res[::-1][2:][l]
    for k in z.files:
        if k in dup:
            continue
        if z[k].dtype != np.float32:
            sys.exit(f"unexpected dtype {z[k].dtype} for {k}")
        add(f"enc.{k}", z[k])
    z = load(f"decoder_nfe{args.nfe}.npz")
    for k in z.files:
        a = z[k]
        if k == "frames_u8":
            a = a.astype(np.float32)
        elif a.dtype != np.float32:
            sys.exit(f"unexpected dtype {a.dtype} for {k}")
        add(f"dec.{k}", a)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dumps", required=True, help="directory written by dump_reference.py")
    ap.add_argument("--nfe", type=int, default=6, help="nfe of the detail run (default 6)")
    ap.add_argument("--sampler-nfe", type=int, nargs="*", default=[6, 10],
                    help="nfe of the sampler runs to include (default 6 10; none: skip)")
    ap.add_argument("--output", required=True, help="baseline GGUF to write")
    ap.add_argument("--component", choices=("flow", "motion"), default="flow",
                    help="flow transformer (default) or motion autoencoder dumps")
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

    if args.component == "motion":
        write_motion(args, load, add)
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

    w.add_array("baseline.sampler_nfe", [int(m) for m in args.sampler_nfe])
    for m in args.sampler_nfe:
        z = load(f"sampler_nfe{m}.npz")
        p = f"sampler{m}"
        n_blocks = 0
        for k in z.files:
            a = z[k]
            if k == "sigma_index":
                w.add_array(f"baseline.{p}.sigma_index", [int(i) for i in a])
            elif k == "a_cfg_scale":
                w.add_float32(f"baseline.{p}.a_cfg_scale", float(a))
            elif k.endswith(".start_pos"):
                w.add_uint32(f"baseline.{p}.{k}", int(a))
                n_blocks += 1
            elif a.dtype == np.float32:
                add(f"{p}.{k}", a)
            else:
                sys.exit(f"unexpected dtype {a.dtype} for sampler_nfe{m}.npz:{k}")
        w.add_uint32(f"baseline.{p}.steady_blocks", n_blocks)

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
