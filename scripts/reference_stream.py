#!/usr/bin/env python3
"""Run the upstream offline streaming path (inference.py ``run_stream`` and
``AvatarForcing.inference_stream``) on the CPU in float32, with injected
noise, and write its uint8 frames as PNG files. The end-to-end gate compares
these frames with the frames of ``avatarforcing-cli --noise``.

    python scripts/reference_stream.py --avatarvoice AvatarVoice \\
        --flow-ckpt flow_transformer.pth --mae-ckpt motion_autoencoder.pth \\
        --wav2vec wav2vec2-base-960h --portrait portrait512.png \\
        --audio speech16k.wav --noise noise.f32 --nfe 6 --out ref-frames/

The noise file is raw little-endian float32: the primer noise [50, 512], then
the noise of each steady block [10, 512], in the order inference_stream draws
it with ``torch.randn``. With ``--noise-seed N`` the script writes such a file
first (numpy ``default_rng(N).standard_normal``) when ``--noise`` does not
exist yet.

What the script changes against ``run_stream``, and why:

- It runs on the CPU in float32, with no bf16 autocast and no
  ``torch.cuda.synchronize`` (run_stream needs CUDA).
- The portrait must be pre-cropped to 512x512. The S3FD face crop of
  ``DataProcessor.preprocess_face`` is skipped, as in the C++ CLI.
- The audio is read as 16-bit PCM / 32768 (what ``librosa.load(sr=16000)``
  returns for a 16 kHz mono 16-bit file), then normalized with
  ``Wav2Vec2FeatureExtractor`` and cut to 30 s, as ``default_aud_loader``
  and ``preprocess`` do.
- There is no user stream. ``user_wa`` is passed as zeros, the user frame
  list holds T placeholders and ``encode_user_motion`` returns zeros. The
  two-branch CFG of ``prepare_cfg_condition`` replaces all user inputs by
  zeros anyway, so this does not change the result.
- ``torch.randn`` returns the injected noise.
- The decoder runs one frame at a time instead of a batch of 10, to bound
  memory. Frames do not interact in the decoder.
- The frames go to PNG files instead of an ffmpeg pipe. The uint8 conversion
  is the one of run_stream (clamp, block-wide min/max, truncate).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dump_reference import build_model, read_portrait, read_wav_16k_mono  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--avatarvoice", required=True, help="AvatarVoice checkout (with the avatarforcing submodule)")
    ap.add_argument("--flow-ckpt", required=True)
    ap.add_argument("--mae-ckpt", required=True)
    ap.add_argument("--wav2vec", required=True, help="local facebook/wav2vec2-base-960h directory")
    ap.add_argument("--portrait", required=True, help="pre-cropped 512x512 RGB portrait")
    ap.add_argument("--audio", required=True, help="16 kHz mono 16-bit WAV")
    ap.add_argument("--noise", required=True, help="raw float32 noise file (read, or written with --noise-seed)")
    ap.add_argument("--noise-seed", type=int, default=None,
                    help="write --noise from numpy default_rng(seed) if it does not exist")
    ap.add_argument("--nfe", type=int, default=6)
    ap.add_argument("--a-cfg-scale", type=float, default=2.0)
    ap.add_argument("--out", required=True, help="directory for frame_NNNNN.png")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import Wav2Vec2FeatureExtractor
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)

    G, opt, _ = build_model(args)
    clip, blk, dim_w = G.num_frames_for_clip, G.num_frames_per_block, opt.dim_w

    # ---- inputs, as DataProcessor.preprocess builds them ----
    _, image = read_portrait(args.portrait)
    pcm = read_wav_16k_mono(args.audio)
    fe = Wav2Vec2FeatureExtractor.from_pretrained(args.wav2vec, local_files_only=True)
    avatar_a = torch.from_numpy(fe(pcm, sampling_rate=16000, return_tensors="pt").input_values[0].numpy())
    avatar_a = avatar_a[: int(30 * opt.sampling_rate)].unsqueeze(0)
    T = math.ceil(avatar_a.shape[-1] * G.fps / G.sampling_rate)
    n_steady = max(0, math.ceil((T - clip) / blk))

    # ---- noise ----
    n_noise = clip * dim_w + n_steady * blk * dim_w
    if not os.path.exists(args.noise):
        if args.noise_seed is None:
            sys.exit(f"{args.noise} does not exist; give --noise-seed to write it")
        rng = np.random.default_rng(args.noise_seed)
        rng.standard_normal(n_noise, dtype=np.float32).astype("<f4").tofile(args.noise)
        print(f"wrote {args.noise}: {n_noise} float32 values (seed {args.noise_seed})")
    flat = np.fromfile(args.noise, dtype="<f4")
    if flat.size < n_noise:
        sys.exit(f"{args.noise}: {flat.size} values, the run needs {n_noise}")
    queue = [flat[: clip * dim_w].reshape(1, clip, dim_w)]
    for b in range(n_steady):
        o = clip * dim_w + b * blk * dim_w
        queue.append(flat[o: o + blk * dim_w].reshape(1, blk, dim_w))
    queue.reverse()

    real_randn = torch.randn

    def injected_randn(*size, **kw):
        shape = tuple(size[0]) if len(size) == 1 and isinstance(size[0], (tuple, list)) else tuple(size)
        if not queue or tuple(queue[-1].shape) != shape:
            raise RuntimeError(f"unexpected torch.randn{shape}; noise left: {len(queue)}")
        return torch.from_numpy(queue.pop().copy())

    # ---- no user stream (zeroed by the two-branch CFG) ----
    G.encode_user_motion = lambda frames: torch.zeros(1, len(frames), dim_w)
    data = {
        "avatar_ref": torch.from_numpy(image),
        "avatar_a": avatar_a,
        "user_a": torch.zeros_like(avatar_a),
        "user_frame": [None] * T,
    }
    user_wa = torch.zeros(1, T, dim_w)

    # Decode one frame at a time to bound memory (a batch of 10 at 512x512
    # needs several GB). The decoder has no cross-frame interaction, so this
    # is the same computation as the batched _decode_one_block.
    def decode_one_block(latent_block, s_r_unsq, s_r_feats_expanded):
        frames = []
        for i in range(latent_block.shape[1]):
            style = (s_r_unsq + latent_block[:, i:i + 1]).reshape(1, -1)
            img, _ = G.motion_autoencoder.dec(style, alpha=None, feats=[f[i:i + 1] for f in s_r_feats_expanded])
            frames.append(img)
        return torch.stack(frames, dim=1)

    G._decode_one_block = decode_one_block

    os.makedirs(args.out, exist_ok=True)
    print(f"T = {T} audio frames, primer {clip} + {n_steady} steady blocks, nfe {args.nfe}, "
          f"a_cfg_scale {args.a_cfg_scale}", flush=True)
    total = 0
    t0 = time.perf_counter()
    torch.randn = injected_randn
    try:
        for block_idx, pixels, is_last in G.inference_stream(
                data=data, a_cfg_scale=args.a_cfg_scale, u_cfg_scale=0.0, nfe=args.nfe,
                user_wa=user_wa):
            # run_stream: [1, k, 3, H, W] -> [k, H, W, 3] uint8
            vid = pixels.squeeze(0).detach().clamp(-1, 1).cpu()
            vid = ((vid - vid.min()) / (vid.max() - vid.min() + 1e-8) * 255.0)
            vid = vid.permute(0, 2, 3, 1).contiguous().to(torch.uint8).numpy()
            for f in vid:
                Image.fromarray(f).save(os.path.join(args.out, f"frame_{total:05d}.png"))
                total += 1
            print(f"\rblock {block_idx}: {total} frames, {time.perf_counter() - t0:.1f} s", end="", flush=True)
    finally:
        torch.randn = real_randn
    print()
    if queue:
        print(f"warning: {len(queue)} noise blocks not used")
    print(f"wrote {total} frames to {args.out} in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
