#!/usr/bin/env python3
"""Convert the AvatarForcing checkpoints and wav2vec2-base-960h into one
self-contained, metadata-driven GGUF file.

Inputs:
  --flow-ckpt  flow_transformer.pth   (flow transformer + audio_projection)
  --mae-ckpt   motion_autoencoder.pth (motion encoder + decoder)
  --wav2vec    directory with config.json + model.safetensors of
               facebook/wav2vec2-base-960h (Wav2Vec2ForCTC checkpoint)
  --config     the upstream configs/inference.yaml (optional; the built-in
               defaults are the values of that file at upstream 566f8e1)

Every dimension and hyperparameter goes into GGUF KV under the
``avatarforcing.`` prefix. Tensor names are stable and documented in
docs/conversion.md. All tensors are F32. The converter applies these exact
transforms (each one is verified by tests/python/check_convert.py):

  flow transformer
    - nn.MultiheadAttention packed in_proj (c_embedder) is split into
      q_proj / k_proj / v_proj.
    - The checkpoint buffers alignment_mask and block_bidir_causal_lookahead_mask
      are dropped. They are the inverse of the masks the model builds and uses,
      and the upstream loader never loads them.
    - Added: ft.rope.inv_freq [head_dim/2] and ft.t_embedder.freqs [128], the
      exact float32 frequency vectors torch computes.
  motion autoencoder
    - EqualConv2d / EqualLinear / ModulatedConv2d weights are multiplied by
      their runtime scale (1/sqrt(fan_in)); EqualLinear biases by lr_mul (1).
    - ModulatedConv2d weight [1,O,I,k,k] is squeezed to [O,I,k,k]. Demodulated
      convs get ``.wsq`` [O,I] = sum over (kh,kw) of the folded weight squared,
      so demod = rsqrt(style^2 @ wsq^T + 1e-8).
    - Direction.weight [512,20] is replaced by ``dec.direction.q`` [512,20]: the
      Q factor of a float64 QR of (weight + 1e-8), where the +1e-8 is done in
      float32 as upstream does it.
    - Bias tensors of shape [1,C,1,1] are flattened to [C].
    - Dropped: NoiseInjection weights (the decoder is always called with
      noise=None) and dec.to_rgb1 (built but never used in forward).
    - Dropped: the Blur / Upsample kernel buffers (35); the converter checks
      that they equal the [1,3,3,1] kernel the KV describes (gain 1 in the
      encoder, 4 in the decoder).
  wav2vec2
    - ``wav2vec2.`` -> ``w2v2.`` (voice-detect.cpp convention); the weight-norm
      tensors of the positional conv stay as weight_g / weight_v; lm_head is
      dropped.
  extras
    - sched.sigmas / sched.timesteps [1000]: the FlowMatchScheduler tables
      (float32) that solve_closed looks timesteps up in.
    - audio.idle_pcm [80000]: the 5 s pink-noise "room tone" PCM (seed 42,
      -50 dBFS) whose wav2vec2 features drive the avatar during silence.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys

import numpy as np

ARCH = "avatarforcing"
K = ARCH + "."

# Values of avatarforcing/configs/inference.yaml at upstream 566f8e1.
DEFAULT_CONFIG = {
    "fps": 25, "sampling_rate": 16000, "input_size": 512, "dim_m": 20, "dim_a": 512,
    "dim_w": 512, "dim_h": 1024, "style_dim": 512, "num_train_timestep": 1000, "nfe": 10,
    "sec": 2, "num_frames_per_block": 10, "only_last_features": False,
    "a_cfg_scale": 2.0, "attention_window": 2, "num_heads": 8, "mlp_ratio": 4.0,
    "transformer_depth": 8, "block_causal": True,
}

# Live-app behaviour (voicechat/avatar_worker.py) that is not in the config file.
APP = {
    "nfe": 6,                 # AvatarWorker(nfe=6)
    "reanchor_every": 112,    # REANCHOR_EVERY
    "reanchor_max_defer": 162,  # MAX_DEFER
    "reanchor_fade_frames": 25,  # REANCHOR_FADE_FRAMES
    "idle_seconds": 5, "idle_level_dbfs": -50.0, "idle_seed": 42,
}

# Hardcoded in the upstream model code (not in the config file).
CODE = {
    "ln_eps": 1e-6,            # DFoTBlock / FinalLayer LayerNorm(eps=1e-6), no affine
    "torch_ln_eps": 1e-5,      # nn.LayerNorm default (c_embedder, audio_projection)
    "cond_heads": 4,           # AudioMotionAttention MultiheadAttention(num_heads=4)
    "lookahead": 2,            # make_block_bidir_causal_lookahead_mask(..., 2)
    "rope_theta": 10000.0,
    "t_max_period": 10000.0,
    "context_frames": 2,
    "sched_shift": 1.0, "sched_sigma_min": 0.0, "sched_sigma_max": 1.0,
    "mae_blur_kernel": [1, 3, 3, 1],
    "mae_leaky_slope": 0.2,
    "mae_act_gain": math.sqrt(2.0),
    "mae_demod_eps": 1e-8,
    "mae_channel_multiplier": 1,
    "group_norm_eps": 1e-5,    # nn.GroupNorm default in the wav2vec2 conv front end
}

MAX_NAME = 63  # ggml GGML_MAX_NAME is 64 including the terminator


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config(path):
    cfg = dict(DEFAULT_CONFIG)
    if path:
        import yaml
        with open(path) as f:
            user = yaml.safe_load(f)
        for k in DEFAULT_CONFIG:
            if k in user:
                cfg[k] = user[k]
    return cfg


def shaped_pink_noise(n_samples, sample_rate, level_dbfs, seed):
    """Pink (1/f) noise at level_dbfs RMS, mono float32. Same algorithm as the
    upstream idle-feature generator (white -> rfft -> 1/sqrt(f) -> irfft ->
    RMS scale)."""
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n_samples).astype(np.float32)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)
    spec *= 1.0 / np.sqrt(np.maximum(freqs, 1.0))
    pink = np.fft.irfft(spec, n=n_samples).astype(np.float32)
    rms = float(np.sqrt(np.mean(pink * pink)))
    pink *= (10.0 ** (level_dbfs / 20.0)) / (rms + 1e-12)
    return pink


def scheduler_tables(num_steps, shift, sigma_min, sigma_max):
    """FlowMatchScheduler(num_steps=1000, shift=1, sigma_min=0,
    extra_one_step=False).set_timesteps(1000), float32 like torch."""
    import torch
    sig = torch.linspace(sigma_min + (sigma_max - sigma_min), sigma_min, num_steps)
    sig = shift * sig / (1 + (shift - 1) * sig)
    return sig.numpy().astype(np.float32), (sig * num_steps).numpy().astype(np.float32)


# --------------------------------------------------------------------------
# flow transformer + audio_projection
# --------------------------------------------------------------------------

def convert_flow(sd):
    import torch
    out = {}
    dropped = []
    for name, t in sd.items():
        if name.startswith("audio_encoder.audio_projection."):
            out["audio_projection." + name[len("audio_encoder.audio_projection."):]] = t.numpy()
            continue
        if not name.startswith("flow_transformer."):
            raise SystemExit(f"unexpected key in flow checkpoint: {name}")
        n = name[len("flow_transformer."):]
        if n in ("alignment_mask", "block_bidir_causal_lookahead_mask"):
            dropped.append(name)
            continue
        if n.endswith(".in_proj_weight") or n.endswith(".in_proj_bias"):
            base, kind = n.rsplit(".", 1)
            suffix = "weight" if kind == "in_proj_weight" else "bias"
            a = t.numpy()
            d = a.shape[0] // 3
            for i, p in enumerate(("q_proj", "k_proj", "v_proj")):
                out[f"ft.{base}.{p}.{suffix}"] = a[i * d:(i + 1) * d]
            continue
        out["ft." + n] = t.numpy()
    return out, dropped


# --------------------------------------------------------------------------
# motion autoencoder
# --------------------------------------------------------------------------

def mae_kind(n):
    """Classify a motion-autoencoder parameter by its upstream module type."""
    if n.endswith(".noise.weight"):
        return "noise"
    if n.endswith(".kernel"):
        return "blur_kernel"                       # Blur / Upsample buffer
    if n.startswith("dec.to_rgb1."):
        return "unused"
    if n == "dec.direction.weight":
        return "direction"
    if n == "dec.input.input":
        return "const"
    if ".modulation." in n:
        return "equal_linear"                      # EqualLinear(style_dim, in_ch, bias_init=1)
    if n.startswith("enc.fc."):
        return "equal_linear"
    if (n.startswith("dec.convs.") or n.startswith("dec.conv1.")) and n.endswith(".conv.weight"):
        return "modconv_demod"                     # StyledConv -> ModulatedConv2d(demodulate=True)
    if n.startswith("dec.to_flows.") and n.endswith(".conv.weight"):
        return "modconv"                           # ToFlow -> ModulatedConv2d(demodulate=False)
    if n.endswith(".weight") and (n.startswith("enc.net_app.convs.") or n.startswith("dec.to_rgbs.")):
        return "equal_conv"
    if n.endswith(".bias"):
        return "bias"
    raise SystemExit(f"motion autoencoder: unclassified parameter {n}")


def convert_mae(sd, cfg):
    import torch
    out = {}
    dropped = []
    for name, t in sd.items():
        if not name.startswith("motion_autoencoder."):
            raise SystemExit(f"unexpected key in motion checkpoint: {name}")
        n = name[len("motion_autoencoder."):]
        kind = mae_kind(n)
        a = t.numpy().astype(np.float32)
        if kind in ("noise", "unused"):
            dropped.append(name)
            continue
        if kind == "blur_kernel":
            # Buffers rebuilt from mae.blur_kernel (the upstream loader copies
            # parameters only). Check that they hold what the KV describes.
            k1 = np.asarray(CODE["mae_blur_kernel"], np.float32)
            k2 = np.outer(k1, k1)
            k2 /= k2.sum()
            gain = 4.0 if n.startswith("dec.") else 1.0
            if not np.array_equal(a, (k2 * gain).astype(np.float32)):
                raise SystemExit(f"motion autoencoder: unexpected blur kernel in {name}")
            dropped.append(name)
            continue
        if kind == "direction":
            w = (t + 1e-8).to(torch.float64)       # upstream adds 1e-8 in float32
            q, _ = torch.linalg.qr(w)
            out["mae.dec.direction.q"] = q.to(torch.float32).numpy()
            continue
        if kind == "const":
            out["mae." + n] = a
            continue
        if kind == "equal_linear":
            if n.endswith(".weight"):
                scale = 1.0 / math.sqrt(a.shape[1])    # lr_mul = 1 everywhere in this model
                out["mae." + n] = (t * scale).numpy()
            else:
                out["mae." + n] = (t * 1).numpy()      # bias * lr_mul
            continue
        if kind == "equal_conv":
            o, i, kh, kw = a.shape
            out["mae." + n] = (t * (1.0 / math.sqrt(i * kh * kw))).numpy()
            continue
        if kind in ("modconv_demod", "modconv"):
            _, o, i, kh, kw = a.shape
            w = t[0] * (1.0 / math.sqrt(i * kh * kw))  # self.scale * self.weight
            out["mae." + n] = w.numpy()
            if kind == "modconv_demod":
                out["mae." + n[: -len("weight")] + "wsq"] = w.pow(2).sum([2, 3]).numpy()
            continue
        if kind == "bias":
            out["mae." + n] = a.reshape(-1)
            continue
    return out, dropped


# --------------------------------------------------------------------------
# wav2vec2
# --------------------------------------------------------------------------

def convert_w2v2(path):
    import json
    from safetensors.numpy import load_file
    with open(os.path.join(path, "config.json")) as f:
        c = json.load(f)
    pre = {}
    pp = os.path.join(path, "preprocessor_config.json")
    if os.path.exists(pp):
        with open(pp) as f:
            pre = json.load(f)
    st = os.path.join(path, "model.safetensors")
    sd = load_file(st)
    out, dropped = {}, []
    for k, v in sd.items():
        if k.startswith("lm_head.") or k == "wav2vec2.masked_spec_embed":
            dropped.append(k)
            continue
        if not k.startswith("wav2vec2."):
            raise SystemExit(f"unexpected wav2vec2 key {k}")
        n = "w2v2." + k[len("wav2vec2."):]
        n = n.replace(".parametrizations.weight.original0", ".weight_g")
        n = n.replace(".parametrizations.weight.original1", ".weight_v")
        out[n] = v.astype(np.float32)
    meta = {
        "hidden_size": int(c["hidden_size"]), "n_layers": int(c["num_hidden_layers"]),
        "n_heads": int(c["num_attention_heads"]), "ff_dim": int(c["intermediate_size"]),
        "num_conv_layers": int(c["num_feat_extract_layers"]),
        "conv_dims": [int(x) for x in c["conv_dim"]], "conv_kernels": [int(x) for x in c["conv_kernel"]],
        "conv_strides": [int(x) for x in c["conv_stride"]],
        "feat_extract_norm": str(c["feat_extract_norm"]),
        "feat_extract_activation": str(c["feat_extract_activation"]),
        "hidden_act": str(c["hidden_act"]), "do_stable_layer_norm": bool(c["do_stable_layer_norm"]),
        "layer_norm_eps": float(c["layer_norm_eps"]), "conv_bias": bool(c.get("conv_bias", False)),
        "num_conv_pos_embeddings": int(c["num_conv_pos_embeddings"]),
        "num_conv_pos_embedding_groups": int(c["num_conv_pos_embedding_groups"]),
        "do_normalize": bool(pre.get("do_normalize", False)),
        "model_id": str(c.get("_name_or_path", "")),
    }
    return out, dropped, meta


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flow-ckpt", required=True)
    ap.add_argument("--mae-ckpt", required=True)
    ap.add_argument("--wav2vec", required=True)
    ap.add_argument("--config", default=None, help="upstream configs/inference.yaml")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    try:
        import gguf
        import torch
    except ImportError as e:
        print(f"AVATARFORCING_CONVERT_DEPS_MISSING: {e}", file=sys.stderr)
        sys.exit(2)

    cfg = load_config(args.config)
    tensors = {}
    dropped = []

    sd = torch.load(args.flow_ckpt, map_location="cpu", weights_only=True, mmap=True)
    t, d = convert_flow(sd)
    tensors.update(t)
    dropped += d
    sd = torch.load(args.mae_ckpt, map_location="cpu", weights_only=True, mmap=True)
    t, d = convert_mae(sd, cfg)
    tensors.update(t)
    dropped += d
    t, d, w2v = convert_w2v2(args.wav2vec)
    tensors.update(t)
    dropped += d

    # ---- shapes -> dims (metadata derived from tensors where possible) ----
    dim_h, dim_2w = tensors["ft.x_embedder.proj.weight"].shape
    dim_w = dim_2w // 2
    depth = len({n.split(".")[2] for n in tensors if n.startswith("ft.blocks.")})
    n_heads = int(cfg["num_heads"])
    head_dim = dim_h // n_heads
    mlp_hidden = tensors["ft.blocks.0.mlp.fc1.weight"].shape[0]
    freq_dim = tensors["ft.t_embedder.mlp.0.weight"].shape[1]
    dim_a = tensors["ft.c_embedder.motion_embedder.0.weight"].shape[0]
    proj_out, proj_in = tensors["audio_projection.0.weight"].shape
    n_taps = proj_in // w2v["hidden_size"]
    clip = int(cfg["sec"] * cfg["fps"])
    block = int(cfg["num_frames_per_block"])
    ctx = CODE["context_frames"]
    spf = int(cfg["sampling_rate"] // cfg["fps"])
    assert dim_w == cfg["dim_w"] and dim_h == cfg["dim_h"] and depth == cfg["transformer_depth"]
    assert mlp_hidden == int(dim_h * cfg["mlp_ratio"]) and dim_a == cfg["dim_a"]
    assert n_taps == (1 if cfg["only_last_features"] else w2v["n_layers"])

    # ---- extra tensors ----
    inv_freq = 1.0 / (CODE["rope_theta"] ** (torch.arange(0, head_dim, 2)[: head_dim // 2].float() / head_dim))
    tensors["ft.rope.inv_freq"] = inv_freq.numpy().astype(np.float32)
    half = freq_dim // 2
    tensors["ft.t_embedder.freqs"] = torch.exp(
        -math.log(CODE["t_max_period"]) * torch.arange(start=0, end=half, dtype=torch.float32) / half).numpy()
    sig, ts = scheduler_tables(int(cfg["num_train_timestep"]), CODE["sched_shift"],
                               CODE["sched_sigma_min"], CODE["sched_sigma_max"])
    tensors["sched.sigmas"] = sig
    tensors["sched.timesteps"] = ts
    idle = shaped_pink_noise(APP["idle_seconds"] * int(cfg["sampling_rate"]), int(cfg["sampling_rate"]),
                             APP["idle_level_dbfs"], APP["idle_seed"])
    tensors["audio.idle_pcm"] = idle

    # motion autoencoder channel tables (upstream EncoderApp / Synthesis dicts)
    size = int(cfg["input_size"])
    log_size = int(math.log2(size))
    mae_enc_ch = [tensors[f"mae.enc.net_app.convs.0.0.weight"].shape[0]]
    i = 1
    while f"mae.enc.net_app.convs.{i}.conv2.1.weight" in tensors:
        mae_enc_ch.append(tensors[f"mae.enc.net_app.convs.{i}.conv2.1.weight"].shape[0])
        i += 1
    n_res = i - 1
    mae_dec_ch = [tensors["mae.dec.input.input"].shape[1]]
    lv = 0
    while f"mae.dec.convs.{2 * lv}.conv.weight" in tensors:
        mae_dec_ch.append(tensors[f"mae.dec.convs.{2 * lv}.conv.weight"].shape[0])
        lv += 1
    n_levels = lv
    motion_dim = tensors["mae.dec.direction.q"].shape[1]
    assert n_levels == log_size - 2 and n_res == log_size - 2 and motion_dim == cfg["dim_m"]

    # ---- write ----
    for n in tensors:
        if len(n) > MAX_NAME:
            raise SystemExit(f"tensor name too long for ggml ({len(n)}): {n}")

    w = gguf.GGUFWriter(args.output, ARCH)
    w.add_name("AvatarForcing")
    w.add_string("general.license", "cc-by-nc-4.0")
    w.add_string("general.source.url", "https://github.com/TaekyungKi/AvatarForcing")
    w.add_string(K + "source.flow_sha256", sha256(args.flow_ckpt))
    w.add_string(K + "source.mae_sha256", sha256(args.mae_ckpt))
    w.add_string(K + "source.wav2vec2", w2v["model_id"])
    w.add_uint32(K + "format_version", 1)

    # timing / streaming
    w.add_uint32(K + "fps", int(cfg["fps"]))
    w.add_uint32(K + "sample_rate", int(cfg["sampling_rate"]))
    w.add_uint32(K + "samples_per_frame", spf)
    w.add_uint32(K + "image_size", size)
    w.add_uint32(K + "clip_frames", clip)
    w.add_uint32(K + "block_frames", block)
    w.add_uint32(K + "context_frames", ctx)
    w.add_uint32(K + "stream.self_cache_len", clip - block - ctx)        # 38
    w.add_uint32(K + "stream.cross_cache_len", int(cfg["attention_window"]))  # 2
    w.add_uint32(K + "stream.kv_shift", block)
    w.add_uint32(K + "stream.first_steady_start_pos", clip - ctx)       # 48
    w.add_uint32(K + "stream.reanchor_every", APP["reanchor_every"])
    w.add_uint32(K + "stream.reanchor_max_defer", APP["reanchor_max_defer"])
    w.add_uint32(K + "stream.reanchor_fade_frames", APP["reanchor_fade_frames"])
    w.add_string(K + "output.u8_mode", "clamp_block_minmax_trunc")

    # sampler
    w.add_uint32(K + "sampler.default_nfe", APP["nfe"])
    w.add_uint32(K + "sampler.config_nfe", int(cfg["nfe"]))
    w.add_uint32(K + "sampler.num_train_timesteps", int(cfg["num_train_timestep"]))
    w.add_string(K + "sampler.steps", "linspace(num_train_timesteps, 0, nfe - 1)")
    w.add_string(K + "sampler.solver", "closed_form_argmin_f64")
    w.add_uint32(K + "sampler.sigma_table_len", int(sig.shape[0]))
    w.add_float32(K + "sampler.shift", CODE["sched_shift"])
    w.add_float32(K + "sampler.sigma_min", CODE["sched_sigma_min"])
    w.add_float32(K + "sampler.sigma_max", CODE["sched_sigma_max"])
    w.add_float32(K + "cfg.audio_scale", float(cfg["a_cfg_scale"]))
    w.add_array(K + "cfg.branches", ["uncond", "audio"])

    # flow transformer
    w.add_uint32(K + "flow.dim_w", dim_w)
    w.add_uint32(K + "flow.dim_h", dim_h)
    w.add_uint32(K + "flow.dim_a", dim_a)
    w.add_uint32(K + "flow.depth", depth)
    w.add_uint32(K + "flow.n_heads", n_heads)
    w.add_uint32(K + "flow.head_dim", head_dim)
    w.add_uint32(K + "flow.mlp_hidden", mlp_hidden)
    w.add_float32(K + "flow.mlp_ratio", float(cfg["mlp_ratio"]))
    w.add_string(K + "flow.mlp_act", "gelu_tanh")
    w.add_float32(K + "flow.ln_eps", CODE["ln_eps"])
    w.add_uint32(K + "flow.freq_embed_dim", int(freq_dim))
    w.add_float32(K + "flow.t_max_period", CODE["t_max_period"])
    w.add_float32(K + "flow.rope.theta", CODE["rope_theta"])
    w.add_uint32(K + "flow.rope.dim", head_dim)
    w.add_string(K + "flow.rope.layout", "interleaved")   # pairs (2i, 2i+1), ggml mode 0
    w.add_uint32(K + "flow.self_mask.block", block)
    w.add_uint32(K + "flow.self_mask.lookahead", CODE["lookahead"])
    w.add_uint32(K + "flow.cross_mask.window", int(cfg["attention_window"]))
    w.add_bool(K + "flow.block_causal", bool(cfg["block_causal"]))
    w.add_array(K + "flow.adaln.order", ["shift_msa", "scale_msa", "gate_msa", "shift_mlp", "scale_mlp", "gate_mlp"])
    w.add_array(K + "flow.final_adaln.order", ["shift", "scale"])
    w.add_uint32(K + "cond.n_heads", CODE["cond_heads"])
    w.add_float32(K + "cond.ln_eps", CODE["torch_ln_eps"])

    # audio
    w.add_uint32(K + "audio.n_hidden_taps", int(n_taps))
    w.add_uint32(K + "audio.proj_in", int(proj_in))
    w.add_uint32(K + "audio.proj_out", int(proj_out))
    w.add_float32(K + "audio.proj_ln_eps", CODE["torch_ln_eps"])
    w.add_string(K + "audio.interp", "linear_align_corners")
    w.add_bool(K + "audio.normalize_input", False)    # live app feeds raw PCM
    w.add_string(K + "audio.pad", "zero_to_whole_frames")
    w.add_uint32(K + "audio.idle.seconds", APP["idle_seconds"])
    w.add_float32(K + "audio.idle.level_dbfs", APP["idle_level_dbfs"])
    w.add_uint32(K + "audio.idle.seed", APP["idle_seed"])

    # wav2vec2 (voice-detect.cpp key names, under this arch's prefix)
    P = K + "w2v2."
    w.add_uint32(P + "hidden_size", w2v["hidden_size"])
    w.add_uint32(P + "n_layers", w2v["n_layers"])
    w.add_uint32(P + "n_heads", w2v["n_heads"])
    w.add_uint32(P + "ff_dim", w2v["ff_dim"])
    w.add_uint32(P + "num_conv_layers", w2v["num_conv_layers"])
    w.add_array(P + "conv_dims", w2v["conv_dims"])
    w.add_array(P + "conv_kernels", w2v["conv_kernels"])
    w.add_array(P + "conv_strides", w2v["conv_strides"])
    w.add_bool(P + "conv_bias", w2v["conv_bias"])
    w.add_string(P + "feat_extract_norm", w2v["feat_extract_norm"])
    w.add_string(P + "feat_extract_activation", w2v["feat_extract_activation"])
    w.add_string(P + "hidden_act", w2v["hidden_act"])
    w.add_bool(P + "do_stable_layer_norm", w2v["do_stable_layer_norm"])
    w.add_float32(P + "layer_norm_eps", w2v["layer_norm_eps"])
    w.add_float32(P + "group_norm_eps", CODE["group_norm_eps"])
    w.add_uint32(P + "num_conv_pos_embeddings", w2v["num_conv_pos_embeddings"])
    w.add_uint32(P + "num_conv_pos_embedding_groups", w2v["num_conv_pos_embedding_groups"])
    w.add_uint32(P + "pos_conv_weight_norm_dim", 2)
    w.add_bool(P + "do_normalize", w2v["do_normalize"])

    # motion autoencoder
    M = K + "mae."
    w.add_uint32(M + "size", size)
    w.add_uint32(M + "log_size", log_size)
    w.add_uint32(M + "style_dim", int(cfg["style_dim"]))
    w.add_uint32(M + "motion_dim", int(motion_dim))
    w.add_uint32(M + "n_latent", log_size * 2 - 2)
    w.add_uint32(M + "enc.n_resblocks", n_res)
    w.add_array(M + "enc.channels", [int(c) for c in mae_enc_ch])
    w.add_uint32(M + "dec.n_levels", n_levels)
    w.add_array(M + "dec.channels", [int(c) for c in mae_dec_ch])
    w.add_array(M + "blur_kernel", CODE["mae_blur_kernel"])
    w.add_float32(M + "leaky_slope", CODE["mae_leaky_slope"])
    w.add_float32(M + "act_gain", CODE["mae_act_gain"])
    w.add_float32(M + "demod_eps", CODE["mae_demod_eps"])
    w.add_float32(M + "resblock_scale", 1.0 / math.sqrt(2.0))
    w.add_array(M + "pad.enc_down_conv3", [2, 2])     # Blur before the 3x3 stride-2 conv
    w.add_array(M + "pad.enc_down_skip1", [1, 1])     # Blur before the 1x1 stride-2 skip conv
    w.add_array(M + "pad.dec_up_conv", [1, 1])        # Blur after the 3x3 stride-2 transposed conv (kernel * 4)
    w.add_array(M + "pad.dec_upsample", [2, 1])       # Upsample of skip / flow (up 2, kernel * 4)
    w.add_string(M + "grid", "linspace(-1,1,n) f64 -> f32, x along width")
    w.add_string(M + "grid_sample", "bilinear, zeros, align_corners=false")
    w.add_bool(M + "folded_scales", True)

    for n in sorted(tensors):
        w.add_tensor(n, np.ascontiguousarray(tensors[n], dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    nbytes = sum(a.size * 4 for a in tensors.values())
    print(f"wrote {args.output}: {len(tensors)} tensors, {nbytes / 1e6:.1f} MB of f32 data")
    print(f"dropped {len(dropped)} source tensors:")
    for d in dropped:
        print(f"  {d}")


if __name__ == "__main__":
    main()
