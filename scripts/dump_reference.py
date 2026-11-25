#!/usr/bin/env python3
"""Run the upstream AvatarForcing PyTorch code on CPU in float32 and write the
gold reference dumps (.npz) that the C++ port is gated against.

The script imports the model classes from an upstream AvatarVoice checkout
(``--avatarvoice``, with its ``avatarforcing`` submodule) and drives them the
same way the live streaming worker does (``voicechat/avatar_worker.py``):
50-frame primer without KV, then steady 10-frame blocks with 2 context frames,
KV read, and the KV update folded into the final (t = 0) step. The only
differences from the live worker are deliberate and documented in
``docs/reference.md``: float32 on CPU (no bf16 autocast), injected noise
(seeded or from a file) instead of an unseeded ``torch.randn``, and audio
features read from a precomputed array instead of time-indexed rings.

Nothing in the upstream checkout is modified. The script only adds forward
hooks and wraps ``forward_with_precomputed`` on the instance to record
intermediate tensors.

Output files (all float32 unless noted; see docs/reference.md):

  inputs.npz              portrait tensor, raw PCM, noise
  motion_encoder.npz      EncoderApp per-layer outputs, s_r, 7 feature maps,
                          alpha, r_s, the fp32 QR basis
  wav2vec_raw.npz         wav2vec2 front end + encoder + projection, raw PCM
  wav2vec_normalized.npz  same, zero-mean/unit-variance input (offline path)
  idle_pink.npz           pink-noise PCM and its 512-d idle features
  adaln_nfe{N}.npz        timestep embeddings and adaLN tables per step
  flow_forward_nfe{N}.npz per-module detail of 4 forwards (primer step 0,
                          primer KV update, steady block 0 step 0, steady
                          block 0 final step with the folded KV update)
  sampler_nfe{N}.npz      primer + steady blocks: x after every step, raw
                          velocities, KV cache after the primer and after
                          every steady block
  decoder_nfe{N}.npz      motion decoder per level (frame 0), final pixels
                          and uint8 frames for steady block 0
  manifest.json           arguments, versions, shapes and SHA-256 of inputs
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import wave

import numpy as np


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_wav_16k_mono(path):
    """16-bit PCM WAV, 16 kHz, mono -> float32 in [-1, 1) (int16 / 32768).

    This is what librosa.load(sr=16000) returns for such a file."""
    with wave.open(path, "rb") as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise SystemExit(f"{path}: need 16 kHz mono 16-bit PCM, got "
                             f"{w.getframerate()} Hz, {w.getnchannels()} ch, "
                             f"{8 * w.getsampwidth()} bit")
        raw = w.readframes(w.getnframes())
    return (np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0)


def read_portrait(path):
    """512x512 RGB image -> [1, 3, 512, 512] float32 in [-1, 1].

    Same as the upstream DataProcessor.transform after its crop/resize:
    A.Normalize(mean=0.5, std=0.5) on uint8, then CHW."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    if img.size != (512, 512):
        raise SystemExit(f"{path}: need a pre-cropped 512x512 portrait "
                         f"(see scripts/prepare_portrait.py), got {img.size}")
    u8 = np.asarray(img, dtype=np.uint8)
    x = (u8.astype(np.float32) / 255.0 - 0.5) / 0.5
    return u8, x.transpose(2, 0, 1)[None].copy()


class Saver:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.manifest = {"files": {}}
        os.makedirs(out_dir, exist_ok=True)

    def save(self, name, arrays):
        clean = {}
        for k, v in arrays.items():
            a = to_np(v)
            clean[k] = a
        path = os.path.join(self.out_dir, name)
        np.savez(path, **clean)
        self.manifest["files"][name] = {k: [list(a.shape), str(a.dtype)] for k, a in clean.items()}
        size = os.path.getsize(path)
        print(f"  wrote {name}: {len(clean)} arrays, {size / 1e6:.1f} MB", flush=True)


def _none(*_):
    """Forward hooks must return None: a returned value replaces the output."""
    return None


def to_np(v):
    import torch
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu()
        if v.dtype == torch.bfloat16 or v.dtype == torch.float16:
            v = v.float()
        return v.numpy().copy()
    return np.asarray(v)


# --------------------------------------------------------------------------
# model construction (mirrors avatarforcing/inference.py InferenceAgent)
# --------------------------------------------------------------------------

def build_model(args):
    import torch
    from omegaconf import OmegaConf
    af_dir = os.path.join(args.avatarvoice, "avatarforcing")
    sys.dont_write_bytecode = True  # leave the upstream checkout untouched
    sys.path.insert(0, af_dir)
    sys.path.insert(0, args.avatarvoice)
    from models.avatarforcing.AvatarForcing import AvatarForcing

    opt = OmegaConf.load(os.path.join(af_dir, "configs", "inference.yaml"))
    opt.rank = "cpu"
    opt.wav2vec_model_path = args.wav2vec
    torch.manual_seed(0)  # only affects the random init that the checkpoints overwrite
    G = AvatarForcing(opt)

    # load_mae_ckpt then load_ckpt: copy matching *parameters* only. The mask
    # buffers in the flow checkpoint are never loaded (inverted, unused).
    # Parameters are pointed at the mmap'd checkpoint tensors (same values as
    # the upstream copy_, without a second resident copy of the weights).
    report = {}
    for tag, path in (("mae", args.mae_ckpt), ("flow", args.flow_ckpt)):
        sd = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        params = dict(G.named_parameters())
        used = 0
        with torch.no_grad():
            for name, p in params.items():
                if name in sd:
                    assert p.shape == sd[name].shape and sd[name].dtype == torch.float32, name
                    p.data = sd[name]
                    used += 1
        report[tag] = {"ckpt_keys": len(sd), "params_loaded": used,
                       "unused_ckpt_keys": sorted(k for k in sd if k not in params)}
        del sd
    report["w2v2_pos_conv"] = check_w2v2_pos_conv(G.audio_encoder.wav2vec2, args.wav2vec)
    import gc
    gc.collect()
    G.eval()
    for p in G.parameters():
        p.requires_grad_(False)
    return G, opt, report


def check_w2v2_pos_conv(w2v, path):
    """Check that the wav2vec2 positional conv holds the pretrained weights.

    On torch >= 2.1, transformers 4.30.2 builds the positional conv with
    nn.utils.parametrizations.weight_norm (``...original0`` / ``...original1``)
    and prints a "newly initialized" warning for them, because the checkpoint
    stores ``weight_g`` / ``weight_v``. The warning is spurious: torch's
    weight_norm parametrization registers a load_state_dict pre-hook that maps
    the legacy names, so the weights do load. This check proves it on every
    run and stops the dump if a future version breaks it.
    """
    import torch
    from safetensors import safe_open
    conv = w2v.encoder.pos_conv_embed.conv
    with safe_open(os.path.join(path, "model.safetensors"), "pt") as f:
        g = f.get_tensor("wav2vec2.encoder.pos_conv_embed.conv.weight_g")
        v = f.get_tensor("wav2vec2.encoder.pos_conv_embed.conv.weight_v")
    if hasattr(conv, "parametrizations"):
        have_g, have_v = conv.parametrizations.weight.original0, conv.parametrizations.weight.original1
        kind = "parametrized weight_norm"
    else:
        have_g, have_v = conv.weight_g, conv.weight_v
        kind = "legacy weight_norm"
    if not (torch.equal(have_g, g) and torch.equal(have_v, v)):
        raise SystemExit("wav2vec2 positional conv does not hold the checkpoint weights")
    return f"{kind}: weight_g/weight_v loaded (checked)"


# --------------------------------------------------------------------------
# dumps
# --------------------------------------------------------------------------

def dump_motion_encoder(G, image, saver):
    import torch
    rec = {}
    hooks = []
    net = G.motion_autoencoder.enc.net_app
    for i, m in enumerate(net.convs):
        hooks.append(m.register_forward_hook(
            lambda mod, inp, out, i=i: rec.__setitem__(f"enc.convs.{i}", out.clone())))
    with torch.inference_mode():
        s_r, alpha, feats = G.encode_image_into_latent(image)
        r_s = G.motion_autoencoder.dec.direction(alpha)
        Q, R = torch.linalg.qr(G.motion_autoencoder.dec.direction.weight + 1e-8)
    for h in hooks:
        h.remove()
    out = {"image": image, "s_r": s_r, "alpha": alpha, "r_s": r_s,
           "direction.q_fp32": Q, "direction.r_fp32": R}
    for k, f in enumerate(feats):
        out[f"feat.{k}"] = f
    out.update(rec)
    saver.save("motion_encoder.npz", out)
    return s_r, r_s, feats


def wav2vec_features(G, pcm, mode, saver, fe_path, name=None):
    """Run AudioEncoder.inference the way the live app (mode='raw') or the
    offline inference.py (mode='normalized') does and record every stage."""
    import torch
    ae = G.audio_encoder
    w2v = ae.wav2vec2
    if mode == "raw":
        # avatar_worker._wav2vec_loop: round up to whole video frames, zero-pad.
        n_frames = max(1, (pcm.size + 639) // 640)
        need = n_frames * 640
        x = np.concatenate([pcm, np.zeros(need - pcm.size, np.float32)]) if pcm.size < need else pcm[:need]
    elif mode == "normalized":
        # inference.py: Wav2Vec2FeatureExtractor (zero mean, unit variance),
        # T = ceil(len * fps / sr); AudioEncoder.inference replicate-pads.
        from transformers import Wav2Vec2FeatureExtractor
        fe = Wav2Vec2FeatureExtractor.from_pretrained(fe_path, local_files_only=True)
        x = fe(pcm, sampling_rate=16000, return_tensors="pt").input_values[0].numpy()
        n_frames = math.ceil(pcm.size * 25 / 16000)
    else:
        raise ValueError(mode)

    rec = {}
    hooks = []
    for i, layer in enumerate(w2v.feature_extractor.conv_layers):
        hooks.append(layer.register_forward_hook(
            lambda m, inp, out, i=i: rec.__setitem__(f"conv.{i}", out[0].clone())))
    hooks.append(w2v.feature_projection.register_forward_hook(
        lambda m, inp, out: _none(rec.__setitem__("interp", inp[0][0].clone()),
                             rec.__setitem__("feature_projection", out[0][0].clone()),
                             rec.__setitem__("feature_projection.norm", out[1][0].clone()))))
    hooks.append(w2v.encoder.pos_conv_embed.register_forward_hook(
        lambda m, inp, out: rec.__setitem__("pos_conv", out[0].clone())))
    hooks.append(w2v.register_forward_hook(
        lambda m, inp, out: rec.__setitem__("_hidden", [h[0].clone() for h in out.hidden_states])))
    proj = ae.audio_projection
    hooks.append(proj[0].register_forward_hook(
        lambda m, inp, out: _none(rec.__setitem__("stacked", inp[0][0].clone()),
                             rec.__setitem__("proj.linear", out[0].clone()))))
    hooks.append(proj[1].register_forward_hook(
        lambda m, inp, out: rec.__setitem__("proj.norm", out[0].clone())))
    with torch.inference_mode():
        a = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).unsqueeze(0)
        feats = ae.inference(a, seq_len=n_frames)
    for h in hooks:
        h.remove()
    hidden = rec.pop("_hidden")
    out = {"pcm_in": pcm, "model_input": x, "n_frames": np.int64(n_frames),
           "features": feats[0]}
    for i, h in enumerate(hidden):
        out[f"hidden.{i}"] = h
    out.update(rec)
    if saver is not None:
        saver.save(name or f"wav2vec_{mode}.npz", out)
    return feats[0]


def dump_idle(G, saver, fe_path):
    import torch
    from voicechat.avatar_worker import _shaped_pink_noise
    n_sec = 5
    pcm = _shaped_pink_noise(n_sec * 16000, sample_rate=16000, level_dbfs=-50.0)
    with torch.inference_mode():
        feats = G.audio_encoder.inference(torch.from_numpy(pcm).unsqueeze(0), seq_len=n_sec * 25)[0]
    saver.save("idle_pink.npz", {"pcm": pcm, "features": feats,
                                 "seed": np.int64(42), "level_dbfs": np.float32(-50.0)})
    return feats


def dump_adaln(G, steps, saver, nfe):
    import torch
    ft = G.flow_transformer
    ts = sorted(set([float(t) for t in steps] + [0.0]), reverse=True)
    out = {"timesteps": np.array(ts, np.float32), "step_list": np.array(steps, np.float32)}
    with torch.inference_mode():
        t = torch.tensor(ts, dtype=torch.float32)
        t_freq = ft.t_embedder.timestep_embedding(t, ft.t_embedder.frequency_embedding_size)
        t_emb = ft.t_embedder(t)
        out["t_freq"] = t_freq
        out["t_embed"] = t_emb
        out["adaln6"] = ft.adaLN_modulation(t_emb)
        out["final_adaln2"] = ft.final_layer.adaLN_modulation(t_emb)
    saver.save(f"adaln_nfe{nfe}.npz", out)


def kv_snapshot(af):
    import torch
    return {
        "self_k": torch.stack([sa["k"].clone() for sa, ca in af.kv_cache]),
        "self_v": torch.stack([sa["v"].clone() for sa, ca in af.kv_cache]),
        "cross_k": torch.stack([ca["k"].clone() for sa, ca in af.kv_cache]),
        "cross_v": torch.stack([ca["v"].clone() for sa, ca in af.kv_cache]),
    }


class FlowRecorder:
    """Wraps forward_with_precomputed on the instance and records (a) every
    call's input and output for the sampler dump and (b) per-module detail for
    the calls tagged in ``detail_tags``."""

    def __init__(self, af):
        self.af = af
        self.ft = af.flow_transformer
        self.orig = self.ft.forward_with_precomputed
        self.calls = []
        self.tag = None           # current tag for detail recording
        self.detail = {}          # tag -> dict
        self.ft.forward_with_precomputed = self._wrapped
        self._hooks = []
        self._install_hooks()

    def _rec(self, key, val):
        if self.tag is not None:
            self.detail[self.tag][key] = val.detach().clone()

    def _install_hooks(self):
        ft = self.ft
        H = self._hooks
        H.append(ft.x_embedder.register_forward_hook(lambda m, i, o: self._rec("x_embed", o)))
        for li, blk in enumerate(ft.blocks):
            H.append(blk.register_forward_hook(lambda m, i, o, li=li: self._rec(f"block.{li}", o)))
        b0 = ft.blocks[0]
        H.append(b0.attn.register_forward_hook(
            lambda m, i, o: _none(self._rec("block.0.attn_in", i[0]), self._rec("block.0.attn_out", o))))
        H.append(b0.cross_attn.register_forward_hook(
            lambda m, i, o: _none(self._rec("block.0.cross_in", i[0]), self._rec("block.0.cross_out", o))))
        H.append(b0.mlp.register_forward_hook(
            lambda m, i, o: _none(self._rec("block.0.mlp_in", i[0]), self._rec("block.0.mlp_out", o))))
        H.append(ft.final_layer.register_forward_hook(lambda m, i, o: self._rec("final", o)))
        ce = ft.c_embedder
        H.append(ce.motion_embedder.register_forward_hook(lambda m, i, o: self._rec("c_embedder.motion_embedder", o)))
        H.append(ce.dual_modal_attn.register_forward_hook(lambda m, i, o: self._rec("c_embedder.dual_modal_attn", o[0])))
        H.append(ce.silu1.register_forward_hook(lambda m, i, o: self._rec("c_embedder.cross_modal_feat", o)))
        H.append(ce.user_agent_attn.register_forward_hook(lambda m, i, o: self._rec("c_embedder.user_agent_attn", o[0])))
        H.append(ce.silu2.register_forward_hook(lambda m, i, o: self._rec("c_embedder.unified", o)))
        H.append(ce.register_forward_hook(
            lambda m, i, o: _none(self._rec("c_embedder.in.wa_user", i[0]), self._rec("c_embedder.in.motion_user", i[1]),
                             self._rec("c_embedder.in.wa", i[2]), self._rec("c_embedder.out", o))))

    def begin(self, tag):
        self.tag = tag
        self.detail.setdefault(tag, {})

    def end(self):
        self.tag = None

    def _wrapped(self, x, precomputed_c, precomputed_adaLN, precomputed_wr,
                 freqs_cis_slice, kv_cache_list, update_kv_cache):
        d = self.detail[self.tag] if self.tag is not None else None
        if d is not None:
            ssa, t_embed, final = precomputed_adaLN
            d["in.x"] = x.clone()
            d["in.c"] = precomputed_c.clone()
            d["in.wr"] = precomputed_wr.clone()
            d["in.adaln6"] = __import__("torch").stack([s[0] for s in ssa])      # batch 0
            d["in.final_adaln2"] = __import__("torch").stack([s[0] for s in final])
            d["in.freqs_cis"] = freqs_cis_slice.clone()
            d["in.update_kv_cache"] = np.bool_(update_kv_cache)
            d["in.kv_read"] = np.bool_(kv_cache_list is not None)
            if kv_cache_list is not None:
                for k, v in kv_snapshot(self.af).items():
                    d[f"kv_before.{k}"] = v
        out = self.orig(x, precomputed_c, precomputed_adaLN, precomputed_wr,
                        freqs_cis_slice, kv_cache_list, update_kv_cache)
        if d is not None:
            d["out.v"] = out.clone()
            if kv_cache_list is not None:
                for k, v in kv_snapshot(self.af).items():
                    d[f"kv_after.{k}"] = v
        self.calls.append(out.detach().clone())
        return out

    def close(self):
        for h in self._hooks:
            h.remove()
        self.ft.forward_with_precomputed = self.orig


def sigma_index(af, t):
    """Reproduce solve_closed's argmin lookup for one timestep value."""
    import torch
    tbl = af.scheduler.timesteps.to(torch.double)
    return int((tbl - float(t)).abs().argmin())


def run_sampler(G, r_s, wa_all, noise_primer, noise_steady, nfe, saver,
                detail, a_cfg=2.0):
    """The live worker's loop (avatar_worker._run_loop), minus rings and
    pacing: primer then len(noise_steady) steady blocks."""
    import torch
    af = G
    clip = af.num_frames_for_clip          # 50
    blk = af.num_frames_per_block          # 10
    dim_w = af.opt.dim_w
    B = 1
    af.denoising_step_list = torch.tensor(np.linspace(af.opt.num_train_timestep, 0, nfe - 1).tolist())
    af.initialize_kv_cache(batch_size=B, dtype=torch.float32, device="cpu")
    steps = [float(t) for t in af.denoising_step_list]

    rec = FlowRecorder(af)
    out = {"step_list": np.array(steps, np.float32),
           "sigma_index": np.array([sigma_index(af, t) for t in steps], np.int64),
           "noise.primer": noise_primer, "noise.steady": noise_steady,
           "r_s": r_s, "a_cfg_scale": np.float32(a_cfg)}
    try:
        with torch.inference_mode():
            # ---------------- primer ----------------
            wa = torch.from_numpy(wa_all[None, 0:clip]).float()
            out["primer.wa"] = wa[0]
            x_t = torch.from_numpy(noise_primer[None]).float()
            pc_c, pc_wr, pc_adaLN = None, None, None
            if detail:
                rec.begin("primer_step0")
            pc_c, pc_wr, pc_adaLN = af.prepare_cfg_condition(
                wa, torch.zeros_like(wa), torch.zeros(1, clip, dim_w), r_s, seq_len=clip, context_len=0)
            out["primer.c"] = pc_c
            for idx, ts in enumerate(af.denoising_step_list):
                is_final = idx == len(af.denoising_step_list) - 1
                out[f"primer.x_in.{idx}"] = x_t
                x_t = af.solve_cfg(B=B, index=idx, current_timestep=ts, x_t=x_t,
                                   precomputed_c=pc_c, precomputed_wr=pc_wr, precomputed_adaLN=pc_adaLN,
                                   start_pos=0, context_len=0, use_kv_cache=False,
                                   a_cfg_scale=a_cfg, u_cfg_scale=0.0)
                rec.end()
                out[f"primer.v.{idx}"] = rec.calls[-1]
                out[f"primer.x.{idx}"] = x_t
                if is_final:
                    if detail:
                        rec.begin("primer_kvupdate")
                    af.update_kv_cache(final_latents=x_t, precomputed_c=pc_c, precomputed_wr=pc_wr,
                                       precomputed_adaLN=pc_adaLN, start_pos=0)
                    rec.end()
                    out["primer.kvupdate.v"] = rec.calls[-1]
            for k, v in kv_snapshot(af).items():
                out[f"primer.kv.{k}"] = v
            out["primer.x_final"] = x_t
            prev_tail = x_t

            # ---------------- steady blocks ----------------
            gen_t = clip
            start_pos = gen_t - 2
            for b in range(noise_steady.shape[0]):
                feats = torch.from_numpy(wa_all[None, gen_t - 2: gen_t + blk]).float()
                assert feats.shape[1] == blk + 2
                out[f"steady.{b}.wa"] = feats[0]
                out[f"steady.{b}.start_pos"] = np.int64(start_pos)
                x_t = torch.cat([prev_tail[:, -2:], torch.from_numpy(noise_steady[b][None]).float()], dim=1)
                if detail and b == 0:
                    rec.begin("steady0_step0")
                pc_c, pc_wr, pc_adaLN = af.prepare_cfg_condition(
                    feats, torch.zeros_like(feats), torch.zeros(1, blk + 2, dim_w), r_s,
                    seq_len=blk, context_len=2)
                out[f"steady.{b}.c"] = pc_c
                for idx, ts in enumerate(af.denoising_step_list):
                    is_final = idx == len(af.denoising_step_list) - 1
                    if detail and b == 0 and is_final:
                        rec.begin("steady0_final")
                    out[f"steady.{b}.x_in.{idx}"] = x_t
                    x_t = af.solve_cfg(B=B, index=idx, current_timestep=ts, x_t=x_t,
                                       precomputed_c=pc_c, precomputed_wr=pc_wr, precomputed_adaLN=pc_adaLN,
                                       start_pos=start_pos, context_len=2, use_kv_cache=True,
                                       a_cfg_scale=a_cfg, u_cfg_scale=0.0,
                                       update_kv_cache=is_final)
                    rec.end()
                    out[f"steady.{b}.v.{idx}"] = rec.calls[-1]
                    out[f"steady.{b}.x.{idx}"] = x_t
                for k, v in kv_snapshot(af).items():
                    out[f"steady.{b}.kv.{k}"] = v
                out[f"steady.{b}.latent"] = x_t[0, -blk:]
                prev_tail = x_t
                start_pos = gen_t + blk - 2
                gen_t += blk
    finally:
        rec.close()

    saver.save(f"sampler_nfe{nfe}.npz", out)
    if detail:
        flat = {}
        for tag, d in rec.detail.items():
            for k, v in d.items():
                flat[f"{tag}.{k}"] = v
        saver.save(f"flow_forward_nfe{nfe}.npz", flat)
    return {b: out[f"steady.{b}.latent"] for b in range(noise_steady.shape[0])}


def dump_decoder(G, s_r, feats, latent, saver, nfe):
    """Decode one steady block (10 frames). Frames run one at a time (batch 1)
    to bound memory; the decoder has no cross-frame interaction, so this is
    the same computation as the worker's batch of 10. Per-level detail is
    recorded for frame 0."""
    import torch
    dec = G.motion_autoencoder.dec
    rec = {}
    on = {"v": False}

    def put(k, v):
        if on["v"]:
            rec[k] = v.detach().clone()

    hooks = [dec.conv1.register_forward_hook(lambda m, i, o: _none(put("const", i[0]), put("conv1", o)))]
    hooks.append(dec.conv1.conv.modulation.register_forward_hook(lambda m, i, o: put("conv1.style", o)))
    for ci, conv in enumerate(dec.convs):
        lvl, which = ci // 2, ("up" if ci % 2 == 0 else "conv")
        hooks.append(conv.register_forward_hook(lambda m, i, o, l=lvl, w=which: put(f"level.{l}.{w}", o)))
        hooks.append(conv.conv.modulation.register_forward_hook(
            lambda m, i, o, l=lvl, w=which: put(f"level.{l}.{w}.style", o)))
    for l, tf in enumerate(dec.to_flows):
        hooks.append(tf.register_forward_hook(
            lambda m, i, o, l=l: _none(put(f"level.{l}.warp", o[0]), put(f"level.{l}.out", o[1]),
                                  put(f"level.{l}.flow_raw", o[2]), put(f"level.{l}.flow", o[3]))))
        hooks.append(tf.conv.modulation.register_forward_hook(
            lambda m, i, o, l=l: put(f"level.{l}.toflow.style", o)))
    for l, tr in enumerate(dec.to_rgbs):
        hooks.append(tr.register_forward_hook(lambda m, i, o, l=l: put(f"level.{l}.skip", o)))

    styles = (s_r + latent[None])[0]           # [10, 512] = s_r + r_d per frame
    frames = []
    with torch.inference_mode():
        for f in range(styles.shape[0]):
            on["v"] = f == 0
            img, _ = dec(styles[f:f + 1], alpha=None, feats=feats)
            frames.append(img)
    for h in hooks:
        h.remove()
    pixels = torch.cat(frames, 0)              # [10, 3, 512, 512]
    # avatar_worker._emit_pixel_block: clamp, block-wide min/max, *255, truncate.
    vid = pixels.clamp(-1, 1)
    vid = (vid - vid.min()) / (vid.max() - vid.min() + 1e-8) * 255.0
    u8 = vid.permute(0, 2, 3, 1).contiguous().to(torch.uint8)
    out = {"latent": latent, "s_r": s_r, "style": styles, "pixels": pixels, "frames_u8": u8}
    out.update(rec)
    saver.save(f"decoder_nfe{nfe}.npz", out)


def build_feature_track(feats_main, feats_idle, n_needed):
    """Audio features for the sampler: the speech features, then cycled idle
    (pink-noise) features if the clip is shorter than needed, as the worker
    does for frames past the ring's write position."""
    if feats_main.shape[0] >= n_needed:
        return feats_main[:n_needed].copy()
    n = n_needed - feats_main.shape[0]
    idx = np.arange(n) % feats_idle.shape[0]
    return np.concatenate([feats_main, feats_idle[idx]], 0)


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--avatarvoice", required=True, help="AvatarVoice checkout (with the avatarforcing submodule)")
    ap.add_argument("--flow-ckpt", required=True)
    ap.add_argument("--mae-ckpt", required=True)
    ap.add_argument("--wav2vec", required=True, help="local facebook/wav2vec2-base-960h directory")
    ap.add_argument("--portrait", required=True, help="pre-cropped 512x512 RGB portrait")
    ap.add_argument("--audio", required=True, help="16 kHz mono 16-bit WAV")
    ap.add_argument("--out", required=True)
    ap.add_argument("--nfe", type=int, nargs="+", default=[6, 10])
    ap.add_argument("--steady-blocks", type=int, default=6)
    ap.add_argument("--audio-modes", nargs="+", default=["raw", "normalized"],
                    choices=["raw", "normalized"], help="wav2vec2 input modes to dump")
    ap.add_argument("--sampler-audio-mode", default="raw", choices=["raw", "normalized"],
                    help="which wav2vec2 features drive the sampler (raw = live app)")
    ap.add_argument("--noise-seed", type=int, default=1234,
                    help="numpy default_rng seed for the injected noise")
    ap.add_argument("--noise-file", default=None,
                    help=".npz with 'primer' [50,512] and 'steady' [K,10,512]; overrides --noise-seed")
    ap.add_argument("--detail-nfe", type=int, default=6,
                    help="nfe whose run also writes flow_forward_*.npz and decoder_*.npz")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    import torch
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)

    saver = Saver(args.out)
    G, opt, load_report = build_model(args)
    for k, v in load_report.items():
        print(f"loaded {k}: " + (json.dumps({kk: vv for kk, vv in v.items() if kk != 'unused_ckpt_keys'})
                                 if isinstance(v, dict) else v), flush=True)

    u8, image = read_portrait(args.portrait)
    pcm = read_wav_16k_mono(args.audio)

    clip, blk = G.num_frames_for_clip, G.num_frames_per_block
    if args.noise_file:
        nz = np.load(args.noise_file)
        noise_primer = nz["primer"].astype(np.float32)
        noise_steady = nz["steady"].astype(np.float32)[: args.steady_blocks]
    else:
        rng = np.random.default_rng(args.noise_seed)
        noise_primer = rng.standard_normal((clip, opt.dim_w), dtype=np.float32)
        noise_steady = rng.standard_normal((args.steady_blocks, blk, opt.dim_w), dtype=np.float32)
    assert noise_steady.shape[0] == args.steady_blocks

    saver.save("inputs.npz", {"portrait_u8": u8, "portrait": image, "pcm": pcm,
                              "noise.primer": noise_primer, "noise.steady": noise_steady})

    print("motion encoder ...", flush=True)
    s_r, r_s, feats = dump_motion_encoder(G, torch.from_numpy(image), saver)

    print("wav2vec2 ...", flush=True)
    feats_by_mode = {}
    for mode in sorted(set(args.audio_modes) | {args.sampler_audio_mode}):
        feats_by_mode[mode] = wav2vec_features(
            G, pcm, mode, saver if mode in args.audio_modes else None, args.wav2vec).numpy()
    idle = dump_idle(G, saver, args.wav2vec).numpy()

    n_needed = clip + args.steady_blocks * blk
    wa_all = build_feature_track(feats_by_mode[args.sampler_audio_mode], idle, n_needed)

    for nfe in args.nfe:
        print(f"sampler nfe={nfe} ...", flush=True)
        steps = np.linspace(opt.num_train_timestep, 0, nfe - 1).tolist()
        dump_adaln(G, steps, saver, nfe)
        detail = nfe == args.detail_nfe
        latents = run_sampler(G, r_s, wa_all, noise_primer, noise_steady, nfe, saver, detail)
        if detail:
            print(f"decoder (steady block 0, nfe={nfe}) ...", flush=True)
            dump_decoder(G, s_r, feats, latents[0], saver, nfe)

    import transformers
    saver.manifest.update({
        "args": vars(args),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "numpy": np.__version__, "python": sys.version.split()[0]},
        "inputs_sha256": {"portrait": sha256(args.portrait), "audio": sha256(args.audio),
                          "flow_ckpt": sha256(args.flow_ckpt), "mae_ckpt": sha256(args.mae_ckpt)},
        "checkpoint_load": load_report,
        "sampler_feature_frames": {"speech": int(feats_by_mode[args.sampler_audio_mode].shape[0]),
                                   "needed": n_needed},
    })
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(saver.manifest, f, indent=1, sort_keys=True, default=str)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
