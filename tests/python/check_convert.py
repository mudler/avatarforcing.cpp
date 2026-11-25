#!/usr/bin/env python3
"""Round-trip check for scripts/convert_avatarforcing_to_gguf.py.

Reads the GGUF back with gguf-py and compares it with the PyTorch state as the
upstream modules hold it, independently of the converter code:

  1. Every required KV key is present, and the values that the upstream
     config and modules define match.
  2. The expected tensor set is rebuilt from the upstream modules
     (FlowTransformer, the motion autoencoder Generator, Wav2Vec2Model) with
     the documented folds, using each module's own ``scale`` / ``lr_mul``.
     The GGUF must hold exactly that set: no tensor missing, none extra.
  3. Every tensor is equal: bit-exact for copies and scale folds, and within
     1e-6 for the float64 QR basis (numpy LAPACK here, torch in the converter).
  4. Semantic checks of the folds: demodulation from ``.wsq`` equals the
     upstream demodulation; Direction from ``dec.direction.q`` equals the
     upstream float32 QR path.
  5. Every upstream parameter is either in the GGUF or on the documented drop
     list.

Exit codes (ctest convention): 0 pass, 1 fail, 77 skip (dependencies or
inputs missing).
"""
import argparse
import math
import os
import sys

try:
    import numpy as np
    import gguf
    import torch
except ImportError as e:
    print(f"check_convert: missing dependency ({e}); skipping", file=sys.stderr)
    sys.exit(77)

REQUIRED_KV = """
general.architecture general.license
fps sample_rate samples_per_frame image_size clip_frames block_frames context_frames
stream.self_cache_len stream.cross_cache_len stream.kv_shift stream.first_steady_start_pos
stream.reanchor_every stream.reanchor_max_defer stream.reanchor_fade_frames output.u8_mode
sampler.default_nfe sampler.config_nfe sampler.num_train_timesteps sampler.steps sampler.solver
sampler.sigma_table_len sampler.shift sampler.sigma_min sampler.sigma_max cfg.audio_scale cfg.branches
flow.dim_w flow.dim_h flow.dim_a flow.depth flow.n_heads flow.head_dim flow.mlp_hidden flow.mlp_ratio
flow.mlp_act flow.ln_eps flow.freq_embed_dim flow.t_max_period flow.rope.theta flow.rope.dim
flow.rope.layout flow.self_mask.block flow.self_mask.lookahead flow.cross_mask.window
flow.block_causal flow.adaln.order flow.final_adaln.order cond.n_heads cond.ln_eps
audio.n_hidden_taps audio.proj_in audio.proj_out audio.proj_ln_eps audio.interp
audio.normalize_input audio.pad audio.idle.seconds audio.idle.level_dbfs audio.idle.seed
w2v2.hidden_size w2v2.n_layers w2v2.n_heads w2v2.ff_dim w2v2.num_conv_layers w2v2.conv_dims
w2v2.conv_kernels w2v2.conv_strides w2v2.conv_bias w2v2.feat_extract_norm
w2v2.feat_extract_activation w2v2.hidden_act w2v2.do_stable_layer_norm w2v2.layer_norm_eps
w2v2.group_norm_eps w2v2.num_conv_pos_embeddings w2v2.num_conv_pos_embedding_groups
w2v2.pos_conv_weight_norm_dim w2v2.do_normalize
mae.size mae.log_size mae.style_dim mae.motion_dim mae.n_latent mae.enc.n_resblocks
mae.enc.channels mae.dec.n_levels mae.dec.channels mae.blur_kernel mae.leaky_slope mae.act_gain
mae.demod_eps mae.resblock_scale mae.pad.enc_down_conv3 mae.pad.enc_down_skip1 mae.pad.dec_up_conv
mae.pad.dec_upsample mae.grid mae.grid_sample mae.folded_scales
""".split()

FAILS = []


def fail(msg):
    FAILS.append(msg)
    print(f"FAIL {msg}")


def kv_value(field):
    """Decode a GGUFReader field into a Python value."""
    t = field.types
    if t[0] == gguf.GGUFValueType.STRING:
        return bytes(field.parts[field.data[0]]).decode()
    if t[0] == gguf.GGUFValueType.ARRAY:
        if t[1] == gguf.GGUFValueType.STRING:
            return [bytes(field.parts[i]).decode() for i in field.data]
        return [field.parts[i].tolist()[0] for i in field.data]
    return field.parts[field.data[0]].tolist()[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--avatarvoice", required=True)
    ap.add_argument("--flow-ckpt", required=True)
    ap.add_argument("--mae-ckpt", required=True)
    ap.add_argument("--wav2vec", required=True)
    args = ap.parse_args()
    for p in (args.gguf, args.avatarvoice, args.flow_ckpt, args.mae_ckpt, args.wav2vec):
        if not os.path.exists(p):
            print(f"check_convert: {p} not found; skipping", file=sys.stderr)
            sys.exit(77)

    af_dir = os.path.join(args.avatarvoice, "avatarforcing")
    sys.dont_write_bytecode = True  # leave the upstream checkout untouched
    sys.path.insert(0, af_dir)
    sys.path.insert(0, args.avatarvoice)
    from omegaconf import OmegaConf
    import models.avatarforcing.styledecoder as sd_mod
    import models.avatarforcing.encoder as enc_mod
    from models import FlowMatchScheduler
    from models.avatarforcing.flow_transformer import FlowTransformer
    from models.avatarforcing.generator import Generator
    from models.wav2vec2 import Wav2VecModel
    from voicechat.avatar_worker import _shaped_pink_noise

    reader = gguf.GGUFReader(args.gguf)
    kv = {name: kv_value(f) for name, f in reader.fields.items() if not name.startswith("GGUF.")}
    gt = {}
    for t in reader.tensors:
        if t.tensor_type != gguf.GGMLQuantizationType.F32:
            fail(f"{t.name}: type {t.tensor_type.name}, expected F32")
        if len(t.name) >= 64:
            fail(f"{t.name}: name too long for ggml")
        gt[t.name] = np.asarray(t.data)
    print(f"gguf: {len(gt)} tensors, {len(kv)} KV")

    # ---------------- 1. KV ----------------
    missing = [k for k in REQUIRED_KV if (k if k.startswith("general.") else "avatarforcing." + k) not in kv]
    for k in missing:
        fail(f"KV missing: {k}")
    print(f"KV: {len(REQUIRED_KV) - len(missing)}/{len(REQUIRED_KV)} required keys present")
    A = lambda k: kv.get("avatarforcing." + k)
    opt = OmegaConf.load(os.path.join(af_dir, "configs", "inference.yaml"))
    opt.rank = "cpu"
    expect_kv = {
        "fps": opt.fps, "sample_rate": opt.sampling_rate, "samples_per_frame": opt.sampling_rate // opt.fps,
        "image_size": opt.input_size, "clip_frames": int(opt.sec * opt.fps),
        "block_frames": opt.num_frames_per_block, "flow.dim_w": opt.dim_w, "flow.dim_h": opt.dim_h,
        "flow.dim_a": opt.dim_a, "flow.depth": opt.transformer_depth, "flow.n_heads": opt.num_heads,
        "flow.head_dim": opt.dim_h // opt.num_heads, "flow.mlp_hidden": int(opt.dim_h * opt.mlp_ratio),
        "flow.cross_mask.window": opt.attention_window, "sampler.config_nfe": opt.nfe,
        "sampler.num_train_timesteps": opt.num_train_timestep, "mae.size": opt.input_size,
        "mae.style_dim": opt.style_dim, "mae.motion_dim": opt.dim_m,
        "stream.self_cache_len": int(opt.sec * opt.fps) - opt.num_frames_per_block - 2,
        "stream.cross_cache_len": opt.attention_window,
    }
    for k, v in expect_kv.items():
        if A(k) != v:
            fail(f"KV {k} = {A(k)}, expected {v}")
    if not math.isclose(A("cfg.audio_scale"), opt.a_cfg_scale):
        fail("KV cfg.audio_scale")

    # ---------------- 2. upstream modules ----------------
    ft = FlowTransformer(opt)
    gen = Generator(size=opt.input_size, style_dim=opt.dim_w, motion_dim=opt.dim_m)
    fsd = torch.load(args.flow_ckpt, map_location="cpu", weights_only=True, mmap=True)
    msd = torch.load(args.mae_ckpt, map_location="cpu", weights_only=True, mmap=True)
    dropped = set()
    with torch.no_grad():
        for n, p in ft.named_parameters():
            p.copy_(fsd["flow_transformer." + n])
        for n, p in gen.named_parameters():
            p.copy_(msd["motion_autoencoder." + n])
    ft_ckpt = {k for k in fsd if k.startswith("flow_transformer.")}
    ft_params = {"flow_transformer." + n for n, _ in ft.named_parameters()}
    for k in sorted(ft_ckpt - ft_params):
        dropped.add(k)
    if dropped != {"flow_transformer.alignment_mask", "flow_transformer.block_bidir_causal_lookahead_mask"}:
        fail(f"unexpected flow checkpoint keys outside the model parameters: {sorted(dropped)}")

    w2v = Wav2VecModel.from_pretrained(args.wav2vec, local_files_only=True)
    w2v.eval()
    # On torch >= 2.1 the positional conv is a weight_norm parametrization
    # (original0 = weight_g, original1 = weight_v). torch maps the checkpoint's
    # legacy names on load, although transformers 4.30.2 warns that they are
    # "newly initialized". Compare under the GGUF (legacy) names.
    pos_conv_rename = {
        "encoder.pos_conv_embed.conv.parametrizations.weight.original0": "encoder.pos_conv_embed.conv.weight_g",
        "encoder.pos_conv_embed.conv.parametrizations.weight.original1": "encoder.pos_conv_embed.conv.weight_v",
    }

    expected = {}
    # flow transformer
    for n, p in ft.named_parameters():
        a = p.detach().numpy()
        if n.endswith("in_proj_weight") or n.endswith("in_proj_bias"):
            base, kind = n.rsplit(".", 1)
            suf = "weight" if kind == "in_proj_weight" else "bias"
            d = a.shape[0] // 3
            for i, q in enumerate(("q_proj", "k_proj", "v_proj")):
                expected[f"ft.{base}.{q}.{suf}"] = a[i * d:(i + 1) * d]
        else:
            expected["ft." + n] = a
    for k, v in fsd.items():
        if k.startswith("audio_encoder.audio_projection."):
            expected["audio_projection." + k[len("audio_encoder.audio_projection."):]] = v.numpy()
    # motion autoencoder, from module attributes
    mae_params_covered = set()
    mae_dropped = set()
    for mname, m in gen.named_modules():
        P = "mae." + mname
        if isinstance(m, (sd_mod.EqualConv2d, enc_mod.EqualConv2d)):
            expected[P + ".weight"] = (m.weight * m.scale).detach().numpy()
            mae_params_covered.add(mname + ".weight")
            if m.bias is not None:
                expected[P + ".bias"] = m.bias.detach().numpy()
                mae_params_covered.add(mname + ".bias")
        elif isinstance(m, (sd_mod.EqualLinear, enc_mod.EqualLinear)):
            expected[P + ".weight"] = (m.weight * m.scale).detach().numpy()
            expected[P + ".bias"] = (m.bias * m.lr_mul).detach().numpy()
            mae_params_covered |= {mname + ".weight", mname + ".bias"}
        elif isinstance(m, sd_mod.ModulatedConv2d):
            w = (m.scale * m.weight)[0]
            expected[P + ".weight"] = w.detach().numpy()
            if m.demodulate:
                expected[P + ".wsq"] = w.pow(2).sum([2, 3]).detach().numpy()
            mae_params_covered.add(mname + ".weight")
        elif isinstance(m, (sd_mod.FusedLeakyReLU, enc_mod.FusedLeakyReLU)):
            expected[P + ".bias"] = m.bias.detach().numpy().reshape(-1)
            mae_params_covered.add(mname + ".bias")
        elif isinstance(m, (sd_mod.ToRGB, sd_mod.ToFlow)):
            expected[P + ".bias"] = m.bias.detach().numpy().reshape(-1)
            mae_params_covered.add(mname + ".bias")
        elif isinstance(m, sd_mod.ConstantInput):
            expected[P + ".input"] = m.input.detach().numpy()
            mae_params_covered.add(mname + ".input")
        elif isinstance(m, sd_mod.Direction):
            w64 = (m.weight + 1e-8).detach().numpy().astype(np.float64)
            q, _ = np.linalg.qr(w64)
            expected[P + ".q"] = q.astype(np.float32)
            mae_params_covered.add(mname + ".weight")
        elif isinstance(m, sd_mod.NoiseInjection):
            mae_dropped.add(mname + ".weight")
    # dec.to_rgb1 is built but unused in Synthesis.forward: the converter drops it.
    for k in [k for k in expected if k.startswith("mae.dec.to_rgb1.")]:
        del expected[k]
    mae_dropped |= {n for n, _ in gen.named_parameters() if n.startswith("dec.to_rgb1.")}
    mae_all = {n for n, _ in gen.named_parameters()}
    uncovered = mae_all - mae_params_covered - mae_dropped
    if uncovered:
        fail(f"motion autoencoder parameters not covered: {sorted(uncovered)}")
    # wav2vec2
    for n, p in w2v.state_dict().items():
        if n == "masked_spec_embed":
            continue
        expected["w2v2." + pos_conv_rename.get(n, n)] = p.detach().numpy()
    # extras
    sched = FlowMatchScheduler(num_steps=1000, shift=1, sigma_min=0.0, extra_one_step=False)
    sched.set_timesteps(opt.num_train_timestep, training=True)
    expected["sched.sigmas"] = sched.sigmas.numpy()
    expected["sched.timesteps"] = sched.timesteps.numpy()
    hd = opt.dim_h // opt.num_heads
    expected["ft.rope.inv_freq"] = (1.0 / (10000.0 ** (torch.arange(0, hd, 2)[: hd // 2].float() / hd))).numpy()
    half = ft.t_embedder.frequency_embedding_size // 2
    expected["ft.t_embedder.freqs"] = torch.exp(
        -math.log(10000) * torch.arange(start=0, end=half, dtype=torch.float32) / half).numpy()
    expected["audio.idle_pcm"] = _shaped_pink_noise(5 * 16000, sample_rate=16000, level_dbfs=-50.0)

    # sanity: the RoPE table the model builds equals cos/sin of positions * inv_freq
    fc = ft.blocks[0].attn.freqs_cis[:64].numpy()
    ang = np.outer(np.arange(64, dtype=np.float32), expected["ft.rope.inv_freq"]).astype(np.float32)
    if not np.array_equal(fc, np.concatenate([np.cos(ang), np.sin(ang)], 1).astype(np.float32)):
        d = float(np.abs(fc - np.concatenate([np.cos(ang), np.sin(ang)], 1)).max())
        print(f"note: RoPE table vs float32 cos/sin(pos*inv_freq): max |d| = {d:.3g}")

    # ---------------- 3. tensor set and equality ----------------
    miss = sorted(set(expected) - set(gt))
    extra = sorted(set(gt) - set(expected))
    for n in miss:
        fail(f"tensor missing from GGUF: {n}")
    for n in extra:
        fail(f"unexpected tensor in GGUF: {n}")
    n_exact, worst = 0, ("", 0.0)
    for n in sorted(set(expected) & set(gt)):
        e, g = np.asarray(expected[n], np.float32), gt[n]
        if e.shape != g.shape:
            fail(f"{n}: shape {g.shape}, expected {e.shape}")
            continue
        if n == "mae.dec.direction.q":
            d = float(np.abs(e - g).max())
            print(f"direction.q: GGUF (torch f64 QR) vs numpy f64 QR: max |d| = {d:.3g}")
            if d > 1e-6:
                fail(f"{n}: max |d| {d}")
            continue
        if np.array_equal(e, g):
            n_exact += 1
        else:
            d = float(np.abs(e - g).max())
            fail(f"{n}: not equal, max |d| = {d:.3g}")
            if d > worst[1]:
                worst = (n, d)
    print(f"tensors: {n_exact}/{len(expected) - 1} bit-exact (+ direction.q within tolerance)")

    # ---------------- 4. semantic checks ----------------
    torch.set_grad_enabled(False)
    g = torch.Generator().manual_seed(0)
    worst_demod = 0.0
    n_demod = 0
    for mname, m in gen.named_modules():
        if isinstance(m, sd_mod.ModulatedConv2d) and m.demodulate and not mname.startswith("dec.to_rgb1"):
            style = torch.randn(3, m.in_channel, generator=g) + 1.0
            w = m.scale * m.weight * style.view(3, 1, m.in_channel, 1, 1)
            ref = torch.rsqrt(w.pow(2).sum([2, 3, 4]) + 1e-8).double()
            wsq = torch.tensor(np.array(gt["mae." + mname + ".wsq"]), dtype=torch.float64)
            got = torch.rsqrt(style.double().pow(2) @ wsq.t() + 1e-8)
            worst_demod = max(worst_demod, float(((got - ref).abs() / ref.abs()).max()))
            n_demod += 1
    print(f"demod from .wsq vs upstream ({n_demod} convs): max rel |d| = {worst_demod:.3g}")
    if worst_demod > 1e-5:
        fail("demodulation from wsq differs from upstream")
    alpha = torch.randn(4, opt.dim_m, generator=g)
    with torch.no_grad():
        ref = gen.dec.direction(alpha).double()
    got = alpha.double() @ torch.tensor(np.array(gt["mae.dec.direction.q"]), dtype=torch.float64).t()
    d = float((got - ref).abs().max())
    print(f"Direction(alpha) via GGUF q vs upstream float32 QR: max |d| = {d:.3g} (|ref| max {float(ref.abs().max()):.3g})")
    if d > 1e-4:
        fail("Direction from GGUF q differs from upstream (sign or basis mismatch)")

    # ---------------- 5. coverage summary ----------------
    print(f"dropped from flow checkpoint: {sorted(dropped)}")
    print(f"dropped from motion checkpoint: {len(mae_dropped)} tensors (NoiseInjection weights, dec.to_rgb1.*)")

    if FAILS:
        print(f"check_convert: {len(FAILS)} failure(s)")
        sys.exit(1)
    print("check_convert: PASS")


if __name__ == "__main__":
    main()
