# GGUF conversion schema

`scripts/convert_avatarforcing_to_gguf.py` writes one self-contained GGUF file
with the three model components: the flow transformer (with the audio
projection), the motion autoencoder, and wav2vec2-base-960h. The C++ engine
reads every dimension and hyperparameter from KV. No config, vocab or side
file ships with it.

```
python scripts/convert_avatarforcing_to_gguf.py \
    --flow-ckpt flow_transformer.pth --mae-ckpt motion_autoencoder.pth \
    --wav2vec wav2vec2-base-960h --config AvatarVoice/avatarforcing/configs/inference.yaml \
    --output avatarforcing-f32.gguf
```

`--config` is optional. Without it, the converter uses the values of the
upstream `configs/inference.yaml` at AvatarVoice commit `566f8e1`. All tensors
are F32 (quantization comes later, see the plan in the README).

The acceptance check is `tests/python/check_convert.py`. It rebuilds the
expected tensors from the upstream PyTorch modules, independently of the
converter, and requires the GGUF to hold exactly that set with equal values.

## Header

- `general.architecture` = `avatarforcing`
- `general.name` = `AvatarForcing`
- `general.license` = `cc-by-nc-4.0`
- `general.source.url` = upstream repository
- `avatarforcing.source.flow_sha256`, `avatarforcing.source.mae_sha256`:
  SHA-256 of the source checkpoints; `avatarforcing.source.wav2vec2`: HF id
- `avatarforcing.format_version` = 1

## KV metadata (`avatarforcing.*`)

### Timing and streaming

| Key | Type | Value | Meaning |
|---|---|---|---|
| `fps` | u32 | 25 | video frame rate |
| `sample_rate` | u32 | 16000 | audio sample rate |
| `samples_per_frame` | u32 | 640 | audio samples per video frame |
| `image_size` | u32 | 512 | portrait and output frame size |
| `clip_frames` | u32 | 50 | primer length (2 s) |
| `block_frames` | u32 | 10 | new frames per steady block |
| `context_frames` | u32 | 2 | frames carried from the previous block |
| `stream.self_cache_len` | u32 | 38 | self-attention KV cache length per layer |
| `stream.cross_cache_len` | u32 | 2 | cross-attention KV cache length per layer |
| `stream.kv_shift` | u32 | 10 | self cache left shift per steady block |
| `stream.first_steady_start_pos` | u32 | 48 | RoPE start position of the first steady block |
| `stream.reanchor_every` | u32 | 112 | live app: reanchor after this many blocks once idle |
| `stream.reanchor_max_defer` | u32 | 162 | live app: forced reanchor |
| `stream.reanchor_fade_frames` | u32 | 25 | live app: crossfade after a reanchor |
| `output.u8_mode` | str | `clamp_block_minmax_trunc` | clamp to [-1,1], block-wide min/max to [0,255], truncate |

### Sampler

| Key | Type | Value | Meaning |
|---|---|---|---|
| `sampler.default_nfe` | u32 | 6 | nfe of the live app (5 forwards per block) |
| `sampler.config_nfe` | u32 | 10 | nfe of the upstream config and offline script |
| `sampler.num_train_timesteps` | u32 | 1000 | |
| `sampler.steps` | str | `linspace(num_train_timesteps, 0, nfe - 1)` | timestep list |
| `sampler.solver` | str | `closed_form_argmin_f64` | `solve_closed`: sigma by argmin over `sched.timesteps`, update in float64 |
| `sampler.sigma_table_len` | u32 | 1000 | length of `sched.sigmas` / `sched.timesteps` |
| `sampler.shift`, `sigma_min`, `sigma_max` | f32 | 1, 0, 1 | FlowMatchScheduler parameters |
| `cfg.audio_scale` | f32 | 2.0 | `v = v_uncond + s * (v_audio - v_uncond)` |
| `cfg.branches` | [str] | `uncond, audio` | batch order of the two CFG branches |

### Flow transformer

| Key | Type | Value |
|---|---|---|
| `flow.dim_w` | u32 | 512 (motion latent) |
| `flow.dim_h` | u32 | 1024 (hidden) |
| `flow.dim_a` | u32 | 512 (condition embedder width) |
| `flow.depth` | u32 | 8 |
| `flow.n_heads`, `flow.head_dim` | u32 | 8, 128 |
| `flow.mlp_hidden`, `flow.mlp_ratio` | u32, f32 | 4096, 4.0 |
| `flow.mlp_act` | str | `gelu_tanh` |
| `flow.ln_eps` | f32 | 1e-6 (block and final LayerNorms, no affine) |
| `flow.freq_embed_dim`, `flow.t_max_period` | u32, f32 | 256, 10000 |
| `flow.rope.theta`, `flow.rope.dim` | f32, u32 | 10000, 128 |
| `flow.rope.layout` | str | `interleaved`: pairs (2i, 2i+1), ggml rope mode 0 |
| `flow.self_mask.block`, `flow.self_mask.lookahead` | u32 | 10, 2: allow `k_blk < q_blk or k <= q + 2` |
| `flow.cross_mask.window` | u32 | 2: allow `abs(k - q) <= 2` |
| `flow.block_causal` | bool | true |
| `flow.adaln.order` | [str] | `shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp` |
| `flow.final_adaln.order` | [str] | `shift, scale` |
| `cond.n_heads`, `cond.ln_eps` | u32, f32 | 4, 1e-5 (c_embedder attention and LayerNorms) |

### Audio

| Key | Type | Value |
|---|---|---|
| `audio.n_hidden_taps` | u32 | 12 (encoder layer outputs 1..12, concatenated) |
| `audio.proj_in`, `audio.proj_out` | u32 | 9216, 512 |
| `audio.proj_ln_eps` | f32 | 1e-5 |
| `audio.interp` | str | `linear_align_corners` (50 Hz conv features to 25 fps) |
| `audio.normalize_input` | bool | false: the live app feeds raw PCM |
| `audio.pad` | str | `zero_to_whole_frames` |
| `audio.idle.seconds`, `.level_dbfs`, `.seed` | u32, f32, u32 | 5, -50, 42 |
| `w2v2.*` | | wav2vec2 config, same key names as voice-detect.cpp: `hidden_size`, `n_layers`, `n_heads`, `ff_dim`, `num_conv_layers`, `conv_dims`, `conv_kernels`, `conv_strides`, `conv_bias`, `feat_extract_norm`, `feat_extract_activation`, `hidden_act`, `do_stable_layer_norm`, `layer_norm_eps`, `group_norm_eps`, `num_conv_pos_embeddings`, `num_conv_pos_embedding_groups`, `pos_conv_weight_norm_dim`, `do_normalize` |

### Motion autoencoder

| Key | Type | Value |
|---|---|---|
| `mae.size`, `mae.log_size` | u32 | 512, 9 |
| `mae.style_dim`, `mae.motion_dim`, `mae.n_latent` | u32 | 512, 20, 16 |
| `mae.enc.n_resblocks`, `mae.enc.channels` | u32, [u32] | 7, [32, 64, 128, 256, 512, 512, 512, 512] (512² down to 4²) |
| `mae.dec.n_levels`, `mae.dec.channels` | u32, [u32] | 7, [512, 512, 512, 512, 256, 128, 64, 32] (4² up to 512²) |
| `mae.blur_kernel` | [i32] | [1, 3, 3, 1] (outer product, normalized to sum 1) |
| `mae.leaky_slope`, `mae.act_gain` | f32 | 0.2, sqrt(2) (FusedLeakyReLU) |
| `mae.demod_eps` | f32 | 1e-8 |
| `mae.resblock_scale` | f32 | 1/sqrt(2) |
| `mae.pad.enc_down_conv3` | [i32] | [2, 2]: blur before the 3x3 stride-2 conv |
| `mae.pad.enc_down_skip1` | [i32] | [1, 1]: blur before the 1x1 stride-2 skip conv |
| `mae.pad.dec_up_conv` | [i32] | [1, 1]: blur (kernel x4) after the 3x3 stride-2 transposed conv |
| `mae.pad.dec_upsample` | [i32] | [2, 1]: upfirdn2d up=2 (kernel x4) for the skip image and flow |
| `mae.grid` | str | base grid `linspace(-1, 1, n)` in float64, cast to float32, x along width |
| `mae.grid_sample` | str | bilinear, zero padding, `align_corners=False` |
| `mae.folded_scales` | bool | true: see the folds below |

## Tensors

All F32. Names are at most 63 characters (ggml limit).

| Prefix | Source | Transform |
|---|---|---|
| `ft.*` | `flow_transformer.*` | verbatim, except the splits below |
| `ft.c_embedder.{dual_modal_attn,user_agent_attn}.{q,k,v}_proj.{weight,bias}` | packed `in_proj_weight` / `in_proj_bias` | split in three along dim 0 (q, k, v) |
| `ft.rope.inv_freq` [64] | computed | `1 / 10000^(arange(0,128,2)/128)` in float32, as torch builds it |
| `ft.t_embedder.freqs` [128] | computed | `exp(-ln(10000) * arange(128) / 128)` in float32 |
| `audio_projection.{0,1}.*` | `audio_encoder.audio_projection.*` | verbatim (Linear 9216->512, LayerNorm) |
| `mae.enc.*`, `mae.dec.*` | `motion_autoencoder.*` | scale folds, see below |
| `mae.dec.direction.q` [512, 20] | `dec.direction.weight` | Q of a float64 QR of `weight + 1e-8` (the add in float32) |
| `w2v2.*` | `wav2vec2.*` of the HF checkpoint | prefix renamed; positional conv kept as `weight_g` [1,1,128] / `weight_v` [768,48,128], weight-norm over dim 2 |
| `sched.sigmas`, `sched.timesteps` [1000] | computed | FlowMatchScheduler tables, float32 |
| `audio.idle_pcm` [80000] | computed | 5 s pink-noise PCM (seed 42, -50 dBFS RMS) |

Motion autoencoder folds (each verified by the round-trip check against the
module's own `scale` and `lr_mul`):

- `EqualConv2d`, `EqualLinear`: `weight * scale`, with
  `scale = 1/sqrt(fan_in)`; `EqualLinear` bias `* lr_mul` (1 everywhere).
- `ModulatedConv2d`: `weight[0] * scale`, shape [O, I, k, k]. Demodulated convs
  (all StyledConv) also get `<name>.wsq` [O, I] = sum over kernel positions of
  the folded weight squared, so `demod[b, o] = rsqrt(sum_i style[b,i]^2 * wsq[o,i] + 1e-8)`.
  ToFlow convs are not demodulated and have no `.wsq`.
- Bias tensors shaped [1, C, 1, 1] (FusedLeakyReLU, ToRGB, ToFlow) are
  flattened to [C].

Dropped from the sources:

- `flow_transformer.alignment_mask`, `flow_transformer.block_bidir_causal_lookahead_mask`:
  bool buffers that are the inverse of the masks the model builds; the
  upstream loader copies parameters only and never loads them.
- `motion_autoencoder.*.noise.weight` (NoiseInjection): the decoder always runs
  with `noise=None`, so these are identity.
- `motion_autoencoder.dec.to_rgb1.*`: built but not used in `Synthesis.forward`.
- `lm_head.*` of the wav2vec2 CTC checkpoint.

## Notes for the C++ loader

- The motion-decoder ToRGB is not modulated: `conv1x1(folded) -> FusedLeakyReLU(bias) -> + bias`.
- `ft.blocks.N.attn.qkv` and `ft.blocks.N.cross_attn.kv` stay packed. The packing
  order is `(3, heads, head_dim)` and `(2, heads, head_dim)` along the output dim.
- Cross-attention has no gate: `x += CrossAttn(LN3(x), c)`.
- The RoPE table the model uses is `cos/sin(float32(pos) * inv_freq)` computed in
  float32. For positions above 1023, the live app extends the table to 65536.
