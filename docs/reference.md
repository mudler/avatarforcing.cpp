# Reference dumps (M0)

`scripts/dump_reference.py` runs the upstream PyTorch code on the CPU in
float32 and writes the gold tensors that each C++ component must match. It
imports the model classes from an AvatarVoice checkout (`--avatarvoice`,
written against commit `566f8e1`) and does not modify it: it only adds forward
hooks and wraps `forward_with_precomputed` on the model instance.

## Running

```sh
python scripts/prepare_portrait.py --input AvatarVoice/data/rumi.jpg --output rumi512.png
python scripts/dump_reference.py --avatarvoice AvatarVoice \
    --flow-ckpt flow_transformer.pth --mae-ckpt motion_autoencoder.pth \
    --wav2vec wav2vec2-base-960h --portrait rumi512.png --audio jfk.wav \
    --out dumps-gold
# determinism self-check: a second run must be bit-identical
python scripts/dump_reference.py ... --out dumps-rerun
python tests/python/check_dumps.py dumps-gold dumps-rerun
```

Options: `--nfe 6 10` (sampler runs; default both), `--steady-blocks 6`,
`--audio-modes raw normalized`, `--sampler-audio-mode raw`,
`--noise-seed 1234` or `--noise-file noise.npz` (keys `primer` [50,512] and
`steady` [K,10,512]), `--detail-nfe 6` (which run also writes the per-module
and decoder files), `--threads 8`.

The output does not change between runs with the same arguments and thread
count (checked bit for bit by `check_dumps.py`). A different `--threads`
changes the low bits of the results: at 16 threads instead of 8, the flow
outputs differ by up to 4e-6 relative, the wav2vec2 hidden states by up to
6e-5 relative, and the motion encoder not at all. Treat this as the noise
floor of the reference itself.

## Inputs used for the gold set

- **Portrait**: `data/rumi.jpg` of AvatarVoice, which comes from the
  AvatarForcing repository (its authors note it is AI-generated). It is
  already a face-centred 512x512 image, so `prepare_portrait.py` only
  re-encodes it as PNG. The upstream pipeline would run an S3FD face crop on
  it first; the C++ engine expects a pre-cropped 512x512 portrait instead.
- **Speech**: `samples/jfk.wav` from the whisper.cpp repository (11.0 s,
  16 kHz mono 16-bit). It is a recording of the 1961 inaugural address of
  J. F. Kennedy, a work of the US government in the public domain. SHA-256
  `59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e`.
- **Noise**: `numpy.random.default_rng(1234)`, `standard_normal` float32:
  first the primer [50, 512], then the steady blocks [6, 10, 512]. The live
  app draws unseeded `torch.randn` noise; the C++ engine takes the noise from
  `inputs.npz` for parity runs.

## How the run follows the live app

The sampler is the loop of `voicechat/avatar_worker.py` (`_run_loop`) without
the rings, the clock and the reanchor:

1. Primer: `x = noise.primer` [1, 50, 512], audio features `wa[0:50]`,
   `start_pos = 0`, no KV read. One forward per step of
   `linspace(1000, 0, nfe - 1)`. After the last step, a separate KV-update
   forward at t = 0 (`update_kv_cache`, the app does the same for the primer).
2. Steady block b (b = 0..5): `x = concat(previous x[-2:], noise.steady[b])`,
   features `wa[50 + 10b - 2 : 50 + 10b + 10]`, `start_pos = 48 + 10b`, KV
   read; the KV update is folded into the last step (`update_kv_cache=True`
   on the t = 0 step, as the app does).
3. Decoder: steady block 0 of the nfe = 6 run, 10 frames, style
   `s_r + x[-10:]`.

The audio features that drive the sampler are the `raw` wav2vec2 features of
the speech clip (the live app path, see below). The clip gives 275 frames and
the run needs 110, so the idle (pink-noise) features are not used here. If a
clip is shorter, the script continues with cycled idle features, as the app
does past the end of the speech.

Differences from the live app, all deliberate:

- float32 on the CPU. The app runs under bf16 autocast on CUDA.
- Injected noise instead of unseeded `torch.randn`.
- `u_cfg_scale` is passed as 0 (the app value). The two-branch CFG ignores it.
- The decoder runs one frame at a time to bound memory. The decoder has no
  cross-frame interaction, so this is the same computation as the app's
  batch of 10.

## wav2vec2 input modes

- `raw` (live app, `avatar_worker._wav2vec_loop`): the PCM as is, zero-padded
  up to a whole number of video frames (`ceil(len / 640) * 640` samples),
  `seq_len = ceil(len / 640)`.
- `normalized` (offline `inference.py`): `Wav2Vec2FeatureExtractor`
  normalization (zero mean, unit variance over the clip, eps 1e-7), then
  `seq_len = ceil(len * 25 / 16000)` and `AudioEncoder.inference` pads by
  replicating the last sample.

transformers 4.30.2 on torch >= 2.1 warns that the positional-conv weights
(`weight_g`, `weight_v`) are "newly initialized". The warning is spurious:
torch's `weight_norm` parametrization maps the legacy names when it loads the
state dict. The dump script checks on every run that the conv holds the
checkpoint weights, and stops if it does not.

## Files

All arrays are float32 unless noted, in PyTorch layout (row-major, the last
dimension is the fastest). `B = 2` is the CFG batch, in the order
`[uncond, audio]`. `.npz` files are written uncompressed (`np.savez`), so a
C++ reader needs only a stored-zip parser.

### `inputs.npz`

| Key | Shape | Content |
|---|---|---|
| `portrait_u8` | [512, 512, 3] uint8 | the PNG pixels |
| `portrait` | [1, 3, 512, 512] | `(u8 / 255 - 0.5) / 0.5` |
| `pcm` | [176000] | speech, `int16 / 32768` |
| `noise.primer` | [50, 512] | primer noise |
| `noise.steady` | [6, 10, 512] | steady-block noise |

### `motion_encoder.npz`

`image`; `enc.convs.{0..8}`: output of each `EncoderApp` layer
(0: 1x1 conv at 512² with 32 channels, 1..7: ResBlocks down to 4², 8: the 4x4
conv to [1, 512, 1, 1]); `s_r` [1, 512]; `feat.{0..6}`: the decoder feature
maps from 8² to 512² (`res[::-1][2:]`); `alpha` [1, 20] (`enc.fc(s_r)`);
`r_s` [1, 512] (`Direction(alpha)`); `direction.q_fp32`, `direction.r_fp32`:
the float32 QR that upstream computes at run time.

### `wav2vec_raw.npz`, `wav2vec_normalized.npz`

| Key | Shape (11 s clip) | Content |
|---|---|---|
| `pcm_in` | [176000] | PCM before the mode's preprocessing |
| `model_input` | [176000] | PCM given to `AudioEncoder.inference` |
| `n_frames` | () int64 | 275 |
| `conv.{0..6}` | [512, T_i] | each feature-extractor conv layer (after its norm/GELU) |
| `interp` | [275, 512] | conv features linearly interpolated to 25 fps |
| `feature_projection`, `feature_projection.norm` | [275, 768], [275, 512] | projection output and its LayerNorm output |
| `pos_conv` | [275, 768] | positional conv embedding |
| `hidden.{0..12}` | [275, 768] | encoder hidden states; 1..12 are used |
| `stacked` | [275, 9216] | `hidden.1..12` concatenated per frame |
| `proj.linear`, `proj.norm` | [275, 512] | audio projection stages |
| `features` | [275, 512] | final 512-d features (after SiLU) |

### `idle_pink.npz`

`pcm` [80000] (5 s of pink noise, seed 42, -50 dBFS RMS, the app's
`_shaped_pink_noise`), `features` [125, 512] (raw-mode wav2vec2 features at
25 fps), `seed`, `level_dbfs`.

### `adaln_nfe{6,10}.npz`

`timesteps`: the step values plus 0, descending; `step_list`; `t_freq`
[n, 256] (sinusoidal embedding); `t_embed` [n, 1024]; `adaln6` [n, 6144]
(shared by all blocks, order shift_msa, scale_msa, gate_msa, shift_mlp,
scale_mlp, gate_mlp); `final_adaln2` [n, 2048] (shift, scale).

### `sampler_nfe{6,10}.npz`

| Key | Shape | Content |
|---|---|---|
| `step_list` | [nfe-1] | `linspace(1000, 0, nfe - 1)` |
| `sigma_index` | [nfe-1] int64 | argmin index into the sigma table per step (nfe 6: 0, 250, 499, 749, 999) |
| `noise.primer`, `noise.steady`, `r_s`, `a_cfg_scale` | | inputs |
| `primer.wa` | [50, 512] | audio features of the primer |
| `primer.c` | [2, 50, 1024] | condition (c_embedder output) |
| `primer.x_in.{i}`, `primer.x.{i}` | [1, 50, 512] | x before / after step i |
| `primer.v.{i}` | [2, 50, 512] | raw transformer output of step i (both branches) |
| `primer.kvupdate.v` | [2, 50, 512] | output of the KV-update forward |
| `primer.kv.{self_k,self_v}` | [8, 2, 8, 38, 128] | self cache after the primer: [layer, branch, head, pos, dim], post-RoPE |
| `primer.kv.{cross_k,cross_v}` | [8, 2, 8, 2, 128] | cross cache after the primer |
| `primer.x_final` | [1, 50, 512] | |
| `steady.{b}.wa` | [12, 512] | audio features of block b |
| `steady.{b}.start_pos` | () int64 | 48 + 10b |
| `steady.{b}.c` | [2, 12, 1024] | condition |
| `steady.{b}.x_in.{i}`, `steady.{b}.x.{i}` | [1, 12, 512] | x before / after step i (2 context + 10 new tokens) |
| `steady.{b}.v.{i}` | [2, 12, 512] | raw transformer output |
| `steady.{b}.kv.*` | as above | caches after block b (after the folded update) |
| `steady.{b}.latent` | [10, 512] | `x[-10:]` after the last step, the decoder input |

The last step (t = 0) does not change x, in the primer and in the steady
blocks (`solve_closed` keeps tokens whose current and next time are both 0).
Its only effect is the KV update.

### `flow_forward_nfe6.npz`

Per-module detail of four forwards, prefixed by tag: `primer_step0` (N = 50,
no KV), `primer_kvupdate` (N = 50, writes the caches), `steady0_step0`
(N = 12, KV read), `steady0_final` (N = 12, KV read and folded update). Each
tag has:

- `in.x` [2, N, 512], `in.c` [2, N, 1024], `in.wr` [2, N, 512] (`r_s`
  repeated), `in.adaln6` [6, N, 1024] and `in.final_adaln2` [2, N, 1024]
  (branch 0; per token, so the steady context tokens show their t = 0
  rows), `in.freqs_cis` [N, 128] (cos ‖ sin of the RoPE slice),
  `in.update_kv_cache`, `in.kv_read` (bool);
- `kv_before.*`, `kv_after.*` when the forward uses the caches;
- `x_embed` [2, N, 1024], `block.{0..7}` (block outputs), block 0 internals
  `block.0.{attn_in, attn_out, cross_in, cross_out, mlp_in, mlp_out}`,
  `final` and `out.v` [2, N, 512];
- for `primer_step0` and `steady0_step0` also the c_embedder stages:
  `c_embedder.in.{wa_user, motion_user, wa}`, `c_embedder.motion_embedder`,
  `.dual_modal_attn`, `.cross_modal_feat`, `.user_agent_attn`, `.unified`,
  `.out`.

### `decoder_nfe6.npz`

`latent` [10, 512], `s_r`, `style` [10, 512] (= `s_r + latent`), `pixels`
[10, 3, 512, 512] (decoder output, not clamped), `frames_u8` [10, 512, 512, 3]
uint8 (the app's conversion: clamp to [-1, 1], block-wide min/max to
[0, 255], truncate). Frame-0 internals: `const` [1, 512, 4, 4], `conv1`,
`conv1.style`, and for each level l = 0..6 (8² .. 512²): `level.l.up`,
`level.l.conv` (StyledConv outputs), `level.l.{up,conv,toflow}.style`
(modulation outputs), `level.l.flow_raw` (ToFlow conv + bias + upsampled
previous flow, 3 channels), `level.l.flow` [1, s, s, 2] (sampling grid),
`level.l.warp` (masked grid_sample output), `level.l.out` (next decoder
input) and `level.l.skip` (RGB skip, the final image at l = 6).

### `manifest.json`

Arguments, library versions, SHA-256 of the portrait, audio and checkpoints,
and every array's shape and dtype.

## Gates for the C++ port

From the scoping study. Per-module float32 relative error <= 2e-5;
primer + 6 blocks with the injected noise <= 1e-3; decoder per level <= 1e-4
absolute; end-to-end per-frame SSIM >= 0.995 against these frames. Keep the
flow transformer in F32: CFG amplifies rounding.
