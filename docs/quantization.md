# Quantized GGUF files

`avatarforcing-quantize` makes an F16 or quantized GGUF from the F32 GGUF of
`scripts/convert_avatarforcing_to_gguf.py`:

```sh
avatarforcing-quantize avatarforcing-f32.gguf avatarforcing-q8_0.gguf q8_0
avatarforcing-quantize avatarforcing-f32.gguf avatarforcing-f16.gguf f16
```

| Option | Meaning |
|---|---|
| `TYPE` | `f16`, `q8_0`, `q6_k`, `q5_k` or `q4_k` |
| `--tensor-type REGEX=T` | give the tensors whose name matches `REGEX` (`std::regex`, searched in the name) the type `T` (`f32`, `f16` or a `TYPE`) instead; repeatable, the first match wins; only tensors that the policy allows |
| `--exclude REGEX` | same as `--tensor-type REGEX=f32` |
| `--threads N` | quantization threads (default: all cores) |

The input must be the F32 file. The tool reads one tensor at a time, so it
needs little memory (about 50 MB). The CLI and the C API load the output
like the F32 file; nothing else changes. `avatarforcing-cli info FILE`
prints the tensor types per component.

## Per-tensor type policy

`src/quant_policy.cpp` holds the rule. The quantizer and the loader use the
same rule: the loader rejects a file that has a type other than F32 on any
other tensor.

| Tensors | Type |
|---|---|
| flow transformer blocks: `attn.qkv`, `attn.proj`, `cross_attn.{q,kv,proj}`, `mlp.{fc1,fc2}` weights | `TYPE` |
| flow transformer: `t_embedder.mlp.{0,2}`, `adaLN_modulation.1`, `final_layer.adaLN_modulation.1`, `final_layer.linear`, `x_embedder.proj` weights | F16 when `TYPE` is a block type (Q8_0, K types), else `TYPE` |
| condition embedder: `{dual_modal,user_agent}_attn.{q,k,v,out}_proj`, `motion_embedder.{0,3}`, `to_cond_user` weights | `TYPE` |
| `audio_projection.0.weight` (Linear 9216 -> 512) | `TYPE` |
| wav2vec2 encoder layers: `attention.{q,k,v,out}_proj`, `feed_forward.{intermediate,output}_dense` weights; `feature_projection.projection.weight` | `TYPE` |
| all biases, LayerNorm and GroupNorm weights | F32 |
| motion autoencoder (`mae.*`): all tensors | F32 |
| wav2vec2 feature encoder convolutions and the positional convolution (`weight_g`, `weight_v`) | F32 |
| `ft.rope.inv_freq`, `ft.t_embedder.freqs`, `sched.*`, `audio.idle_pcm` | F32 |

147 of the 547 tensors are not F32; in a Q8_0 file, 141 of them are Q8_0
and 6 are F16. All of them are 2-D weight matrices that
the graphs use only as the first operand of `ggml_mul_mat`, the operand that
ggml supports in F16 and in every block type on the CPU and on CUDA. Their
row sizes (512, 768, 1024, 3072, 4096, 9216) are multiples of 256, so the K
types fit too.

The six F16 tensors of a block-type file are the input and output
projections of the flow transformer and the timestep and adaLN layers. In
Q8_0 they cause most of the error of one forward (see "Ablations" below);
in F16 they make the Q8_0 file 11 MB larger.

The rest stays F32 for these reasons:

- **Motion autoencoder.** The encoder and decoder run `CONV_2D`,
  `CONV_TRANSPOSE_2D` and `CONV_2D_DW` and the fused CUDA decoder kernels of
  `patches/ggml-0003-decoder-kernels.patch`. These kernels read F32 weights
  only. The decoder also reads the folded and transposed weights on the host
  at load time. Its 189 MB are 16 % of the F32 file.
- **wav2vec2 convolutions.** The feature encoder builds an F32 `im2col` and
  multiplies it with the kernel as the second operand of `ggml_mul_mat`,
  which must be F32. The positional convolution is weight-normed on the host
  at load time.
- **Biases, norms and tables.** They are small, and the host reads some of
  them directly as float arrays.

## GGUF metadata

The quantizer copies all KV of the input and adds:

| Key | Type | Value |
|---|---|---|
| `general.file_type` | u32 | llama.cpp numbering: 1 (F16), 7 (Q8_0), 18 (Q6_K), 16 (Q5_K_S), 14 (Q4_K_S) |
| `general.quantization_version` | u32 | `GGML_QNT_VERSION` (2) |
| `avatarforcing.quant.type` | str | `f16`, `q8_0`, `q6_k`, `q5_k` or `q4_k` |
| `avatarforcing.quant.policy` | str | one-line description of the rule above |
| `avatarforcing.quant.n_tensors` | u32 | number of tensors that are not F32 (147) |
| `avatarforcing.quant.overrides` | [str] | the `--tensor-type` and `--exclude` options (`REGEX=T`), when given |

## Summary

| File | Size | E2E min SSIM vs F32 (7.9 s clip) | Gate | Use |
|---|---:|---|---|---|
| `avatarforcing-f32.gguf` | 1181 MB | 1 (reference) | | reference, parity tests |
| `avatarforcing-f16.gguf` | 704 MB | 0.9941 (CUDA), 0.9988 (CPU) | pass | default |
| `avatarforcing-q8_0.gguf` | 491 MB | 0.940 (CPU) | fail | not published |

F16 passes every gate. Q8_0 does not: its frames are clean, but the motion
drifts away from the F32 motion over the blocks (see "Why Q8_0 drifts").
The K types are not published for the same reason, with more drift.

## Quality gates

All comparisons are against the F32 model on the same inputs: the same
portrait (`rumi512.png`), the same audio and the same noise file
(`noise-1234.f32`, the noise of the end-to-end gate in `docs/cli.md`), nfe 6.
The parity tests stay F32.

### Component level

`tests/test_quant.cpp` runs the F32 model, frees it, then runs the
quantized model on the same inputs: the four dumped flow forwards of the
flow baseline (with their dumped KV caches), the condition embedder, the
adaLN tables, the audio projection, and wav2vec2 on the dumped speech in raw
and normalized mode. It reports, per output:

- `max_rel = max |q - f| / max |f|`
- `mean_rel = mean |q - f| / mean |f|`

```sh
AVATARFORCING_TEST_QUANT_GGUF=avatarforcing-q8_0.gguf \
AVATARFORCING_TEST_GGUF=avatarforcing-f32.gguf \
AVATARFORCING_TEST_BASELINE=flow-baseline.gguf \
AVATARFORCING_TEST_W2V2_BASELINE=w2v2-baseline.gguf \
ctest --test-dir build -R test_quant --output-on-failure
```

The test skips when `AVATARFORCING_TEST_QUANT_GGUF` is unset. The gates
depend on `avatarforcing.quant.type` and are about twice the values below.
The velocity of the two t = 0 forwards (`primer_kvupdate`, `steady0_final`)
is printed but not gated: the sampler uses only the KV cache that they
write.

CPU backend, 8 threads, `GGML_NATIVE=ON` (x86-64, AVX2). Largest value over
the outputs of each group:

| Outputs | F16 max / mean | Q8_0 max / mean | Q5_K max / mean | Q4_K max / mean |
|---|---|---|---|---|
| adaLN tables | 3.2e-4 / 2.7e-4 | 3.2e-4 / 2.7e-4 | 3.2e-4 / 2.7e-4 | 3.2e-4 / 2.7e-4 |
| condition embedder | 4.4e-4 / 7.3e-4 | 9.5e-3 / 2.0e-2 | 4.0e-2 / 9.1e-2 | 5.8e-2 / 1.7e-1 |
| velocity (`primer_step0`, `steady0_step0`) | 1.6e-3 / 6.6e-4 | 1.8e-2 / 2.9e-3 | 2.9e-2 / 1.2e-2 | 4.8e-2 / 2.0e-2 |
| KV caches written (`self_k`, `cross_k`) | 1.7e-3 / 1.7e-3 | 1.3e-2 / 1.0e-2 | 4.0e-2 / 4.0e-2 | 6.5e-2 / 7.3e-2 |
| audio projection | 2.0e-4 / 2.3e-4 | 7.4e-3 / 6.4e-3 | 2.2e-2 / 2.9e-2 | 4.5e-2 / 5.0e-2 |
| wav2vec2 hidden states 1, 6, 12 | 1.6e-3 / 9.5e-4 | 8.6e-2 / 6.7e-2 | 4.0e-1 / 2.2e-1 | 5.1e-1 / 3.4e-1 |
| audio features (raw, normalized, idle) | 6.4e-4 / 3.8e-4 | 2.8e-2 / 2.0e-2 | 8.9e-2 / 8.0e-2 | 2.3e-1 / 3.2e-1 |
| `test_quant` | pass | pass | fail (`hidden.12`) | fail (`hidden.12`, idle features) |

The adaLN tables are the same in all four files: their weights are F16 in
each of them.

These gates check that a file is loaded and computed correctly. They do not
predict the video: an error that looks small in one forward grows over the
blocks (next section).

### Sampler drift

`test_sampler_parity` runs the primer and six steady blocks from the dumped
noise and audio features and compares the motion latents with the PyTorch
values. With a quantized file it fails its F32 gate (1e-3) by design, but its
numbers show how the error grows. `max_rel` of the latents at nfe 6 (F32:
below 1e-5):

| File | primer | steady 0 | steady 2 | steady 5 |
|---|---:|---:|---:|---:|
| F16 | 1.2e-3 | 2.3e-3 | 4.7e-3 | 2.5e-3 |
| Q8_0, all 147 tensors Q8_0 | 3.4e-2 | 3.9e-2 | 7.2e-2 | 1.1e-1 |
| Q8_0, the policy above (6 tensors F16) | 3.2e-2 | 3.7e-2 | 4.6e-2 | 4.2e-2 |
| Q8_0, the policy above, MLP in F32 | 1.0e-2 | 6.5e-3 | 2.7e-2 | 3.5e-2 |

### End to end

`avatarforcing-cli --noise noise-1234.f32 --frames-dir DIR` with each file,
then `scripts/compare_frames.py` against the F32 frames of the same backend
(SSIM as in `docs/parity-motion.md`, PSNR over all pixels, peak 255).

| Clip | Backend | File | min SSIM | mean SSIM | min PSNR | mean PSNR |
|---|---|---|---:|---:|---:|---:|
| jfk 7.9 s (198 frames) | CUDA | F16 | 0.9941 | 0.9981 | 37.9 dB | 50.5 dB |
| jfk 7.9 s | CUDA | Q8_0, all 147 tensors Q8_0 | 0.9175 | 0.9664 | 30.1 dB | 37.2 dB |
| jfk 7.9 s | CPU | F16 | 0.9988 | 0.9997 | 56.2 dB | 61.1 dB |
| jfk 7.9 s | CPU | Q8_0 (the policy above) | 0.9400 | 0.9791 | 30.8 dB | 38.4 dB |
| jfk 7.9 s | CPU | Q8_0, wav2vec2 and audio projection in F16 | 0.9545 | 0.9866 | 31.8 dB | 42.1 dB |
| jfk 7.9 s | CPU | Q8_0, wav2vec2, audio projection and MLP in F16 (638 MB) | 0.9873 | 0.9962 | 37.6 dB | 48.2 dB |
| jfk 7.9 s | CPU | F16, wav2vec2 in Q8_0 (624 MB) | 0.9674 | 0.9925 | 33.0 dB | 46.2 dB |
| jfk 2.5 s, 44.1 kHz stereo (63 frames) | CUDA | F16 | 0.9985 | 0.9994 | 54.7 dB | 57.5 dB |
| jfk 2.5 s | CUDA | Q8_0, all 147 tensors Q8_0 | 0.9665 | 0.9890 | 34.4 dB | 41.5 dB |

The CPU reference frames are those of the M6 end-to-end gate (F32, CPU,
`noise-1234`); the CUDA reference frames come from an F32 run on the same
GPU, right before the F16 and Q8_0 runs. The Q8_0 rows marked "all 147 tensors" were made before the six F16
tensors were added to the policy.

Mean SSIM of the Q8_0 file (the policy above) per 25 frames, CPU, 7.9 s
clip: 0.997, 0.995, 0.991, 0.989, 0.979, 0.975, 0.955, 0.951. F16 on CUDA:
0.999, 1.000, 0.997, 0.996, 0.997, 0.998, 0.999, 0.999; F16 on the CPU:
1.000, 1.000, 1.000, 0.999, 0.999, 1.000, 1.000, 1.000. The Q8_0 error
grows over the clip; the F16 error does not.

On the contact sheets, the Q8_0 frames are sharp and have no artefacts. The
differences are small offsets of the head, the eyes and the mouth: the face
moves along a slightly different path. The F16 frames show no visible
change. One F16 frame block (frames 110 to 119 of the CUDA run) has a small
brightness offset over the whole frame: the uint8 conversion scales each
block with its own minimum and maximum (`output.u8_mode`), so a small change
of one extreme value shifts all pixels of the block.

### Gates

| Gate | F16 | Q8_0 target |
|---|---|---|
| `test_quant` | pass (values above) | pass (values above) |
| E2E min SSIM, both clips, CPU and CUDA | >= 0.99 | >= 0.99 |
| E2E mean SSIM | >= 0.995 | >= 0.995 |
| contact sheet | no visible change | no visible change |

F16 passes all gates. Q8_0 passes `test_quant` and fails the E2E gates.

## Why Q8_0 drifts

The sampler is autoregressive: each 10-frame block starts from the last
two latents of the previous block and reads a KV cache that the previous
blocks wrote, and classifier-free guidance (scale 2) amplifies the
difference between two forwards. An error in one block changes the input
of every later block, so the per-forward error of the weights adds up over
the clip.

ggml also rounds the activations: a Q8_0 `mul_mat` quantizes its F32 input
to Q8_0 on the CPU (Q8_1 on CUDA) before the dot products. An F16 `mul_mat`
rounds the input to F16, which keeps 3 more bits.

### Ablations

Single forwards (`test_quant`, flow part, CPU), `mean_rel` of the
velocity of `steady0_step0` and of the self-attention K cache that the
primer KV update writes, with groups of tensors kept in F32:

| Q8_0 file | velocity | primer `self_k` |
|---|---:|---:|
| all 147 tensors Q8_0 | 1.7e-2 | 5.1e-2 |
| blocks (`ft.blocks.*`) in F32 | 1.7e-2 | 5.0e-2 |
| `attn.qkv` in F32 | 1.7e-2 | 5.1e-2 |
| `mlp.fc1`, `mlp.fc2` in F32 | 1.7e-2 | 5.1e-2 |
| condition embedder in F32 | 1.7e-2 | 5.1e-2 |
| `adaLN_modulation` (both) in F32 | 1.6e-2 | 4.1e-2 |
| `t_embedder`, `x_embedder.proj`, `final_layer.linear` in F32 | 4.7e-3 | 2.7e-2 |
| the five above plus the condition embedder in F32 | 2.7e-3 | 1.0e-2 |
| the policy above (the six tensors in F16) | 2.9e-3 | 1.0e-2 |
| the policy above, MLP in F32 | 1.3e-3 | 8.4e-3 |
| the policy above, blocks in F32 | 6.4e-4 | 1.7e-3 |
| F16 file, for comparison | 6.6e-4 | 1.7e-3 |

The input and output projections and the timestep and adaLN layers cause
most of the error of one forward; the policy keeps them in F16. After that,
the block weights are the largest source, the MLP first.

wav2vec2 is sensitive too: with the flow transformer in F16 and only
wav2vec2 in Q8_0, the end-to-end min SSIM is 0.967 (table above). Its last
hidden states have the largest relative error of all outputs, and the audio
features condition every forward of the sampler. Keeping both the MLP and
wav2vec2 in F16 brings the end-to-end min SSIM to 0.987, but the file is
then 638 MB, only 9 % smaller than the F16 file.

## Speed and memory

### CUDA

NVIDIA Thor, `GGML_NATIVE=OFF`, nfe 6, 7.9 s clip, `--frames-dir` output.
Median times of a steady block (10 frames, 400 ms of video):

| File | sampler | decoder | peak host RSS |
|---|---:|---:|---:|
| F32 | 175 ms | 206 ms | 674 MB |
| F16 | 126 ms | 178 ms | 783 MB |
| Q8_0 (all 147 tensors) | 128 ms | 176 ms | 726 MB |

The decoder is F32 in all files; its time changes by the run-to-run noise
only. The sampler is 28 % faster with F16 or Q8_0 weights. The memory that
the device reports in use is 2.5 GB to 2.9 GB for all three files: the
weights are a small part of the device memory, and Thor shares one memory
between the CPU and the GPU, so the numbers move with other processes.

### CPU

16-core x86-64 CPU, 8 threads, `GGML_NATIVE=ON`, 7.9 s clip. The machine
ran other heavy jobs during these measurements (load average above 35), so
the times are much slower than in `docs/cli.md` and only the ratios mean
something.

| File | Size | Peak RSS of `avatarforcing-cli` | One flow forward, steady block (`test_flow_parity`, 2 runs) |
|---|---:|---:|---:|
| F32 | 1181 MB | at least 1555 MiB (the runs were stopped by the OOM killer of the loaded machine) | 2.73 s, 3.38 s |
| F16 | 704 MB | 1109 MiB | 3.41 s, 3.26 s |
| Q8_0 | 491 MB | 905 MiB | 2.43 s, 2.17 s |

- The weights are in memory once (no mmap), so the peak RSS falls with the
  file size.
- On this CPU, an F16 `mul_mat` is not faster than F32. A Q8_0 `mul_mat` is
  about 25 % faster.
- On the CPU the motion decoder, which is F32 in every file, takes 90 % of
  the generation time (`docs/cli.md`). The type of the flow transformer and
  wav2vec2 weights therefore changes the CPU generation time by a few percent
  at most.

## K types

`avatarforcing-quantize` makes Q6_K, Q5_K and Q4_K files with the same
policy (the six sensitive tensors in F16). They are not published:

- Their error is larger than that of Q8_0 (`test_quant` above), and Q8_0
  already fails the end-to-end gates.
- They save little. The motion autoencoder (189 MB) and the wav2vec2
  convolutions stay F32 in every file, so the Q5_K file is 405 MB and the
  Q4_K file 377 MB, against 491 MB for Q8_0 and 704 MB for F16.

## Choosing a file

- **F16** for normal use: 40 % smaller than F32, the same video within
  the gates, and a faster sampler on CUDA.
- **F32** for parity work and as the reference.
- Q8_0 and the K types: only when the file size matters more than following
  the F32 motion. Make them with `avatarforcing-quantize`.
