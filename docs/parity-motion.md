# Parity (M4: motion autoencoder)

`tests/test_motion_parity.cpp` runs the C++ motion autoencoder
(`src/motion.cpp`, `src/grid_sample.cpp`) on the inputs of the reference dumps
(`motion_encoder.npz`, `decoder_nfe6.npz`, see `docs/reference.md`) and
compares every dumped stage with the PyTorch values.

## Running it

```sh
python scripts/dumps_to_baseline.py --dumps dumps-gold --nfe 6 \
    --component motion --output motion-baseline-nfe6.gguf
AVATARFORCING_TEST_GGUF=avatarforcing-f32.gguf \
AVATARFORCING_TEST_MOTION_BASELINE=motion-baseline-nfe6.gguf \
AVATARFORCING_TEST_THREADS=8 \
ctest --test-dir build -R test_motion_parity --output-on-failure
```

Optional: `AVATARFORCING_TEST_CHUNK` (frames per decoder graph, default 1,
0 = the whole block) and `AVATARFORCING_TEST_RUNS` (timing runs, the fastest
is reported). Without the two paths the test is skipped (exit code 77).

## What the test covers

- **Encoder** on `image`: the last two `EncoderApp` layers (`enc.convs.7`,
  `enc.convs.8`), the seven decoder feature maps `feat.{0..6}` (the outputs of
  `enc.convs.{6..0}`), `s_r`, `alpha` and `r_s`. Gate `max_rel <= 2e-5`.
- **Decoder** on the reference `s_r`, `feat.*` and `latent` (so encoder
  differences do not enter): frame 0 `const`, `conv1`, `conv1.style` and, for
  each level, `up`, `up.style`, `conv`, `conv.style`, `toflow.style`,
  `flow_raw`, `flow`, `warp`, `out`, `skip`; then all 10 frames of `pixels`.
  Gate `max_abs <= 1e-4`, with one exception (see below).
- **grid_sample alone**: the reference flow and features of each level
  through the C++ operator, times the mask, against the reference `warp`.
  Gate `max_abs <= 1e-6`.
- **Frames**: SSIM of every frame against the reference (gate `>= 0.999`), and
  the uint8 frames of the app's conversion against `frames_u8` (number of
  differing values and the largest difference; gate: difference <= 1).
- **End to end**: the decoder on the C++ encoder outputs, against the
  reference pixels (SSIM gate `>= 0.999`).

## Results

CPU backend, F32, 8 threads, `GGML_NATIVE=ON`, 97 checks, all pass. The
worst check per group (`max_rel = max_abs / max |ref|`):

| Group | Worst check | max_abs | max_rel |
|---|---|---|---|
| encoder layers | `enc.convs.7` | 5.0e-06 | 9.6e-07 |
| encoder features | `feat.4` | 3.0e-05 | 3.7e-06 |
| encoder `s_r` | | 3.3e-06 | 1.6e-06 |
| encoder `alpha` | | 4.9e-06 | 1.6e-06 |
| encoder `r_s` | | 1.5e-06 | 1.4e-06 |
| modulation styles | `level.0.conv.style` | 3.8e-06 | 8.0e-08 |
| StyledConv (up) | `level.6.up` | 1.5e-05 | 6.2e-06 |
| StyledConv | `level.2.conv` | 1.4e-05 | 2.6e-06 |
| ToFlow raw output | `level.6.flow_raw` | 2.1e-05 | 1.1e-05 |
| flow field | `level.6.flow` | 6.3e-07 | 6.3e-07 |
| warp, levels 0-5 | `level.5.warp` | 3.2e-05 | 1.0e-05 |
| warp, level 6 | `level.6.warp` | 1.05e-04 | 2.3e-05 |
| next input, level 6 | `level.6.out` | 1.06e-04 | 2.3e-05 |
| RGB skip | `level.6.skip` | 1.2e-05 | 1.0e-05 |
| pixels (10 frames) | | 2.1e-05 | 1.9e-05 |
| grid_sample alone | `level.5.warp` | 4.8e-07 | 1.6e-07 |
| pixels, end to end | | 9.1e-05 | 8.1e-05 |

SSIM is above 0.99999999 for every frame, decoder only and end to end. The
uint8 frames differ in 195 of 7,864,320 values (210 with 10 frames per
graph), each by 1: a pixel that lands within the rounding error of an integer
boundary truncates to the other side.

## The level-6 warp gate

The last level samples a 512x512 feature map, so a flow error of 1e-7 moves
the sample by about 2.6e-5 pixels, and the error of `warp` and `out` at that
level is the largest in the decoder. The float32 reference has the same
problem. Measured on frame 0, all runs fed the same inputs:

| Comparison | `level.6.warp` max_abs |
|---|---|
| float32 reference (8 threads) vs the same code in float64 | 7.1e-05 |
| float32 reference, 1 thread vs 8 threads | 9.5e-05 |
| C++ vs the float64 run | 7.6e-05 |
| C++ vs the float32 reference | 1.05e-04 |

The C++ port is as close to the float64 result as the float32 reference is.
The 1e-4 gate is at the noise floor of the reference for these two tensors,
so the test gates them at 1.5e-4. Every other decoder check uses 1e-4.

## Precision notes

- `grid_sample` matches the PyTorch CPU kernel bit for bit when given the
  same flow and features (checked against `F.grid_sample` on all seven levels
  of the dumps). The x86-64 PyTorch build contracts the bilinear sum into
  fused multiply-adds; `src/grid_sample.cpp` writes that sum with explicit
  `fmaf` calls and is compiled with `-ffp-contract=off`. Its remaining
  difference to the reference `warp` (<= 4.8e-7) is the mask, where `expf`
  differs from the PyTorch sigmoid in the last bit.
- The base grids are built like numpy `linspace(-1, 1, n)` in float64 and
  cast to float32 (`motion.cpp`, also built with `-ffp-contract=off`).
- Decoding in chunks of 1 or 10 frames gives results that differ in the last
  bits (the GEMM tiling depends on the batch). Both pass the same gates.
- `r_s` uses the float64 QR basis from the GGUF; the reference computes the QR
  in float32 at run time. The difference is below 1.5e-6.

## Timing and memory

CPU, 8 threads, best of 3 runs, on a shared 16-core machine that ran other
jobs at the same time (load average 11 to 19), so these are upper bounds:

| Work | Time |
|---|---|
| encoder, once per portrait | 0.46 to 1.07 s |
| decoder, 1 frame | 0.68 s |
| decoder, 10-frame block, 1 frame per graph | 7.96 s (0.80 s/frame) |
| decoder, 10-frame block, 10 frames per graph | 5.6 to 6.8 s (0.56 to 0.68 s/frame) |

Peak RSS of the whole test process: 1.04 GB with 1 frame per graph, 1.82 GB
with 10. Of this, about 0.25 GB are the motion weights and 0.35 GB the
encoder pass; the decoder graph adds about 0.45 GB (1 frame) or 1.2 GB
(10 frames).

Per frame the decoder does about 40 GFLOP. The hot spots are the convolutions
at 256x256 and 512x512 (`ggml_conv_2d_direct`, 40 to 115 GFLOPS here), the
depthwise blur after each transposed conv (`ggml_conv_2d_dw_direct`, scalar
on the CPU: 34 ms per frame at 512x512) and the element-wise passes around
them (modulation, demodulation, phase interleave, bias and activation).

## Graph rewrites

Each is the same function as the reference; only the float rounding differs.

- Modulated conv: `conv(x * s, W) / sqrt(s^2 @ wsq^T + eps)` instead of a
  per-sample weight. `wsq` comes from the GGUF.
- Transposed 3x3 conv with stride 2: four phase convolutions on the input
  grid (sub-kernels 2x2, 1x2, 2x1, 1x1, built at load time), interleaved with
  `ggml_concat` and reshapes into the 2H+1 output. The ggml CPU
  `conv_transpose_2d` handles batch 1 only.
- `upfirdn2d` without upsampling (the blurs): depthwise convolution with the
  normalized 2-D kernel.
- `upfirdn2d` with up = 2 (the skip image and flow): zero insertion with
  `ggml_pad` and reshapes, padding, depthwise convolution.
- The activation gain of FusedLeakyReLU (g = sqrt(2) > 0) is folded into the
  demodulation and the bias: `leaky(y * (g / d) + g * b)` instead of
  `leaky(y / d + b) * g`, one pass fewer over each StyledConv output.
- The zero insertion of `upfirdn2d` (up = 2) and the phase interleave of the
  transposed conv use `ggml_concat` on 3-D shapes (all outer dimensions
  folded into `ne[2]`); `x * 0` provides the inserted zeros. On CUDA a 4-D
  concat launches one kernel per `ne[3]` slice and a pad of a 1-wide row
  leaves most threads idle.
- The portrait features live in their own backend buffer
  (`MotionModel::set_portrait`) and are broadcast across the frames of a
  graph instead of repeated.
