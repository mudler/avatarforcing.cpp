# Parity (M5: wav2vec2 audio features)

`tests/test_wav2vec2_parity.cpp` runs the C++ audio feature path
(`src/wav2vec2.cpp`) on the PCM of the reference dumps and compares every
stage with the PyTorch values in `wav2vec_raw.npz`, `wav2vec_normalized.npz`
and `idle_pink.npz` (see `docs/reference.md`).

## The path

```
phrase PCM, 16 kHz mono float
  -> model input: Normalized (zero mean, unit variance, replicate-pad) or
     Raw (as is, zero-pad), to n = ceil(len / 640) whole frames
  -> 7-layer conv feature encoder (GroupNorm on layer 0, exact GELU), 50 Hz
  -> linear interpolation to n frames (25 fps), align_corners
  -> feature projection (LayerNorm, Linear 512 -> 768)
  -> + positional conv (grouped, kernel 128, weight norm folded at load)
  -> encoder LayerNorm, 12 post-LN encoder layers
  -> hidden states 1..12 stacked per frame (n x 9216)
  -> audio projection (Linear, LayerNorm, SiLU), n x 512
```

The encoder runs on the interpolated 25 fps frames, as the reference
`Wav2VecModel.forward` does. The interpolation runs on the host in float32
with the index and weight arithmetic of torch's `upsample_linear1d`.

The idle features come from the 5 s pink-noise PCM stored in the GGUF
(`audio.idle_pcm`), in raw mode, as the live worker computes them. They are
computed once in `AudioEncoder::load`.

## Metric and gate

Per stage: `max_abs = max |got - ref|`, `max_rel = max_abs / max |ref|`. A
stage passes when `max_rel <= 1e-4` (`hidden.12`: `1.5e-4`, see below) and
every value is finite. The reference itself moves by up to 6e-5 relative on
the hidden states between 8 and 16 threads.

## Run

```sh
python scripts/dumps_to_baseline.py --dumps dumps-gold --w2v2 \
    --output w2v2-baseline.gguf
AVATARFORCING_TEST_GGUF=avatarforcing-f32.gguf \
AVATARFORCING_TEST_W2V2_BASELINE=w2v2-baseline.gguf \
AVATARFORCING_TEST_THREADS=8 \
ctest --test-dir build -R wav2vec2 --output-on-failure
```

`AVATARFORCING_TEST_LONG_SECONDS` (default 30, 0 turns it off) sets the length
of an extra run on the speech tiled to that length, which checks that a long
phrase runs in one graph and stays finite.

## Results

CPU backend, F32, `GGML_NATIVE=ON`, 11 s of speech (275 frames). 51 checks,
all pass. The results are the same at 4, 8 and 16 threads.

| Stage | raw max_rel | normalized max_rel |
|---|---|---|
| `model_input` | 0 | 0 |
| `conv.0` | 6.5e-07 | 1.2e-06 |
| `conv.6` | 2.9e-06 | 3.0e-06 |
| `interp` | 3.0e-06 | 3.1e-06 |
| `feature_projection.norm` | 7.4e-06 | 6.5e-06 |
| `feature_projection` | 4.0e-06 | 4.8e-06 |
| `pos_conv` | 1.9e-06 | 1.8e-06 |
| `hidden.0` | 4.9e-06 | 5.5e-06 |
| `hidden.1` .. `hidden.11` (worst) | 7.9e-06 | 7.6e-06 |
| `hidden.12` | 6.2e-05 | 9.1e-05 |
| `stacked` | 1.0e-05 | 1.5e-05 |
| `proj.linear` | 4.4e-06 | 6.2e-06 |
| `proj.norm` | 8.5e-06 | 1.1e-05 |
| `features` | 6.9e-06 | 6.3e-06 |

Idle: `idle.pcm` 0 (bit-exact), `idle.features` 4.4e-06.

`hidden.12` has the smallest range of the hidden states (max |ref| about
2.4), so its relative error is the largest. It is in the range of the
reference's own thread-count noise. The projected features, which is what the
model consumes, are at 7e-06.

On the CUDA backend `normalized.hidden.12` is at 2.5e-4 absolute, 1.05e-4
relative (CPU: 2.2e-4, 9.1e-5): the same absolute error with another GEMM
reduction order. Its gate is 1.5e-4 relative since the CUDA backend landed;
every other stage keeps 1e-4. See `docs/gpu.md`.

## Timing

8 threads, shared 16-core machine (other jobs running), median of 3: 10 s
of audio in 0.55 to 0.9 s; 30 s of audio (750 frames, one graph) in 2.0 to
3.2 s. Loading the wav2vec2 tensors plus the idle features takes 0.7 to
2.9 s, mostly file reads. The 30 s run raises the peak RSS of the test
process by about 0.3 GB (from 1.6 GB, which includes the flow and wav2vec2
weights and the baseline file).
