# avatarforcing-cli

`avatarforcing-cli` turns a portrait and a speech recording into a lip-synced
talking-head video. It runs the whole model in C++ on the ggml CPU or CUDA
backend (`docs/gpu.md`) and calls `ffmpeg` to encode the mp4. The C API
(`docs/capi.md`) runs the same pipeline.

```sh
avatarforcing-cli --model avatarforcing-f32.gguf --portrait portrait512.png \
    --audio speech.wav -o out.mp4
avatarforcing-cli info avatarforcing-f32.gguf
avatarforcing-cli --model avatarforcing-f32.gguf --portrait portrait512.png \
    --live call.txt -o call.mp4
```

## Options

| Option | Default | Meaning |
|---|---|---|
| `--model PATH` | required | the avatarforcing GGUF |
| `--portrait PATH` | required | square face image at the model size (512x512), PNG, JPEG or BMP |
| `--audio PATH` | required | speech WAV: any sample rate, any channel count, PCM or float |
| `--live SCRIPT` | | live mode instead of `--audio`: simulates a live call (see "Live mode") |
| `-o`, `--output PATH` | | mp4 output (H.264 video, AAC audio); needs `ffmpeg` on `PATH` |
| `--frames-dir DIR` | | also writes every frame as `DIR/frame_NNNNN.png` |
| `--nfe N` | GGUF `sampler.default_nfe` (6) | number of sampler time points; N - 1 denoising forwards per block |
| `--cfg-scale S` | GGUF `cfg.audio_scale` (2.0) | audio classifier-free guidance scale |
| `--seed N` | 42 | seed of the built-in noise generator |
| `--noise PATH` | | reads the noise from a file instead (see below) |
| `--audio-mode M` | `normalized` | `normalized` (offline path) or `raw` (live app) wav2vec2 input |
| `--max-seconds N` | 30 | uses only the first N s of audio (0: no limit) |
| `--threads N` | all cores | CPU threads |
| `--backend B` | `cpu` | compute device: `cpu`, `cuda` or `cuda:N`; `--device` is an alias |
| `--decode-batch N` | 1 on the CPU, 10 on a GPU | frames per decoder graph (0: a whole block) |
| `--overlap`, `--no-overlap` | on for a GPU | decode block N in a second thread while the sampler computes block N + 1; on CUDA the sampler then runs on high-priority streams |
| `--profile` | off | per-op time profile of the decoder graphs (node by node, slow) |
| `--realtime` | off | live mode: steps on the wall clock, one block every 400 ms, and counts the late blocks |
| `--queue-blocks N` | 3 | live mode: blocks that may wait for the writer |
| `--reanchor-every N` | GGUF (112) | live mode: steady blocks before a reanchor when idle; 0: none |
| `--reanchor-max-defer N` | GGUF (162) | live mode: steady blocks before a reanchor also in speech; 0: none |
| `--reanchor-fade M` | `latent` | live mode: fade after a reanchor: `latent` (motion), `pixel` (the live app's RGB cross-fade) or `none` |
| `--reanchor-fade-frames N` | GGUF (25) | live mode: fade frames after a reanchor; 0: none |
| `--reanchor-hold` | off | live mode: the reanchor block repeats the last frame, as the live app |

Give `-o`, `--frames-dir` or both. At the end the tool prints the time of
each stage, the sampler and decoder time of a steady block and the period
between two finished blocks (with `--overlap` the two stages run at the same
time, so the period is less than their sum), and the realtime factor: the generation time (portrait encoder,
audio features, sampler, decoder and output) divided by the video length.
A value below 1 is faster than real time. The tool also prints the factor
with the model load included.

`info` prints the GGUF metadata and the tensor count and size per component.

### Inputs

- **Portrait.** The tool does not detect or crop the face. The image must
  already be a face-centred square at the model size.
  `scripts/prepare_portrait.py` makes one. Pixels map to
  `(u8 / 255 - 0.5) / 0.5` in float32.
- **Audio.** The tool reads the WAV with dr_wav and mixes the channels down
  to mono by averaging them, as librosa does. 16-bit samples convert as
  `int16 / 32768`. When the rate is not 16 kHz, a Kaiser-windowed sinc
  resampler (16 zero crossings, beta 8.6, cutoff at 0.97 of the lower Nyquist
  frequency) converts it. Upstream resamples with `librosa.load` (soxr), so
  the features of a resampled file differ slightly from upstream. A 16 kHz
  file is used as is.
- **Noise.** The built-in generator is xoshiro256\*\* seeded through
  splitmix64, with a Box-Muller transform in double precision. It does not
  reproduce `torch.randn`, so a seed does not give the upstream video for the
  same seed. For a parity run, `--noise` reads raw little-endian float32
  values: the primer noise `[clip_frames, dim_w]` (50 x 512) first, then the
  noise of each steady block `[block_frames, dim_w]` (10 x 512), in the order
  of the blocks. The file must hold at least that many values for the run.

## The pipeline

The tool follows the offline streaming path of the upstream code,
`inference.py` `run_stream` and `AvatarForcing.inference_stream` in
AvatarVoice:

1. **Portrait encoder.** The motion encoder gives the appearance code `s_r`,
   the identity code `r_s` and the decoder feature maps.
2. **Audio features.** In `normalized` mode, the PCM is normalized to zero
   mean and unit variance over the whole file (`Wav2Vec2FeatureExtractor`),
   then cut to the first 30 s (`DataProcessor.preprocess`). The video has
   `T = ceil(samples * 25 / 16000)` frames. wav2vec2 runs once on the whole
   clip, with the last sample repeated up to `T * 640` samples
   (`AudioEncoder.inference`).
3. **Primer.** 50 frames of noise, audio features `0..49` (the last feature
   row repeated when `T < 50`), denoised without the KV cache, then one
   forward at t = 0 that writes the cache. All 50 primer frames go to the
   video, decoded in 5 blocks of 10.
4. **Steady blocks.** For `t = 50, 60, ...` while `t < T`: the last 2 tokens
   of the previous block as context, 10 tokens of new noise, audio features
   `t - 2 .. t + 9` (the last row repeated past the end), RoPE start position
   `t - 2`, KV read. The block keeps only `min(10, T - t)` frames, so the
   last block can be short.
5. **Decoder.** With style `s_r + latent`; one frame per graph on the CPU, a
   block of 10 frames per graph on a GPU. With `--overlap` the decoder has
   its own backend instance and thread, and a queue of at most two blocks.
6. **uint8 conversion**, per block of frames as upstream yields them: clamp to
   [-1, 1], scale with the minimum and maximum of the whole block to
   [0, 255] (eps 1e-8), truncate.
7. **Output.** Raw RGB frames go to `ffmpeg` on a pipe, at 25 fps, encoded
   with libx264 (yuv420p). The input audio file is muxed as AAC with
   `-shortest`. Audio shorter than the 2 s primer is padded with silence
   to 2 s (`apad=whole_dur`).

The video has `max(50, T)` frames. When the audio is shorter than 2 s, the
mp4 has the 50 primer frames and 2 s of audio: the speech, then silence.
Upstream cuts the video to the audio length with `-shortest`.

### Differences from upstream `run_stream`

- No face detection and crop: the portrait must be pre-cropped.
- No user stream (listener audio and video). The AvatarVoice two-branch CFG
  replaces the user inputs by zeros, so they do not change the result.
- float32 on the CPU or CUDA. Upstream runs under bf16 autocast on CUDA.
- The noise comes from the built-in generator or from a file, not from
  `torch.randn`.
- The steady-block KV update is folded into the last (t = 0) step of the
  block, as the live app does. `inference_stream` runs a separate forward at
  t = 0 after the last step. Both forwards see the same tokens, the same
  t = 0 adaLN rows and the same cache, so the result is the same with one
  forward fewer.
- One ffmpeg call encodes and muxes. Upstream encodes to a temporary file,
  then muxes it with the audio (`-vcodec copy -acodec aac -shortest`).
- The upstream default nfe of `inference.py` is 10. The CLI default comes
  from the GGUF, which stores the live app value (6).

### Differences in the live app (`voicechat/avatar_worker.py`)

The live app runs the same model in a different loop. The offline mode of
the CLI does not implement these. Live mode (`--live`, next section)
implements all of them, with one change: it fades the reanchor in the
motion space by default, not in RGB (see below):

- wav2vec2 runs per TTS phrase on the raw PCM (`--audio-mode raw`): no
  normalization, zero padding, `max(1, ceil(samples / 640))` frames.
- Past the end of the speech, the app feeds cycled pink-noise idle features
  (`audio.idle.*` in the GGUF) instead of repeating the last feature row.
- The primer only fills the KV cache; its frames are not shown.
- Every 112 blocks (about 45 s) once idle, or after 162 blocks at the
  latest, the app re-runs a primer to reset the cache (reanchor) and
  crossfades over 25 frames.
- The video length follows the session, not the audio length.

## Live mode

`--live SCRIPT` simulates a live call on a live session of the library
(`docs/capi.md`, "Live mode"). The script lists timed events, one per line:

```
# seconds  event  [WAV file]
4.0   say        hello.wav
9.0   say        answer.wav
11.0  interrupt
14.0  end
```

- `say` pushes the WAV as one phrase that starts at that time. A relative
  path is relative to the script.
- `interrupt` is a barge-in at that time: the phrase audio stops there and
  the face goes idle.
- `end` ends the call. Without it, the call ends 2 s after the last event.

The reanchor works as in the library (`docs/capi.md`, "Reanchor"): by
default every 112 steady blocks when idle, at the latest after 162, with a
25-frame fade of the motion. A short call does not reach it.
`--reanchor-every N`, `--reanchor-max-defer N` and
`--reanchor-fade-frames N` change the policy and the fade length (0 turns
each off). `--reanchor-fade pixel` uses the RGB cross-fade of the live app
instead, and `--reanchor-hold` repeats the last frame in the reanchor
block, as the live app does. The tool prints each reanchor and its step
time.

Use `--nfe 6`, the default, for live mode. At nfe 10 a reanchor step takes
391 ms of the 400 ms block on an NVIDIA Thor: a live caller must then keep
two blocks of frames ahead of playback.

Time 0 is frame 0, the first frame after the hidden primer. The tool steps
one block (10 frames, 400 ms) at a time. Before each step it sends the
events whose frame is in that block, as a caller that runs one block ahead
of playback does. So a phrase starts at its exact frame, unless it would
overlap an earlier phrase. In that case it starts at the end of the earlier
phrase, and the tool prints the frame where it starts.

The mp4 has the frames of the whole call. Its audio track holds each phrase
at its start frame, cut at the interrupts. wav2vec2 gets the raw PCM
(`--audio-mode raw`) unless `--audio-mode` says otherwise. At the end the
tool prints the time from the start of the session to the first frame,
the step time per block, for idle blocks, for blocks with speech and for
reanchor blocks, the reanchor frames, and the session statistics. With `--realtime` it waits for the time
of each block, as a live caller does, and counts the blocks that were not
ready in time. The writer runs in the same loop, so the count includes the
output time: `--frames-dir` writes PNG files at about 50 ms per frame and
makes most blocks late. Use `-o` for a meaningful count.

## End-to-end gate

`scripts/reference_stream.py` runs the upstream `inference_stream` on the CPU
in float32 with injected noise and writes its uint8 frames as PNG files.
`scripts/compare_frames.py` compares them with the CLI frames. The gate is a
per-frame SSIM >= 0.995 on every frame.

```sh
python scripts/reference_stream.py --avatarvoice AvatarVoice \
    --flow-ckpt flow_transformer.pth --mae-ckpt motion_autoencoder.pth \
    --wav2vec wav2vec2-base-960h --portrait rumi512.png --audio jfk-7.9s.wav \
    --noise noise-1234.f32 --noise-seed 1234 --nfe 6 --out ref-frames
avatarforcing-cli --model avatarforcing-f32.gguf --portrait rumi512.png \
    --audio jfk-7.9s.wav --noise noise-1234.f32 --nfe 6 --frames-dir cpp-frames
python scripts/compare_frames.py --got cpp-frames --ref ref-frames --min-ssim 0.995
```

`reference_stream.py` decodes one frame at a time (a batch of 10 frames at
512x512 needs several GB on the CPU) and reads the WAV as `int16 / 32768`
instead of `librosa.load`, which gives the same samples for a 16 kHz mono
16-bit file. Use such a file for the gate.

### Result

Portrait `rumi512.png` (see `docs/reference.md`), the first 7.9 s of
`jfk.wav` (126400 samples, T = 198 frames: the primer, 14 full steady blocks
and a last block of 8 frames), nfe 6, CFG 2.0, normalized audio, noise from
`numpy.random.default_rng(1234).standard_normal`, 8 threads:

| Frames | Min SSIM | Mean SSIM | Max pixel diff |
|---|---|---|---|
| all 198 | 0.999988 | 0.999999 | 1 |
| primer (0..49) | 0.999999 | 0.999999 | 1 |
| steady (50..197) | 0.999988 | 0.999998 | 1 |
| last block (190..197) | 0.999999 | 0.999999 | 1 |

The mean absolute difference is 0.0002 uint8 levels: almost all pixels are
identical, and no pixel differs by more than one level.

### Timing

16-core x86-64 CPU, 8 threads, F32, the full 11 s `jfk.wav` (275 frames),
nfe 6:

| Stage | Time |
|---|---|
| load | 1.8 s |
| portrait encoder | 0.51 s |
| audio features | 0.69 s |
| sampler | 4.95 s (primer 0.52 s, steady block median 0.19 s) |
| decoder | 153.9 s (0.56 s per frame) |
| uint8 conversion and output (PNG files and ffmpeg) | 11.6 s |
| generation total | 171.7 s |

The realtime factor is 15.6 (15.8 with the load). The decoder takes 90 % of
the time.

### Video length

`-shortest` makes ffmpeg stop at the end of the audio stream. With 198 frames
(7.92 s) and 7.90 s of audio, the mp4 has 197 frames. Upstream muxes with
`-shortest` too. Audio shorter than the 2 s primer is padded with silence to
2 s, so the mp4 always has the 50 primer frames.
