# C API

`libavatarforcing` is a shared library with a flat C API, declared in
`include/avatarforcing.h`. It is for programs that load the library at run
time, for example Go with [purego](https://github.com/ebitengine/purego)
(no cgo) or Python with `ctypes`, and for C programs.

The library runs the same pipeline as `avatarforcing-cli` (`docs/cli.md`).
Both use the code in `src/pipeline.cpp`, so for the same inputs and options
the API gives the same frames as the CLI.

## Build

CMake builds the library by default (`AVATARFORCING_BUILD_SHARED=ON`):

```sh
cmake -B build
cmake --build build -j8
```

The result is `build/libavatarforcing.so`. It contains ggml: CMake builds
ggml as static libraries with position-independent code, so the library
needs only the C and C++ runtimes, OpenMP and, in a CUDA build, the CUDA
runtime. Set `-DBUILD_SHARED_LIBS=ON` to link ggml as shared libraries
instead. On Linux the library exports only the `avatarforcing_*` functions
(a linker version script hides everything else):

```sh
nm -D --defined-only build/libavatarforcing.so
```

For CUDA, configure with `-DGGML_CUDA=ON` as in `docs/gpu.md` and load the
context with `"backend": "cuda"`.

## Design

- **Plain C types only.** Handles are opaque pointers. Arguments are
  pointers, `int`, `float` and NUL-terminated UTF-8 strings. No function
  takes or returns a struct by value. No function takes a callback.
- **Errors.** A function that returns `int` returns 0 on success and -1 on
  error. A function that returns a pointer returns `NULL` on error.
  `avatarforcing_last_error(ctx)` gives the message. `avatarforcing_load` has
  no context yet, so it returns its message through `error_out`.
- **Memory.** The library allocates returned strings and frame buffers with
  `malloc`. Release them with `avatarforcing_free_string` and
  `avatarforcing_free_frames`.
- **Options** are flat JSON objects. `NULL` or `""` means no options. An
  unknown key is an error, so a misspelled option does not go unnoticed.
- **ABI version.** `avatarforcing_abi_version()` returns the version of the
  library. A change to a signature or to the meaning of a function
  increments it. Compare it with `AVATARFORCING_ABI_VERSION` from the header
  that your code was written for.
- **No C++ exceptions** cross the API. Every function catches them and
  returns an error.

## Functions

| Function | Purpose |
|---|---|
| `avatarforcing_abi_version` | ABI version of the library |
| `avatarforcing_load` | load a GGUF into a new context, with context options |
| `avatarforcing_free` | release a context (and its open session) |
| `avatarforcing_last_error` | message of the last error on a context |
| `avatarforcing_free_string` | release a string from the library |
| `avatarforcing_info_json` | frame size, frame rate, sample rate, block sizes, defaults, device |
| `avatarforcing_cancel` | stop the running generation before its next block |
| `avatarforcing_generate_frames` | portrait and WAV file to RGB frames |
| `avatarforcing_generate_frames_pcm` | portrait and float PCM to RGB frames |
| `avatarforcing_free_frames` | release a frame buffer |
| `avatarforcing_generate_mp4` | portrait and WAV file to an mp4 file (ffmpeg) |
| `avatarforcing_generate_mp4_pcm` | portrait and float PCM to an mp4 file (ffmpeg) |
| `avatarforcing_session_open` | start a session with a portrait |
| `avatarforcing_session_push_pcm` | add one phrase of audio and compute the blocks it completes |
| `avatarforcing_session_flush` | end the audio and compute the rest |
| `avatarforcing_session_frames_ready` | number of frames ready to pull |
| `avatarforcing_session_pull_frames` | copy ready frames to a caller buffer |
| `avatarforcing_session_close` | release a session |
| `avatarforcing_session_push_phrase` | live mode: add one phrase; returns its start frame |
| `avatarforcing_session_step` | live mode: compute the next block (speech or idle) |
| `avatarforcing_session_interrupt` | live mode: barge-in; idle from a frame on |
| `avatarforcing_session_stats_json` | live mode: frame clock, queue and counters |

The header documents each function.

## Options

Context options go to `avatarforcing_load`:

| Key | Type | Default | Meaning |
|---|---|---|---|
| `backend` | string | `"cpu"` | compute device: `cpu`, `cuda`, `cuda:N` or `gpu` |
| `threads` | int | 0 (all cores) | CPU threads |
| `decode_batch` | int | 1 on the CPU, 10 on a GPU | frames per decoder graph; 0: a whole block |
| `overlap` | bool | true on a GPU | decoder in a second thread while the sampler computes the next block |
| `ffmpeg` | string | `"ffmpeg"` | ffmpeg program for mp4 output, looked up on `PATH` |

A context option object can also hold request options. They become the
defaults of every request on the context.

Request options go to the generate functions and to
`avatarforcing_session_open`:

| Key | Type | Default | Meaning |
|---|---|---|---|
| `nfe` | int | 0 (GGUF value, 6) | sampler time points, 3 or more; nfe - 1 denoising steps per block |
| `cfg_scale` | number | -1 (GGUF value, 2.0) | audio classifier-free guidance scale |
| `seed` | int | 42 | seed of the built-in noise generator |
| `noise_file` | string | | raw float32 noise instead of the generator (`docs/cli.md`) |
| `audio_mode` | string | `"normalized"` | `normalized` (upstream offline path) or `raw` (upstream live app) |
| `max_seconds` | int | 30 | offline only: use the first N s of audio; 0: no limit |

The options map to the CLI options of the same names (`docs/cli.md`).
The frame rate is not an option: it comes from the GGUF (25 fps).

Session options go to `avatarforcing_session_open` only, beside the request
options:

| Key | Type | Default | Meaning |
|---|---|---|---|
| `live` | bool | false | open a live session (see "Live mode") |
| `live_queue_blocks` | int | 3 | live: blocks of frames that may wait for pull, 1 to 1000 |
| `live_max_ahead_seconds` | int | 30 | live: a phrase must end at most this far after the next block |

In a live session `audio_mode` defaults to `raw`, as in the upstream live
app, unless the request or the context options set it.

## Offline generation

```c
#include "avatarforcing.h"

char* err = NULL;
avatarforcing_ctx* ctx = avatarforcing_load("avatarforcing-f32.gguf",
                                            "{\"backend\": \"cpu\"}", &err);
if (!ctx) {
    fprintf(stderr, "load: %s\n", err);
    avatarforcing_free_string(err);
    return 1;
}

uint8_t* frames = NULL;
int n = 0, w = 0, h = 0, fps = 0;
if (avatarforcing_generate_frames(ctx, "portrait512.png", "speech.wav",
                                  "{\"nfe\": 6, \"seed\": 42}",
                                  &frames, &n, &w, &h, &fps) != 0) {
    fprintf(stderr, "generate: %s\n", avatarforcing_last_error(ctx));
} else {
    /* n frames of h * w * 3 bytes (RGB), at fps frames per second */
    avatarforcing_free_frames(frames);
}

if (avatarforcing_generate_mp4(ctx, "portrait512.png", "speech.wav",
                               "out.mp4", NULL) != 0)
    fprintf(stderr, "mp4: %s\n", avatarforcing_last_error(ctx));

avatarforcing_free(ctx);
```

The inputs are the same as for the CLI. The portrait is a face-centred
square image at the model size (512x512): PNG, JPEG or BMP. The library does
not detect faces. A WAV file can have any sample rate and channel count. PCM
is mono float in [-1, 1] at any sample rate. The library resamples audio
that is not at 16 kHz.

The video has `max(50, T)` frames for `T = ceil(seconds * 25)` audio frames:
the primer of 50 frames (2 s) is always in the video. A frame is
`height * width * 3` bytes of RGB, row-major. The frame buffer of a 30 s
clip at 512x512 is 590 MB. To receive the frames block by block, use a
session.

The mp4 functions start `ffmpeg` before they do any other work. When ffmpeg
cannot start, they fail at once with an error that names ffmpeg. They mux
the input audio as AAC with `-shortest`, as the CLI does. The PCM variant
writes the PCM to a temporary float WAV file next to the output file for
ffmpeg, and removes it afterwards.

## Sessions

A session generates one video while its audio arrives in parts:

1. Open the session with `avatarforcing_session_open`.
2. Add each phrase of audio with `avatarforcing_session_push_pcm`.
3. Pull the ready frames with `avatarforcing_session_pull_frames`.
4. End the audio with `avatarforcing_session_flush`, and pull the last
   frames.
5. Close the session with `avatarforcing_session_close`.

wav2vec2 runs on each phrase on its own, so a phrase must be a whole
utterance, for example one TTS sentence. In `normalized` mode the library
normalizes each phrase on its own. The session computes every block whose
audio is present. The primer (50 frames) needs 2 s of audio. After the
primer, each block of 10 frames needs 0.4 s more. The flush computes the
rest: the last block repeats the last audio frame and keeps only the frames
that the audio covers.

One phrase of a whole clip, then a flush, gives the same frames as
`avatarforcing_generate_frames_pcm` for a clip that is not longer than
`max_seconds`. The session does not apply `max_seconds`.

With `overlap`, the decoder works in its own thread. `push_pcm` then returns
when the sampler has computed the ready blocks, and their frames become
ready later. Another thread can call `frames_ready` and `pull_frames` while
`push_pcm` or `flush` runs. Without `overlap`, `push_pcm` returns when the
frames are ready.

## Live mode

A live session drives the face for as long as a call lasts, as the
upstream live app (`voicechat/avatar_worker.py` of AvatarVoice) does. Open
it with `{"live": true}`. Then:

1. Call `avatarforcing_session_step` once per block: 10 frames, 400 ms of
   video at 25 fps. Each call computes exactly one block.
2. Add each TTS phrase with `avatarforcing_session_push_phrase` when it
   arrives. It returns the frame where the phrase starts.
3. Pull the frames with `avatarforcing_session_pull_frames`.
4. On barge-in, call `avatarforcing_session_interrupt`.
5. Close the session with `avatarforcing_session_close`. A live session
   has no flush.

```c
avatarforcing_session* s = avatarforcing_session_open(ctx, "portrait512.png",
                                                      "{\"live\": true}");
/* when a phrase arrives (any thread) */
int64_t start = avatarforcing_session_push_phrase(s, pcm, n, 24000, -1);
/* play the phrase audio from frame `start` on */

/* every 400 ms, a little before the block is due */
int64_t first = avatarforcing_session_step(s);
if (first == AVATARFORCING_LIVE_QUEUE_FULL) { /* pull, then step again */ }
int got = avatarforcing_session_pull_frames(s, buf, 10);

/* the user starts to speak */
int64_t idle_from = avatarforcing_session_interrupt(s, playback_frame);
```

### Clock

The library has no timer and reads no clock. The caller owns the clock: it
calls `step` when the next block is due, minus the time a step takes plus a
margin. If the caller is ahead, it
sleeps. Frame indices are `int64_t` and count from 0. Frame `f` is due at
`f / 25` s after frame 0. The frames of a session come out in order, 10 per
step, with no gap. The library is deterministic: the same portrait, phrases,
call order, start frames and seed give the same frames.

`step` gives each frame of its block the features of the phrase that covers
the frame (lip-sync). A frame that no phrase covers gets the idle
features: wav2vec2 on the 5 s of pink-noise room tone in the GGUF
(`audio.idle_pcm`), the features that the live app uses. Zero features make
the mouth move, so the idle features are not zeros. Frame `f` takes idle
row `(f + 50) mod 125`. So the idle features run as one continuous stream,
and a frame has the same features in every block that reads it (a block
reads 2 frames of context and its 10 new frames).

### Hidden primer

The model starts with a primer of 50 frames. The live app runs it once on
idle features, writes the caches and does not decode or show its frames.
A live session does the same in `avatarforcing_session_open`: the primer
covers frames -50 to -1 and nobody sees them. The first `step` gives
frame 0. So `session_open` costs a primer (151 ms on an NVIDIA Thor at nfe
6, `docs/gpu.md`) and the first frames come after one more step.

### Phrases

`push_phrase` runs wav2vec2 on the whole phrase, as the live app does. The
live app measured that features of rolling windows match the whole-phrase
features badly, so a phrase must be a whole utterance, for example one TTS
sentence. The phrase then starts:

- with `at_frame < 0`: right after the last phrase, or at the next block
  when no phrase is pending (the live app does this);
- with `at_frame >= 0`: at that frame, or later when a step has computed
  that frame already, or when an earlier phrase covers it.

Phrases never overlap. The return value is the start frame: play the
phrase audio from that frame on, on the same clock as the frames. A phrase
that would end more than `live_max_ahead_seconds` after the next block
fails with -1. It does not change the session.

### Interrupt

`avatarforcing_session_interrupt(s, at_frame)` drops the phrase audio from
`at_frame` on: phrases that start there or later go, and the phrase that
runs over `at_frame` is cut there. From that frame on, `step` uses idle
features. Frames that a step has computed already do not change, so the
cut is at `max(at_frame, next block)`, and the function returns that frame.
Up to `live_queue_blocks` blocks can be computed ahead of playback. Those
frames still show the speech. To cut them too, drop them in the caller.
The live app lets them play out. There is no
cross-fade, as in the live app: within the next block the model moves the
face from speech to idle.

`interrupt` does not wait for a running `step` or `push_phrase`. A
`push_phrase` that has started when the interrupt comes, also one that
waits for the context lock, returns `AVATARFORCING_LIVE_DROPPED` (-3) and
places nothing. This is what the live app does with phrases in its wav2vec2
queue.

### Backpressure

At most `live_queue_blocks` blocks of frames wait for pull. That counts
the frames that are ready and the frames still in the decoder thread. When
the next block does not fit, `step` computes nothing and returns
`AVATARFORCING_LIVE_QUEUE_FULL` (-2). The frame clock of the session does
not move and nothing is dropped. Pull frames, then call `step` again. A
caller that stops pulling thus stops the session after
`live_queue_blocks` blocks, with bounded memory (3 blocks at 512x512 are
23 MB).

`avatarforcing_session_stats_json` gives the next frame, the end of the
scheduled speech, the queued frames, the number of steps and of steps with
speech, the queue-full count, the phrases, the dropped phrases, the
interrupts, and the time of the last and the slowest step.

### Not in live mode yet

- **Reanchor.** The live app re-runs the primer every 112 blocks (about
  45 s) when it is idle, or after 162 blocks at the latest, and
  cross-fades over 25 frames. Without it the quality drifts after about a
  minute. The schedule keeps the features of the last 50 frames for it.
- **Skip ahead.** When the live app falls behind, it moves its block
  clock forward. A live session computes every block in order; a caller
  that falls behind gets late frames.
- Different user video or audio: the user inputs are zeros, as in the live
  app with `u_cfg_scale = 0`.

## Threads

- Contexts are independent. Different contexts can run at the same time in
  different threads. Each context holds its own copy of the model.
- The calls on one context are safe from any thread. A lock in the context
  serializes them, so they run one at a time.
- A context has at most one open session. While a session is open, the
  generate functions on its context fail.
- `avatarforcing_cancel` does not wait for the lock. Call it from any thread
  to stop the running generation before its next block. The generation then
  fails with the error `cancelled`.
- `avatarforcing_session_frames_ready` and
  `avatarforcing_session_pull_frames` do not wait for the lock either. Do not
  call `pull_frames` from two threads at the same time.
- In live mode, `step` and `push_phrase` take the lock: they run one at a
  time, from any threads. `push_phrase` runs wav2vec2 on the model device,
  so a `step` waits for a `push_phrase` that runs, and the other way round.
  `interrupt` and `stats_json` do not wait for the lock. A Go program can
  thus run `step` on a ticker goroutine, `push_phrase` on the TTS goroutine,
  `interrupt` on the VAD goroutine and `pull_frames` on the encoder
  goroutine.
- `avatarforcing_cancel` makes the next `step` of a live session fail, and
  the session stays failed. Use `interrupt` for barge-in.
- `avatarforcing_last_error` returns a pointer into the context. The next
  failing call on the context overwrites it, so copy the message before you
  make that call.
- Do not call `avatarforcing_session_close` or `avatarforcing_free` while
  another thread uses the session or the context.

## Logs

The library writes its log lines to stderr with the prefix
`[avatarforcing]`, as the CLI does. When a step fails, the error message
of the API holds the last log line of that step.

## Test

`tests/test_capi.c` is a C program that uses the API. It needs
`AVATARFORCING_TEST_GGUF` and is skipped without it. It writes a synthetic
portrait and a 2.5 s WAV file to the current directory, then:

- checks the errors for a bad GGUF, a missing GGUF, bad options, a missing
  image, an invalid `nfe` and a missing ffmpeg;
- generates 63 frames from PCM and from the WAV file, and checks that the
  frames are the same;
- runs the same audio through a session and checks that the frames are the
  same as the offline frames.

`tests/test_capi_live.c` tests live mode. It needs `AVATARFORCING_TEST_GGUF`
too. It runs one call of 150 frames twice:

- the queue bound: 3 steps without a pull, then
  `AVATARFORCING_LIVE_QUEUE_FULL`, with the clock unchanged;
- phrase placement: a start frame, an append, a start inside an earlier
  phrase, and a phrase too far ahead;
- an interrupt, and an interrupt while a push waits for the lock (the
  push is dropped);
- that idle frames are stable;
- that the two runs give the same frames;
- that an offline clip gives the same frames before and after the live
  sessions.

With `AVATARFORCING_TEST_PORTRAIT` (a real face) it also checks that the
mouth moves more in the phrases than when idle.
`AVATARFORCING_TEST_WAV` gives it real speech (16-bit or float WAV, at
least 3.6 s). `AVATARFORCING_TEST_OVERLAP=1` runs it with the decoder
thread. `tests/test_live_schedule.cpp` tests the feature schedule of
live mode and needs no model.

```sh
AVATARFORCING_TEST_GGUF=avatarforcing-f32.gguf \
ctest --test-dir build -R test_capi --output-on-failure
AVATARFORCING_TEST_GGUF=avatarforcing-f32.gguf \
AVATARFORCING_TEST_PORTRAIT=portrait512.png AVATARFORCING_TEST_WAV=speech.wav \
ctest --test-dir build -R test_capi_live --output-on-failure
```
