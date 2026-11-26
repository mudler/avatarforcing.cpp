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

### Not implemented yet

The session is the base for real-time use. These parts of the upstream live
app (`docs/cli.md`) are not in it yet:

- Idle frames. When no audio arrives, the live app feeds pink-noise idle
  features (`audio.idle_pcm` in the GGUF), so the avatar keeps moving. A
  session now waits for audio.
- Reanchor. The live app re-runs the primer every 112 blocks once idle and
  crossfades over 25 frames. A session has no length limit, but it does not
  reset its cache.
- A hidden primer. The live app does not show the 50 primer frames. A
  session outputs them, as the offline path does.

These change the frames that a session outputs, so they will come with a
new ABI version or new functions.

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

```sh
AVATARFORCING_TEST_GGUF=avatarforcing-f32.gguf \
ctest --test-dir build -R test_capi --output-on-failure
```
