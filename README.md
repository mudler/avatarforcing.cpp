# avatarforcing.cpp

A C++/ggml port of the AvatarForcing talking-head avatar model: a portrait
image and speech audio in, lip-synced 512x512 video frames out, with no Python
at inference time.

The model has three parts:

- **Motion flow transformer.** A DiT-style flow-matching transformer (8 blocks,
  hidden size 1024) that generates 512-d motion latents in blocks of 10 frames,
  with a rolling KV cache and classifier-free guidance on the audio.
- **Motion autoencoder.** A LIA/StyleGAN2-style encoder (portrait to identity
  code and feature maps) and decoder (motion latent to pixels, with flow-field
  warping of the portrait features).
- **Audio features.** `facebook/wav2vec2-base-960h`, all 12 hidden states,
  resampled to 25 fps and projected to 512-d.

## Status

| Milestone | Content | State |
|---|---|---|
| M0 | Reference dumps from the PyTorch code, CPU float32, injected noise | done |
| M1 | Metadata-driven GGUF converter and round-trip check | done |
| M2 | C++/ggml flow transformer forward (condition embedder, adaLN tables, 8 blocks, KV caches), parity-gated | done |
| M3 | Sampler loop: primer, steady blocks, folded KV update, reanchor, CFG, closed-form solver, parity-gated | done |
| M4 | Motion autoencoder: encoder, decoder with grid_sample warping, uint8 conversion, parity-gated | done |
| M5 | wav2vec2 audio features (raw and normalized input), idle features, audio projection, parity-gated | done |
| M6 | `avatarforcing-cli`: portrait + speech WAV to mp4, end-to-end gate against the upstream streaming path | done |
| M7 | CUDA backend: the whole pipeline on the GPU in float32, parity-gated; faster than real time on an NVIDIA Thor (one 400 ms block every 202 ms at nfe 6, 241 ms at nfe 10) | done |
| M8 | C API: `libavatarforcing` with `include/avatarforcing.h`, offline generation to RGB frames or mp4, streaming sessions; the CLI and the API share one pipeline | done |

The C++ library holds all three parts in F32 on the ggml CPU or CUDA backend
(`docs/gpu.md`): the
flow transformer and its sampler loop (primer and steady blocks, rolling KV
cache, two-branch CFG), the motion autoencoder, and the wav2vec2 audio
features. The parity results against the PyTorch dumps are in
`docs/parity.md` (flow transformer and sampler), `docs/parity-motion.md` and
`docs/parity-wav2vec2.md`.

## Usage

```sh
avatarforcing-cli --model avatarforcing-f32.gguf --portrait portrait512.png \
    --audio speech.wav -o out.mp4
```

The portrait must be a face-centred 512x512 image (`scripts/prepare_portrait.py`
makes one); the tool does not detect faces. The WAV can have any sample rate
and channel count. `ffmpeg` must be on `PATH` for the mp4 output;
`--frames-dir DIR` writes PNG frames instead or as well. `docs/cli.md`
describes all options, the pipeline and how it differs from the upstream
code.

## C API

`libavatarforcing.so` exports a flat C API (`include/avatarforcing.h`) for
programs that load the library at run time, for example Go with purego or
Python with ctypes:

```c
char* err = NULL;
avatarforcing_ctx* ctx = avatarforcing_load("avatarforcing-f32.gguf", "{\"backend\": \"cpu\"}", &err);
uint8_t* frames = NULL;
int n = 0, w = 0, h = 0, fps = 0;
avatarforcing_generate_frames(ctx, "portrait512.png", "speech.wav", "{\"nfe\": 6}",
                              &frames, &n, &w, &h, &fps);
avatarforcing_free_frames(frames);
avatarforcing_free(ctx);
```

It generates RGB frames or an mp4 file from a portrait and a WAV file or
float PCM, and has sessions that take the audio phrase by phrase. The CLI
and the API run the same pipeline (`src/pipeline.cpp`), so they give the
same frames. `docs/capi.md` describes the functions, the options, the
sessions and the thread rules.

## Build and test

ggml is a git submodule (`third_party/ggml`, pinned at v0.15.2).

```sh
git submodule update --init
cmake -B build -DGGML_NATIVE=ON
cmake --build build -j8
```

The tool is `build/examples/cli/avatarforcing-cli` and the library
`build/libavatarforcing.so` (`-DAVATARFORCING_BUILD_SHARED=OFF` skips it).
CMake builds ggml as static libraries inside the library and the tool;
`-DBUILD_SHARED_LIBS=ON` links them as shared libraries instead.
stb_image, stb_image_write and dr_wav are vendored under `third_party/`.

For the CUDA backend add `-DGGML_CUDA=ON` (and `-DCMAKE_CUDA_ARCHITECTURES=...`
for your GPU) and run the tool with `--backend cuda`; see `docs/gpu.md`.
CMake applies the ggml patches in `patches/` to a copy of the submodule in
the build tree (a `grid_sample` operator and strict float32 settings for
CUDA), so `git` must be on `PATH` at configure time.

The parity tests need the model GGUF and baseline GGUFs made from the
reference dumps (see below). A test whose files are not set is skipped.

```sh
python scripts/dumps_to_baseline.py --dumps dumps-gold --nfe 6 \
    --sampler-nfe 6 10 --output flow-baseline.gguf
python scripts/dumps_to_baseline.py --dumps dumps-gold --nfe 6 \
    --component motion --output motion-baseline.gguf
python scripts/dumps_to_baseline.py --dumps dumps-gold --w2v2 \
    --output w2v2-baseline.gguf
AVATARFORCING_TEST_GGUF=avatarforcing-f32.gguf \
AVATARFORCING_TEST_BASELINE=flow-baseline.gguf \
AVATARFORCING_TEST_MOTION_BASELINE=motion-baseline.gguf \
AVATARFORCING_TEST_W2V2_BASELINE=w2v2-baseline.gguf \
AVATARFORCING_TEST_THREADS=8 \
ctest --test-dir build --output-on-failure
```

`AVATARFORCING_TEST_BACKEND=cuda` runs the tests on the GPU.

`test_flow_parity` and `test_sampler_parity` read the flow baseline,
`test_motion_parity` the motion baseline and `test_wav2vec2_parity` the
wav2vec2 baseline. `test_capi` needs only the model GGUF (`docs/capi.md`).

## Reference tooling

The tooling needs the upstream source, the checkpoints and a small Python
environment. None of this is needed at inference time.

1. Get the upstream code. The scripts import the model classes directly from a
   checkout of [AvatarVoice](https://github.com/johnrobinsn/AvatarVoice) with
   its `avatarforcing` submodule. The scripts were written against commit
   `566f8e1`.
2. Get the weights (see `docs/reference.md` for the sources):
   - `flow_transformer.pth` and `motion_autoencoder.pth` from the AvatarForcing
     authors' Google Drive;
   - `facebook/wav2vec2-base-960h` from Hugging Face.
3. Install the dependencies in `scripts/requirements.txt` (CPU PyTorch is
   enough).

Then:

```sh
# GGUF (all three components in one self-contained file)
python scripts/convert_avatarforcing_to_gguf.py \
    --flow-ckpt flow_transformer.pth --mae-ckpt motion_autoencoder.pth \
    --wav2vec wav2vec2-base-960h --output avatarforcing-f32.gguf

# round-trip check: GGUF tensors and KV against the PyTorch state
python tests/python/check_convert.py --gguf avatarforcing-f32.gguf \
    --avatarvoice AvatarVoice --flow-ckpt flow_transformer.pth \
    --mae-ckpt motion_autoencoder.pth --wav2vec wav2vec2-base-960h

# reference dumps (gold .npz files for the C++ parity tests)
python scripts/dump_reference.py --avatarvoice AvatarVoice \
    --flow-ckpt flow_transformer.pth --mae-ckpt motion_autoencoder.pth \
    --wav2vec wav2vec2-base-960h --portrait portrait512.png \
    --audio speech16k.wav --out dumps/
```

`docs/reference.md` describes every dump file and its contents.
`docs/conversion.md` describes the GGUF schema.

## Licence and attribution

The code in this repository is licensed under the MIT licence. See `LICENSE`.

The AvatarForcing model weights and the upstream AvatarForcing and AvatarVoice
code are licensed under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
(non-commercial use only). GGUF files converted with this repository carry the
same licence, so you can use them only for non-commercial purposes.

- **AvatarForcing**: Taekyung Ki, Sangwon Jang, Jaehyeong Jo, Jaehong Yoon,
  Sung Ju Hwang. *Avatar Forcing: Real-Time Interactive Head Avatar Generation
  for Natural Conversation.* CVPR 2026.
  [Project page](https://taekyungki.github.io/AvatarForcing/),
  [code](https://github.com/TaekyungKi/AvatarForcing),
  [arXiv:2601.00664](https://arxiv.org/abs/2601.00664).
- **AvatarVoice** by John Robinson: the streaming, real-time variant of the
  AvatarForcing inference code that this port follows (block loop, KV-cache
  handling, idle audio features).
  [code](https://github.com/johnrobinsn/AvatarVoice).
- **wav2vec 2.0** (`facebook/wav2vec2-base-960h`), Apache-2.0.

Vendored third-party code keeps its own licence:

- `third_party/ggml`: MIT.
- `third_party/stb`: MIT or public domain (Unlicense), at your choice.
- `third_party/dr_libs`: public domain (Unlicense) or MIT-0, at your choice.
