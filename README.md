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
| M3+ | Sampler loop, motion decoder, wav2vec2, CLI | not started |

The C++ library holds the flow transformer forward pass: the primer shape
(50 tokens, no cache read) and the steady shape (12 tokens, cache read), with
the KV cache update, for the two CFG branches, in F32 on the ggml CPU backend.
`docs/parity.md` gives the parity results against the PyTorch dumps.

## Build and test

ggml is a git submodule (`third_party/ggml`, pinned at v0.15.2).

```sh
git submodule update --init
cmake -B build -DGGML_NATIVE=ON
cmake --build build -j8
```

The parity test needs the model GGUF and a baseline GGUF made from the
reference dumps (see below). Without them it is skipped.

```sh
python scripts/dumps_to_baseline.py --dumps dumps-gold --nfe 6 \
    --output flow-baseline-nfe6.gguf
AVATARFORCING_TEST_GGUF=avatarforcing-f32.gguf \
AVATARFORCING_TEST_BASELINE=flow-baseline-nfe6.gguf \
AVATARFORCING_TEST_THREADS=8 \
ctest --test-dir build --output-on-failure
```

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

The model weights and the upstream code are licensed under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
(non-commercial use only). GGUF files converted from these weights carry the
same licence.

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

The licence of the code in this repository is not decided yet. See `LICENSE`.
