# Parity (M2: flow transformer)

`tests/test_flow_parity.cpp` runs the C++ flow transformer on the inputs of
the reference dumps (`docs/reference.md`) and compares every module output
with the PyTorch values.

## Metric and gate

For each module: `max_abs = max |got - ref|` and
`max_rel = max_abs / max |ref|`. A module passes when `max_rel <= 2e-5` and
every value is finite. The reference itself moves by up to 4e-6 relative
between thread counts (see `docs/reference.md`).

## What the test covers

All four forwards of `flow_forward_nfe6.npz`:

| Tag | Shape | Cache |
|---|---|---|
| `primer_step0` | N = 50 | none |
| `primer_kvupdate` | N = 50, t = 0 | write (from zero caches) |
| `steady0_step0` | N = 12 (2 context + 10 new) | read |
| `steady0_final` | N = 12, t = 0 | read and folded write |

For each tag the test checks:

- the inputs the engine builds itself: the RoPE table (`in.freqs_cis`) and
  the per-token adaLN rows (`in.adaln6`, `in.final_adaln2`; the steady
  context tokens use the t = 0 rows);
- the condition embedder stages, where dumped (`c_embedder.*`);
- `x_embed`, the block 0 internals (`attn_in`, `attn_out`, `cross_in`,
  `cross_out`, `mlp_in`, `mlp_out`), every block output and `out.v`;
- the caches after a write (`kv_after.*`).

The forwards above start from the dumped caches (`kv_before.*`). A second
pass (`chain.*`) starts from zero caches and carries the engine's own caches
through `primer_kvupdate`, `steady0_step0` and `steady0_final`.

The test also checks the timestep embedding and adaLN tables
(`adaln_nfe6.npz`) and the audio projection (`wav2vec_raw.npz`: `stacked` to
`proj.linear`, `proj.norm`, `features`).

## Results

CPU backend, F32, 8 threads, `GGML_NATIVE=ON`. 113 modules, all pass. The
worst module per group:

| Group | Worst module | max_abs | max_rel |
|---|---|---|---|
| adaLN tables | `adaln.t_embed` | 5.7e-06 | 4.0e-07 |
| RoPE table | `in.freqs_cis` | 6.0e-08 | 6.0e-08 |
| per-token adaLN | `primer_kvupdate.in.final_adaln2` | 2.7e-06 | 2.5e-06 |
| condition embedder | `steady0_step0.c_embedder.out` | 6.0e-08 | 8.7e-07 |
| primer step 0 | `primer_step0.block.0.attn_out` | 5.8e-05 | 1.6e-06 |
| primer KV update | `primer_kvupdate.block.7` | 8.9e-04 | 9.2e-06 |
| primer caches | `primer_kvupdate.kv_after.self_k` | 7.3e-05 | 6.4e-06 |
| steady step 0 | `steady0_step0.block.0.mlp_in` | 7.6e-06 | 1.4e-06 |
| steady final | `steady0_final.out.v` | 4.9e-06 | 3.0e-06 |
| steady caches | `steady0_final.kv_after.self_v` | 1.9e-05 | 2.5e-06 |
| chain | `chain.primer_kvupdate.kv_after.self_k` | 7.3e-05 | 6.4e-06 |
| audio projection | `audio.proj.norm` | 2.9e-05 | 2.6e-06 |

The test prints the full table on every run.

## Precision notes

- RoPE runs as element-wise ggml ops on a host table built as the reference
  builds it (`cos/sin(float32(pos) * inv_freq)`). `ggml_rope_ext` derives the
  angles by repeated float32 multiplication. A numpy model of that
  recurrence differs from the reference table by up to 7.6e-6 in cos/sin at
  position 108, 1.2e-4 at position 1000 and 7.3e-4 at position 5000, and the
  live stream reaches positions above 1000.
- GELU (tanh) is built from `ggml_tanh` and element-wise ops. `ggml_gelu`
  uses an fp16 lookup table on the CPU, which is too coarse for this gate.
- The masks are built from the KV (`flow.self_mask.*`,
  `flow.cross_mask.window`, cache lengths), never from the checkpoint
  buffers.

# Parity (M3: sampler loop)

`tests/test_sampler_parity.cpp` runs `af::Sampler` on the inputs of
`sampler_nfe6.npz` and `sampler_nfe10.npz` (`docs/reference.md`): the primer
and 6 steady blocks, from the injected noise, with the engine's own x and
caches carried from step to step. Nothing is re-injected from the reference
between steps or blocks.

## Metric and gate

The same metric as above. A tensor passes when `max_rel <= 1e-3` and every
value is finite. The test checks:

- the step list and the sigma index of each step (exact);
- the start position of each steady block (exact);
- x after each step, the raw transformer output of each step (both CFG
  branches), the primer's KV-update output, `primer.x_final` and the latents
  of each steady block;
- the four caches after the primer and after each steady block.

The dumps do not cover the reanchor. The test checks it against the engine's
own run instead: after the 6 steady blocks, a reanchor on the primer inputs
must reproduce the primer x and caches, the start position 48 and the
latents of steady block 0 bit for bit.

## Results

CPU backend, F32, 8 threads, `GGML_NATIVE=ON`. 278 tensors, all pass. Worst
tensor per block, `max_abs / max_rel`:

| Block | nfe 6: x and latents | nfe 6: caches | nfe 10: x and latents | nfe 10: caches |
|---|---|---|---|---|
| primer | 2.6e-06 / 1.1e-06 | 7.5e-05 / 7.9e-06 | 2.1e-06 / 7.7e-07 | 7.2e-05 / 6.3e-06 |
| steady 0 | 3.8e-06 / 2.8e-06 | 7.3e-05 / 7.8e-06 | 2.7e-06 / 1.4e-06 | 7.2e-05 / 6.3e-06 |
| steady 1 | 5.5e-06 / 4.7e-06 | 7.3e-05 / 8.4e-06 | 4.0e-06 / 3.6e-06 | 7.1e-05 / 6.2e-06 |
| steady 2 | 7.8e-06 / 6.9e-06 | 7.3e-05 / 1.0e-05 | 4.3e-06 / 4.0e-06 | 7.1e-05 / 6.2e-06 |
| steady 3 | 8.7e-06 / 4.9e-06 | 7.3e-05 / 9.2e-06 | 4.2e-06 / 2.5e-06 | 5.1e-05 / 5.4e-06 |
| steady 4 | 7.6e-06 / 4.5e-06 | 7.3e-05 / 9.2e-06 | 3.3e-06 / 1.7e-06 | 5.1e-05 / 5.4e-06 |
| steady 5 | 5.8e-06 / 3.2e-06 | 7.3e-05 / 9.2e-06 | 9.9e-06 / 5.8e-06 | 7.5e-05 / 6.5e-06 |

The error does not grow over the blocks: it stays between 1e-6 and 1e-5
relative, two orders of magnitude under the gate. The raw transformer
outputs stay under 1.5e-5 relative.

Time on the same machine (median of 5 steady blocks, each with the
condition embedder, all forwards and the folded KV update):

| nfe | Forwards per steady block | Steady block | Per frame | Primer |
|---|---|---|---|---|
| 6 | 5 | 204 ms | 20 ms | 0.62 s |
| 10 | 9 | 334 ms | 33 ms | 0.99 s |

## Sampler notes

- Step list: `linspace(num_train_timesteps, 0, nfe - 1)` in float64, stored
  as float32. Sigma: argmin of `|sched.timesteps - t|` in float64 over the
  float32 tables, forced to 0 at t = 0 (t = 750 gives index 250,
  sigma 0.74975).
- CFG runs in float32 (`v_u + s * (v_a - v_u)`, uncond branch first); the
  closed-form update runs in float64, as `solve_closed` does.
- The t = 0 step does not change x. The primer skips it and runs the
  separate KV-update forward instead; a steady block runs it only for the
  folded KV update.
