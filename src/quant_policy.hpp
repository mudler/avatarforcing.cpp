#pragma once
#include <string>

namespace af {

// Per-tensor type policy of the quantized GGUF files (docs/quantization.md).
//
// A tensor can hold a type other than F32 only when quant_allowed() is true:
// the 2-D weight matrices that the graphs use only as the first operand of
// ggml_mul_mat (the linear layers of the flow transformer, the condition
// embedder, the audio projection and the wav2vec2 encoder). Everything else
// stays F32: biases, norms, the motion autoencoder (its convolutions and the
// fused CUDA decoder kernels take F32 weights only), the wav2vec2 feature
// encoder and positional convolution, and the tables the host reads.
//
// The quantizer (avatarforcing-quantize) and the loader use the same rule.
bool quant_allowed(const std::string& name);

// The input and output projections of the flow transformer and its
// timestep and adaLN layers: x_embedder.proj, final_layer.linear,
// t_embedder.mlp.{0,2}, adaLN_modulation.1, final_layer.adaLN_modulation.1.
// A block type (Q8_0, K types) on these tensors makes the sampler drift
// (docs/quantization.md), so the quantizer stores them as F16 in files of
// a block type.
bool quant_sensitive(const std::string& name);

// One-line description of the rule, stored as avatarforcing.quant.policy.
const char* quant_policy_text();

} // namespace af
