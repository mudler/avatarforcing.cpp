#pragma once
#include "backend.hpp"
#include "model_loader.hpp"

#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace af {

class FlowModel;

// wav2vec2 configuration, read from the `avatarforcing.w2v2.*` and
// `avatarforcing.audio.*` KV (docs/conversion.md).
struct W2V2Config {
    uint32_t hidden_size = 0, n_layers = 0, n_heads = 0, ff_dim = 0;
    uint32_t num_conv_layers = 0;
    std::vector<int32_t> conv_dims, conv_kernels, conv_strides;
    bool conv_bias = false;
    std::string feat_extract_norm, feat_extract_activation, hidden_act;
    bool do_stable_layer_norm = false;
    float layer_norm_eps = 0.f, group_norm_eps = 0.f;
    uint32_t num_conv_pos_embeddings = 0, num_conv_pos_embedding_groups = 0;
    uint32_t pos_conv_weight_norm_dim = 0;
    // audio front end
    uint32_t sample_rate = 0, samples_per_frame = 0;
    uint32_t n_hidden_taps = 0;
    std::string interp;
    uint32_t idle_seconds = 0;
};

// How the phrase PCM is fed to wav2vec2.
//   Normalized: zero mean, unit variance over the phrase (the
//               Wav2Vec2FeatureExtractor of the upstream offline inference),
//               n = ceil(len / samples_per_frame) frames, replicate-padded.
//   Raw:        the PCM as is (the live voice-chat worker),
//               n = max(1, ceil(len / samples_per_frame)) frames, zero-padded.
enum class AudioInputMode { Normalized, Raw };

// Debug outputs, keyed like the reference dump (conv.N, interp, hidden.N, ...).
using W2V2Taps = std::map<std::string, std::vector<float>>;

// The audio feature path: phrase PCM (16 kHz mono float) to the 25 fps
// 512-d audio features the flow transformer consumes.
//
//   PCM -> conv feature encoder (50 Hz) -> linear interpolation to n frames
//   (align_corners) -> feature projection -> positional conv -> 12 post-LN
//   encoder layers -> hidden states 1..12 stacked (n x 9216) ->
//   audio projection (Linear + LayerNorm + SiLU, owned by FlowModel) -> n x 512
//
// The encoder runs on the interpolated 25 fps frames, as the reference does.
class AudioEncoder {
public:
    // `flow` provides the audio projection; it must be loaded first and outlive
    // the encoder.
    AudioEncoder(Backend& backend, FlowModel& flow) : be_(backend), flow_(flow) {}

    // Loads the `w2v2.` tensors and the idle PCM, then computes the idle
    // (pink-noise room tone) features.
    bool load(const std::string& gguf_path);

    const W2V2Config& config() const { return cfg_; }

    // Number of 25 fps frames the reference produces for `n_samples` of PCM.
    int frames_for(size_t n_samples) const;

    // Phrase features: out = [n_frames, 512] row-major. Any phrase length; one
    // graph per phrase. `taps` (optional) receives the intermediate stages.
    bool features(const float* pcm, size_t n_samples, AudioInputMode mode,
                  std::vector<float>& out, int& n_frames, W2V2Taps* taps = nullptr);

    // Idle features (raw mode on the stored pink-noise PCM): [idle_frames(), 512].
    const std::vector<float>& idle_features() const { return idle_feats_; }
    int idle_frames() const { return idle_frames_; }
    const std::vector<float>& idle_pcm() const { return idle_pcm_; }

    // Lower-level stages, exposed for the parity test.
    // Model input as the reference builds it (normalization and padding).
    void model_input(const float* pcm, size_t n_samples, AudioInputMode mode,
                     std::vector<float>& x, int& n_frames) const;
    // Model input -> 12 stacked hidden states [n_frames, 12 * hidden].
    bool hidden_states(const std::vector<float>& x, int n_frames, std::vector<float>& stacked,
                       W2V2Taps* taps = nullptr);

private:
    Backend& be_;
    FlowModel& flow_;
    ModelLoader loader_;
    W2V2Config cfg_;
    std::vector<float> pos_w_;      // weight-normed positional conv kernel [H][icg][K]
    std::vector<float> idle_pcm_, idle_feats_;
    int idle_frames_ = 0;
};

} // namespace af
