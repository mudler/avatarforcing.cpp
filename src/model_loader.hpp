#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct ggml_backend;
typedef struct ggml_backend* ggml_backend_t;
struct ggml_backend_buffer;
typedef struct ggml_backend_buffer* ggml_backend_buffer_t;

namespace af {

// Flow transformer and stream configuration. Every value comes from the
// `avatarforcing.*` KV of the GGUF (see docs/conversion.md); the loader fails
// when a required key is missing.
struct FlowConfig {
    // timing and streaming
    uint32_t clip_frames = 0;       // primer length
    uint32_t block_frames = 0;      // new frames per steady block
    uint32_t context_frames = 0;    // frames carried from the previous block
    uint32_t self_cache_len = 0;    // self-attention KV cache length
    uint32_t cross_cache_len = 0;   // cross-attention KV cache length
    uint32_t kv_shift = 0;          // self cache left shift per steady block
    uint32_t first_steady_start_pos = 0;
    // flow transformer
    uint32_t dim_w = 0, dim_h = 0, dim_a = 0;
    uint32_t depth = 0, n_heads = 0, head_dim = 0, mlp_hidden = 0;
    std::string mlp_act;
    float ln_eps = 0.f;
    uint32_t freq_embed_dim = 0;
    float rope_theta = 0.f;
    uint32_t rope_dim = 0;
    std::string rope_layout;
    uint32_t self_mask_block = 0, self_mask_lookahead = 0, cross_mask_window = 0;
    std::vector<std::string> adaln_order, final_adaln_order;
    // condition embedder
    uint32_t cond_n_heads = 0;
    float cond_ln_eps = 0.f;
    // audio projection
    uint32_t audio_proj_in = 0, audio_proj_out = 0;
    float audio_proj_ln_eps = 0.f;
    // sampler (used by the sampler loop, M3)
    float cfg_audio_scale = 0.f;
};

// Loads the KV and the tensors whose names start with one of `prefixes` into
// a single backend buffer. Only those tensors are read from the file.
class ModelLoader {
public:
    ModelLoader() = default;
    ~ModelLoader();
    ModelLoader(const ModelLoader&) = delete;
    ModelLoader& operator=(const ModelLoader&) = delete;

    bool load(const std::string& path, ggml_backend_t backend,
              const std::vector<std::string>& prefixes);

    const FlowConfig& config() const { return cfg_; }
    ggml_tensor* tensor(const std::string& name) const;   // nullptr if absent
    ggml_tensor* require(const std::string& name);         // logs and records a miss
    bool missing() const { return missing_; }
    size_t n_tensors() const { return tensors_.size(); }
    size_t n_bytes() const { return n_bytes_; }

private:
    FlowConfig cfg_;
    ggml_context* ctx_ = nullptr;
    ggml_backend_buffer_t buf_ = nullptr;
    std::unordered_map<std::string, ggml_tensor*> tensors_;
    size_t n_bytes_ = 0;
    bool missing_ = false;
};

} // namespace af
