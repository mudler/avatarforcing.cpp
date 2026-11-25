#pragma once
#include "backend.hpp"
#include "model_loader.hpp"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace af {

// Rolling KV caches of the flow transformer, kept on the host between
// forwards. Layout [layer, branch, head, pos, head_dim] (row-major), the same
// layout as the reference dumps. The self cache holds post-RoPE keys.
struct KvCache {
    int depth = 0, branches = 0, heads = 0, head_dim = 0;
    int self_len = 0, cross_len = 0;
    std::vector<float> self_k, self_v, cross_k, cross_v;

    void init(const FlowConfig& cfg, int n_branches);   // zero-filled
    size_t self_layer_size() const { return (size_t)branches * heads * self_len * head_dim; }
    size_t cross_layer_size() const { return (size_t)branches * heads * cross_len * head_dim; }
};

// Timestep embeddings and adaLN rows, one row per timestep.
struct AdaLNTable {
    int dim_h = 0, freq_dim = 0;
    std::vector<float> timesteps;   // [n]
    std::vector<float> t_freq;      // [n, freq_dim]
    std::vector<float> t_embed;     // [n, dim_h]
    std::vector<float> adaln6;      // [n, 6 * dim_h]: shift/scale/gate msa, shift/scale/gate mlp
    std::vector<float> final2;      // [n, 2 * dim_h]: shift, scale
    int size() const { return (int)timesteps.size(); }
    int row(float t) const;         // index of timestep t, -1 if absent

    // Per-token adaLN inputs for a forward: token j uses row rows[j].
    void gather(const std::vector<int>& rows, std::vector<float>& adaln6_tok,
                std::vector<float>& final2_tok) const;
};

// Debug outputs, keyed like the reference dump (x_embed, block.N, ...).
using Taps = std::map<std::string, std::vector<float>>;

struct ForwardArgs {
    int n_tokens = 0;              // N: clip_frames (primer) or block + context (steady)
    int n_branches = 2;            // CFG batch, [uncond, audio]
    const float* x = nullptr;      // [B, N, dim_w]
    const float* c = nullptr;      // [B, N, dim_h] condition (c_embedder output)
    const float* wr = nullptr;     // [dim_w] identity code, shared by all tokens and branches
    const float* adaln6 = nullptr; // [N, 6 * dim_h] per token, shared by the branches
    const float* final2 = nullptr; // [N, 2 * dim_h] per token
    int start_pos = 0;             // RoPE position of token 0
    // nullptr: no cache (primer denoising steps). Otherwise the cache is read
    // when N == block + context, and written when update_kv is set.
    KvCache* kv = nullptr;
    bool update_kv = false;
};

class FlowModel {
public:
    explicit FlowModel(Backend& backend) : be_(backend) {}
    ~FlowModel();

    bool load(const std::string& gguf_path);
    const FlowConfig& config() const { return loader_.config(); }
    const ModelLoader& loader() const { return loader_; }

    // Timestep embedding and adaLN rows for the given timesteps.
    bool timestep_tables(const std::vector<float>& timesteps, AdaLNTable& out);

    // Condition embedder: c = c_embedder(wa_user, motion_user, wa).
    // Inputs [B, N, dim_a]; output [B, N, dim_h].
    bool condition(int n_branches, int n_tokens, const float* wa_user, const float* motion_user,
                   const float* wa, std::vector<float>& c, Taps* taps = nullptr);

    // One flow transformer forward. Output v [B, N, dim_w]. Updates args.kv
    // when args.update_kv is set.
    bool forward(const ForwardArgs& args, std::vector<float>& v, Taps* taps = nullptr);

    // Audio projection: SiLU(LN(Linear(stacked))). Input [T, proj_in], output [T, proj_out].
    bool audio_projection(int n_frames, const float* stacked, std::vector<float>& out,
                          Taps* taps = nullptr);

    // RoPE table as the reference builds it: [n, head_dim], cos || sin of
    // float32(pos) * inv_freq for pos in [start_pos, start_pos + n).
    void rope_table(int start_pos, int n, std::vector<float>& cos_sin) const;

private:
    struct Block {
        ggml_tensor *qkv_w, *qkv_b, *proj_w, *proj_b;
        ggml_tensor *cq_w, *cq_b, *ckv_w, *ckv_b, *cproj_w, *cproj_b;
        ggml_tensor *fc1_w, *fc1_b, *fc2_w, *fc2_b;
    };
    struct Mha {
        ggml_tensor *q_w, *q_b, *k_w, *k_b, *v_w, *v_b, *o_w, *o_b;
    };
    Backend& be_;
    ModelLoader loader_;
    std::vector<Block> blocks_;
    Mha dual_{}, user_{};
    std::vector<float> rope_inv_freq_, t_freqs_;
};

} // namespace af
