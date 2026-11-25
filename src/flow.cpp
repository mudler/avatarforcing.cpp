#include "flow.hpp"
#include "common.hpp"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstring>

namespace af {

// ---------------------------------------------------------------------------
// Host-side state

struct KvCache::Device {
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<ggml_tensor*> t[2][4];   // [copy][kind][layer]
    int cur = 0;
    ~Device() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
    }
};

void KvCache::init(const FlowConfig& cfg, int n_branches) {
    depth = (int)cfg.depth;
    branches = n_branches;
    heads = (int)cfg.n_heads;
    head_dim = (int)cfg.head_dim;
    self_len = (int)cfg.self_cache_len;
    cross_len = (int)cfg.cross_cache_len;
    self_k.assign((size_t)depth * self_layer_size(), 0.f);
    self_v.assign((size_t)depth * self_layer_size(), 0.f);
    cross_k.assign((size_t)depth * cross_layer_size(), 0.f);
    cross_v.assign((size_t)depth * cross_layer_size(), 0.f);
    if (dev_) {
        ggml_backend_buffer_clear(dev_->buf, 0);
        dev_->cur = 0;
    }
    host_stale_ = false;
}

bool KvCache::alloc_device(ggml_backend_t backend) {
    auto d = std::make_shared<Device>();
    ggml_init_params p = { ggml_tensor_overhead() * (size_t)(8 * depth + 1), nullptr, true };
    d->ctx = ggml_init(p);
    if (!d->ctx) return false;
    for (int c = 0; c < 2; ++c)
        for (int k = 0; k < 4; ++k)
            for (int l = 0; l < depth; ++l)
                d->t[c][k].push_back(ggml_new_tensor_4d(d->ctx, GGML_TYPE_F32, head_dim, k < 2 ? self_len : cross_len,
                                                        heads, branches));
    d->buf = ggml_backend_alloc_ctx_tensors(d->ctx, backend);
    if (!d->buf) { AF_LOG("KV cache: device buffer allocation failed"); return false; }
    ggml_backend_buffer_clear(d->buf, 0);
    dev_ = std::move(d);
    host_stale_ = false;
    // The host vectors may hold a cache already: start from it.
    std::vector<float>* h[4] = {&self_k, &self_v, &cross_k, &cross_v};
    for (int k = 0; k < 4; ++k) {
        const size_t per = k < 2 ? self_layer_size() : cross_layer_size();
        for (int l = 0; l < depth; ++l)
            ggml_backend_tensor_set(dev_->t[0][k][l], h[k]->data() + (size_t)l * per, 0, per * sizeof(float));
    }
    return true;
}

ggml_tensor* KvCache::dev_tensor(int copy, int kind, int layer) const {
    return dev_->t[(dev_->cur + copy) & 1][kind][layer];
}

void KvCache::swap_device() {
    dev_->cur ^= 1;
    host_stale_ = true;
}

void KvCache::sync_host() {
    if (!dev_ || !host_stale_) return;
    std::vector<float>* h[4] = {&self_k, &self_v, &cross_k, &cross_v};
    for (int k = 0; k < 4; ++k) {
        const size_t per = k < 2 ? self_layer_size() : cross_layer_size();
        for (int l = 0; l < depth; ++l)
            ggml_backend_tensor_get(dev_tensor(0, k, l), h[k]->data() + (size_t)l * per, 0, per * sizeof(float));
    }
    host_stale_ = false;
}

int AdaLNTable::row(float t) const {
    for (int i = 0; i < size(); ++i)
        if (timesteps[i] == t) return i;
    return -1;
}

void AdaLNTable::gather(const std::vector<int>& rows, std::vector<float>& adaln6_tok,
                        std::vector<float>& final2_tok) const {
    const size_t n6 = (size_t)6 * dim_h, n2 = (size_t)2 * dim_h;
    adaln6_tok.resize(rows.size() * n6);
    final2_tok.resize(rows.size() * n2);
    for (size_t j = 0; j < rows.size(); ++j) {
        std::memcpy(&adaln6_tok[j * n6], &adaln6[(size_t)rows[j] * n6], n6 * sizeof(float));
        std::memcpy(&final2_tok[j * n2], &final2[(size_t)rows[j] * n2], n2 * sizeof(float));
    }
}

// ---------------------------------------------------------------------------
// Graph helpers

namespace {

ggml_tensor* linear(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b) {
    ggml_tensor* y = ggml_mul_mat(ctx, w, x);
    return b ? ggml_add(ctx, y, b) : y;
}

ggml_tensor* layer_norm(ggml_context* ctx, ggml_tensor* x, float eps, ggml_tensor* w, ggml_tensor* b) {
    ggml_tensor* y = ggml_norm(ctx, x, eps);
    if (w) y = ggml_mul(ctx, y, w);
    if (b) y = ggml_add(ctx, y, b);
    return y;
}

// x * (1 + scale) + shift, with (1 + scale) precomputed as scale1.
ggml_tensor* modulate(ggml_context* ctx, ggml_tensor* x, ggml_tensor* shift, ggml_tensor* scale1) {
    return ggml_add(ctx, ggml_mul(ctx, x, scale1), shift);
}

// GELU, tanh approximation, in float32 as torch computes it:
// 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
// ggml_gelu goes through an fp16 lookup table on the CPU, too coarse here.
ggml_tensor* gelu_tanh(ggml_context* ctx, ggml_tensor* x) {
    const float kBeta = (float)(M_SQRT2 * M_2_SQRTPI * 0.5);
    const float kKappa = 0.044715f;
    ggml_tensor* x3 = ggml_mul(ctx, ggml_mul(ctx, x, x), x);
    ggml_tensor* inner = ggml_scale(ctx, ggml_add(ctx, x, ggml_scale(ctx, x3, kKappa)), kBeta);
    ggml_tensor* t1 = ggml_scale_bias(ctx, ggml_tanh(ctx, inner), 1.0f, 1.0f);
    return ggml_mul(ctx, ggml_scale(ctx, x, 0.5f), t1);
}

// Scaled dot-product attention. q [hd, Nq, H, B], k and v [hd, Nk, H, B],
// mask [Nk, Nq] (0 or -inf) or nullptr. Returns [hd * H, Nq, B].
ggml_tensor* sdpa(ggml_context* ctx, ggml_tensor* q, ggml_tensor* k, ggml_tensor* v,
                  ggml_tensor* mask, float scale) {
    const int64_t hd = q->ne[0], nq = q->ne[1], H = q->ne[2], B = q->ne[3];
    ggml_tensor* kq = ggml_mul_mat(ctx, k, q);                            // [Nk, Nq, H, B]
    kq = ggml_soft_max_ext(ctx, kq, mask, scale, 0.0f);
    ggml_tensor* vt = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));   // [Nk, hd, H, B]
    ggml_tensor* o = ggml_mul_mat(ctx, vt, kq);                           // [hd, Nq, H, B]
    o = ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3));                 // [hd, H, Nq, B]
    return ggml_reshape_3d(ctx, o, hd * H, nq, B);
}

// Split heads: [hd * H, N, B] (or a strided view of it) to [hd, N, H, B].
ggml_tensor* heads_view(ggml_context* ctx, ggml_tensor* src, int64_t hd, int64_t H, int64_t N,
                        int64_t B, size_t offset) {
    ggml_tensor* t = ggml_view_4d(ctx, src, hd, H, N, B, hd * sizeof(float), src->nb[1], src->nb[2], offset);
    return t;   // [hd, H, N, B], not contiguous
}

ggml_tensor* to_heads_first(ggml_context* ctx, ggml_tensor* t) {
    return ggml_cont(ctx, ggml_permute(ctx, t, 0, 2, 1, 3));   // [hd, H, N, B] -> [hd, N, H, B]
}

// Tokens [t0, t0 + n) along dim 1 of a contiguous [hd, N, H, B] tensor.
ggml_tensor* token_range(ggml_context* ctx, ggml_tensor* t, int64_t t0, int64_t n) {
    ggml_tensor* v = ggml_view_4d(ctx, t, t->ne[0], n, t->ne[2], t->ne[3], t->nb[1], t->nb[2], t->nb[3],
                                  t0 * t->nb[1]);
    return ggml_cont(ctx, v);
}

// Interleaved RoPE: pairs (2i, 2i+1) rotate by the angle of frequency i.
// x [hd, H, N, B] contiguous; cs [hd, 1, N, 1] holds (cos_i, cos_i) per pair
// and sn [hd, 1, N, 1] holds (-sin_i, sin_i). out = x * cs + swap(x) * sn,
// which is the reference's (a cos - b sin, a sin + b cos) operation by operation.
ggml_tensor* rope_interleaved(ggml_context* ctx, ggml_tensor* x, ggml_tensor* cs, ggml_tensor* sn) {
    const int64_t m = ggml_nelements(x) / 2;
    ggml_tensor* x2 = ggml_reshape_2d(ctx, x, 2, m);
    ggml_tensor* even = ggml_view_2d(ctx, x2, 1, m, x2->nb[1], 0);
    ggml_tensor* odd = ggml_view_2d(ctx, x2, 1, m, x2->nb[1], sizeof(float));
    ggml_tensor* sw = ggml_concat(ctx, odd, even, 0);                    // [2, m]
    sw = ggml_reshape_4d(ctx, sw, x->ne[0], x->ne[1], x->ne[2], x->ne[3]);
    return ggml_add(ctx, ggml_mul(ctx, x, cs), ggml_mul(ctx, sw, sn));
}

void tap(Backend::Graph& g, std::vector<std::pair<std::string, ggml_tensor*>>& list, Taps* taps,
         const std::string& name, ggml_tensor* t) {
    if (!taps) return;
    ggml_set_output(t);
    ggml_build_forward_expand(g.gf, t);
    list.emplace_back(name, t);
}

void read_taps(const std::vector<std::pair<std::string, ggml_tensor*>>& list, Taps* taps) {
    if (!taps) return;
    for (const auto& kv : list) {
        std::vector<float>& dst = (*taps)[kv.first];
        dst.resize(ggml_nelements(kv.second));
        ggml_backend_tensor_get(kv.second, dst.data(), 0, ggml_nbytes(kv.second));
    }
}

void read_tensor(ggml_tensor* t, std::vector<float>& out) {
    out.resize(ggml_nelements(t));
    ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t));
}

} // namespace

// ---------------------------------------------------------------------------
// Load

FlowModel::~FlowModel() = default;

bool FlowModel::load(const std::string& gguf_path) {
    if (!loader_.load(gguf_path, be_.handle(), {"ft.", "audio_projection.", "sched."})) return false;
    const FlowConfig& c = loader_.config();
    auto T = [&](const std::string& n) { return loader_.require(n); };

    blocks_.resize(c.depth);
    for (uint32_t i = 0; i < c.depth; ++i) {
        const std::string p = "ft.blocks." + std::to_string(i) + ".";
        Block& b = blocks_[i];
        b.qkv_w = T(p + "attn.qkv.weight");         b.qkv_b = T(p + "attn.qkv.bias");
        b.proj_w = T(p + "attn.proj.weight");       b.proj_b = T(p + "attn.proj.bias");
        b.cq_w = T(p + "cross_attn.q.weight");      b.cq_b = T(p + "cross_attn.q.bias");
        b.ckv_w = T(p + "cross_attn.kv.weight");    b.ckv_b = T(p + "cross_attn.kv.bias");
        b.cproj_w = T(p + "cross_attn.proj.weight"); b.cproj_b = T(p + "cross_attn.proj.bias");
        b.fc1_w = T(p + "mlp.fc1.weight");          b.fc1_b = T(p + "mlp.fc1.bias");
        b.fc2_w = T(p + "mlp.fc2.weight");          b.fc2_b = T(p + "mlp.fc2.bias");
    }
    auto mha = [&](const std::string& p, Mha& m) {
        m.q_w = T(p + "q_proj.weight");   m.q_b = T(p + "q_proj.bias");
        m.k_w = T(p + "k_proj.weight");   m.k_b = T(p + "k_proj.bias");
        m.v_w = T(p + "v_proj.weight");   m.v_b = T(p + "v_proj.bias");
        m.o_w = T(p + "out_proj.weight"); m.o_b = T(p + "out_proj.bias");
    };
    mha("ft.c_embedder.dual_modal_attn.", dual_);
    mha("ft.c_embedder.user_agent_attn.", user_);
    for (const char* n : {"ft.x_embedder.proj.weight", "ft.x_embedder.proj.bias",
                          "ft.t_embedder.mlp.0.weight", "ft.t_embedder.mlp.0.bias",
                          "ft.t_embedder.mlp.2.weight", "ft.t_embedder.mlp.2.bias",
                          "ft.adaLN_modulation.1.weight", "ft.adaLN_modulation.1.bias",
                          "ft.final_layer.adaLN_modulation.1.weight", "ft.final_layer.adaLN_modulation.1.bias",
                          "ft.final_layer.linear.weight", "ft.final_layer.linear.bias",
                          "ft.c_embedder.motion_embedder.0.weight", "ft.c_embedder.motion_embedder.0.bias",
                          "ft.c_embedder.motion_embedder.1.weight", "ft.c_embedder.motion_embedder.1.bias",
                          "ft.c_embedder.motion_embedder.3.weight", "ft.c_embedder.motion_embedder.3.bias",
                          "ft.c_embedder.motion_embedder.4.weight", "ft.c_embedder.motion_embedder.4.bias",
                          "ft.c_embedder.ln1.weight", "ft.c_embedder.ln1.bias",
                          "ft.c_embedder.ln2.weight", "ft.c_embedder.ln2.bias",
                          "ft.c_embedder.to_cond_user.weight", "ft.c_embedder.to_cond_user.bias",
                          "audio_projection.0.weight", "audio_projection.0.bias",
                          "audio_projection.1.weight", "audio_projection.1.bias"})
        T(n);

    ggml_tensor* inv = T("ft.rope.inv_freq");
    ggml_tensor* tf = T("ft.t_embedder.freqs");
    if (loader_.missing()) return false;

    // Shape checks against the KV.
    bool ok = true;
    auto expect = [&](ggml_tensor* t, int64_t ne0, int64_t ne1, const char* what) {
        if (t->ne[0] != ne0 || t->ne[1] != ne1) {
            AF_LOG("shape mismatch: %s [%lld, %lld], expected [%lld, %lld]", what, (long long)t->ne[0],
                   (long long)t->ne[1], (long long)ne0, (long long)ne1);
            ok = false;
        }
    };
    expect(inv, c.rope_dim / 2, 1, "ft.rope.inv_freq");
    expect(tf, c.freq_embed_dim / 2, 1, "ft.t_embedder.freqs");
    expect(T("ft.x_embedder.proj.weight"), 2 * c.dim_w, c.dim_h, "ft.x_embedder.proj.weight");
    expect(T("ft.adaLN_modulation.1.weight"), c.dim_h, 6 * c.dim_h, "ft.adaLN_modulation.1.weight");
    expect(T("ft.final_layer.linear.weight"), c.dim_h, c.dim_w, "ft.final_layer.linear.weight");
    expect(T("ft.c_embedder.to_cond_user.weight"), c.dim_a, c.dim_h, "ft.c_embedder.to_cond_user.weight");
    expect(T("audio_projection.0.weight"), c.audio_proj_in, c.audio_proj_out, "audio_projection.0.weight");
    for (const Block& b : blocks_) {
        expect(b.qkv_w, c.dim_h, 3 * c.dim_h, "attn.qkv.weight");
        expect(b.ckv_w, c.dim_h, 2 * c.dim_h, "cross_attn.kv.weight");
        expect(b.fc1_w, c.dim_h, c.mlp_hidden, "mlp.fc1.weight");
    }
    if (!ok) return false;

    rope_inv_freq_.resize(ggml_nelements(inv));
    ggml_backend_tensor_get(inv, rope_inv_freq_.data(), 0, ggml_nbytes(inv));
    t_freqs_.resize(ggml_nelements(tf));
    ggml_backend_tensor_get(tf, t_freqs_.data(), 0, ggml_nbytes(tf));
    AF_LOG("flow transformer: %zu tensors, %.1f MiB", loader_.n_tensors(), loader_.n_bytes() / 1048576.0);
    return true;
}

// ---------------------------------------------------------------------------
// RoPE and masks

void FlowModel::rope_table(int start_pos, int n, std::vector<float>& cos_sin) const {
    const int half = (int)rope_inv_freq_.size();
    cos_sin.resize((size_t)n * 2 * half);
    for (int j = 0; j < n; ++j) {
        const float pos = (float)(start_pos + j);
        for (int i = 0; i < half; ++i) {
            const float a = pos * rope_inv_freq_[i];
            cos_sin[(size_t)j * 2 * half + i] = std::cos(a);
            cos_sin[(size_t)j * 2 * half + half + i] = std::sin(a);
        }
    }
}

// ---------------------------------------------------------------------------
// Timestep tables

bool FlowModel::timestep_tables(const std::vector<float>& timesteps, AdaLNTable& out) {
    const FlowConfig& c = config();
    const int n = (int)timesteps.size();
    const int half = (int)c.freq_embed_dim / 2;
    out.dim_h = (int)c.dim_h;
    out.freq_dim = (int)c.freq_embed_dim;
    out.timesteps = timesteps;
    // Sinusoidal embedding as the reference computes it: float32 t * freqs,
    // then cat(cos, sin).
    out.t_freq.resize((size_t)n * c.freq_embed_dim);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < half; ++i) {
            const float a = timesteps[j] * t_freqs_[i];
            out.t_freq[(size_t)j * c.freq_embed_dim + i] = std::cos(a);
            out.t_freq[(size_t)j * c.freq_embed_dim + half + i] = std::sin(a);
        }

    ggml_tensor *temb = nullptr, *ada = nullptr, *fin = nullptr;
    auto build = [&](Backend::Graph& g) {
        ggml_context* ctx = g.ctx;
        auto W = [&](const char* name) { return loader_.tensor(name); };
        ggml_tensor* tf = g.input(out.t_freq.data(), c.freq_embed_dim, n);
        ggml_tensor* h = linear(ctx, tf, W("ft.t_embedder.mlp.0.weight"), W("ft.t_embedder.mlp.0.bias"));
        temb = linear(ctx, ggml_silu(ctx, h), W("ft.t_embedder.mlp.2.weight"), W("ft.t_embedder.mlp.2.bias"));
        ggml_tensor* s = ggml_silu(ctx, temb);
        ada = linear(ctx, s, W("ft.adaLN_modulation.1.weight"), W("ft.adaLN_modulation.1.bias"));
        fin = linear(ctx, s, W("ft.final_layer.adaLN_modulation.1.weight"),
                     W("ft.final_layer.adaLN_modulation.1.bias"));
        for (ggml_tensor* t : {temb, ada, fin}) {
            ggml_set_output(t);
            ggml_build_forward_expand(g.gf, t);
        }
        return true;
    };
    auto read = [&](Backend::Graph&) {
        read_tensor(temb, out.t_embed);
        read_tensor(ada, out.adaln6);
        read_tensor(fin, out.final2);
    };
    return be_.compute(build, read);
}

// ---------------------------------------------------------------------------
// Condition embedder

bool FlowModel::condition(int B, int N, const float* wa_user, const float* motion_user, const float* wa,
                          std::vector<float>& c_out, Taps* taps) {
    const FlowConfig& c = config();
    const int64_t E = c.dim_a, H = c.cond_n_heads, hd = E / H;
    const float eps = c.cond_ln_eps;
    std::vector<std::pair<std::string, ggml_tensor*>> list;
    ggml_tensor* out = nullptr;

    auto build = [&](Backend::Graph& g) {
        ggml_context* ctx = g.ctx;
        auto W = [&](const char* name) { return loader_.tensor(name); };
        // nn.MultiheadAttention (batch_first, no mask). The query is scaled
        // before the product, as torch does.
        auto mha = [&](ggml_tensor* q_in, ggml_tensor* kv_in, const Mha& m) {
            const int64_t nq = q_in->ne[1], nk = kv_in->ne[1];
            ggml_tensor* q = ggml_scale(ctx, linear(ctx, q_in, m.q_w, m.q_b), 1.0f / std::sqrt((float)hd));
            ggml_tensor* k = linear(ctx, kv_in, m.k_w, m.k_b);
            ggml_tensor* v = linear(ctx, kv_in, m.v_w, m.v_b);
            q = to_heads_first(ctx, ggml_reshape_4d(ctx, q, hd, H, nq, B));
            k = to_heads_first(ctx, ggml_reshape_4d(ctx, k, hd, H, nk, B));
            v = to_heads_first(ctx, ggml_reshape_4d(ctx, v, hd, H, nk, B));
            ggml_tensor* o = sdpa(ctx, q, k, v, nullptr, 1.0f);
            return linear(ctx, o, m.o_w, m.o_b);
        };
        ggml_tensor* x_wu = g.input(wa_user, E, N, B);
        ggml_tensor* x_mu = g.input(motion_user, E, N, B);
        ggml_tensor* x_wa = g.input(wa, E, N, B);

        ggml_tensor* m = linear(ctx, x_mu, W("ft.c_embedder.motion_embedder.0.weight"),
                                W("ft.c_embedder.motion_embedder.0.bias"));
        m = ggml_silu(ctx, layer_norm(ctx, m, eps, W("ft.c_embedder.motion_embedder.1.weight"),
                                      W("ft.c_embedder.motion_embedder.1.bias")));
        m = linear(ctx, m, W("ft.c_embedder.motion_embedder.3.weight"), W("ft.c_embedder.motion_embedder.3.bias"));
        m = ggml_silu(ctx, layer_norm(ctx, m, eps, W("ft.c_embedder.motion_embedder.4.weight"),
                                      W("ft.c_embedder.motion_embedder.4.bias")));
        tap(g, list, taps, "c_embedder.motion_embedder", m);

        ggml_tensor* d = mha(m, x_wu, dual_);
        tap(g, list, taps, "c_embedder.dual_modal_attn", d);
        ggml_tensor* cmf = ggml_silu(ctx, layer_norm(ctx, ggml_add(ctx, m, d), eps, W("ft.c_embedder.ln1.weight"),
                                                     W("ft.c_embedder.ln1.bias")));
        tap(g, list, taps, "c_embedder.cross_modal_feat", cmf);

        ggml_tensor* u = mha(x_wa, cmf, user_);
        tap(g, list, taps, "c_embedder.user_agent_attn", u);
        ggml_tensor* un = ggml_silu(ctx, layer_norm(ctx, ggml_add(ctx, x_wa, u), eps, W("ft.c_embedder.ln2.weight"),
                                                    W("ft.c_embedder.ln2.bias")));
        tap(g, list, taps, "c_embedder.unified", un);
        out = linear(ctx, un, W("ft.c_embedder.to_cond_user.weight"), W("ft.c_embedder.to_cond_user.bias"));
        ggml_set_output(out);
        ggml_build_forward_expand(g.gf, out);
        return true;
    };
    auto read = [&](Backend::Graph&) {
        read_tensor(out, c_out);
        read_taps(list, taps);
    };
    return be_.compute(build, read);
}

// ---------------------------------------------------------------------------
// Flow transformer forward

bool FlowModel::forward(const ForwardArgs& a, std::vector<float>& v_out, Taps* taps) {
    const FlowConfig& c = config();
    const int64_t N = a.n_tokens, B = a.n_branches;
    const int64_t D = c.dim_h, H = c.n_heads, hd = c.head_dim;
    const int64_t ctxf = c.context_frames;
    const int64_t L = c.self_cache_len, Lc = c.cross_cache_len;
    const bool steady = N == (int64_t)(c.block_frames + c.context_frames);
    // The reference concatenates the caches only for the steady shape.
    const bool kv_read = a.kv && steady;
    const bool kv_write = a.kv && a.update_kv;
    const bool kv_dev = a.kv && a.kv->on_device();

    if (!(N == (int64_t)c.clip_frames || steady)) {
        AF_LOG("forward: N = %lld is neither the primer nor the steady shape", (long long)N);
        return false;
    }
    if (a.kv && (a.kv->branches != B || a.kv->depth != (int)c.depth)) {
        AF_LOG("forward: KV cache shape does not match");
        return false;
    }
    if (kv_write && !steady && N - ctxf - L < 0) { AF_LOG("forward: too few tokens for the cache"); return false; }
    if (kv_write && steady && N - ctxf != (int64_t)c.kv_shift) { AF_LOG("forward: kv_shift mismatch"); return false; }

    // Masks, 0 where attention is allowed and -inf elsewhere.
    const float NEG = -INFINITY;
    const int64_t Nk = kv_read ? L + N : N;
    const int64_t Nkc = kv_read ? Lc + N : N;
    std::vector<float> self_mask((size_t)Nk * N), cross_mask((size_t)Nkc * N);
    {
        // Self: the block-causal lookahead mask of the clip, rows of the last
        // N queries when the cache is read (key index = position in the clip).
        const int64_t q0 = kv_read ? (int64_t)c.clip_frames - N : 0;
        const int64_t blk = c.self_mask_block, la = c.self_mask_lookahead;
        for (int64_t qi = 0; qi < N; ++qi)
            for (int64_t k = 0; k < Nk; ++k) {
                const int64_t q = q0 + qi;
                const bool allow = (k / blk < q / blk) || (k <= q + la);
                self_mask[(size_t)qi * Nk + k] = allow ? 0.f : NEG;
            }
        // Cross: band |k - q| <= window, with the keys shifted by the cache
        // length when the cache is read.
        const int64_t off = kv_read ? Lc : 0, w = c.cross_mask_window;
        for (int64_t qi = 0; qi < N; ++qi)
            for (int64_t k = 0; k < Nkc; ++k)
                cross_mask[(size_t)qi * Nkc + k] = std::llabs(k - (qi + off)) <= w ? 0.f : NEG;
    }
    // RoPE multipliers [N, hd]: (cos, cos) and (-sin, sin) per pair.
    std::vector<float> rope_cs((size_t)N * hd), rope_sn((size_t)N * hd);
    {
        std::vector<float> tbl;
        rope_table(a.start_pos, (int)N, tbl);
        const int64_t half = hd / 2;
        for (int64_t j = 0; j < N; ++j)
            for (int64_t i = 0; i < half; ++i) {
                const float cv = tbl[j * hd + i], sv = tbl[j * hd + half + i];
                rope_cs[j * hd + 2 * i] = cv;
                rope_cs[j * hd + 2 * i + 1] = cv;
                rope_sn[j * hd + 2 * i] = -sv;
                rope_sn[j * hd + 2 * i + 1] = sv;
            }
    }

    std::vector<std::pair<std::string, ggml_tensor*>> list;
    ggml_tensor* out = nullptr;
    struct KvOut { ggml_tensor *sk, *sv, *ck, *cv; };
    std::vector<KvOut> kv_out(c.depth, KvOut{nullptr, nullptr, nullptr, nullptr});

    auto build = [&](Backend::Graph& g) {
        ggml_context* ctx = g.ctx;
        auto W = [&](const char* name) { return loader_.tensor(name); };
        auto out_expand = [&](ggml_tensor* t) {
            ggml_set_output(t);
            ggml_build_forward_expand(g.gf, t);
        };

        ggml_tensor* x = g.input(a.x, c.dim_w, N, B);
        ggml_tensor* cond = g.input(a.c, D, N, B);
        ggml_tensor* wr = g.input(a.wr, c.dim_w, 1, 1);
        ggml_tensor* ada = g.input(a.adaln6, 6 * D, N);
        ggml_tensor* fin = g.input(a.final2, 2 * D, N);
        ggml_tensor* smask = g.input(self_mask.data(), Nk, N);
        ggml_tensor* cmask = g.input(cross_mask.data(), Nkc, N);
        ggml_tensor* rcs = g.input(rope_cs.data(), hd, 1, N, 1);
        ggml_tensor* rsn = g.input(rope_sn.data(), hd, 1, N, 1);

        // adaLN chunks, shared by all blocks: [D, N] each, broadcast over B.
        auto chunk = [&](ggml_tensor* t, int k) {
            return ggml_cont(ctx, ggml_view_2d(ctx, t, D, N, t->nb[1], (size_t)k * D * sizeof(float)));
        };
        ggml_tensor* shift_msa = chunk(ada, 0);
        ggml_tensor* scale1_msa = ggml_scale_bias(ctx, chunk(ada, 1), 1.0f, 1.0f);
        ggml_tensor* gate_msa = chunk(ada, 2);
        ggml_tensor* shift_mlp = chunk(ada, 3);
        ggml_tensor* scale1_mlp = ggml_scale_bias(ctx, chunk(ada, 4), 1.0f, 1.0f);
        ggml_tensor* gate_mlp = chunk(ada, 5);
        ggml_tensor* shift_fin = chunk(fin, 0);
        ggml_tensor* scale1_fin = ggml_scale_bias(ctx, chunk(fin, 1), 1.0f, 1.0f);

        // x_embedder(concat(x, wr)).
        ggml_tensor* wr_rep = ggml_repeat(ctx, wr, x);
        ggml_tensor* h = linear(ctx, ggml_concat(ctx, x, wr_rep, 0), W("ft.x_embedder.proj.weight"),
                                W("ft.x_embedder.proj.bias"));
        tap(g, list, taps, "x_embed", h);

        const float scale = 1.0f / std::sqrt((float)hd);
        for (uint32_t li = 0; li < c.depth; ++li) {
            const Block& bw = blocks_[li];
            const bool t0 = li == 0;
            const std::string bp = "block.0.";

            // Self-attention.
            ggml_tensor* a_in = modulate(ctx, ggml_norm(ctx, h, c.ln_eps), shift_msa, scale1_msa);
            if (t0) tap(g, list, taps, bp + "attn_in", a_in);
            ggml_tensor* qkv = linear(ctx, a_in, bw.qkv_w, bw.qkv_b);            // [3D, N, B]
            const size_t es = sizeof(float);
            ggml_tensor* q = ggml_cont(ctx, heads_view(ctx, qkv, hd, H, N, B, 0));
            ggml_tensor* k = ggml_cont(ctx, heads_view(ctx, qkv, hd, H, N, B, (size_t)D * es));
            ggml_tensor* v = heads_view(ctx, qkv, hd, H, N, B, (size_t)2 * D * es);
            q = to_heads_first(ctx, rope_interleaved(ctx, q, rcs, rsn));         // [hd, N, H, B]
            k = to_heads_first(ctx, rope_interleaved(ctx, k, rcs, rsn));
            v = to_heads_first(ctx, v);
            ggml_tensor *k_attn = k, *v_attn = v;
            ggml_tensor *k_cache = nullptr, *v_cache = nullptr;
            if (kv_read) {
                if (kv_dev) {
                    k_cache = a.kv->dev_tensor(0, 0, li);
                    v_cache = a.kv->dev_tensor(0, 1, li);
                } else {
                    const size_t off = (size_t)li * a.kv->self_layer_size();
                    k_cache = g.input(a.kv->self_k.data() + off, hd, L, H, B);
                    v_cache = g.input(a.kv->self_v.data() + off, hd, L, H, B);
                }
                k_attn = ggml_concat(ctx, k_cache, k, 1);
                v_attn = ggml_concat(ctx, v_cache, v, 1);
            }
            ggml_tensor* att = linear(ctx, sdpa(ctx, q, k_attn, v_attn, smask, scale), bw.proj_w, bw.proj_b);
            if (t0) tap(g, list, taps, bp + "attn_out", att);
            if (kv_write) {
                if (steady) {
                    // Shift left by kv_shift, append the new non-context tokens.
                    const int64_t sh = c.kv_shift;
                    kv_out[li].sk = ggml_concat(ctx, token_range(ctx, k_cache, sh, L - sh), token_range(ctx, k, 0, N - ctxf), 1);
                    kv_out[li].sv = ggml_concat(ctx, token_range(ctx, v_cache, sh, L - sh), token_range(ctx, v, 0, N - ctxf), 1);
                } else {
                    kv_out[li].sk = token_range(ctx, k, N - ctxf - L, L);
                    kv_out[li].sv = token_range(ctx, v, N - ctxf - L, L);
                }
            }
            h = ggml_add(ctx, h, ggml_mul(ctx, att, gate_msa));

            // Cross-attention to the condition (no gate).
            ggml_tensor* c_in = ggml_norm(ctx, h, c.ln_eps);
            if (t0) tap(g, list, taps, bp + "cross_in", c_in);
            ggml_tensor* cq = to_heads_first(ctx, ggml_reshape_4d(ctx, linear(ctx, c_in, bw.cq_w, bw.cq_b), hd, H, N, B));
            ggml_tensor* ckv = linear(ctx, cond, bw.ckv_w, bw.ckv_b);           // [2D, N, B]
            ggml_tensor* ck = to_heads_first(ctx, heads_view(ctx, ckv, hd, H, N, B, 0));
            ggml_tensor* cv = to_heads_first(ctx, heads_view(ctx, ckv, hd, H, N, B, (size_t)D * es));
            ggml_tensor *ck_attn = ck, *cv_attn = cv;
            if (kv_read) {
                ggml_tensor *ckc, *cvc;
                if (kv_dev) {
                    ckc = a.kv->dev_tensor(0, 2, li);
                    cvc = a.kv->dev_tensor(0, 3, li);
                } else {
                    const size_t off = (size_t)li * a.kv->cross_layer_size();
                    ckc = g.input(a.kv->cross_k.data() + off, hd, Lc, H, B);
                    cvc = g.input(a.kv->cross_v.data() + off, hd, Lc, H, B);
                }
                ck_attn = ggml_concat(ctx, ckc, ck, 1);
                cv_attn = ggml_concat(ctx, cvc, cv, 1);
            }
            ggml_tensor* catt = linear(ctx, sdpa(ctx, cq, ck_attn, cv_attn, cmask, scale), bw.cproj_w, bw.cproj_b);
            if (t0) tap(g, list, taps, bp + "cross_out", catt);
            if (kv_write) {
                kv_out[li].ck = token_range(ctx, ck, N - ctxf - Lc, Lc);
                kv_out[li].cv = token_range(ctx, cv, N - ctxf - Lc, Lc);
            }
            h = ggml_add(ctx, h, catt);

            // MLP.
            ggml_tensor* m_in = modulate(ctx, ggml_norm(ctx, h, c.ln_eps), shift_mlp, scale1_mlp);
            if (t0) tap(g, list, taps, bp + "mlp_in", m_in);
            ggml_tensor* m = linear(ctx, gelu_tanh(ctx, linear(ctx, m_in, bw.fc1_w, bw.fc1_b)), bw.fc2_w, bw.fc2_b);
            if (t0) tap(g, list, taps, bp + "mlp_out", m);
            h = ggml_add(ctx, h, ggml_mul(ctx, m, gate_mlp));
            tap(g, list, taps, "block." + std::to_string(li), h);

            if (kv_write) {
                if (kv_dev) {
                    // Into the other copy; the current one is still read by
                    // this graph.
                    ggml_tensor* src[4] = {kv_out[li].sk, kv_out[li].sv, kv_out[li].ck, kv_out[li].cv};
                    for (int k = 0; k < 4; ++k)
                        ggml_build_forward_expand(g.gf, ggml_cpy(ctx, src[k], a.kv->dev_tensor(1, k, li)));
                } else {
                    for (ggml_tensor* t : {kv_out[li].sk, kv_out[li].sv, kv_out[li].ck, kv_out[li].cv}) out_expand(t);
                }
            }
        }

        // Final layer.
        ggml_tensor* f = modulate(ctx, ggml_norm(ctx, h, c.ln_eps), shift_fin, scale1_fin);
        out = linear(ctx, f, W("ft.final_layer.linear.weight"), W("ft.final_layer.linear.bias"));
        out_expand(out);
        return true;
    };
    auto read = [&](Backend::Graph&) {
        read_tensor(out, v_out);
        read_taps(list, taps);
        if (kv_write && !kv_dev) {
            for (uint32_t li = 0; li < c.depth; ++li) {
                const size_t so = (size_t)li * a.kv->self_layer_size(), co = (size_t)li * a.kv->cross_layer_size();
                ggml_backend_tensor_get(kv_out[li].sk, a.kv->self_k.data() + so, 0, ggml_nbytes(kv_out[li].sk));
                ggml_backend_tensor_get(kv_out[li].sv, a.kv->self_v.data() + so, 0, ggml_nbytes(kv_out[li].sv));
                ggml_backend_tensor_get(kv_out[li].ck, a.kv->cross_k.data() + co, 0, ggml_nbytes(kv_out[li].ck));
                ggml_backend_tensor_get(kv_out[li].cv, a.kv->cross_v.data() + co, 0, ggml_nbytes(kv_out[li].cv));
            }
        }
    };
    if (!be_.compute(build, read)) return false;
    if (kv_write && kv_dev) a.kv->swap_device();
    return true;
}

// ---------------------------------------------------------------------------
// Audio projection

bool FlowModel::audio_projection(int T, const float* stacked, std::vector<float>& out_v, Taps* taps) {
    const FlowConfig& c = config();
    std::vector<std::pair<std::string, ggml_tensor*>> list;
    ggml_tensor* out = nullptr;
    auto build = [&](Backend::Graph& g) {
        ggml_context* ctx = g.ctx;
        auto W = [&](const char* name) { return loader_.tensor(name); };
        ggml_tensor* x = g.input(stacked, c.audio_proj_in, T);
        ggml_tensor* l = linear(ctx, x, W("audio_projection.0.weight"), W("audio_projection.0.bias"));
        tap(g, list, taps, "proj.linear", l);
        ggml_tensor* n = layer_norm(ctx, l, c.audio_proj_ln_eps, W("audio_projection.1.weight"),
                                    W("audio_projection.1.bias"));
        tap(g, list, taps, "proj.norm", n);
        out = ggml_silu(ctx, n);
        ggml_set_output(out);
        ggml_build_forward_expand(g.gf, out);
        return true;
    };
    auto read = [&](Backend::Graph&) {
        read_tensor(out, out_v);
        read_taps(list, taps);
    };
    return be_.compute(build, read);
}

} // namespace af
