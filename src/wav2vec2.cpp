// wav2vec2 audio features in ggml.
//
// The conv feature encoder, the positional conv and the post-LN encoder are
// adapted from the parametric wav2vec2 of voice-detect.cpp
// (src/analyze_w2v2.cpp, same author, same `w2v2.` tensor names). Changes for
// AvatarForcing: the linear interpolation between the conv encoder and the
// feature projection (the reference Wav2VecModel.forward), the two input
// modes, all hidden states as outputs, the manual F32 attention only, and the
// group-norm eps from KV.

#include "wav2vec2.hpp"
#include "common.hpp"
#include "flow.hpp"
#include "quant_policy.hpp"

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace af {

namespace {

// ---------------------------------------------------------------------------
// KV

struct Kv {
    gguf_context* g;
    bool ok = true;

    int64_t find(const char* key, gguf_type type) {
        int64_t id = gguf_find_key(g, key);
        if (id < 0) { AF_LOG("missing KV: %s", key); ok = false; return -1; }
        if (gguf_get_kv_type(g, id) != type) { AF_LOG("KV %s: unexpected type", key); ok = false; return -1; }
        return id;
    }
    uint32_t u32(const char* key) {
        int64_t id = find(key, GGUF_TYPE_UINT32);
        return id < 0 ? 0 : gguf_get_val_u32(g, id);
    }
    float f32(const char* key) {
        int64_t id = find(key, GGUF_TYPE_FLOAT32);
        return id < 0 ? 0.f : gguf_get_val_f32(g, id);
    }
    bool b(const char* key) {
        int64_t id = find(key, GGUF_TYPE_BOOL);
        return id < 0 ? false : gguf_get_val_bool(g, id);
    }
    std::string str(const char* key) {
        int64_t id = find(key, GGUF_TYPE_STRING);
        return id < 0 ? std::string() : std::string(gguf_get_val_str(g, id));
    }
    std::vector<int32_t> i32_arr(const char* key) {
        std::vector<int32_t> out;
        int64_t id = find(key, GGUF_TYPE_ARRAY);
        if (id < 0) return out;
        const gguf_type t = gguf_get_arr_type(g, id);
        const size_t n = gguf_get_arr_n(g, id);
        const void* d = gguf_get_arr_data(g, id);
        if (t == GGUF_TYPE_INT32) out.assign((const int32_t*)d, (const int32_t*)d + n);
        else if (t == GGUF_TYPE_UINT32) for (size_t i = 0; i < n; ++i) out.push_back((int32_t)((const uint32_t*)d)[i]);
        else { AF_LOG("KV %s: not an integer array", key); ok = false; }
        return out;
    }
};

bool read_w2v2_config(const std::string& path, W2V2Config& c) {
    gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    gguf_context* g = gguf_init_from_file(path.c_str(), gp);
    if (!g) { AF_LOG("cannot open GGUF: %s", path.c_str()); return false; }
    Kv r{g};
    c.hidden_size             = r.u32("avatarforcing.w2v2.hidden_size");
    c.n_layers                = r.u32("avatarforcing.w2v2.n_layers");
    c.n_heads                 = r.u32("avatarforcing.w2v2.n_heads");
    c.ff_dim                  = r.u32("avatarforcing.w2v2.ff_dim");
    c.num_conv_layers         = r.u32("avatarforcing.w2v2.num_conv_layers");
    c.conv_dims               = r.i32_arr("avatarforcing.w2v2.conv_dims");
    c.conv_kernels            = r.i32_arr("avatarforcing.w2v2.conv_kernels");
    c.conv_strides            = r.i32_arr("avatarforcing.w2v2.conv_strides");
    c.conv_bias               = r.b("avatarforcing.w2v2.conv_bias");
    c.feat_extract_norm       = r.str("avatarforcing.w2v2.feat_extract_norm");
    c.feat_extract_activation = r.str("avatarforcing.w2v2.feat_extract_activation");
    c.hidden_act              = r.str("avatarforcing.w2v2.hidden_act");
    c.do_stable_layer_norm    = r.b("avatarforcing.w2v2.do_stable_layer_norm");
    c.layer_norm_eps          = r.f32("avatarforcing.w2v2.layer_norm_eps");
    c.group_norm_eps          = r.f32("avatarforcing.w2v2.group_norm_eps");
    c.num_conv_pos_embeddings = r.u32("avatarforcing.w2v2.num_conv_pos_embeddings");
    c.num_conv_pos_embedding_groups = r.u32("avatarforcing.w2v2.num_conv_pos_embedding_groups");
    c.pos_conv_weight_norm_dim = r.u32("avatarforcing.w2v2.pos_conv_weight_norm_dim");
    c.sample_rate             = r.u32("avatarforcing.sample_rate");
    c.samples_per_frame       = r.u32("avatarforcing.samples_per_frame");
    c.n_hidden_taps           = r.u32("avatarforcing.audio.n_hidden_taps");
    c.interp                  = r.str("avatarforcing.audio.interp");
    c.idle_seconds            = r.u32("avatarforcing.audio.idle.seconds");
    gguf_free(g);
    if (!r.ok) return false;

    bool ok = true;
    auto check = [&](bool cond, const char* what) {
        if (!cond) { AF_LOG("unsupported wav2vec2 config: %s", what); ok = false; }
    };
    const size_t nc = c.num_conv_layers;
    check(c.conv_dims.size() == nc && c.conv_kernels.size() == nc && c.conv_strides.size() == nc,
          "conv array sizes");
    check(c.feat_extract_norm == "group", "feat_extract_norm != group");
    check(c.feat_extract_activation == "gelu" && c.hidden_act == "gelu", "activation != gelu");
    check(!c.do_stable_layer_norm, "do_stable_layer_norm (pre-LN) not supported");
    check(c.pos_conv_weight_norm_dim == 2, "pos_conv_weight_norm_dim != 2");
    check(c.n_heads > 0 && c.hidden_size % c.n_heads == 0, "hidden_size % n_heads");
    check(c.num_conv_pos_embedding_groups > 0 &&
          c.hidden_size % c.num_conv_pos_embedding_groups == 0, "hidden_size % pos groups");
    check(c.interp == "linear_align_corners", "audio.interp != linear_align_corners");
    check(c.n_hidden_taps == c.n_layers, "audio.n_hidden_taps != n_layers");
    check(c.samples_per_frame > 0, "samples_per_frame");
    return ok;
}

// ---------------------------------------------------------------------------
// Graph helpers

ggml_tensor* linear(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b) {
    ggml_tensor* y = ggml_mul_mat(ctx, w, x);
    return b ? ggml_add(ctx, y, b) : y;
}

ggml_tensor* layer_norm(ggml_context* ctx, ggml_tensor* x, float eps, ggml_tensor* w, ggml_tensor* b) {
    return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, eps), w), b);
}

// Strided Conv1d without padding as F32 im2col + mul_mat (ggml_conv_1d builds
// an F16 im2col). w ne = [K, IC, OC], x ne = [L, IC]; returns [OL, OC].
ggml_tensor* conv1d_strided(ggml_context* ctx, ggml_tensor* w, ggml_tensor* x, int stride) {
    ggml_tensor* im = ggml_im2col(ctx, w, x, stride, 0, 0, 0, 1, 0, false, GGML_TYPE_F32);   // [IC*K, OL]
    ggml_tensor* col = ggml_reshape_2d(ctx, im, im->ne[0], im->ne[1]);
    ggml_tensor* ker = ggml_reshape_2d(ctx, w, w->ne[0] * w->ne[1], w->ne[2]);
    return ggml_mul_mat(ctx, col, ker);                                                   // [OL, OC]
}

void tap(Backend::Graph& g, std::vector<std::pair<std::string, ggml_tensor*>>& list, W2V2Taps* taps,
         const std::string& name, ggml_tensor* t) {
    if (!taps) return;
    ggml_set_output(t);
    ggml_build_forward_expand(g.gf, t);
    list.emplace_back(name, t);
}

void read_taps(const std::vector<std::pair<std::string, ggml_tensor*>>& list, W2V2Taps* taps) {
    if (!taps) return;
    for (const auto& kv : list) {
        std::vector<float>& dst = (*taps)[kv.first];
        dst.resize(ggml_nelements(kv.second));
        ggml_backend_tensor_get(kv.second, dst.data(), 0, ggml_nbytes(kv.second));
    }
}

} // namespace

// ---------------------------------------------------------------------------
// Load

bool AudioEncoder::load(const std::string& gguf_path) {
    if (!read_w2v2_config(gguf_path, cfg_)) return false;
    if (!loader_.load(gguf_path, be_.handle(), {"w2v2.", "audio.idle_pcm"})) return false;

    const W2V2Config& c = cfg_;
    const int H = (int)c.hidden_size, K = (int)c.num_conv_pos_embeddings;
    const int icg = H / (int)c.num_conv_pos_embedding_groups;
    bool ok = true;
    auto expect = [&](const std::string& name, std::vector<int64_t> ne) {
        ggml_tensor* t = loader_.require(name);
        if (!t) { ok = false; return; }
        for (size_t i = 0; i < ne.size(); ++i)
            // ModelLoader has checked the type against the quantization policy.
            if (t->ne[i] != ne[i] || (t->type != GGML_TYPE_F32 && !quant_allowed(name))) {
                AF_LOG("tensor %s: unexpected shape or type", name.c_str());
                ok = false;
                return;
            }
    };
    int in_ch = 1;
    for (uint32_t i = 0; i < c.num_conv_layers; ++i) {
        const std::string p = "w2v2.feature_extractor.conv_layers." + std::to_string(i);
        expect(p + ".conv.weight", {c.conv_kernels[i], in_ch, c.conv_dims[i]});
        if (c.conv_bias) expect(p + ".conv.bias", {c.conv_dims[i]});
        in_ch = c.conv_dims[i];
    }
    expect("w2v2.feature_extractor.conv_layers.0.layer_norm.weight", {c.conv_dims[0]});
    expect("w2v2.feature_extractor.conv_layers.0.layer_norm.bias", {c.conv_dims[0]});
    expect("w2v2.feature_projection.layer_norm.weight", {in_ch});
    expect("w2v2.feature_projection.layer_norm.bias", {in_ch});
    expect("w2v2.feature_projection.projection.weight", {in_ch, H});
    expect("w2v2.feature_projection.projection.bias", {H});
    expect("w2v2.encoder.pos_conv_embed.conv.weight_g", {K, 1, 1});
    expect("w2v2.encoder.pos_conv_embed.conv.weight_v", {K, icg, H});
    expect("w2v2.encoder.pos_conv_embed.conv.bias", {H});
    expect("w2v2.encoder.layer_norm.weight", {H});
    expect("w2v2.encoder.layer_norm.bias", {H});
    for (uint32_t l = 0; l < c.n_layers; ++l) {
        const std::string p = "w2v2.encoder.layers." + std::to_string(l);
        for (const char* n : {"q_proj", "k_proj", "v_proj", "out_proj"}) {
            expect(p + ".attention." + n + ".weight", {H, H});
            expect(p + ".attention." + n + ".bias", {H});
        }
        for (const char* n : {"layer_norm", "final_layer_norm"}) {
            expect(p + "." + n + ".weight", {H});
            expect(p + "." + n + ".bias", {H});
        }
        expect(p + ".feed_forward.intermediate_dense.weight", {H, (int64_t)c.ff_dim});
        expect(p + ".feed_forward.intermediate_dense.bias", {(int64_t)c.ff_dim});
        expect(p + ".feed_forward.output_dense.weight", {(int64_t)c.ff_dim, H});
        expect(p + ".feed_forward.output_dense.bias", {H});
    }
    const FlowConfig& fc = flow_.config();
    if (fc.audio_proj_in != c.n_hidden_taps * c.hidden_size) {
        AF_LOG("audio.proj_in %u != n_hidden_taps * hidden_size", fc.audio_proj_in);
        ok = false;
    }
    ggml_tensor* idle = loader_.require("audio.idle_pcm");
    if (!ok || !idle) return false;

    // Weight norm of the positional conv (dim 2, the kernel axis):
    // w[oc, ic, k] = g[k] * v[oc, ic, k] / ||v[:, :, k]||. Static, so once on the host.
    {
        std::vector<float> g(K), v((size_t)K * icg * H);
        ggml_backend_tensor_get(loader_.tensor("w2v2.encoder.pos_conv_embed.conv.weight_g"), g.data(), 0,
                                g.size() * sizeof(float));
        ggml_backend_tensor_get(loader_.tensor("w2v2.encoder.pos_conv_embed.conv.weight_v"), v.data(), 0,
                                v.size() * sizeof(float));
        std::vector<double> sumsq(K, 0.0);
        for (size_t j = 0; j < v.size(); ++j) sumsq[j % K] += (double)v[j] * v[j];
        pos_w_.resize(v.size());
        for (size_t j = 0; j < v.size(); ++j) {
            const double n = std::sqrt(sumsq[j % K]);
            pos_w_[j] = n > 0 ? (float)((double)v[j] * ((double)g[j % K] / n)) : 0.f;
        }
    }

    // Idle (pink-noise room tone) features, raw input as the live worker does.
    idle_pcm_.resize(ggml_nelements(idle));
    ggml_backend_tensor_get(idle, idle_pcm_.data(), 0, ggml_nbytes(idle));
    if (idle_pcm_.size() != (size_t)c.idle_seconds * c.sample_rate) {
        AF_LOG("audio.idle_pcm: %zu samples, expected %u s", idle_pcm_.size(), c.idle_seconds);
        return false;
    }
    if (!features(idle_pcm_.data(), idle_pcm_.size(), AudioInputMode::Raw, idle_feats_, idle_frames_))
        return false;
    return true;
}

// ---------------------------------------------------------------------------
// Input

int AudioEncoder::frames_for(size_t n) const {
    const size_t spf = cfg_.samples_per_frame;
    return (int)std::max<size_t>(1, (n + spf - 1) / spf);
}

void AudioEncoder::model_input(const float* pcm, size_t n, AudioInputMode mode, std::vector<float>& x,
                               int& n_frames) const {
    n_frames = frames_for(n);
    const size_t need = (size_t)n_frames * cfg_.samples_per_frame;
    x.assign(pcm, pcm + n);
    if (mode == AudioInputMode::Normalized && n > 0) {
        // Wav2Vec2FeatureExtractor: (x - mean) / sqrt(var + 1e-7), population
        // variance, float32 result. Mean and variance accumulate in double.
        double s = 0;
        for (size_t i = 0; i < n; ++i) s += pcm[i];
        const float mean = (float)(s / (double)n);
        double ss = 0;
        for (size_t i = 0; i < n; ++i) {
            const double d = (double)pcm[i] - (double)mean;
            ss += d * d;
        }
        const float var = (float)(ss / (double)n);
        const float denom = std::sqrt(var + 1e-7f);
        for (size_t i = 0; i < n; ++i) x[i] = (pcm[i] - mean) / denom;
        // AudioEncoder.inference: replicate-pad to whole frames.
        x.resize(need, x.back());
    } else {
        // avatar_worker: zero-pad to whole frames.
        x.resize(need, 0.f);
    }
}

// ---------------------------------------------------------------------------
// Forward

bool AudioEncoder::hidden_states(const std::vector<float>& x, int T, std::vector<float>& stacked,
                                 W2V2Taps* taps) {
    const W2V2Config& c = cfg_;
    auto W = [&](const std::string& name) { return loader_.tensor(name); };
    std::vector<std::pair<std::string, ggml_tensor*>> list;

    // Graph 1: conv feature encoder, [L] -> [L', conv_dim] (50 Hz).
    std::vector<float> conv;
    int64_t L1 = 0, C1 = 0;
    {
        ggml_tensor* out = nullptr;
        auto build = [&](Backend::Graph& g) {
            ggml_context* ctx = g.ctx;
            ggml_tensor* h = g.input(x.data(), (int64_t)x.size(), 1);   // [L, IC = 1]
            for (uint32_t i = 0; i < c.num_conv_layers; ++i) {
                const std::string p = "w2v2.feature_extractor.conv_layers." + std::to_string(i);
                h = conv1d_strided(ctx, W(p + ".conv.weight"), h, c.conv_strides[i]);   // [OL, OC]
                if (c.conv_bias)
                    h = ggml_add(ctx, h, ggml_reshape_2d(ctx, W(p + ".conv.bias"), 1, c.conv_dims[i]));
                if (i == 0) {
                    // GroupNorm with one group per channel: normalize each channel
                    // over time (ne[0]), then the per-channel affine.
                    ggml_tensor* gw = W(p + ".layer_norm.weight");
                    ggml_tensor* gb = W(p + ".layer_norm.bias");
                    h = ggml_norm(ctx, h, c.group_norm_eps);
                    h = ggml_mul(ctx, h, ggml_reshape_2d(ctx, gw, 1, gw->ne[0]));
                    h = ggml_add(ctx, h, ggml_reshape_2d(ctx, gb, 1, gb->ne[0]));
                }
                h = ggml_gelu_erf(ctx, h);   // nn.GELU(), exact erf
                tap(g, list, taps, "conv." + std::to_string(i), h);
            }
            out = h;
            ggml_set_output(out);
            ggml_build_forward_expand(g.gf, out);
            return true;
        };
        auto read = [&](Backend::Graph&) {
            L1 = out->ne[0];
            C1 = out->ne[1];
            conv.resize(ggml_nelements(out));
            ggml_backend_tensor_get(out, conv.data(), 0, ggml_nbytes(out));
            read_taps(list, taps);
        };
        if (x.size() < 400) { AF_LOG("audio too short for the conv encoder: %zu samples", x.size()); return false; }
        if (!be_.compute(build, read)) return false;
    }
    if (L1 < 1) { AF_LOG("conv encoder produced no frames"); return false; }

    // Linear interpolation [L1] -> [T] with align_corners, in float32 as
    // torch's upsample_linear1d computes it. Output [T, C1] row-major.
    std::vector<float> interp((size_t)T * C1);
    {
        const float scale = T > 1 ? (float)(L1 - 1) / (float)(T - 1) : 0.f;
        for (int t = 0; t < T; ++t) {
            const float src = scale * (float)t;
            const int64_t i0 = std::min<int64_t>((int64_t)src, L1 - 1);
            const int64_t i1 = i0 + (i0 < L1 - 1 ? 1 : 0);
            const float l1 = std::min(std::max(src - (float)i0, 0.f), 1.f);
            const float l0 = 1.f - l1;
            float* dst = &interp[(size_t)t * C1];
            for (int64_t ch = 0; ch < C1; ++ch)
                dst[ch] = l0 * conv[(size_t)ch * L1 + i0] + l1 * conv[(size_t)ch * L1 + i1];
        }
    }
    if (taps) (*taps)["interp"] = interp;

    // Graph 2: feature projection, positional conv, encoder; all hidden states.
    list.clear();
    const int H = (int)c.hidden_size;
    const int n_heads = (int)c.n_heads, hd = H / n_heads;
    const int K = (int)c.num_conv_pos_embeddings, groups = (int)c.num_conv_pos_embedding_groups;
    const int icg = H / groups;
    const float eps = c.layer_norm_eps;
    std::vector<ggml_tensor*> states;
    auto build = [&](Backend::Graph& g) {
        ggml_context* ctx = g.ctx;
        ggml_tensor* f = g.input(interp.data(), C1, T);                     // [C1, T]
        ggml_tensor* fn = layer_norm(ctx, f, eps, W("w2v2.feature_projection.layer_norm.weight"),
                                     W("w2v2.feature_projection.layer_norm.bias"));
        tap(g, list, taps, "feature_projection.norm", fn);
        ggml_tensor* proj = linear(ctx, fn, W("w2v2.feature_projection.projection.weight"),
                                   W("w2v2.feature_projection.projection.bias"));   // [H, T]
        tap(g, list, taps, "feature_projection", proj);

        // Positional conv: grouped Conv1d (kernel K, padding K/2) with the
        // weight-normed kernel, drop the last step (even K), GELU.
        ggml_tensor* wpos = g.input(pos_w_.data(), K, icg, H);              // [K, icg, H]
        ggml_tensor* xt = ggml_cont(ctx, ggml_transpose(ctx, proj));         // [T, H]
        ggml_tensor* pos = nullptr;
        for (int gi = 0; gi < groups; ++gi) {
            ggml_tensor* xg = ggml_view_2d(ctx, xt, T, icg, xt->nb[1], (size_t)gi * icg * xt->nb[1]);
            ggml_tensor* wg = ggml_view_3d(ctx, wpos, K, icg, icg, wpos->nb[1], wpos->nb[2],
                                           (size_t)gi * icg * wpos->nb[2]);
            ggml_tensor* im = ggml_im2col(ctx, wg, xg, 1, 0, K / 2, 0, 1, 0, false, GGML_TYPE_F32);
            ggml_tensor* yg = ggml_mul_mat(ctx, ggml_reshape_2d(ctx, im, im->ne[0], im->ne[1]),
                                           ggml_reshape_2d(ctx, ggml_cont(ctx, wg), (int64_t)K * icg, icg));
            pos = gi == 0 ? yg : ggml_concat(ctx, pos, yg, 1);               // [OL, H]
        }
        ggml_tensor* pb = W("w2v2.encoder.pos_conv_embed.conv.bias");
        pos = ggml_add(ctx, pos, ggml_reshape_2d(ctx, pb, 1, H));
        const int64_t keep = pos->ne[0] - (K % 2 == 0 ? 1 : 0);
        pos = ggml_view_2d(ctx, pos, keep, H, pos->nb[1], 0);                // [T, H]
        pos = ggml_gelu_erf(ctx, ggml_cont(ctx, pos));
        pos = ggml_cont(ctx, ggml_transpose(ctx, pos));                      // [H, T]
        tap(g, list, taps, "pos_conv", pos);

        // Post-LN encoder: LN(proj + pos), then per layer
        // h = LN(h + attn(h)); h = final_LN(h + ff(h)).
        ggml_tensor* h = layer_norm(ctx, ggml_add(ctx, proj, pos), eps, W("w2v2.encoder.layer_norm.weight"),
                                    W("w2v2.encoder.layer_norm.bias"));
        tap(g, list, taps, "hidden.0", h);
        const float scaling = 1.0f / std::sqrt((float)hd);
        for (uint32_t l = 0; l < c.n_layers; ++l) {
            const std::string p = "w2v2.encoder.layers." + std::to_string(l);
            const std::string a = p + ".attention.";
            ggml_tensor* q = linear(ctx, h, W(a + "q_proj.weight"), W(a + "q_proj.bias"));
            q = ggml_scale(ctx, q, scaling);   // HF scales q before q.k
            ggml_tensor* k = linear(ctx, h, W(a + "k_proj.weight"), W(a + "k_proj.bias"));
            ggml_tensor* v = linear(ctx, h, W(a + "v_proj.weight"), W(a + "v_proj.bias"));
            q = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, q, hd, n_heads, T), 0, 2, 1, 3)); // [hd, T, nh]
            k = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, k, hd, n_heads, T), 0, 2, 1, 3)); // [hd, T, nh]
            v = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, v, hd, n_heads, T), 1, 2, 0, 3)); // [T, hd, nh]
            ggml_tensor* kq = ggml_soft_max(ctx, ggml_mul_mat(ctx, k, q));   // [Tk, Tq, nh]
            ggml_tensor* o = ggml_mul_mat(ctx, v, kq);                       // [hd, Tq, nh]
            o = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3)), H, T);
            o = linear(ctx, o, W(a + "out_proj.weight"), W(a + "out_proj.bias"));
            h = layer_norm(ctx, ggml_add(ctx, h, o), eps, W(p + ".layer_norm.weight"), W(p + ".layer_norm.bias"));
            ggml_tensor* ff = linear(ctx, h, W(p + ".feed_forward.intermediate_dense.weight"),
                                     W(p + ".feed_forward.intermediate_dense.bias"));
            ff = ggml_gelu_erf(ctx, ff);
            ff = linear(ctx, ff, W(p + ".feed_forward.output_dense.weight"),
                        W(p + ".feed_forward.output_dense.bias"));
            h = layer_norm(ctx, ggml_add(ctx, h, ff), eps, W(p + ".final_layer_norm.weight"),
                           W(p + ".final_layer_norm.bias"));
            ggml_set_output(h);
            ggml_build_forward_expand(g.gf, h);
            states.push_back(h);
        }
        return true;
    };
    auto read = [&](Backend::Graph&) {
        // stacked[t, l * H + j] = hidden_states[l + 1][t, j]
        const size_t row = (size_t)states.size() * H;
        stacked.assign((size_t)T * row, 0.f);
        std::vector<float> s((size_t)H * T);
        for (size_t l = 0; l < states.size(); ++l) {
            ggml_backend_tensor_get(states[l], s.data(), 0, ggml_nbytes(states[l]));
            for (int t = 0; t < T; ++t)
                std::memcpy(&stacked[(size_t)t * row + l * H], &s[(size_t)t * H], H * sizeof(float));
            if (taps) (*taps)["hidden." + std::to_string(l + 1)] = s;
        }
        read_taps(list, taps);
    };
    return be_.compute(build, read);
}

bool AudioEncoder::features(const float* pcm, size_t n, AudioInputMode mode, std::vector<float>& out,
                            int& n_frames, W2V2Taps* taps) {
    std::vector<float> x, stacked;
    model_input(pcm, n, mode, x, n_frames);
    if (taps) (*taps)["model_input"] = x;
    if (!hidden_states(x, n_frames, stacked, taps)) return false;
    if (taps) (*taps)["stacked"] = stacked;
    Taps pt;
    if (!flow_.audio_projection(n_frames, stacked.data(), out, taps ? &pt : nullptr)) return false;
    if (taps)
        for (auto& kv : pt) (*taps)[kv.first] = std::move(kv.second);
    return true;
}

} // namespace af
