#include "motion.hpp"
#include "common.hpp"
#include "grid_sample.hpp"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstring>

// Motion autoencoder graph (LIA / StyleGAN2 style), ggml layout [W, H, C, B].
//
// Exact rewrites of the reference (each one is the same function, only the
// float rounding differs):
// - Modulated conv: conv(x * s, W) / sqrt(s^2 @ wsq^T + eps) instead of a
//   per-sample weight; wsq = sum over kernel positions of W^2 (from the GGUF).
// - Stride-2 transposed 3x3 conv: four phase convolutions on the input grid
//   (2x2, 2x1, 1x2 and 1x1 sub-kernels), interleaved into the 2H+1 output.
// - upfirdn2d (blur, up = 1): depthwise convolution with the 2-D kernel.
// - upfirdn2d (up = 2): zero insertion, padding, depthwise convolution.
// - The base grid of each ToFlow level is built once at load time.

namespace af {

namespace {

struct KvReader {
    gguf_context* g;
    bool ok = true;
    int64_t find(const char* key) {
        int64_t id = gguf_find_key(g, key);
        if (id < 0) { AF_LOG("missing KV: %s", key); ok = false; }
        return id;
    }
    uint32_t u32(const char* key) {
        int64_t id = find(key);
        if (id < 0) return 0;
        if (gguf_get_kv_type(g, id) != GGUF_TYPE_UINT32) { AF_LOG("KV %s: not u32", key); ok = false; return 0; }
        return gguf_get_val_u32(g, id);
    }
    float f32(const char* key) {
        int64_t id = find(key);
        if (id < 0) return 0.f;
        if (gguf_get_kv_type(g, id) != GGUF_TYPE_FLOAT32) { AF_LOG("KV %s: not f32", key); ok = false; return 0.f; }
        return gguf_get_val_f32(g, id);
    }
    std::string str(const char* key) {
        int64_t id = find(key);
        if (id < 0) return {};
        if (gguf_get_kv_type(g, id) != GGUF_TYPE_STRING) { AF_LOG("KV %s: not str", key); ok = false; return {}; }
        return gguf_get_val_str(g, id);
    }
    // Integer array (int32 or uint32 elements).
    std::vector<int64_t> ints(const char* key) {
        std::vector<int64_t> out;
        int64_t id = find(key);
        if (id < 0) return out;
        if (gguf_get_kv_type(g, id) != GGUF_TYPE_ARRAY) { AF_LOG("KV %s: not an array", key); ok = false; return out; }
        const enum gguf_type at = gguf_get_arr_type(g, id);
        const size_t n = gguf_get_arr_n(g, id);
        const void* d = gguf_get_arr_data(g, id);
        for (size_t i = 0; i < n; ++i) {
            if (at == GGUF_TYPE_INT32) out.push_back(((const int32_t*)d)[i]);
            else if (at == GGUF_TYPE_UINT32) out.push_back(((const uint32_t*)d)[i]);
            else { AF_LOG("KV %s: not an int32/uint32 array", key); ok = false; out.clear(); break; }
        }
        return out;
    }
};

std::string idx(const char* fmt, int i) {
    char b[160];
    std::snprintf(b, sizeof(b), fmt, i);
    return b;
}

std::vector<float> get_host(ggml_tensor* t) {
    std::vector<float> v((size_t)ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
    return v;
}

// First frame of a [.., .., .., B] output as host data.
std::vector<float> get_frame0(ggml_tensor* t) {
    const size_t n = (size_t)(t->ne[0] * t->ne[1] * t->ne[2]);
    std::vector<float> v(n);
    ggml_backend_tensor_get(t, v.data(), 0, n * sizeof(float));
    return v;
}

} // namespace

MotionModel::~MotionModel() {
    if (pbuf_) ggml_backend_buffer_free(pbuf_);
    if (pctx_) ggml_free(pctx_);
    if (dbuf_) ggml_backend_buffer_free(dbuf_);
    if (dctx_) ggml_free(dctx_);
}

bool MotionModel::read_config(const std::string& path) {
    gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    gguf_context* g = gguf_init_from_file(path.c_str(), gp);
    if (!g) { AF_LOG("cannot open GGUF: %s", path.c_str()); return false; }
    KvReader r{g};
    MaeConfig& c = cfg_;
    c.size            = r.u32("avatarforcing.mae.size");
    c.log_size        = r.u32("avatarforcing.mae.log_size");
    c.style_dim       = r.u32("avatarforcing.mae.style_dim");
    c.motion_dim      = r.u32("avatarforcing.mae.motion_dim");
    c.enc_n_resblocks = r.u32("avatarforcing.mae.enc.n_resblocks");
    c.dec_n_levels    = r.u32("avatarforcing.mae.dec.n_levels");
    c.leaky_slope     = r.f32("avatarforcing.mae.leaky_slope");
    c.act_gain        = r.f32("avatarforcing.mae.act_gain");
    c.demod_eps       = r.f32("avatarforcing.mae.demod_eps");
    c.resblock_scale  = r.f32("avatarforcing.mae.resblock_scale");
    c.grid_sample     = r.str("avatarforcing.mae.grid_sample");
    c.block_frames    = r.u32("avatarforcing.block_frames");
    for (int64_t v : r.ints("avatarforcing.mae.enc.channels")) c.enc_channels.push_back((uint32_t)v);
    for (int64_t v : r.ints("avatarforcing.mae.dec.channels")) c.dec_channels.push_back((uint32_t)v);
    for (int64_t v : r.ints("avatarforcing.mae.blur_kernel")) c.blur_kernel.push_back((int32_t)v);
    auto pad2 = [&](const char* key, int32_t* dst) {
        std::vector<int64_t> v = r.ints(key);
        if (v.size() != 2) { AF_LOG("KV %s: expected 2 values", key); r.ok = false; return; }
        dst[0] = (int32_t)v[0];
        dst[1] = (int32_t)v[1];
    };
    pad2("avatarforcing.mae.pad.enc_down_conv3", c.pad_enc_down_conv3);
    pad2("avatarforcing.mae.pad.enc_down_skip1", c.pad_enc_down_skip1);
    pad2("avatarforcing.mae.pad.dec_up_conv", c.pad_dec_up_conv);
    pad2("avatarforcing.mae.pad.dec_upsample", c.pad_dec_upsample);
    int64_t fid = gguf_find_key(g, "avatarforcing.mae.folded_scales");
    const bool folded = fid >= 0 && gguf_get_kv_type(g, fid) == GGUF_TYPE_BOOL && gguf_get_val_bool(g, fid);
    gguf_free(g);
    if (!r.ok) return false;

    bool ok = true;
    auto check = [&](bool cond, const char* what) {
        if (!cond) { AF_LOG("unsupported motion config: %s", what); ok = false; }
    };
    const int kb = (int)c.blur_kernel.size();
    check(folded, "mae.folded_scales is not true");
    check(c.log_size >= 3 && c.size == (1u << c.log_size), "size != 2^log_size");
    check(c.enc_channels.size() == c.log_size - 1, "enc.channels length");
    check(c.dec_channels.size() == c.log_size - 1, "dec.channels length");
    check(c.enc_n_resblocks == c.log_size - 2, "enc.n_resblocks != log_size - 2");
    check(c.dec_n_levels == c.log_size - 2, "dec.n_levels != log_size - 2");
    check(kb >= 2, "blur_kernel");
    // Output sizes the graph relies on (stride-2 down / up by exactly 2).
    check(c.pad_enc_down_conv3[0] + c.pad_enc_down_conv3[1] == kb - 1 + 3 - 2, "pad.enc_down_conv3");
    check(c.pad_enc_down_skip1[0] + c.pad_enc_down_skip1[1] == kb - 1 + 1 - 2, "pad.enc_down_skip1");
    check(c.pad_dec_up_conv[0] + c.pad_dec_up_conv[1] == kb - 2 && c.pad_dec_up_conv[1] >= 1, "pad.dec_up_conv");
    check(c.pad_dec_upsample[0] + c.pad_dec_upsample[1] == kb - 1, "pad.dec_upsample");
    check(c.grid_sample == "bilinear, zeros, align_corners=false", "grid_sample mode");
    check(c.block_frames > 0, "block_frames");
    return ok;
}

bool MotionModel::load(const std::string& gguf_path) {
    if (!read_config(gguf_path)) return false;
    if (!loader_.load(gguf_path, be_.handle(), {"mae."})) return false;
    const int L = (int)cfg_.dec_n_levels;

    enc_in_w_ = loader_.require("mae.enc.net_app.convs.0.0.weight");
    enc_in_b_ = loader_.require("mae.enc.net_app.convs.0.1.bias");
    for (int i = 1; i <= (int)cfg_.enc_n_resblocks; ++i) {
        ResBlock b;
        b.c1_w   = loader_.require(idx("mae.enc.net_app.convs.%d.conv1.0.weight", i));
        b.c1_b   = loader_.require(idx("mae.enc.net_app.convs.%d.conv1.1.bias", i));
        b.c2_w   = loader_.require(idx("mae.enc.net_app.convs.%d.conv2.1.weight", i));
        b.c2_b   = loader_.require(idx("mae.enc.net_app.convs.%d.conv2.2.bias", i));
        b.skip_w = loader_.require(idx("mae.enc.net_app.convs.%d.skip.1.weight", i));
        res_.push_back(b);
    }
    enc_out_w_ = loader_.require(idx("mae.enc.net_app.convs.%d.weight", (int)cfg_.enc_n_resblocks + 1));
    for (int i = 0; loader_.tensor(idx("mae.enc.fc.%d.weight", i)); ++i)
        fc_.emplace_back(loader_.require(idx("mae.enc.fc.%d.weight", i)), loader_.require(idx("mae.enc.fc.%d.bias", i)));
    if (fc_.empty()) { AF_LOG("missing tensor: mae.enc.fc.0.weight"); return false; }
    q_ = loader_.require("mae.dec.direction.q");

    const_in_ = loader_.require("mae.dec.input.input");
    auto styled = [&](const std::string& p) {
        StyledConv c;
        c.w     = loader_.require(p + ".conv.weight");
        c.wsq   = loader_.require(p + ".conv.wsq");
        c.mod_w = loader_.require(p + ".conv.modulation.weight");
        c.mod_b = loader_.require(p + ".conv.modulation.bias");
        c.act_b = loader_.require(p + ".activate.bias");
        return c;
    };
    conv1_ = styled("mae.dec.conv1");
    for (int i = 0; i < 2 * L; ++i) convs_.push_back(styled(idx("mae.dec.convs.%d", i)));
    for (int l = 0; l < L; ++l) {
        ToFlow f;
        f.w     = loader_.require(idx("mae.dec.to_flows.%d.conv.weight", l));
        f.mod_w = loader_.require(idx("mae.dec.to_flows.%d.conv.modulation.weight", l));
        f.mod_b = loader_.require(idx("mae.dec.to_flows.%d.conv.modulation.bias", l));
        f.b     = loader_.require(idx("mae.dec.to_flows.%d.bias", l));
        to_flows_.push_back(f);
        ToRGB t;
        t.w     = loader_.require(idx("mae.dec.to_rgbs.%d.conv.0.weight", l));
        t.act_b = loader_.require(idx("mae.dec.to_rgbs.%d.conv.1.bias", l));
        t.b     = loader_.require(idx("mae.dec.to_rgbs.%d.bias", l));
        to_rgbs_.push_back(t);
    }
    if (loader_.missing()) return false;

    // Shape checks against the config.
    bool ok = true;
    auto check = [&](bool cond, const char* what) {
        if (!cond) { AF_LOG("motion tensor shape mismatch: %s", what); ok = false; }
    };
    check(q_->ne[0] == cfg_.motion_dim && q_->ne[1] == cfg_.style_dim, "dec.direction.q");
    check(const_in_->ne[0] == 4 && const_in_->ne[1] == 4 && const_in_->ne[2] == cfg_.dec_channels[0], "dec.input");
    for (int i = 0; i < 2 * L; i += 2)
        check(convs_[i].w->ne[0] == 3 && convs_[i].w->ne[1] == 3, "transposed conv kernel is not 3x3");
    return ok && build_derived();
}

bool MotionModel::build_derived() {
    const int L = (int)cfg_.dec_n_levels;
    const int kb = (int)cfg_.blur_kernel.size();
    uint32_t cmax = 3;
    for (uint32_t c : cfg_.enc_channels) cmax = std::max(cmax, c);
    for (uint32_t c : cfg_.dec_channels) cmax = std::max(cmax, c);

    ggml_init_params p = { ggml_tensor_overhead() * (4 * L + L + 8), nullptr, true };
    dctx_ = ggml_init(p);
    blur_down_ = ggml_new_tensor_4d(dctx_, GGML_TYPE_F32, kb, kb, 1, cmax);
    blur_up_ = ggml_new_tensor_4d(dctx_, GGML_TYPE_F32, kb, kb, 1, cmax);
    for (int l = 0; l < L; ++l) {
        const StyledConv& c = convs_[2 * l];
        const int64_t ci = c.w->ne[2], co = c.w->ne[3];
        for (int ey = 0; ey < 2; ++ey)
            for (int ex = 0; ex < 2; ++ex)
                convs_[2 * l].phase[ey][ex] =
                    ggml_new_tensor_4d(dctx_, GGML_TYPE_F32, ex == 0 ? 2 : 1, ey == 0 ? 2 : 1, ci, co);
        const int n = 8 << l;
        grids_.push_back(ggml_new_tensor_4d(dctx_, GGML_TYPE_F32, n, n, 2, 1));
    }
    dbuf_ = ggml_backend_alloc_ctx_tensors(dctx_, be_.handle());
    if (!dbuf_) { AF_LOG("motion: derived buffer allocation failed"); return false; }

    // Blur kernel as make_kernel builds it: outer product in float32, divided
    // by its sum. The upsampling variant is multiplied by factor^2 = 4.
    {
        std::vector<float> k2(kb * kb);
        float sum = 0.f;
        for (int y = 0; y < kb; ++y)
            for (int x = 0; x < kb; ++x) {
                k2[y * kb + x] = (float)cfg_.blur_kernel[y] * (float)cfg_.blur_kernel[x];
            }
        for (float v : k2) sum += v;
        for (float& v : k2) v /= sum;
        std::vector<float> down((size_t)kb * kb * cmax), up(down.size());
        for (uint32_t ch = 0; ch < cmax; ++ch)
            for (int i = 0; i < kb * kb; ++i) {
                down[ch * kb * kb + i] = k2[i];
                up[ch * kb * kb + i] = k2[i] * 4.0f;
            }
        ggml_backend_tensor_set(blur_down_, down.data(), 0, ggml_nbytes(blur_down_));
        ggml_backend_tensor_set(blur_up_, up.data(), 0, ggml_nbytes(blur_up_));
    }

    // Phase kernels of the stride-2 transposed conv. For output parity e the
    // contributing taps are k = 2, 0 (e = 0, as a 2-tap correlation over the
    // input padded by 1) or k = 1 (e = 1).
    for (int l = 0; l < L; ++l) {
        StyledConv& c = convs_[2 * l];
        const int64_t ci = c.w->ne[2], co = c.w->ne[3];
        std::vector<float> w = get_host(c.w);   // [co][ci][3][3]
        static const int taps0[2] = {2, 0}, taps1[1] = {1};
        for (int ey = 0; ey < 2; ++ey)
            for (int ex = 0; ex < 2; ++ex) {
                const int* ty = ey == 0 ? taps0 : taps1;
                const int* tx = ex == 0 ? taps0 : taps1;
                const int kh = ey == 0 ? 2 : 1, kw = ex == 0 ? 2 : 1;
                std::vector<float> k((size_t)kw * kh * ci * co);
                for (int64_t o = 0; o < co; ++o)
                    for (int64_t i = 0; i < ci; ++i)
                        for (int y = 0; y < kh; ++y)
                            for (int x = 0; x < kw; ++x)
                                k[((o * ci + i) * kh + y) * kw + x] = w[((o * ci + i) * 3 + ty[y]) * 3 + tx[x]];
                ggml_backend_tensor_set(c.phase[ey][ex], k.data(), 0, ggml_nbytes(c.phase[ey][ex]));
            }
    }

    // Base grids: numpy linspace(-1, 1, n) in float64, cast to float32;
    // plane 0 varies along W (x), plane 1 along H (y).
    for (int l = 0; l < L; ++l) {
        const int n = 8 << l;
        std::vector<float> xs(n);
        const double step = 2.0 / (double)(n - 1);
        for (int i = 0; i < n; ++i) {
            double v = (double)i * step;
            v = v + -1.0;
            xs[i] = (float)v;
        }
        xs[n - 1] = 1.0f;
        std::vector<float> g((size_t)2 * n * n);
        for (int y = 0; y < n; ++y)
            for (int x = 0; x < n; ++x) {
                g[(size_t)y * n + x] = xs[x];
                g[(size_t)n * n + (size_t)y * n + x] = xs[y];
            }
        ggml_backend_tensor_set(grids_[l], g.data(), 0, ggml_nbytes(grids_[l]));
    }
    return true;
}

// FusedLeakyReLU: leaky_relu(x + bias, slope) * gain.
ggml_tensor* MotionModel::lrelu_bias(ggml_context* ctx, ggml_tensor* x, ggml_tensor* bias) const {
    x = ggml_add(ctx, x, ggml_reshape_4d(ctx, bias, 1, 1, bias->ne[0], 1));
    x = ggml_leaky_relu(ctx, x, cfg_.leaky_slope, false);
    return ggml_scale(ctx, x, cfg_.act_gain);
}

// upfirdn2d with up = down = 1: depthwise correlation with the 2-D kernel
// (the kernel is symmetric, so the flip of upfirdn2d does not matter).
ggml_tensor* MotionModel::blur(ggml_context* ctx, ggml_tensor* x, ggml_tensor* kernel, int pad) const {
    const int kb = (int)kernel->ne[0];
    ggml_tensor* k = ggml_view_4d(ctx, kernel, kb, kb, 1, x->ne[2], kernel->nb[1], kernel->nb[2],
                                  kernel->nb[3], 0);
    return ggml_conv_2d_dw_direct(ctx, k, x, 1, 1, pad, pad, 1, 1);
}

// upfirdn2d with up = 2 (the Upsample module of ToRGB and ToFlow).
ggml_tensor* MotionModel::upsample(ggml_context* ctx, ggml_tensor* x) const {
    const int64_t w = x->ne[0], h = x->ne[1], c = x->ne[2], b = x->ne[3];
    // zero insertion: x[i] -> position 2i, zeros at 2i + 1
    ggml_tensor* t = ggml_reshape_4d(ctx, x, 1, w, h, c * b);
    t = ggml_pad(ctx, t, 1, 0, 0, 0);
    t = ggml_reshape_4d(ctx, t, 2 * w, 1, h, c * b);
    t = ggml_pad(ctx, t, 0, 1, 0, 0);
    t = ggml_reshape_4d(ctx, t, 2 * w, 2 * h, c, b);
    const int p0 = cfg_.pad_dec_upsample[0], p1 = cfg_.pad_dec_upsample[1];
    t = ggml_pad_ext(ctx, t, p0, p1, p0, p1, 0, 0, 0, 0);
    const int kb = (int)blur_up_->ne[0];
    ggml_tensor* k = ggml_view_4d(ctx, blur_up_, kb, kb, 1, c, blur_up_->nb[1], blur_up_->nb[2],
                                  blur_up_->nb[3], 0);
    return ggml_conv_2d_dw_direct(ctx, k, t, 1, 1, 0, 0, 1, 1);
}

// StyledConv: modulated, demodulated conv (3x3, or stride-2 transposed 3x3 +
// blur when up), then FusedLeakyReLU. Noise injection is identity (noise is
// never given).
ggml_tensor* MotionModel::styled_conv(ggml_context* ctx, ggml_tensor* x, ggml_tensor* style,
                                      const StyledConv& c, bool up, ggml_tensor** s_out) const {
    const int64_t B = x->ne[3], ci = c.w->ne[2], co = c.w->ne[3];
    ggml_tensor* s = ggml_add(ctx, ggml_mul_mat(ctx, c.mod_w, style), c.mod_b);   // [ci, B]
    if (s_out) *s_out = s;
    ggml_tensor* xs = ggml_mul(ctx, x, ggml_reshape_4d(ctx, s, 1, 1, ci, B));
    ggml_tensor* y;
    if (!up) {
        const int pad = (int)(c.w->ne[0] / 2);
        y = ggml_conv_2d_direct(ctx, c.w, xs, 1, 1, pad, pad, 1, 1);
    } else {
        // Transposed conv by output phase, each phase padded to H + 1.
        ggml_tensor* ph[2][2];
        for (int ey = 0; ey < 2; ++ey)
            for (int ex = 0; ex < 2; ++ex) {
                ggml_tensor* p = ggml_conv_2d_direct(ctx, c.phase[ey][ex], xs, 1, 1, ex == 0 ? 1 : 0,
                                                     ey == 0 ? 1 : 0, 1, 1);
                if (ex == 1 || ey == 1) p = ggml_pad(ctx, p, ex, ey, 0, 0);
                ph[ey][ex] = p;
            }
        const int64_t hp = ph[0][0]->ne[1], wp = ph[0][0]->ne[0];
        ggml_tensor* rows[2];
        for (int ey = 0; ey < 2; ++ey) {
            ggml_tensor* a = ggml_reshape_4d(ctx, ph[ey][0], 1, wp, hp, co * B);
            ggml_tensor* b = ggml_reshape_4d(ctx, ph[ey][1], 1, wp, hp, co * B);
            ggml_tensor* r = ggml_concat(ctx, a, b, 0);                  // [2, wp, hp, co*B]
            rows[ey] = ggml_reshape_4d(ctx, r, 2 * wp, 1, hp, co * B);
        }
        y = ggml_concat(ctx, rows[0], rows[1], 1);                       // [2wp, 2, hp, co*B]
        y = ggml_reshape_4d(ctx, y, 2 * wp, 2 * hp, co, B);              // 2H+1 data + one zero row/col
    }
    // demod[o, b] = 1 / sqrt(sum_i s[i, b]^2 * wsq[o, i] + eps)
    ggml_tensor* d = ggml_mul_mat(ctx, c.wsq, ggml_sqr(ctx, s));                  // [co, B]
    d = ggml_sqrt(ctx, ggml_scale_bias(ctx, d, 1.0f, cfg_.demod_eps));
    y = ggml_div(ctx, y, ggml_reshape_4d(ctx, d, 1, 1, co, B));
    if (up) {
        // Blur with padding (p0, p1) of the 2H+1 output; the trailing zero row
        // and column of the phase layout already provide one unit of p1.
        const int p0 = cfg_.pad_dec_up_conv[0], p1 = cfg_.pad_dec_up_conv[1] - 1;
        y = ggml_pad_ext(ctx, y, p0, p1, p0, p1, 0, 0, 0, 0);
        y = blur(ctx, y, blur_up_, 0);
    }
    return lrelu_bias(ctx, y, c.act_b);
}

bool MotionModel::encode(const float* image, Portrait& out, MotionTaps* taps) {
    const int S = (int)cfg_.size;
    const int L = (int)cfg_.dec_n_levels;
    std::vector<ggml_tensor*> res;   // outputs of every encoder layer
    ggml_tensor *sr = nullptr, *alpha = nullptr, *rs = nullptr;

    auto build = [&](Backend::Graph& g) {
        ggml_context* ctx = g.ctx;
        ggml_tensor* x = g.input(image, S, S, 3, 1);
        ggml_tensor* h = ggml_conv_2d_direct(ctx, enc_in_w_, x, 1, 1, 0, 0, 1, 1);
        h = lrelu_bias(ctx, h, enc_in_b_);
        res.push_back(h);
        for (const ResBlock& b : res_) {
            const int pad1 = (int)(b.c1_w->ne[0] / 2);
            ggml_tensor* a = ggml_conv_2d_direct(ctx, b.c1_w, h, 1, 1, pad1, pad1, 1, 1);
            a = lrelu_bias(ctx, a, b.c1_b);
            const int pc = cfg_.pad_enc_down_conv3[0], ps = cfg_.pad_enc_down_skip1[0];
            if (cfg_.pad_enc_down_conv3[0] != cfg_.pad_enc_down_conv3[1])
                a = blur(ctx, ggml_pad_ext(ctx, a, pc, cfg_.pad_enc_down_conv3[1], pc, cfg_.pad_enc_down_conv3[1], 0, 0, 0, 0), blur_down_, 0);
            else
                a = blur(ctx, a, blur_down_, pc);
            a = ggml_conv_2d_direct(ctx, b.c2_w, a, 2, 2, 0, 0, 1, 1);
            a = lrelu_bias(ctx, a, b.c2_b);
            ggml_tensor* sk;
            if (cfg_.pad_enc_down_skip1[0] != cfg_.pad_enc_down_skip1[1])
                sk = blur(ctx, ggml_pad_ext(ctx, h, ps, cfg_.pad_enc_down_skip1[1], ps, cfg_.pad_enc_down_skip1[1], 0, 0, 0, 0), blur_down_, 0);
            else
                sk = blur(ctx, h, blur_down_, ps);
            sk = ggml_conv_2d_direct(ctx, b.skip_w, sk, 2, 2, 0, 0, 1, 1);
            h = ggml_scale(ctx, ggml_add(ctx, a, sk), cfg_.resblock_scale);
            res.push_back(h);
        }
        h = ggml_conv_2d_direct(ctx, enc_out_w_, h, 1, 1, 0, 0, 1, 1);   // [1, 1, style_dim, 1]
        res.push_back(h);
        sr = ggml_reshape_2d(ctx, h, cfg_.style_dim, 1);
        ggml_tensor* v = sr;
        for (auto& f : fc_) v = ggml_add(ctx, ggml_mul_mat(ctx, f.first, v), f.second);
        alpha = v;
        rs = ggml_mul_mat(ctx, q_, alpha);   // sum_k alpha_k Q[:, k]
        for (ggml_tensor* t : res) { ggml_set_output(t); ggml_build_forward_expand(g.gf, t); }
        ggml_set_output(alpha); ggml_set_output(rs);
        ggml_build_forward_expand(g.gf, rs);
        return true;
    };
    auto read = [&](Backend::Graph&) {
        out.s_r = get_host(sr);
        out.alpha = get_host(alpha);
        out.r_s = get_host(rs);
        // feat.l = res[::-1][2:][l]: the layer output at resolution 8 << l.
        out.feats.assign(L, {});
        const int n_res = (int)res.size();
        for (int l = 0; l < L; ++l) out.feats[l] = get_host(res[n_res - 3 - l]);
        if (taps) {
            for (size_t i = 0; i < res.size(); ++i) (*taps)[idx("enc.convs.%d", (int)i)] = get_host(res[i]);
            (*taps)["s_r"] = out.s_r;
            (*taps)["alpha"] = out.alpha;
            (*taps)["r_s"] = out.r_s;
            for (int l = 0; l < L; ++l) (*taps)[idx("feat.%d", l)] = out.feats[l];
        }
    };
    return be_.compute(build, read);
}

bool MotionModel::set_portrait(const Portrait& p) {
    const int L = (int)cfg_.dec_n_levels;
    if ((int)p.feats.size() != L || p.s_r.size() != cfg_.style_dim) {
        AF_LOG("set_portrait: wrong portrait shape");
        return false;
    }
    if (pbuf_) { ggml_backend_buffer_free(pbuf_); pbuf_ = nullptr; }
    if (pctx_) { ggml_free(pctx_); pctx_ = nullptr; }
    feats_.clear();
    ggml_init_params ip = { ggml_tensor_overhead() * (L + 1), nullptr, true };
    pctx_ = ggml_init(ip);
    for (int l = 0; l < L; ++l) {
        const int n = 8 << l;
        const int64_t c = cfg_.dec_channels[l + 1];
        if (p.feats[l].size() != (size_t)n * n * c) { AF_LOG("set_portrait: feat.%d size", l); return false; }
        feats_.push_back(ggml_new_tensor_4d(pctx_, GGML_TYPE_F32, n, n, c, 1));
    }
    pbuf_ = ggml_backend_alloc_ctx_tensors(pctx_, be_.handle());
    if (!pbuf_) { AF_LOG("set_portrait: buffer allocation failed"); return false; }
    for (int l = 0; l < L; ++l) ggml_backend_tensor_set(feats_[l], p.feats[l].data(), 0, ggml_nbytes(feats_[l]));
    s_r_ = p.s_r;
    return true;
}

bool MotionModel::decode(const float* latent, int n, std::vector<float>& pixels, MotionTaps* taps) {
    if (s_r_.empty()) { AF_LOG("decode: no portrait set"); return false; }
    const size_t D = cfg_.style_dim;
    std::vector<float> style((size_t)n * D);
    for (int f = 0; f < n; ++f)
        for (size_t i = 0; i < D; ++i) style[f * D + i] = s_r_[i] + latent[f * D + i];
    return decode_styles(style.data(), n, pixels, taps);
}

bool MotionModel::decode_styles(const float* style, int n, std::vector<float>& pixels, MotionTaps* taps) {
    if (feats_.empty()) { AF_LOG("decode: no portrait set"); return false; }
    const int S = (int)cfg_.size, L = (int)cfg_.dec_n_levels;
    const size_t D = cfg_.style_dim, frame = (size_t)3 * S * S;
    pixels.assign((size_t)n * frame, 0.f);
    const int chunk = chunk_frames_ > 0 ? chunk_frames_ : n;

    for (int f0 = 0; f0 < n; f0 += chunk) {
        const int B = std::min(chunk, n - f0);
        const bool want_taps = taps && f0 == 0;
        std::vector<std::pair<std::string, ggml_tensor*>> tapped;
        ggml_tensor* img = nullptr;
        auto tap = [&](const std::string& name, ggml_tensor* t) {
            if (!want_taps) return;
            ggml_set_output(t);
            tapped.emplace_back(name, t);
        };
        auto build = [&](Backend::Graph& g) {
            ggml_context* ctx = g.ctx;
            ggml_tensor* st = g.input(style + (size_t)f0 * D, D, B);
            ggml_tensor* x = ggml_repeat(ctx, const_in_, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 4, 4, const_in_->ne[2], B));
            ggml_tensor* s = nullptr;
            x = styled_conv(ctx, x, st, conv1_, false, &s);
            tap("conv1.style", s);
            tap("conv1", x);
            ggml_tensor *flow_raw = nullptr, *skip = nullptr;
            for (int l = 0; l < L; ++l) {
                const std::string p = idx("level.%d.", l);
                x = styled_conv(ctx, x, st, convs_[2 * l], true, &s);
                tap(p + "up.style", s);
                tap(p + "up", x);
                x = styled_conv(ctx, x, st, convs_[2 * l + 1], false, &s);
                tap(p + "conv.style", s);
                tap(p + "conv", x);

                // ToFlow: 1x1 modulated conv without demodulation, + bias, + up(previous)
                const ToFlow& tf = to_flows_[l];
                const int64_t C = x->ne[2], H = x->ne[1], W = x->ne[0];
                s = ggml_add(ctx, ggml_mul_mat(ctx, tf.mod_w, st), tf.mod_b);
                tap(p + "toflow.style", s);
                ggml_tensor* fr = ggml_conv_2d_direct(ctx, tf.w, ggml_mul(ctx, x, ggml_reshape_4d(ctx, s, 1, 1, C, B)),
                                                      1, 1, 0, 0, 1, 1);
                fr = ggml_add(ctx, fr, ggml_reshape_4d(ctx, tf.b, 1, 1, tf.b->ne[0], 1));
                if (flow_raw) fr = ggml_add(ctx, fr, upsample(ctx, flow_raw));
                flow_raw = fr;
                tap(p + "flow_raw", fr);
                ggml_tensor* samp = ggml_cont(ctx, ggml_view_4d(ctx, fr, W, H, 2, B, fr->nb[1], fr->nb[2], fr->nb[3], 0));
                ggml_tensor* flow = ggml_add(ctx, ggml_tanh(ctx, samp), grids_[l]);
                tap(p + "flow", flow);
                ggml_tensor* mask = ggml_cont(ctx, ggml_view_4d(ctx, fr, W, H, 1, B, fr->nb[1], fr->nb[2], fr->nb[3],
                                                                2 * fr->nb[2]));
                mask = ggml_sigmoid(ctx, mask);
                ggml_tensor* warp = ggml_mul(ctx, grid_sample_bilinear_zeros(ctx, feats_[l], flow), mask);
                tap(p + "warp", warp);
                x = ggml_add(ctx, warp, ggml_mul(ctx, x, ggml_scale_bias(ctx, mask, -1.0f, 1.0f)));
                tap(p + "out", x);

                // ToRGB (not modulated): 1x1 conv, FusedLeakyReLU, + bias, + up(skip)
                const ToRGB& tr = to_rgbs_[l];
                ggml_tensor* r = lrelu_bias(ctx, ggml_conv_2d_direct(ctx, tr.w, warp, 1, 1, 0, 0, 1, 1), tr.act_b);
                r = ggml_add(ctx, r, ggml_reshape_4d(ctx, tr.b, 1, 1, tr.b->ne[0], 1));
                if (skip) r = ggml_add(ctx, r, upsample(ctx, skip));
                skip = r;
                tap(p + "skip", r);
            }
            img = skip;
            ggml_set_output(img);
            for (auto& t : tapped) ggml_build_forward_expand(g.gf, t.second);
            ggml_build_forward_expand(g.gf, img);
            return true;
        };
        auto read = [&](Backend::Graph&) {
            ggml_backend_tensor_get(img, pixels.data() + (size_t)f0 * frame, 0, ggml_nbytes(img));
            if (!want_taps) return;
            (*taps)["const"] = get_host(const_in_);
            for (auto& t : tapped) {
                std::vector<float> v = get_frame0(t.second);
                if (t.first.size() > 5 && t.first.compare(t.first.size() - 5, 5, ".flow") == 0) {
                    // planar [2, H, W] -> [H, W, 2]
                    const size_t hw = v.size() / 2;
                    std::vector<float> il(v.size());
                    for (size_t i = 0; i < hw; ++i) { il[2 * i] = v[i]; il[2 * i + 1] = v[hw + i]; }
                    v.swap(il);
                }
                (*taps)[t.first] = std::move(v);
            }
        };
        if (!be_.compute(build, read)) return false;
    }
    return true;
}

void MotionModel::to_u8(const float* pixels, int n, int h, int w, std::vector<uint8_t>& out) {
    const size_t total = (size_t)n * 3 * h * w;
    float lo = 1.0f, hi = -1.0f;
    for (size_t i = 0; i < total; ++i) {
        const float v = std::min(1.0f, std::max(-1.0f, pixels[i]));
        lo = std::min(lo, v);
        hi = std::max(hi, v);
    }
    const float denom = (hi - lo) + 1e-8f;
    out.assign(total, 0);
    const size_t plane = (size_t)h * w;
    for (int f = 0; f < n; ++f)
        for (int c = 0; c < 3; ++c) {
            const float* src = pixels + ((size_t)f * 3 + c) * plane;
            uint8_t* dst = out.data() + (size_t)f * 3 * plane + c;
            for (size_t i = 0; i < plane; ++i) {
                const float v = std::min(1.0f, std::max(-1.0f, src[i]));
                const float s = ((v - lo) / denom) * 255.0f;
                dst[i * 3] = (uint8_t)(int)s;   // truncation, s is in [0, 255]
            }
        }
}

} // namespace af
