#include "grid_sample.hpp"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <vector>

// Arithmetic, chosen to match the PyTorch CPU kernel bit for bit
// (aten/src/ATen/native/cpu/GridSamplerKernel.cpp, float32, built with FMA
// contraction, the default x86-64 AVX2 build):
//
//   ix = (gx + 1) * (W_in / 2) - 0.5          (unnormalize, align_corners = false)
//   x0 = floor(ix), w = ix - x0, e = 1 - w    (same for y: y0, n, s = 1 - n)
//   nw = s*e, ne = s*w, sw = n*e, se = n*w
//   out = fma(v_se, se, fma(v_sw, sw, fma(v_ne, ne, v_nw * nw)))
//
// A corner outside the input reads 0 (padding_mode = zeros).
// This file is built with -ffp-contract=off so that the compiler adds no
// other fused operations; the fmaf calls are explicit.

namespace af {

namespace {

struct RowParams {
    std::vector<int> x0, y0;
    std::vector<float> nw, ne, sw, se;
    void resize(int n) {
        x0.resize(n); y0.resize(n);
        nw.resize(n); ne.resize(n); sw.resize(n); se.resize(n);
    }
};

inline void corner_weights(float gx, float gy, float half_w, float half_h,
                           int& x0, int& y0, float& nw, float& ne, float& sw, float& se) {
    const float ix = (gx + 1.0f) * half_w - 0.5f;
    const float iy = (gy + 1.0f) * half_h - 0.5f;
    const float fx = std::floor(ix);
    const float fy = std::floor(iy);
    const float w = ix - fx;
    const float e = 1.0f - w;
    const float n = iy - fy;
    const float s = 1.0f - n;
    nw = s * e;
    ne = s * w;
    sw = n * e;
    se = n * w;
    x0 = (int)fx;
    y0 = (int)fy;
}

// One output row (fixed b, y) for all channels. Rows are independent, so the
// threads split the B * H rows.
void sample_row(const float* feat_b, int w_in, int h_in, int c,
                const float* gx_row, const float* gy_row, int w,
                float* out_row, size_t out_c_stride, RowParams& rp) {
    const float half_w = (float)w_in * 0.5f;
    const float half_h = (float)h_in * 0.5f;
    for (int x = 0; x < w; ++x)
        corner_weights(gx_row[x], gy_row[x], half_w, half_h, rp.x0[x], rp.y0[x],
                       rp.nw[x], rp.ne[x], rp.sw[x], rp.se[x]);
    const size_t plane = (size_t)w_in * h_in;
    for (int ch = 0; ch < c; ++ch) {
        const float* src = feat_b + ch * plane;
        float* dst = out_row + ch * out_c_stride;
        for (int x = 0; x < w; ++x) {
            const int x0 = rp.x0[x], y0 = rp.y0[x], x1 = x0 + 1, y1 = y0 + 1;
            const bool in_x0 = x0 >= 0 && x0 < w_in, in_x1 = x1 >= 0 && x1 < w_in;
            const bool in_y0 = y0 >= 0 && y0 < h_in, in_y1 = y1 >= 0 && y1 < h_in;
            const float v_nw = (in_y0 && in_x0) ? src[(size_t)y0 * w_in + x0] : 0.0f;
            const float v_ne = (in_y0 && in_x1) ? src[(size_t)y0 * w_in + x1] : 0.0f;
            const float v_sw = (in_y1 && in_x0) ? src[(size_t)y1 * w_in + x0] : 0.0f;
            const float v_se = (in_y1 && in_x1) ? src[(size_t)y1 * w_in + x1] : 0.0f;
            float acc = v_nw * rp.nw[x];
            acc = std::fmaf(v_ne, rp.ne[x], acc);
            acc = std::fmaf(v_sw, rp.sw[x], acc);
            acc = std::fmaf(v_se, rp.se[x], acc);
            dst[x] = acc;
        }
    }
}

void run_rows(const float* feat, int w_in, int h_in, int c, int b_feat,
              const float* grid, int w, int h, float* out, int64_t row_begin, int64_t row_end) {
    RowParams rp;
    rp.resize(w);
    const size_t feat_batch = (size_t)w_in * h_in * c;
    const size_t grid_plane = (size_t)w * h;
    for (int64_t r = row_begin; r < row_end; ++r) {
        const int bi = (int)(r / h), y = (int)(r % h);
        const float* feat_b = feat + (b_feat == 1 ? 0 : (size_t)bi * feat_batch);
        const float* g = grid + (size_t)bi * 2 * grid_plane;
        float* o = out + (size_t)bi * c * grid_plane + (size_t)y * w;
        sample_row(feat_b, w_in, h_in, c, g + (size_t)y * w, g + grid_plane + (size_t)y * w, w,
                   o, grid_plane, rp);
    }
}

void grid_sample_op(ggml_tensor* dst, int ith, int nth, void* /*userdata*/) {
    const ggml_tensor* feat = dst->src[0];
    const ggml_tensor* grid = dst->src[1];
    const int w = (int)dst->ne[0], h = (int)dst->ne[1], b = (int)dst->ne[3];
    const int64_t rows = (int64_t)b * h;
    const int64_t per = (rows + nth - 1) / nth;
    const int64_t r0 = std::min<int64_t>(rows, per * ith);
    const int64_t r1 = std::min<int64_t>(rows, r0 + per);
    run_rows((const float*)feat->data, (int)feat->ne[0], (int)feat->ne[1], (int)feat->ne[2],
             (int)feat->ne[3], (const float*)grid->data, w, h, (float*)dst->data, r0, r1);
}

} // namespace

ggml_tensor* grid_sample_bilinear_zeros(ggml_context* ctx, ggml_tensor* feat, ggml_tensor* grid) {
    GGML_ASSERT(feat->type == GGML_TYPE_F32 && grid->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(feat) && ggml_is_contiguous(grid));
    GGML_ASSERT(grid->ne[2] == 2);
    GGML_ASSERT(feat->ne[3] == 1 || feat->ne[3] == grid->ne[3]);
    ggml_tensor* args[2] = { feat, grid };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, grid->ne[0], grid->ne[1], feat->ne[2], grid->ne[3],
                          args, 2, grid_sample_op, GGML_N_TASKS_MAX, nullptr);
}

void grid_sample_bilinear_zeros_host(const float* feat, int w_in, int h_in, int c, int b_feat,
                                     const float* grid, int w, int h, int b, float* out) {
    run_rows(feat, w_in, h_in, c, b_feat, grid, w, h, out, 0, (int64_t)b * h);
}

} // namespace af
