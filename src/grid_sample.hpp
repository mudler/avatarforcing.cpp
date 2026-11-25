#pragma once

struct ggml_context;
struct ggml_tensor;

namespace af {

// Bilinear grid sampling with zero padding and align_corners = false, the
// mode of torch.nn.functional.grid_sample that the motion decoder uses.
//
//   feat: [W_in, H_in, C, B_f] F32, contiguous. B_f is 1 (broadcast over the
//         batch of grid) or equal to the batch of grid.
//   grid: [W, H, 2, B] F32, contiguous. Plane 0 holds x (along W_in), plane 1
//         holds y (along H_in), both normalized to [-1, 1]. This is the
//         channel-planar form of the [B, H, W, 2] grid that PyTorch takes.
//   result: [W, H, C, B] F32.
//
// ggml has no grid_sample operator. This is a CPU custom op
// (ggml_custom_4d), so the graph that uses it must run on the CPU backend.
// A GPU backend needs its own kernel with the same arithmetic (see
// grid_sample.cpp for the exact rounding).
ggml_tensor* grid_sample_bilinear_zeros(ggml_context* ctx, ggml_tensor* feat, ggml_tensor* grid);

// Host reference of the same computation, for tests. Same layouts as above.
void grid_sample_bilinear_zeros_host(const float* feat, int w_in, int h_in, int c, int b_feat,
                                     const float* grid, int w, int h, int b, float* out);

} // namespace af
