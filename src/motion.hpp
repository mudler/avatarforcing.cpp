#pragma once
#include "backend.hpp"
#include "model_loader.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct ggml_cgraph;
struct ggml_backend_buffer;
typedef struct ggml_backend_buffer* ggml_backend_buffer_t;

namespace af {

// Motion autoencoder configuration, read from the `avatarforcing.mae.*` KV
// (see docs/conversion.md).
struct MaeConfig {
    uint32_t size = 0, log_size = 0;
    uint32_t style_dim = 0, motion_dim = 0;
    uint32_t enc_n_resblocks = 0;
    std::vector<uint32_t> enc_channels;   // per resolution, size .. 4 (log_size - 1 values)
    uint32_t dec_n_levels = 0;
    std::vector<uint32_t> dec_channels;   // per resolution, 4 .. size (log_size - 1 values)
    std::vector<int32_t> blur_kernel;     // 1-D taps, outer product normalized to sum 1
    float leaky_slope = 0.f, act_gain = 0.f, demod_eps = 0.f, resblock_scale = 0.f;
    int32_t pad_enc_down_conv3[2] = {0, 0};
    int32_t pad_enc_down_skip1[2] = {0, 0};
    int32_t pad_dec_up_conv[2] = {0, 0};
    int32_t pad_dec_upsample[2] = {0, 0};
    std::string grid_sample;
    uint32_t block_frames = 0;            // frames per steady block (u8 conversion unit)
};

// Debug outputs keyed like the reference dumps.
using MotionTaps = std::map<std::string, std::vector<float>>;

// Result of the encoder for one portrait.
struct Portrait {
    std::vector<float> s_r;     // [style_dim] appearance code
    std::vector<float> alpha;   // [motion_dim] enc.fc(s_r)
    std::vector<float> r_s;     // [style_dim] Direction(alpha) = Q alpha, the flow transformer identity input
    // Decoder feature maps, level 0 (8x8) .. dec_n_levels - 1 (size x size),
    // each [C, H, W] row-major (ggml [W, H, C]).
    std::vector<std::vector<float>> feats;
};

// Motion autoencoder: encoder (once per portrait) and decoder (per frame).
// All arithmetic runs in F32, on the CPU or a GPU backend. grid_sample is
// GGML_OP_GRID_SAMPLE of the patched ggml (grid_sample.hpp).
class MotionModel {
public:
    explicit MotionModel(Backend& backend) : be_(backend) {}
    ~MotionModel();
    MotionModel(const MotionModel&) = delete;
    MotionModel& operator=(const MotionModel&) = delete;

    bool load(const std::string& gguf_path);
    const MaeConfig& config() const { return cfg_; }

    // image: [3, size, size] row-major, values in [-1, 1].
    // taps: enc.convs.{0..}, s_r, alpha, r_s, feat.{l}.
    bool encode(const float* image, Portrait& out, MotionTaps* taps = nullptr);

    // Upload the portrait features and s_r to the backend for decode().
    bool set_portrait(const Portrait& p);

    // Frames per decoder graph. Peak memory grows linearly with it; the
    // default 1 keeps the 512x512 levels small. 0 means all frames at once.
    void set_chunk_frames(int n) { chunk_frames_ = n; }

    // Decode n frames with style[f] = s_r + latent[f]. latent: [n, style_dim].
    // pixels: [n, 3, size, size] row-major, not clamped.
    // taps (frame 0 only): const, conv1, conv1.style, level.{l}.{up, up.style,
    // conv, conv.style, toflow.style, flow_raw, flow, warp, out, skip}.
    // level.{l}.flow is [H, W, 2] like the reference.
    bool decode(const float* latent, int n, std::vector<float>& pixels, MotionTaps* taps = nullptr);

    // Same with the styles given directly. style: [n, style_dim].
    bool decode_styles(const float* style, int n, std::vector<float>& pixels,
                       MotionTaps* taps = nullptr);

    // The app's uint8 conversion of one block of frames: clamp to [-1, 1],
    // block-wide min/max scaling to [0, 255] with eps 1e-8, truncate.
    // pixels: [n, 3, H, W]; out: [n, H, W, 3].
    static void to_u8(const float* pixels, int n, int h, int w, std::vector<uint8_t>& out);

private:
    struct StyledConv {
        ggml_tensor *w, *wsq, *mod_w, *mod_b, *act_b;
        ggml_tensor* gbias = nullptr;   // act_gain * act_b as [1, 1, C, 1] (derived)
        ggml_tensor* phase[2][2] = {{nullptr, nullptr}, {nullptr, nullptr}};  // transposed conv, CPU graph
        ggml_tensor* wt = nullptr;   // GPU graph: the 3x3 weight as [OC, KW, KH, IC] (derived)
    };
    struct ToFlow { ggml_tensor *w, *mod_w, *mod_b, *b; };
    struct ToRGB { ggml_tensor *w, *act_b, *b; };
    struct ResBlock { ggml_tensor *c1_w, *c1_b, *c2_w, *c2_b, *skip_w; };

    bool read_config(const std::string& path);
    bool build_derived();
    ggml_tensor* conv2d(ggml_context* ctx, ggml_tensor* w, ggml_tensor* x, int s0, int s1, int p0, int p1,
                        int d0, int d1) const;
    ggml_tensor* lrelu_bias(ggml_context* ctx, ggml_tensor* x, ggml_tensor* bias) const;
    ggml_tensor* blur(ggml_context* ctx, ggml_tensor* x, ggml_tensor* kernel, int pad) const;
    ggml_tensor* upsample(ggml_context* ctx, ggml_tensor* x) const;
    ggml_tensor* styled_conv(ggml_context* ctx, ggml_cgraph* gf, ggml_tensor* x, ggml_tensor* style,
                             const StyledConv& c, bool up, ggml_tensor** s_out) const;
    ggml_tensor* styled_conv_gpu(ggml_context* ctx, ggml_cgraph* gf, ggml_tensor* x, ggml_tensor* s,
                                 const StyledConv& c, bool up) const;

    Backend& be_;
    ModelLoader loader_;
    MaeConfig cfg_;
    int chunk_frames_ = 1;
    bool im2col_ = false;   // conv path, see conv2d()
    bool gpu_graph_ = false;   // decoder graph for the GPU backend, see styled_conv()

    // encoder
    ggml_tensor *enc_in_w_ = nullptr, *enc_in_b_ = nullptr, *enc_out_w_ = nullptr, *q_ = nullptr;
    std::vector<ResBlock> res_;
    std::vector<std::pair<ggml_tensor*, ggml_tensor*>> fc_;
    // decoder
    ggml_tensor* const_in_ = nullptr;
    StyledConv conv1_{};
    std::vector<StyledConv> convs_;   // 2 per level: transposed (up), plain
    std::vector<ToFlow> to_flows_;
    std::vector<ToRGB> to_rgbs_;

    // Derived constants: transposed-conv phase kernels, blur kernels, base grids.
    ggml_context* dctx_ = nullptr;
    ggml_backend_buffer_t dbuf_ = nullptr;
    ggml_tensor *blur_down_ = nullptr, *blur_up_ = nullptr;
    std::vector<ggml_tensor*> grids_;

    // Portrait state for decode().
    ggml_context* pctx_ = nullptr;
    ggml_backend_buffer_t pbuf_ = nullptr;
    std::vector<ggml_tensor*> feats_;
    std::vector<float> s_r_;
};

} // namespace af
