#pragma once
#include "flow.hpp"

#include <vector>

namespace af {

// Flow-matching schedule for one nfe: the step list, the sigma of each step
// and the closed-form update coefficients of the denoised tokens.
struct Schedule {
    int nfe = 0;
    std::vector<float> steps;       // linspace(num_train_timesteps, 0, nfe - 1), float32
    std::vector<int> sigma_index;   // argmin index into the sigma table, per step
    // Closed-form update of the tokens at time steps[i], in float64:
    //   x0 = x - sigma[i] * v;  x_next = ratio[i] * x + c_x0[i] * x0
    // with ratio = sigma_next / sigma and c_x0 = (1 - sigma_next) - ratio * (1 - sigma).
    // keep[i]: current and next time are both 0, x is unchanged (no-op step).
    // Context tokens (time 0, next time 0) always keep x.
    std::vector<double> sigma, sigma_next, ratio, c_x0;
    std::vector<char> keep;

    // Builds the schedule as the reference solve_closed() does: sigma by
    // argmin in float64 over the float32 tables, sigma = 0 at t = 0.
    bool build(int nfe, int num_train_timesteps, const std::vector<float>& sched_timesteps,
               const std::vector<float>& sched_sigmas);
};

// Optional per-step outputs of one primer or steady block, for parity tests.
struct SamplerTrace {
    std::vector<std::vector<float>> x;   // x after step i, [N, dim_w]
    std::vector<std::vector<float>> v;   // raw transformer output of step i, [2, N, dim_w]
                                         // (empty for a step that is skipped)
    std::vector<float> kvupdate_v;       // primer only: output of the KV-update forward
};

// The streaming sampler loop of the live app: a primer of clip_frames tokens,
// then steady blocks of block_frames new tokens with context_frames tokens
// carried over, a rolling KV cache, and two-branch CFG [uncond, audio].
class Sampler {
public:
    explicit Sampler(FlowModel& model) : model_(model) {}

    // nfe <= 0 uses sampler.default_nfe; cfg_scale < 0 uses cfg.audio_scale.
    bool init(int nfe = 0, float cfg_scale = -1.0f);

    // Identity code r_s [dim_w], shared by every token and branch.
    void set_identity(const float* wr);

    // Primer (session start and reanchor): resets the caches, denoises
    // clip_frames tokens with no cache read, then writes the caches with one
    // forward at t = 0. wa [clip_frames, dim_a] audio features, noise
    // [clip_frames, dim_w]. x_out [clip_frames, dim_w]. Afterwards the next
    // steady block starts at RoPE position first_steady_start_pos.
    bool primer(const float* wa, const float* noise, std::vector<float>& x_out,
                SamplerTrace* trace = nullptr);

    // A reanchor is a primer on the last clip_frames frames of features.
    bool reanchor(const float* wa, const float* noise, std::vector<float>& x_out,
                  SamplerTrace* trace = nullptr) {
        return primer(wa, noise, x_out, trace);
    }

    // One steady block: wa [context_frames + block_frames, dim_a] (features of
    // the two context frames, then the new frames), noise [block_frames, dim_w].
    // latent [block_frames, dim_w]: the new tokens, the decoder input. The KV
    // update is folded into the last (t = 0) step.
    bool steady(const float* wa, const float* noise, std::vector<float>& latent,
                SamplerTrace* trace = nullptr);

    bool primed() const { return primed_; }
    int start_pos() const { return start_pos_; }
    int nfe() const { return sched_.nfe; }
    float cfg_scale() const { return cfg_; }
    const Schedule& schedule() const { return sched_; }
    // The caches live on the compute device; this refreshes the host copy.
    const KvCache& kv() { kv_.sync_host(); return kv_; }
    const AdaLNTable& adaln() const { return tbl_; }

private:
    // Denoise x [N, dim_w] in place over all steps. ctx: leading tokens that
    // stay at t = 0. kv: cache (nullptr: none); fold_update: write
    // the cache in the last step.
    bool denoise(int N, int ctx, const std::vector<float>& c, std::vector<float>& x, int start_pos,
                 KvCache* kv, bool fold_update, SamplerTrace* trace);
    bool cond(int N, const float* wa, std::vector<float>& c);

    FlowModel& model_;
    Schedule sched_;
    AdaLNTable tbl_;
    std::vector<int> step_row_;   // adaLN table row per step
    int zero_row_ = -1;           // adaLN table row of t = 0
    float cfg_ = 0.f;
    std::vector<float> wr_;
    KvCache kv_;
    std::vector<float> prev_x_;   // x of the last primer or steady block
    int start_pos_ = 0;
    bool primed_ = false;
};

} // namespace af
