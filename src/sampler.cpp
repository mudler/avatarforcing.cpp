#include "sampler.hpp"
#include "common.hpp"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace af {

// ---------------------------------------------------------------------------
// Schedule

namespace {

// numpy.linspace(start, stop, num) in float64.
std::vector<double> linspace(double start, double stop, int num) {
    std::vector<double> y(num);
    if (num == 1) { y[0] = start; return y; }
    const double step = (stop - start) / (num - 1);
    for (int i = 0; i < num; ++i) y[i] = i * step + start;
    y[num - 1] = stop;
    return y;
}

} // namespace

bool Schedule::build(int nfe_, int num_train, const std::vector<float>& sched_t,
                     const std::vector<float>& sched_s) {
    if (nfe_ < 3) { AF_LOG("sampler: nfe must be >= 3 (got %d)", nfe_); return false; }
    if (sched_t.empty() || sched_t.size() != sched_s.size()) { AF_LOG("sampler: bad sigma tables"); return false; }
    nfe = nfe_;
    const int n = nfe - 1;
    // The reference builds the list in float64, then stores it as float32.
    steps.clear();
    for (double v : linspace((double)num_train, 0.0, n)) steps.push_back((float)v);

    // sigma(t): argmin of |timesteps - t| in float64 (first index on ties),
    // forced to 0 at t = 0.
    auto lookup = [&](float t, int* idx) {
        const double td = t;
        int best = 0;
        double bd = std::fabs((double)sched_t[0] - td);
        for (size_t k = 1; k < sched_t.size(); ++k) {
            const double d = std::fabs((double)sched_t[k] - td);
            if (d < bd) { bd = d; best = (int)k; }
        }
        if (idx) *idx = best;
        return t == 0.0f ? 0.0 : (double)sched_s[best];
    };
    sigma_index.assign(n, 0);
    sigma.assign(n, 0.0);
    sigma_next.assign(n, 0.0);
    ratio.assign(n, 0.0);
    c_x0.assign(n, 0.0);
    keep.assign(n, 0);
    for (int i = 0; i < n; ++i) {
        const float t = steps[i];
        const float tn = i < n - 1 ? steps[i + 1] : steps[i];
        sigma[i] = lookup(t, &sigma_index[i]);
        sigma_next[i] = lookup(tn, nullptr);
        ratio[i] = sigma[i] > 0 ? sigma_next[i] / std::max(sigma[i], 1e-8) : 0.0;
        c_x0[i] = (1.0 - sigma_next[i]) - ratio[i] * (1.0 - sigma[i]);
        keep[i] = (t == 0.0f && tn == 0.0f) ? 1 : 0;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Sampler

bool Sampler::init(int nfe, float cfg_scale) {
    const FlowConfig& c = model_.config();
    if (nfe <= 0) nfe = (int)c.default_nfe;
    cfg_ = cfg_scale < 0 ? c.cfg_audio_scale : cfg_scale;

    auto host = [&](const char* name, std::vector<float>& out) {
        ggml_tensor* t = model_.loader().tensor(name);
        if (!t || t->type != GGML_TYPE_F32 || ggml_nelements(t) != (int64_t)c.sigma_table_len) {
            AF_LOG("sampler: missing or malformed tensor %s", name);
            return false;
        }
        out.resize(ggml_nelements(t));
        ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t));
        return true;
    };
    std::vector<float> sched_t, sched_s;
    if (!host("sched.timesteps", sched_t) || !host("sched.sigmas", sched_s)) return false;
    if (!sched_.build(nfe, (int)c.num_train_timesteps, sched_t, sched_s)) return false;

    // adaLN rows for every step, plus t = 0 for the context tokens and the
    // primer KV update.
    std::vector<float> ts = sched_.steps;
    if (std::find(ts.begin(), ts.end(), 0.0f) == ts.end()) ts.push_back(0.0f);
    if (!model_.timestep_tables(ts, tbl_)) return false;
    step_row_.clear();
    for (float t : sched_.steps) step_row_.push_back(tbl_.row(t));
    zero_row_ = tbl_.row(0.0f);

    wr_.assign(c.dim_w, 0.f);
    kv_.init(c, 2);
    prev_x_.clear();
    start_pos_ = 0;
    primed_ = false;
    return true;
}

void Sampler::set_identity(const float* wr) {
    wr_.assign(wr, wr + model_.config().dim_w);
}

bool Sampler::cond(int N, const float* wa, std::vector<float>& c) {
    const size_t E = model_.config().dim_a, per = (size_t)N * E;
    // Branches [uncond, audio]: the uncond branch has zero audio. The user
    // audio and the user motion are zero in both branches.
    std::vector<float> zeros(2 * per, 0.f), wa2(2 * per, 0.f);
    std::memcpy(wa2.data() + per, wa, per * sizeof(float));
    return model_.condition(2, N, zeros.data(), zeros.data(), wa2.data(), c);
}

bool Sampler::denoise(int N, int ctx, const std::vector<float>& c, std::vector<float>& x, int start_pos,
                      KvCache* kv, bool fold_update, SamplerTrace* trace) {
    const int D = (int)model_.config().dim_w;
    const size_t per = (size_t)N * D;
    const int n_steps = (int)sched_.steps.size();
    std::vector<float> x2(2 * per), v, a6, f2;
    std::vector<int> rows(N);
    if (trace) {
        trace->x.assign(n_steps, {});
        trace->v.assign(n_steps, {});
    }
    for (int i = 0; i < n_steps; ++i) {
        const bool last = i == n_steps - 1;
        const bool update = fold_update && last;
        // A step that keeps x does nothing unless it carries the KV update.
        if (sched_.keep[i] && !update) {
            if (trace) trace->x[i] = x;
            continue;
        }
        for (int j = 0; j < N; ++j) rows[j] = j < ctx ? zero_row_ : step_row_[i];
        tbl_.gather(rows, a6, f2);
        std::memcpy(x2.data(), x.data(), per * sizeof(float));
        std::memcpy(x2.data() + per, x.data(), per * sizeof(float));

        ForwardArgs a;
        a.n_tokens = N;
        a.n_branches = 2;
        a.x = x2.data();
        a.c = c.data();
        a.wr = wr_.data();
        a.adaln6 = a6.data();
        a.final2 = f2.data();
        a.start_pos = start_pos;
        a.kv = kv;
        a.update_kv = update;
        if (!model_.forward(a, v)) return false;

        if (!sched_.keep[i]) {
            // CFG in float32 as the reference does, then the closed-form
            // update in float64.
            const float s = cfg_;
            const double ratio = sched_.ratio[i], cx0 = sched_.c_x0[i], sig = sched_.sigma[i];
            for (int j = ctx; j < N; ++j)
                for (int d = 0; d < D; ++d) {
                    const size_t k = (size_t)j * D + d;
                    const float vu = v[k], va = v[per + k];
                    const float vc = vu + s * (va - vu);
                    const double xd = x[k];
                    const double x0 = xd - sig * (double)vc;
                    x[k] = (float)(ratio * xd + cx0 * x0);
                }
        }
        if (trace) {
            trace->x[i] = x;
            trace->v[i] = v;
        }
    }
    return true;
}

bool Sampler::primer(const float* wa, const float* noise, std::vector<float>& x_out, SamplerTrace* trace) {
    const FlowConfig& c = model_.config();
    const int N = (int)c.clip_frames, D = (int)c.dim_w;
    if (sched_.nfe == 0) { AF_LOG("sampler: init() not called"); return false; }

    std::vector<float> cnd;
    if (!cond(N, wa, cnd)) return false;
    std::vector<float> x(noise, noise + (size_t)N * D);
    // Denoising steps with no cache. The t = 0 step keeps x and is skipped.
    if (!denoise(N, 0, cnd, x, 0, nullptr, false, trace)) return false;

    // KV update: one forward at t = 0 from zero caches.
    kv_.init(c, 2);
    std::vector<int> rows(N, zero_row_);
    std::vector<float> a6, f2, x2(2 * (size_t)N * D), v;
    tbl_.gather(rows, a6, f2);
    std::memcpy(x2.data(), x.data(), x.size() * sizeof(float));
    std::memcpy(x2.data() + x.size(), x.data(), x.size() * sizeof(float));
    ForwardArgs a;
    a.n_tokens = N;
    a.n_branches = 2;
    a.x = x2.data();
    a.c = cnd.data();
    a.wr = wr_.data();
    a.adaln6 = a6.data();
    a.final2 = f2.data();
    a.start_pos = 0;
    a.kv = &kv_;
    a.update_kv = true;
    if (!model_.forward(a, v)) return false;
    if (trace) trace->kvupdate_v = v;

    prev_x_ = x;
    x_out = x;
    start_pos_ = (int)c.first_steady_start_pos;
    primed_ = true;
    return true;
}

bool Sampler::steady(const float* wa, const float* noise, std::vector<float>& latent, SamplerTrace* trace) {
    const FlowConfig& c = model_.config();
    const int ctx = (int)c.context_frames, nb = (int)c.block_frames, N = ctx + nb, D = (int)c.dim_w;
    if (!primed_) { AF_LOG("sampler: steady() before primer()"); return false; }

    std::vector<float> cnd;
    if (!cond(N, wa, cnd)) return false;
    // x = concat(previous x[-ctx:], noise).
    std::vector<float> x((size_t)N * D);
    const size_t prev_n = prev_x_.size() / D;
    std::memcpy(x.data(), prev_x_.data() + (prev_n - ctx) * D, (size_t)ctx * D * sizeof(float));
    std::memcpy(x.data() + (size_t)ctx * D, noise, (size_t)nb * D * sizeof(float));
    if (!denoise(N, ctx, cnd, x, start_pos_, &kv_, true, trace)) return false;

    latent.assign(x.begin() + (size_t)ctx * D, x.end());
    prev_x_ = std::move(x);
    start_pos_ += nb;
    return true;
}

} // namespace af
