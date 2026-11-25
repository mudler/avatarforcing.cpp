// Sampler loop parity against the PyTorch reference dumps.
//
// Runs the primer and the steady blocks of sampler_nfe{M}.npz end to end
// from the injected noise, with this engine's own x and caches carried from
// step to step, and compares x after every step, the latents and the caches
// after every block.
//
// Environment:
//   AVATARFORCING_TEST_GGUF      model GGUF (convert_avatarforcing_to_gguf.py)
//   AVATARFORCING_TEST_BASELINE  baseline GGUF (scripts/dumps_to_baseline.py,
//                                with the sampler runs)
//   AVATARFORCING_TEST_THREADS   CPU threads (default 8)
// Exits 77 (skip) when a path is unset or missing.
//
// Metric: max_abs = max |got - ref|, max_rel = max_abs / max |ref|.
// Gate: max_rel <= 1e-3 for x, latents and caches.

#include "backend.hpp"
#include "flow.hpp"
#include "sampler.hpp"

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr double kGate = 1e-3;

class Baseline {
public:
    bool open(const char* path) {
        gguf_init_params p = { /*.no_alloc =*/ false, /*.ctx =*/ &ctx_ };
        g_ = gguf_init_from_file(path, p);
        return g_ != nullptr;
    }
    ~Baseline() {
        if (g_) gguf_free(g_);
        if (ctx_) ggml_free(ctx_);
    }
    bool has(const std::string& name) const { return ggml_get_tensor(ctx_, name.c_str()) != nullptr; }
    bool has_key(const std::string& key) const { return gguf_find_key(g_, key.c_str()) >= 0; }
    const float* get(const std::string& name, size_t* n = nullptr) const {
        ggml_tensor* t = ggml_get_tensor(ctx_, name.c_str());
        if (!t) {
            std::fprintf(stderr, "baseline: missing tensor %s\n", name.c_str());
            std::exit(1);
        }
        if (n) *n = (size_t)ggml_nelements(t);
        return (const float*)t->data;
    }
    int64_t id(const std::string& key) const {
        int64_t i = gguf_find_key(g_, key.c_str());
        if (i < 0) { std::fprintf(stderr, "baseline: missing KV %s\n", key.c_str()); std::exit(1); }
        return i;
    }
    uint32_t u32(const std::string& key) const { return gguf_get_val_u32(g_, id(key)); }
    float f32(const std::string& key) const { return gguf_get_val_f32(g_, id(key)); }
    std::vector<int64_t> ints(const std::string& key) const {
        const int64_t i = id(key);
        std::vector<int64_t> out;
        const size_t n = gguf_get_arr_n(g_, i);
        const void* d = gguf_get_arr_data(g_, i);
        for (size_t k = 0; k < n; ++k) {
            switch (gguf_get_arr_type(g_, i)) {
                case GGUF_TYPE_INT32: out.push_back(((const int32_t*)d)[k]); break;
                case GGUF_TYPE_UINT32: out.push_back(((const uint32_t*)d)[k]); break;
                case GGUF_TYPE_INT64: out.push_back(((const int64_t*)d)[k]); break;
                default: std::fprintf(stderr, "baseline: %s: unexpected array type\n", key.c_str()); std::exit(1);
            }
        }
        return out;
    }

private:
    gguf_context* g_ = nullptr;
    ggml_context* ctx_ = nullptr;
};

struct Row {
    std::string name;
    double max_abs, max_rel;
    bool pass;
};
std::vector<Row> g_rows;

Row compare(const std::string& name, const float* got, size_t n_got, const float* ref, size_t n_ref,
            double gate) {
    if (n_got != n_ref) {
        std::fprintf(stderr, "%s: size mismatch got %zu ref %zu\n", name.c_str(), n_got, n_ref);
        return {name, INFINITY, INFINITY, false};
    }
    double max_abs = 0, max_ref = 0;
    bool finite = true;
    for (size_t i = 0; i < n_ref; ++i) {
        if (!std::isfinite(got[i])) finite = false;
        max_abs = std::max(max_abs, std::fabs((double)got[i] - (double)ref[i]));
        max_ref = std::max(max_ref, std::fabs((double)ref[i]));
    }
    const double rel = max_ref > 0 ? max_abs / max_ref : max_abs;
    return {name, max_abs, rel, finite && rel <= gate};
}

Row check(const std::string& name, const std::vector<float>& got, const Baseline& bl, const std::string& key) {
    size_t n = 0;
    const float* ref = bl.get(key, &n);
    Row r = compare(name, got.data(), got.size(), ref, n, kGate);
    g_rows.push_back(r);
    return r;
}

// Bit-for-bit equality (the reanchor self-check).
void check_same(const std::string& name, const std::vector<float>& a, const std::vector<float>& b) {
    bool same = a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
    Row r = compare(name, a.data(), a.size(), b.data(), b.size(), 0.0);
    r.pass = same;
    g_rows.push_back(r);
}

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

bool file_exists(const char* p) {
    FILE* f = p ? std::fopen(p, "rb") : nullptr;
    if (f) std::fclose(f);
    return f != nullptr;
}

// Worst values of the rows added since index `from` whose name contains
// `filter` (all rows when empty).
Row worst_since(size_t from, const std::string& label, const std::string& filter = "") {
    Row w{label, 0, 0, true};
    for (size_t i = from; i < g_rows.size(); ++i) {
        if (!filter.empty() && g_rows[i].name.find(filter) == std::string::npos) continue;
        w.max_abs = std::max(w.max_abs, g_rows[i].max_abs);
        w.max_rel = std::max(w.max_rel, g_rows[i].max_rel);
        w.pass = w.pass && g_rows[i].pass;
    }
    return w;
}

} // namespace

int main() {
    const char* gguf_path = std::getenv("AVATARFORCING_TEST_GGUF");
    const char* bl_path = std::getenv("AVATARFORCING_TEST_BASELINE");
    if (!file_exists(gguf_path) || !file_exists(bl_path)) {
        std::fprintf(stderr, "AVATARFORCING_TEST_GGUF / AVATARFORCING_TEST_BASELINE unset or missing; skipping\n");
        return 77;
    }
    const char* th = std::getenv("AVATARFORCING_TEST_THREADS");
    const int n_threads = th ? std::atoi(th) : 8;

    af::Backend be(n_threads);
    af::FlowModel model(be);
    if (!model.load(gguf_path)) { std::fprintf(stderr, "model load failed\n"); return 1; }
    const af::FlowConfig& cfg = model.config();
    const int D = (int)cfg.dim_w, nb = (int)cfg.block_frames, clip = (int)cfg.clip_frames;

    Baseline bl;
    if (!bl.open(bl_path)) { std::fprintf(stderr, "cannot open baseline\n"); return 1; }
    if (!bl.has_key("baseline.sampler_nfe")) {
        std::fprintf(stderr, "baseline has no sampler runs (rebuild it with --sampler-nfe)\n");
        return 1;
    }
    const std::vector<int64_t> nfes = bl.ints("baseline.sampler_nfe");
    if (nfes.empty()) { std::fprintf(stderr, "baseline: empty sampler_nfe\n"); return 1; }

    struct BlockRow { std::string name; Row x, v, kv; };
    std::vector<BlockRow> summary;
    auto summarize = [&](size_t from, const std::string& label) {
        summary.push_back({label, worst_since(from, "", ".x"), worst_since(from, "", ".v"),
                           worst_since(from, "", ".kv.")});
        // Latents count with x.
        Row l = worst_since(from, "", "latent");
        Row& x = summary.back().x;
        x.max_abs = std::max(x.max_abs, l.max_abs);
        x.max_rel = std::max(x.max_rel, l.max_rel);
        x.pass = x.pass && l.pass;
    };
    std::vector<std::string> timings;
    std::vector<Row> reanchor_rows;
    bool sched_ok = true;

    for (int64_t nfe : nfes) {
        const std::string p = "sampler" + std::to_string(nfe);
        const std::string kp = "baseline." + p;

        af::Sampler s(model);
        if (!s.init((int)nfe, bl.f32(kp + ".a_cfg_scale"))) return 1;
        s.set_identity(bl.get(p + ".r_s"));

        // Schedule: the step list and the sigma indices must match exactly.
        {
            size_t n = 0;
            const float* ref = bl.get(p + ".step_list", &n);
            const std::vector<int64_t> idx = bl.ints(kp + ".sigma_index");
            const af::Schedule& sc = s.schedule();
            bool ok = n == sc.steps.size() && idx.size() == sc.sigma_index.size();
            for (size_t i = 0; ok && i < n; ++i)
                ok = ref[i] == sc.steps[i] && idx[i] == sc.sigma_index[i];
            std::printf("nfe %lld schedule:", (long long)nfe);
            for (size_t i = 0; i < sc.steps.size(); ++i)
                std::printf(" t=%g idx=%d sigma=%.6f", sc.steps[i], sc.sigma_index[i], sc.sigma[i]);
            std::printf("  %s\n", ok ? "matches" : "MISMATCH");
            sched_ok = sched_ok && ok;
        }
        const int n_steps = (int)s.schedule().steps.size();

        auto check_steps = [&](const std::string& blk, const af::SamplerTrace& tr) {
            for (int i = 0; i < n_steps; ++i) {
                check(p + "." + blk + ".x." + std::to_string(i), tr.x[i], bl, p + "." + blk + ".x." + std::to_string(i));
                if (!tr.v[i].empty())
                    check(p + "." + blk + ".v." + std::to_string(i), tr.v[i], bl, p + "." + blk + ".v." + std::to_string(i));
            }
        };
        auto check_kv = [&](const std::string& blk) {
            const af::KvCache& kv = s.kv();
            check(p + "." + blk + ".kv.self_k", kv.self_k, bl, p + "." + blk + ".kv.self_k");
            check(p + "." + blk + ".kv.self_v", kv.self_v, bl, p + "." + blk + ".kv.self_v");
            check(p + "." + blk + ".kv.cross_k", kv.cross_k, bl, p + "." + blk + ".kv.cross_k");
            check(p + "." + blk + ".kv.cross_v", kv.cross_v, bl, p + "." + blk + ".kv.cross_v");
        };

        // Primer.
        size_t mark = g_rows.size();
        af::SamplerTrace tr;
        std::vector<float> x, primer_x;
        const float* primer_wa = bl.get(p + ".primer.wa");
        const float* primer_noise = bl.get(p + ".noise.primer");
        auto t0 = std::chrono::steady_clock::now();
        if (!s.primer(primer_wa, primer_noise, x, &tr)) return 1;
        const double t_primer = ms_since(t0);
        primer_x = x;
        check_steps("primer", tr);
        check(p + ".primer.kvupdate.v", tr.kvupdate_v, bl, p + ".primer.kvupdate.v");
        check(p + ".primer.x_final", x, bl, p + ".primer.x_final");
        check_kv("primer");
        summarize(mark, p + " primer");
        af::KvCache primer_kv = s.kv();

        // Steady blocks.
        const int n_blocks = (int)bl.u32(kp + ".steady_blocks");
        size_t n_noise = 0;
        const float* steady_noise = bl.get(p + ".noise.steady", &n_noise);
        std::vector<float> latent, block0_latent;
        for (int b = 0; b < n_blocks; ++b) {
            const std::string blk = "steady." + std::to_string(b);
            mark = g_rows.size();
            const int ref_pos = (int)bl.u32(kp + "." + blk + ".start_pos");
            if (s.start_pos() != ref_pos) {
                std::fprintf(stderr, "%s.%s: start_pos %d, reference %d\n", p.c_str(), blk.c_str(), s.start_pos(), ref_pos);
                g_rows.push_back({p + "." + blk + ".start_pos", INFINITY, INFINITY, false});
            }
            if (!s.steady(bl.get(p + "." + blk + ".wa"), steady_noise + (size_t)b * nb * D, latent, &tr)) return 1;
            if (b == 0) block0_latent = latent;
            check_steps(blk, tr);
            check(p + "." + blk + ".latent", latent, bl, p + "." + blk + ".latent");
            check_kv(blk);
            summarize(mark, p + " " + blk);
        }

        // Reanchor. The dumps do not cover it, so check it against this
        // engine's own run: a reanchor on the primer inputs must reset the
        // caches and start_pos and reproduce the primer and steady block 0
        // bit for bit.
        mark = g_rows.size();
        if (!s.reanchor(primer_wa, primer_noise, x)) return 1;
        check_same(p + ".reanchor.x", x, primer_x);
        check_same(p + ".reanchor.kv.self_k", s.kv().self_k, primer_kv.self_k);
        check_same(p + ".reanchor.kv.self_v", s.kv().self_v, primer_kv.self_v);
        check_same(p + ".reanchor.kv.cross_k", s.kv().cross_k, primer_kv.cross_k);
        check_same(p + ".reanchor.kv.cross_v", s.kv().cross_v, primer_kv.cross_v);
        if (s.start_pos() != (int)cfg.first_steady_start_pos)
            g_rows.push_back({p + ".reanchor.start_pos", INFINITY, INFINITY, false});
        if (!s.steady(bl.get(p + ".steady.0.wa"), steady_noise, latent)) return 1;
        check_same(p + ".reanchor.steady0.latent", latent, block0_latent);
        reanchor_rows.push_back(worst_since(mark, p + " reanchor (bit-exact)"));

        // Timing: one full steady block (condition, all steps, folded KV
        // update), median of 5 after one warm-up.
        std::vector<double> ms;
        for (int r = 0; r < 6; ++r) {
            t0 = std::chrono::steady_clock::now();
            if (!s.steady(bl.get(p + ".steady.1.wa"), steady_noise + (size_t)nb * D, latent)) return 1;
            if (r > 0) ms.push_back(ms_since(t0));
        }
        std::sort(ms.begin(), ms.end());
        char buf[256];
        std::snprintf(buf, sizeof(buf),
                      "nfe %lld (%d threads): primer %.0f ms (%d forwards + KV update), steady block %.1f ms "
                      "(median of 5, %d forwards), %.0f ms per frame",
                      (long long)nfe, n_threads, t_primer, n_steps - 1, ms[ms.size() / 2], n_steps,
                      ms[ms.size() / 2] / nb);
        timings.push_back(buf);
        (void)clip;
    }

    bool all = sched_ok;
    std::printf("%-40s %12s %12s  %s\n", "tensor", "max_abs", "max_rel", "gate");
    for (const Row& r : g_rows) {
        std::printf("%-40s %12.3e %12.3e  %s\n", r.name.c_str(), r.max_abs, r.max_rel, r.pass ? "pass" : "FAIL");
        all = all && r.pass;
    }
    std::printf("\nper block, worst tensor (max_abs / max_rel): x = x after each step and latents, "
                "v = raw transformer outputs, kv = caches after the block\n");
    std::printf("%-22s %21s %21s %21s\n", "block", "x", "v", "kv");
    for (const BlockRow& r : summary)
        std::printf("%-22s %10.2e/%-10.2e %10.2e/%-10.2e %10.2e/%-10.2e %s\n", r.name.c_str(), r.x.max_abs,
                    r.x.max_rel, r.v.max_abs, r.v.max_rel, r.kv.max_abs, r.kv.max_rel,
                    r.x.pass && r.v.pass && r.kv.pass ? "pass" : "FAIL");
    for (const Row& r : reanchor_rows) std::printf("%-22s %s\n", r.name.c_str(), r.pass ? "pass" : "FAIL");
    for (const std::string& t : timings) std::printf("timing: %s\n", t.c_str());
    std::printf("%s: %zu tensors, gate max_rel <= %.0e (reanchor: bit-exact)\n", all ? "PASS" : "FAIL",
                g_rows.size(), kGate);
    return all ? 0 : 1;
}
