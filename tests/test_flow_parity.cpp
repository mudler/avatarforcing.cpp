// Flow transformer parity against the PyTorch reference dumps.
//
// Environment:
//   AVATARFORCING_TEST_GGUF      model GGUF (convert_avatarforcing_to_gguf.py)
//   AVATARFORCING_TEST_BASELINE  baseline GGUF (scripts/dumps_to_baseline.py)
//   AVATARFORCING_TEST_THREADS   CPU threads (default 8)
// Exits 77 (skip) when a path is unset or missing.
//
// Metric per module: max_abs = max |got - ref|, max_rel = max_abs / max |ref|.
// Gate: max_rel <= 2e-5.

#include "backend.hpp"
#include "flow.hpp"

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

constexpr double kGate = 2e-5;

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
    // Tensor data in row-major order (the numpy layout of the dump).
    const float* get(const std::string& name, size_t* n = nullptr) const {
        ggml_tensor* t = ggml_get_tensor(ctx_, name.c_str());
        if (!t) {
            std::fprintf(stderr, "baseline: missing tensor %s\n", name.c_str());
            std::exit(1);
        }
        if (n) *n = (size_t)ggml_nelements(t);
        return (const float*)t->data;
    }
    std::vector<float> vec(const std::string& name) const {
        size_t n = 0;
        const float* p = get(name, &n);
        return std::vector<float>(p, p + n);
    }
    uint32_t u32(const char* key) const {
        int64_t id = gguf_find_key(g_, key);
        if (id < 0) { std::fprintf(stderr, "baseline: missing KV %s\n", key); std::exit(1); }
        return gguf_get_val_u32(g_, id);
    }
    bool flag(const std::string& key) const {
        int64_t id = gguf_find_key(g_, key.c_str());
        if (id < 0) { std::fprintf(stderr, "baseline: missing KV %s\n", key.c_str()); std::exit(1); }
        return gguf_get_val_bool(g_, id);
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

void check(const std::string& name, const float* got, size_t n_got, const float* ref, size_t n_ref) {
    if (n_got != n_ref) {
        std::fprintf(stderr, "%s: size mismatch got %zu ref %zu\n", name.c_str(), n_got, n_ref);
        g_rows.push_back({name, INFINITY, INFINITY, false});
        return;
    }
    double max_abs = 0, max_ref = 0;
    bool finite = true;
    for (size_t i = 0; i < n_ref; ++i) {
        if (!std::isfinite(got[i])) finite = false;
        max_abs = std::max(max_abs, std::fabs((double)got[i] - (double)ref[i]));
        max_ref = std::max(max_ref, std::fabs((double)ref[i]));
    }
    const double rel = max_ref > 0 ? max_abs / max_ref : max_abs;
    g_rows.push_back({name, max_abs, rel, finite && rel <= kGate});
}

void check(const std::string& name, const std::vector<float>& got, const Baseline& bl, const std::string& key) {
    size_t n = 0;
    const float* ref = bl.get(key, &n);
    check(name, got.data(), got.size(), ref, n);
}

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// adaLN row indices per token for one dumped forward.
std::vector<int> adaln_rows(const af::AdaLNTable& tbl, const std::string& tag, int N, int ctx_frames,
                            float step0_t) {
    const int r0 = tbl.row(0.0f), rs = tbl.row(step0_t);
    std::vector<int> rows(N, rs);
    if (tag == "primer_kvupdate" || tag == "steady0_final") std::fill(rows.begin(), rows.end(), r0);
    if (tag == "steady0_step0")
        for (int j = 0; j < ctx_frames; ++j) rows[j] = r0;
    return rows;
}

// [N, k * D] (token-major) to [k, N, D] (the dump layout).
std::vector<float> chunk_major(const std::vector<float>& tok, int N, int k, int D) {
    std::vector<float> out(tok.size());
    for (int j = 0; j < N; ++j)
        for (int c = 0; c < k; ++c)
            std::memcpy(&out[((size_t)c * N + j) * D], &tok[((size_t)j * k + c) * D], D * sizeof(float));
    return out;
}

void load_kv(const Baseline& bl, const std::string& prefix, af::KvCache& kv) {
    kv.self_k = bl.vec(prefix + "self_k");
    kv.self_v = bl.vec(prefix + "self_v");
    kv.cross_k = bl.vec(prefix + "cross_k");
    kv.cross_v = bl.vec(prefix + "cross_v");
}

void check_kv(const std::string& name, const af::KvCache& kv, const Baseline& bl, const std::string& prefix) {
    check(name + ".self_k", kv.self_k, bl, prefix + "self_k");
    check(name + ".self_v", kv.self_v, bl, prefix + "self_v");
    check(name + ".cross_k", kv.cross_k, bl, prefix + "cross_k");
    check(name + ".cross_v", kv.cross_v, bl, prefix + "cross_v");
}

bool file_exists(const char* p) {
    FILE* f = p ? std::fopen(p, "rb") : nullptr;
    if (f) std::fclose(f);
    return f != nullptr;
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
    auto t_load = std::chrono::steady_clock::now();
    if (!model.load(gguf_path)) { std::fprintf(stderr, "model load failed\n"); return 1; }
    std::fprintf(stderr, "load: %.0f ms\n", ms_since(t_load));
    const af::FlowConfig& cfg = model.config();
    const int D = (int)cfg.dim_h, B = 2;

    Baseline bl;
    if (!bl.open(bl_path)) { std::fprintf(stderr, "cannot open baseline\n"); return 1; }
    const int start_steady = (int)bl.u32("baseline.steady0.start_pos");

    // Timestep embedding and adaLN tables.
    size_t n_ts = 0, n_steps = 0;
    const float* ts = bl.get("adaln.timesteps", &n_ts);
    const float* steps = bl.get("adaln.step_list", &n_steps);
    af::AdaLNTable tbl;
    if (!model.timestep_tables(std::vector<float>(ts, ts + n_ts), tbl)) return 1;
    check("adaln.t_freq", tbl.t_freq, bl, "adaln.t_freq");
    check("adaln.t_embed", tbl.t_embed, bl, "adaln.t_embed");
    check("adaln.adaln6", tbl.adaln6, bl, "adaln.adaln6");
    check("adaln.final_adaln2", tbl.final2, bl, "adaln.final_adaln2");

    struct Prepared {
        int N, start_pos;
        std::vector<float> adaln6, final2;
    };
    auto prepare = [&](const std::string& tag) {
        Prepared p;
        size_t nx = 0;
        bl.get(tag + ".in.x", &nx);
        p.N = (int)(nx / ((size_t)B * cfg.dim_w));
        p.start_pos = tag.rfind("steady", 0) == 0 ? start_steady : 0;
        tbl.gather(adaln_rows(tbl, tag, p.N, (int)cfg.context_frames, steps[0]), p.adaln6, p.final2);
        return p;
    };

    const char* tags[] = {"primer_step0", "primer_kvupdate", "steady0_step0", "steady0_final"};
    for (const char* tag_c : tags) {
        const std::string tag = tag_c;
        Prepared p = prepare(tag);

        // Inputs the forward builds itself: RoPE table and per-token adaLN.
        std::vector<float> rope;
        model.rope_table(p.start_pos, p.N, rope);
        check(tag + ".in.freqs_cis", rope, bl, tag + ".in.freqs_cis");
        std::vector<float> a6 = chunk_major(p.adaln6, p.N, 6, D), f2 = chunk_major(p.final2, p.N, 2, D);
        check(tag + ".in.adaln6", a6, bl, tag + ".in.adaln6");
        check(tag + ".in.final_adaln2", f2, bl, tag + ".in.final_adaln2");

        // Condition embedder, where the dump has it.
        if (bl.has(tag + ".c_embedder.out")) {
            af::Taps ct;
            std::vector<float> c_out;
            if (!model.condition(B, p.N, bl.get(tag + ".c_embedder.in.wa_user"),
                                 bl.get(tag + ".c_embedder.in.motion_user"), bl.get(tag + ".c_embedder.in.wa"),
                                 c_out, &ct))
                return 1;
            for (const char* k : {"motion_embedder", "dual_modal_attn", "cross_modal_feat", "user_agent_attn",
                                  "unified"})
                check(tag + ".c_embedder." + k, ct[std::string("c_embedder.") + k], bl,
                      tag + ".c_embedder." + k);
            check(tag + ".c_embedder.out", c_out, bl, tag + ".c_embedder.out");
        }

        // Forward from the dumped inputs and caches.
        const bool kv_read = bl.flag("baseline." + tag + ".in.kv_read");
        const bool update = bl.flag("baseline." + tag + ".in.update_kv_cache");
        af::KvCache kv;
        kv.init(cfg, B);
        if (kv_read) load_kv(bl, tag + ".kv_before.", kv);
        af::ForwardArgs a;
        a.n_tokens = p.N;
        a.n_branches = B;
        a.x = bl.get(tag + ".in.x");
        a.c = bl.get(tag + ".in.c");
        a.wr = bl.get(tag + ".in.wr");   // row 0 of [2, N, dim_w]: r_s
        a.adaln6 = p.adaln6.data();
        a.final2 = p.final2.data();
        a.start_pos = p.start_pos;
        a.kv = kv_read ? &kv : nullptr;
        a.update_kv = update;
        af::Taps t;
        std::vector<float> v;
        if (!model.forward(a, v, &t)) return 1;
        check(tag + ".x_embed", t["x_embed"], bl, tag + ".x_embed");
        for (const char* k : {"attn_in", "attn_out", "cross_in", "cross_out", "mlp_in", "mlp_out"})
            check(tag + ".block.0." + k, t[std::string("block.0.") + k], bl, tag + ".block.0." + k);
        for (uint32_t i = 0; i < cfg.depth; ++i) {
            const std::string k = "block." + std::to_string(i);
            check(tag + "." + k, t[k], bl, tag + "." + k);
        }
        check(tag + ".out.v", v, bl, tag + ".out.v");
        if (kv_read && update) check_kv(tag + ".kv_after", kv, bl, tag + ".kv_after.");
    }

    // Chain: carry this engine's own caches through the primer KV update and
    // steady block 0, starting from zero caches as the reference does.
    {
        af::KvCache kv;
        kv.init(cfg, B);
        std::vector<float> v;
        auto run = [&](const std::string& tag, bool update) {
            Prepared p = prepare(tag);
            af::ForwardArgs a;
            a.n_tokens = p.N;
            a.n_branches = B;
            a.x = bl.get(tag + ".in.x");
            a.c = bl.get(tag + ".in.c");
            a.wr = bl.get(tag + ".in.wr");
            a.adaln6 = p.adaln6.data();
            a.final2 = p.final2.data();
            a.start_pos = p.start_pos;
            a.kv = &kv;
            a.update_kv = update;
            return model.forward(a, v);
        };
        if (!run("primer_kvupdate", true)) return 1;
        check_kv("chain.primer_kvupdate.kv_after", kv, bl, "primer_kvupdate.kv_after.");
        if (!run("steady0_step0", false)) return 1;
        check("chain.steady0_step0.out.v", v, bl, "steady0_step0.out.v");
        if (!run("steady0_final", true)) return 1;
        check("chain.steady0_final.out.v", v, bl, "steady0_final.out.v");
        check_kv("chain.steady0_final.kv_after", kv, bl, "steady0_final.kv_after.");
    }

    // Audio projection.
    {
        size_t n = 0;
        const float* stacked = bl.get("audio.stacked", &n);
        const int T = (int)(n / cfg.audio_proj_in);
        af::Taps t;
        std::vector<float> f;
        if (!model.audio_projection(T, stacked, f, &t)) return 1;
        check("audio.proj.linear", t["proj.linear"], bl, "audio.proj.linear");
        check("audio.proj.norm", t["proj.norm"], bl, "audio.proj.norm");
        check("audio.features", f, bl, "audio.features");
    }

    // Timings: one primer forward (N = clip, no cache) and one steady forward
    // (N = block + context, cache read), median of 5 after one warm-up.
    auto time_forward = [&](const std::string& tag, bool use_kv) {
        Prepared p = prepare(tag);
        af::KvCache kv;
        kv.init(cfg, B);
        if (use_kv) load_kv(bl, tag + ".kv_before.", kv);
        af::ForwardArgs a;
        a.n_tokens = p.N;
        a.n_branches = B;
        a.x = bl.get(tag + ".in.x");
        a.c = bl.get(tag + ".in.c");
        a.wr = bl.get(tag + ".in.wr");
        a.adaln6 = p.adaln6.data();
        a.final2 = p.final2.data();
        a.start_pos = p.start_pos;
        a.kv = use_kv ? &kv : nullptr;
        std::vector<float> v;
        std::vector<double> ms;
        for (int r = 0; r < 6; ++r) {
            auto t0 = std::chrono::steady_clock::now();
            model.forward(a, v);
            if (r > 0) ms.push_back(ms_since(t0));
        }
        std::sort(ms.begin(), ms.end());
        return ms[ms.size() / 2];
    };
    const double t_primer = time_forward("primer_step0", false);
    const double t_steady = time_forward("steady0_step0", true);

    bool all = true;
    std::printf("%-46s %12s %12s  %s\n", "module", "max_abs", "max_rel", "gate");
    for (const Row& r : g_rows) {
        std::printf("%-46s %12.3e %12.3e  %s\n", r.name.c_str(), r.max_abs, r.max_rel, r.pass ? "pass" : "FAIL");
        all = all && r.pass;
    }
    std::printf("timing (%d threads, B = %d): primer forward N = %u: %.1f ms, steady forward N = %u: %.1f ms\n",
                n_threads, B, cfg.clip_frames, t_primer, cfg.block_frames + cfg.context_frames, t_steady);
    std::printf("%s: %zu modules, gate max_rel <= %.0e\n", all ? "PASS" : "FAIL", g_rows.size(), kGate);
    return all ? 0 : 1;
}
