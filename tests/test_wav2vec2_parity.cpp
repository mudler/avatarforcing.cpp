// wav2vec2 audio feature parity against the PyTorch reference dumps.
//
// Environment:
//   AVATARFORCING_TEST_GGUF            model GGUF (convert_avatarforcing_to_gguf.py)
//   AVATARFORCING_TEST_W2V2_BASELINE   wav2vec2 baseline GGUF
//                                      (scripts/dumps_to_baseline.py --w2v2)
//   AVATARFORCING_TEST_THREADS         CPU threads (default 8)
//   AVATARFORCING_TEST_BACKEND         cpu (default) or cuda
//   AVATARFORCING_TEST_LONG_SECONDS    length of the long-phrase run (default 30, 0 = off)
// Exits 77 (skip) when a path is unset or missing.
//
// Metric per module: max_abs = max |got - ref|, max_rel = max_abs / max |ref|.
// Gate: max_rel <= 1e-4 (the reference moves by about 6e-5 relative between
// thread counts on these stages).

#include "backend.hpp"
#include "flow.hpp"
#include "wav2vec2.hpp"

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

constexpr double kGate = 1e-4;

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
    const float* get(const std::string& name, size_t* n = nullptr) const {
        ggml_tensor* t = ggml_get_tensor(ctx_, name.c_str());
        if (!t) {
            std::fprintf(stderr, "baseline: missing tensor %s\n", name.c_str());
            std::exit(1);
        }
        if (n) *n = (size_t)ggml_nelements(t);
        return (const float*)t->data;
    }
    uint32_t u32(const std::string& key) const {
        int64_t id = gguf_find_key(g_, key.c_str());
        if (id < 0) { std::fprintf(stderr, "baseline: missing KV %s\n", key.c_str()); std::exit(1); }
        return gguf_get_val_u32(g_, id);
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

} // namespace

int main() {
    const char* gguf_path = std::getenv("AVATARFORCING_TEST_GGUF");
    const char* bl_path = std::getenv("AVATARFORCING_TEST_W2V2_BASELINE");
    if (!gguf_path || !bl_path) {
        std::printf("skip: set AVATARFORCING_TEST_GGUF and AVATARFORCING_TEST_W2V2_BASELINE\n");
        return 77;
    }
    if (FILE* f = std::fopen(gguf_path, "rb")) std::fclose(f); else { std::printf("skip: %s missing\n", gguf_path); return 77; }
    if (FILE* f = std::fopen(bl_path, "rb")) std::fclose(f); else { std::printf("skip: %s missing\n", bl_path); return 77; }
    const char* th = std::getenv("AVATARFORCING_TEST_THREADS");
    const int n_threads = th ? std::atoi(th) : 8;
    const char* ls = std::getenv("AVATARFORCING_TEST_LONG_SECONDS");
    const int long_seconds = ls ? std::atoi(ls) : 30;

    Baseline bl;
    if (!bl.open(bl_path)) { std::fprintf(stderr, "cannot open baseline %s\n", bl_path); return 1; }

    af::Backend backend(n_threads, af::Backend::device_from_env("AVATARFORCING_TEST_BACKEND"));
    if (!backend.ok()) return 1;
    std::fprintf(stderr, "backend: %s\n", backend.device().c_str());
    af::FlowModel flow(backend);
    if (!flow.load(gguf_path)) { std::fprintf(stderr, "flow load failed\n"); return 1; }
    af::AudioEncoder enc(backend, flow);
    auto t_load = std::chrono::steady_clock::now();
    if (!enc.load(gguf_path)) { std::fprintf(stderr, "wav2vec2 load failed\n"); return 1; }
    const double ms_load = ms_since(t_load);

    // Both input modes, every stage.
    const char* stages[] = {"model_input", "conv.0", "conv.6", "interp", "feature_projection.norm",
                            "feature_projection", "pos_conv", "hidden.0", "hidden.1", "hidden.2",
                            "hidden.3", "hidden.4", "hidden.5", "hidden.6", "hidden.7", "hidden.8",
                            "hidden.9", "hidden.10", "hidden.11", "hidden.12", "stacked",
                            "proj.linear", "proj.norm"};
    for (const char* mode : {"raw", "normalized"}) {
        const std::string m = mode;
        const af::AudioInputMode am = m == "raw" ? af::AudioInputMode::Raw : af::AudioInputMode::Normalized;
        size_t n = 0;
        const float* pcm = bl.get(m + ".pcm_in", &n);
        af::W2V2Taps taps;
        std::vector<float> feats;
        int n_frames = 0;
        if (!enc.features(pcm, n, am, feats, n_frames, &taps)) return 1;
        const uint32_t ref_frames = bl.u32("baseline." + m + ".n_frames");
        if ((uint32_t)n_frames != ref_frames) {
            std::fprintf(stderr, "%s: n_frames %d, reference %u\n", mode, n_frames, ref_frames);
            g_rows.push_back({m + ".n_frames", INFINITY, INFINITY, false});
        }
        for (const char* s : stages) check(m + "." + s, taps[s], bl, m + "." + s);
        check(m + ".features", feats, bl, m + ".features");
    }

    // Idle features, computed at load time from the stored pink-noise PCM.
    check("idle.pcm", enc.idle_pcm(), bl, "idle.pcm");
    check("idle.features", enc.idle_features(), bl, "idle.features");

    // Timing: 10 s of speech (normalized mode), median of 3 after one warm-up.
    double ms10 = 0;
    {
        size_t n = 0;
        const float* pcm = bl.get("normalized.pcm_in", &n);
        const size_t n10 = std::min<size_t>(n, 10 * 16000);
        std::vector<float> feats;
        int nf = 0;
        std::vector<double> ms;
        for (int r = 0; r < 4; ++r) {
            auto t0 = std::chrono::steady_clock::now();
            if (!enc.features(pcm, n10, af::AudioInputMode::Normalized, feats, nf)) return 1;
            if (r > 0) ms.push_back(ms_since(t0));
        }
        std::sort(ms.begin(), ms.end());
        ms10 = ms[ms.size() / 2];
    }

    // Long phrase: the speech tiled to `long_seconds`, one graph. Checks that it
    // runs and stays finite (no reference for this length).
    double ms_long = 0;
    int long_frames = 0;
    if (long_seconds > 0) {
        size_t n = 0;
        const float* pcm = bl.get("normalized.pcm_in", &n);
        std::vector<float> x((size_t)long_seconds * 16000);
        for (size_t i = 0; i < x.size(); ++i) x[i] = pcm[i % n];
        std::vector<float> feats;
        auto t0 = std::chrono::steady_clock::now();
        if (!enc.features(x.data(), x.size(), af::AudioInputMode::Normalized, feats, long_frames)) return 1;
        ms_long = ms_since(t0);
        bool finite = feats.size() == (size_t)long_frames * flow.config().audio_proj_out;
        for (float v : feats) finite = finite && std::isfinite(v);
        g_rows.push_back({"long." + std::to_string(long_seconds) + "s.finite", 0, 0, finite});
    }

    bool all = true;
    std::printf("%-36s %12s %12s  %s\n", "module", "max_abs", "max_rel", "gate");
    for (const Row& r : g_rows) {
        std::printf("%-36s %12.3e %12.3e  %s\n", r.name.c_str(), r.max_abs, r.max_rel, r.pass ? "pass" : "FAIL");
        all = all && r.pass;
    }
    std::printf("timing (%d threads): load + idle features %.1f ms, 10 s of audio %.1f ms", n_threads, ms_load, ms10);
    if (long_seconds > 0) std::printf(", %d s of audio (%d frames) %.1f ms", long_seconds, long_frames, ms_long);
    std::printf("\n%s: %zu modules, gate max_rel <= %.0e\n", all ? "PASS" : "FAIL", g_rows.size(), kGate);
    return all ? 0 : 1;
}
