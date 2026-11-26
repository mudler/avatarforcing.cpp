// Quantized model against the F32 model, component by component.
//
// Environment:
//   AVATARFORCING_TEST_QUANT_GGUF  quantized GGUF (avatarforcing-quantize)
//   AVATARFORCING_TEST_GGUF        the F32 GGUF it was made from
//   AVATARFORCING_TEST_BASELINE    flow baseline GGUF: inputs of the flow forwards
//   AVATARFORCING_TEST_W2V2_BASELINE  wav2vec2 baseline GGUF: the speech PCM
//   AVATARFORCING_TEST_THREADS     CPU threads (default 8)
//   AVATARFORCING_TEST_BACKEND     cpu (default) or cuda
// Exits 77 (skip) when the quant GGUF, the F32 GGUF or both baselines are
// unset or missing. A baseline that is unset skips its part.
//
// Both models run on the same inputs (the dumped inputs of the flow
// forwards, the dumped speech PCM); the F32 model runs first and is freed
// before the quantized one loads. Metrics per output, against the F32 output:
//   max_rel  = max |q - f| / max |f|
//   mean_rel = mean |q - f| / mean |f|
// The gates depend on avatarforcing.quant.type (docs/quantization.md). The
// velocity of the t = 0 forwards (primer_kvupdate, steady0_final) is printed
// but not gated: the sampler uses only the KV cache those forwards write.

#include "backend.hpp"
#include "flow.hpp"
#include "wav2vec2.hpp"

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace {

struct Gate {
    const char* type;
    double flow_max_rel, flow_mean_rel;   // flow transformer, condition embedder, adaLN, audio projection
    double w2v2_max_rel, w2v2_mean_rel;   // wav2vec2 hidden states and the audio features
};

// About twice the values measured on the CPU (docs/quantization.md).
const Gate kGates[] = {
    {"f16", 1e-2, 5e-3, 5e-3, 3e-3},
    {"q8_0", 1e-1, 8e-2, 1.5e-1, 1e-1},
    {"q6_k", 1.5e-1, 1e-1, 2e-1, 1.5e-1},
    {"q5_k", 2e-1, 1.5e-1, 3e-1, 2e-1},
    {"q4_k", 3e-1, 2e-1, 4e-1, 3e-1},
};

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

using Outputs = std::map<std::string, std::vector<float>>;

bool file_exists(const char* p) {
    FILE* f = p ? std::fopen(p, "rb") : nullptr;
    if (f) std::fclose(f);
    return f != nullptr;
}

std::string quant_type(const char* path) {
    gguf_init_params p = {/*no_alloc=*/true, /*ctx=*/nullptr};
    gguf_context* g = gguf_init_from_file(path, p);
    if (!g) return "";
    const int64_t id = gguf_find_key(g, "avatarforcing.quant.type");
    std::string t = id >= 0 ? gguf_get_val_str(g, id) : "";
    gguf_free(g);
    return t;
}

// adaLN rows per token of a dumped forward (as in test_flow_parity).
std::vector<int> adaln_rows(const af::AdaLNTable& tbl, const std::string& tag, int N, int ctx_frames,
                            float step0_t) {
    const int r0 = tbl.row(0.0f), rs = tbl.row(step0_t);
    std::vector<int> rows(N, rs);
    if (tag == "primer_kvupdate" || tag == "steady0_final") std::fill(rows.begin(), rows.end(), r0);
    if (tag == "steady0_step0")
        for (int j = 0; j < ctx_frames; ++j) rows[j] = r0;
    return rows;
}

// Runs every component of one model on the baseline inputs.
bool run_model(const char* gguf_path, const Baseline* flow_bl, const Baseline* w2v2_bl, int n_threads,
               Outputs& out) {
    af::Backend be(n_threads, af::Backend::device_from_env("AVATARFORCING_TEST_BACKEND"));
    if (!be.ok()) return false;
    af::FlowModel model(be);
    if (!model.load(gguf_path)) { std::fprintf(stderr, "flow load failed: %s\n", gguf_path); return false; }
    const af::FlowConfig& cfg = model.config();
    const int B = 2;

    if (flow_bl) {
        const Baseline& bl = *flow_bl;
        const int start_steady = (int)bl.u32("baseline.steady0.start_pos");
        size_t n_ts = 0, n_steps = 0;
        const float* ts = bl.get("adaln.timesteps", &n_ts);
        const float* steps = bl.get("adaln.step_list", &n_steps);
        af::AdaLNTable tbl;
        if (!model.timestep_tables(std::vector<float>(ts, ts + n_ts), tbl)) return false;
        out["flow.adaln6"] = tbl.adaln6;
        out["flow.final_adaln2"] = tbl.final2;

        for (const char* tag_c : {"primer_step0", "primer_kvupdate", "steady0_step0", "steady0_final"}) {
            const std::string tag = tag_c;
            size_t nx = 0;
            bl.get(tag + ".in.x", &nx);
            const int N = (int)(nx / ((size_t)B * cfg.dim_w));
            std::vector<float> a6, f2;
            tbl.gather(adaln_rows(tbl, tag, N, (int)cfg.context_frames, steps[0]), a6, f2);

            if (bl.has(tag + ".c_embedder.out")) {
                std::vector<float> c_out;
                if (!model.condition(B, N, bl.get(tag + ".c_embedder.in.wa_user"),
                                     bl.get(tag + ".c_embedder.in.motion_user"), bl.get(tag + ".c_embedder.in.wa"),
                                     c_out))
                    return false;
                out["flow." + tag + ".c_embedder.out"] = c_out;
            }

            const bool kv_read = bl.flag("baseline." + tag + ".in.kv_read");
            const bool update = bl.flag("baseline." + tag + ".in.update_kv_cache");
            af::KvCache kv;
            kv.init(cfg, B);
            if (kv_read) {
                kv.self_k = bl.vec(tag + ".kv_before.self_k");
                kv.self_v = bl.vec(tag + ".kv_before.self_v");
                kv.cross_k = bl.vec(tag + ".kv_before.cross_k");
                kv.cross_v = bl.vec(tag + ".kv_before.cross_v");
            }
            af::ForwardArgs a;
            a.n_tokens = N;
            a.n_branches = B;
            a.x = bl.get(tag + ".in.x");
            a.c = bl.get(tag + ".in.c");
            a.wr = bl.get(tag + ".in.wr");
            a.adaln6 = a6.data();
            a.final2 = f2.data();
            a.start_pos = tag.rfind("steady", 0) == 0 ? start_steady : 0;
            a.kv = kv_read ? &kv : nullptr;
            a.update_kv = update;
            std::vector<float> v;
            if (!model.forward(a, v)) return false;
            out["flow." + tag + ".out.v"] = v;
            if (kv_read && update) {
                out["flow." + tag + ".kv_after.self_k"] = kv.self_k;
                out["flow." + tag + ".kv_after.cross_k"] = kv.cross_k;
            }
        }

        size_t n = 0;
        const float* stacked = bl.get("audio.stacked", &n);
        std::vector<float> f;
        if (!model.audio_projection((int)(n / cfg.audio_proj_in), stacked, f)) return false;
        out["flow.audio_projection"] = f;
    }

    if (w2v2_bl) {
        af::AudioEncoder enc(be, model);
        if (!enc.load(gguf_path)) { std::fprintf(stderr, "wav2vec2 load failed: %s\n", gguf_path); return false; }
        for (const char* mode : {"raw", "normalized"}) {
            const std::string m = mode;
            size_t n = 0;
            const float* pcm = w2v2_bl->get(m + ".pcm_in", &n);
            af::W2V2Taps taps;
            std::vector<float> feats;
            int n_frames = 0;
            if (!enc.features(pcm, n, m == "raw" ? af::AudioInputMode::Raw : af::AudioInputMode::Normalized, feats,
                              n_frames, &taps))
                return false;
            for (const char* s : {"hidden.1", "hidden.6", "hidden.12"}) out["w2v2." + m + "." + s] = taps[s];
            out["w2v2." + m + ".features"] = feats;
        }
        out["w2v2.idle.features"] = enc.idle_features();
    }
    return true;
}

} // namespace

int main() {
    const char* q_path = std::getenv("AVATARFORCING_TEST_QUANT_GGUF");
    const char* f_path = std::getenv("AVATARFORCING_TEST_GGUF");
    const char* fb = std::getenv("AVATARFORCING_TEST_BASELINE");
    const char* wb = std::getenv("AVATARFORCING_TEST_W2V2_BASELINE");
    if (!file_exists(q_path) || !file_exists(f_path) || (!file_exists(fb) && !file_exists(wb))) {
        std::printf("skip: set AVATARFORCING_TEST_QUANT_GGUF, AVATARFORCING_TEST_GGUF and a baseline\n");
        return 77;
    }
    const char* th = std::getenv("AVATARFORCING_TEST_THREADS");
    const int n_threads = th ? std::atoi(th) : 8;

    const std::string qt = quant_type(q_path);
    const Gate* gate = nullptr;
    for (const Gate& g : kGates)
        if (qt == g.type) gate = &g;
    if (!gate) {
        std::fprintf(stderr, "%s: no gate for avatarforcing.quant.type '%s'\n", q_path, qt.c_str());
        return 1;
    }

    std::unique_ptr<Baseline> flow_bl, w2v2_bl;
    if (file_exists(fb)) {
        flow_bl.reset(new Baseline);
        if (!flow_bl->open(fb)) { std::fprintf(stderr, "cannot open %s\n", fb); return 1; }
    }
    if (file_exists(wb)) {
        w2v2_bl.reset(new Baseline);
        if (!w2v2_bl->open(wb)) { std::fprintf(stderr, "cannot open %s\n", wb); return 1; }
    }

    Outputs ref, got;
    if (!run_model(f_path, flow_bl.get(), w2v2_bl.get(), n_threads, ref)) return 1;
    if (!run_model(q_path, flow_bl.get(), w2v2_bl.get(), n_threads, got)) return 1;

    bool all = true;
    std::printf("quant type %s\n%-44s %12s %12s  %s\n", qt.c_str(), "output", "max_rel", "mean_rel", "gate");
    for (const auto& kv : ref) {
        const std::vector<float>& f = kv.second;
        const std::vector<float>& q = got[kv.first];
        const bool w2v2 = kv.first.rfind("w2v2.", 0) == 0;
        const double g_max = w2v2 ? gate->w2v2_max_rel : gate->flow_max_rel;
        const double g_mean = w2v2 ? gate->w2v2_mean_rel : gate->flow_mean_rel;
        double max_abs = 0, max_ref = 0, sum_abs = 0, sum_ref = 0;
        bool finite = q.size() == f.size() && !f.empty();
        for (size_t i = 0; finite && i < f.size(); ++i) {
            if (!std::isfinite(q[i])) finite = false;
            const double d = std::fabs((double)q[i] - (double)f[i]);
            max_abs = std::max(max_abs, d);
            max_ref = std::max(max_ref, std::fabs((double)f[i]));
            sum_abs += d;
            sum_ref += std::fabs((double)f[i]);
        }
        const double max_rel = max_ref > 0 ? max_abs / max_ref : max_abs;
        const double mean_rel = sum_ref > 0 ? sum_abs / sum_ref : sum_abs;
        const bool info = kv.first.find("kvupdate.out.v") != std::string::npos ||
                          kv.first.find("steady0_final.out.v") != std::string::npos;
        const bool pass = finite && (info || (max_rel <= g_max && mean_rel <= g_mean));
        if (info)
            std::printf("%-44s %12.3e %12.3e  (t = 0, not gated) %s\n", kv.first.c_str(), max_rel, mean_rel,
                        finite ? "" : "NOT FINITE");
        else
            std::printf("%-44s %12.3e %12.3e  <=%.1e/%.1e %s\n", kv.first.c_str(), max_rel, mean_rel, g_max, g_mean,
                        pass ? "pass" : "FAIL");
        all = all && pass;
    }
    std::printf("%s: %zu outputs, %s against F32\n", all ? "PASS" : "FAIL", ref.size(), qt.c_str());
    return all ? 0 : 1;
}
