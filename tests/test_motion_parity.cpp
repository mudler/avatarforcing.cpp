// Motion autoencoder parity against the PyTorch reference dumps.
//
// Environment:
//   AVATARFORCING_TEST_GGUF             model GGUF (convert_avatarforcing_to_gguf.py)
//   AVATARFORCING_TEST_MOTION_BASELINE  baseline GGUF
//                                       (scripts/dumps_to_baseline.py --component motion)
//   AVATARFORCING_TEST_THREADS          CPU threads (default 8)
//   AVATARFORCING_TEST_BACKEND          cpu (default) or cuda
//   AVATARFORCING_TEST_CHUNK            frames per decoder graph (default 1, 0 = whole block)
//   AVATARFORCING_TEST_RUNS             timing runs, the fastest is reported (default 1)
//   AVATARFORCING_TEST_DUMP_DIR         writes the frame-0 decoder taps there as raw float32
// Exits 77 (skip) when a path is unset or missing.
//
// Gates:
//   encoder outputs       max_rel <= 2e-5   (max_rel = max |got - ref| / max |ref|)
//   decoder taps, pixels  max_abs <= 1e-4   (decoder fed the reference s_r and features),
//                         except the grid_sample outputs of the last level
//                         (level.6.warp, level.6.out): max_abs <= 1.5e-4,
//                         see docs/parity.md (reference noise floor)
//   grid_sample alone     max_abs <= 1e-6   (reference flow and features in,
//                         compared with the reference warp)
//   final frames          SSIM >= 0.999 per frame (also from the C++ encoder outputs)
//   uint8 frames          reported: differing values and max difference

#include "backend.hpp"
#include "grid_sample.hpp"
#include "motion.hpp"

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr double kEncGate = 2e-5;
constexpr double kDecGate = 1e-4;
// The last-level warp samples a 512x512 feature map: a flow error of one
// float32 ulp moves the sample by 1/8 pixel-ulp * 256. The float32 reference
// itself differs from a float64 run of the same code by 7.1e-5 there, and from
// its own single-thread run by 9.5e-5 (docs/parity.md).
constexpr double kDecGateLastWarp = 1.5e-4;
constexpr double kGsGate = 1e-6;
constexpr double kSsimGate = 0.999;

// Baseline GGUF, tensor data read on demand (the file is a few hundred MB).
class Baseline {
public:
    bool open(const char* path) {
        path_ = path;
        gguf_init_params p = { /*.no_alloc =*/ true, /*.ctx =*/ &ctx_ };
        g_ = gguf_init_from_file(path, p);
        return g_ != nullptr;
    }
    ~Baseline() {
        if (g_) gguf_free(g_);
        if (ctx_) ggml_free(ctx_);
    }
    bool has(const std::string& name) const { return gguf_find_tensor(g_, name.c_str()) >= 0; }
    std::vector<float> vec(const std::string& name) const {
        int64_t id = gguf_find_tensor(g_, name.c_str());
        if (id < 0) {
            std::fprintf(stderr, "baseline: missing tensor %s\n", name.c_str());
            std::exit(1);
        }
        ggml_tensor* t = ggml_get_tensor(ctx_, name.c_str());
        std::vector<float> v((size_t)ggml_nelements(t));
        FILE* f = std::fopen(path_.c_str(), "rb");
        const size_t off = gguf_get_data_offset(g_) + gguf_get_tensor_offset(g_, id);
        if (!f || std::fseek(f, (long)off, SEEK_SET) != 0 ||
            std::fread(v.data(), sizeof(float), v.size(), f) != v.size()) {
            std::fprintf(stderr, "baseline: read failed for %s\n", name.c_str());
            std::exit(1);
        }
        std::fclose(f);
        return v;
    }

private:
    std::string path_;
    gguf_context* g_ = nullptr;
    ggml_context* ctx_ = nullptr;
};

struct Row {
    std::string name;
    double max_abs, max_rel;
    std::string gate;
    bool pass;
};
std::vector<Row> g_rows;

void check(const std::string& name, const std::vector<float>& got, const std::vector<float>& ref, bool rel_gate,
           double abs_gate = kDecGate) {
    if (got.size() != ref.size()) {
        std::fprintf(stderr, "%s: size mismatch got %zu ref %zu\n", name.c_str(), got.size(), ref.size());
        g_rows.push_back({name, INFINITY, INFINITY, "size", false});
        return;
    }
    double max_abs = 0, max_ref = 0;
    bool finite = true;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (!std::isfinite(got[i])) finite = false;
        max_abs = std::max(max_abs, std::fabs((double)got[i] - (double)ref[i]));
        max_ref = std::max(max_ref, std::fabs((double)ref[i]));
    }
    const double rel = max_ref > 0 ? max_abs / max_ref : max_abs;
    const bool pass = finite && (rel_gate ? rel <= kEncGate : max_abs <= abs_gate);
    char gate[32];
    if (rel_gate) std::snprintf(gate, sizeof(gate), "rel<=%.0e", kEncGate);
    else std::snprintf(gate, sizeof(gate), "abs<=%.1e", abs_gate);
    g_rows.push_back({name, max_abs, rel, gate, pass});
}

void check_info(const std::string& name, const std::vector<float>& got, const std::vector<float>& ref) {
    check(name, got, ref, true);
    g_rows.back().gate = "info";
    g_rows.back().pass = true;
}

// SSIM (Wang et al. 2004): Gaussian window 11, sigma 1.5, valid region,
// on [0, 1] images (pixels clamped to [-1, 1], then (x + 1) / 2), data range
// 1, mean over the 3 channels.
double ssim_frame(const float* a, const float* b, int h, int w) {
    const int K = 11;
    double g[K], gs = 0;
    for (int i = 0; i < K; ++i) { g[i] = std::exp(-((i - 5) * (i - 5)) / (2 * 1.5 * 1.5)); gs += g[i]; }
    for (double& v : g) v /= gs;
    const double C1 = 0.01 * 0.01, C2 = 0.03 * 0.03;
    const int oh = h - K + 1, ow = w - K + 1;
    double total = 0;
    for (int c = 0; c < 3; ++c) {
        const float* pa = a + (size_t)c * h * w;
        const float* pb = b + (size_t)c * h * w;
        auto norm = [](float v) { return ((double)std::min(1.0f, std::max(-1.0f, v)) + 1.0) * 0.5; };
        // five filtered maps: x, y, x^2, y^2, xy (horizontal pass then vertical)
        std::vector<double> hz[5];
        for (auto& v : hz) v.assign((size_t)h * ow, 0.0);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < ow; ++x) {
                double s[5] = {0, 0, 0, 0, 0};
                for (int k = 0; k < K; ++k) {
                    const double u = norm(pa[(size_t)y * w + x + k]), v = norm(pb[(size_t)y * w + x + k]);
                    s[0] += g[k] * u; s[1] += g[k] * v; s[2] += g[k] * u * u; s[3] += g[k] * v * v; s[4] += g[k] * u * v;
                }
                for (int m = 0; m < 5; ++m) hz[m][(size_t)y * ow + x] = s[m];
            }
        double acc = 0;
        for (int y = 0; y < oh; ++y)
            for (int x = 0; x < ow; ++x) {
                double s[5] = {0, 0, 0, 0, 0};
                for (int k = 0; k < K; ++k)
                    for (int m = 0; m < 5; ++m) s[m] += g[k] * hz[m][(size_t)(y + k) * ow + x];
                const double mu_a = s[0], mu_b = s[1];
                const double va = s[2] - mu_a * mu_a, vb = s[3] - mu_b * mu_b, cov = s[4] - mu_a * mu_b;
                acc += ((2 * mu_a * mu_b + C1) * (2 * cov + C2)) /
                       ((mu_a * mu_a + mu_b * mu_b + C1) * (va + vb + C2));
            }
        total += acc / ((double)oh * ow);
    }
    return total / 3.0;
}

double check_ssim(const std::string& name, const std::vector<float>& got, const std::vector<float>& ref, int n, int s) {
    double worst = 1.0;
    const size_t frame = (size_t)3 * s * s;
    for (int f = 0; f < n; ++f) worst = std::min(worst, ssim_frame(got.data() + f * frame, ref.data() + f * frame, s, s));
    g_rows.push_back({name + " (min SSIM)", 1.0 - worst, worst, "ssim>=0.999", worst >= kSsimGate});
    return worst;
}

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

double peak_rss_mb() {
    rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_maxrss / 1024.0;   // Linux: KiB
}

bool file_exists(const char* p) {
    FILE* f = p ? std::fopen(p, "rb") : nullptr;
    if (f) std::fclose(f);
    return f != nullptr;
}

} // namespace

int main() {
    const char* gguf_path = std::getenv("AVATARFORCING_TEST_GGUF");
    const char* bl_path = std::getenv("AVATARFORCING_TEST_MOTION_BASELINE");
    if (!file_exists(gguf_path) || !file_exists(bl_path)) {
        std::fprintf(stderr, "AVATARFORCING_TEST_GGUF / AVATARFORCING_TEST_MOTION_BASELINE unset or missing; skipping\n");
        return 77;
    }
    const char* th = std::getenv("AVATARFORCING_TEST_THREADS");
    const int n_threads = th ? std::atoi(th) : 8;
    const char* ch = std::getenv("AVATARFORCING_TEST_CHUNK");
    const int chunk = ch ? std::atoi(ch) : 1;
    const char* rn = std::getenv("AVATARFORCING_TEST_RUNS");
    const int runs = std::max(1, rn ? std::atoi(rn) : 1);

    af::Backend backend(n_threads, af::Backend::device_from_env("AVATARFORCING_TEST_BACKEND"));
    if (!backend.ok()) return 1;
    std::fprintf(stderr, "backend: %s\n", backend.device().c_str());
    af::MotionModel model(backend);
    auto t_load = std::chrono::steady_clock::now();
    if (!model.load(gguf_path)) { std::fprintf(stderr, "model load failed\n"); return 1; }
    std::fprintf(stderr, "load: %.0f ms, peak RSS %.0f MB\n", ms_since(t_load), peak_rss_mb());
    model.set_chunk_frames(chunk);
    const af::MaeConfig& cfg = model.config();
    const int S = (int)cfg.size, L = (int)cfg.dec_n_levels;

    Baseline bl;
    if (!bl.open(bl_path)) { std::fprintf(stderr, "cannot open baseline\n"); return 1; }

    // ---- encoder ----
    af::Portrait mine;
    double enc_ms = 0;
    {
        std::vector<float> image = bl.vec("enc.image");
        af::MotionTaps taps;
        if (!model.encode(image.data(), mine, &taps)) return 1;   // warm-up and taps
        enc_ms = INFINITY;
        for (int r = 0; r < runs; ++r) {
            auto t0 = std::chrono::steady_clock::now();
            af::Portrait again;
            if (!model.encode(image.data(), again, nullptr)) return 1;
            enc_ms = std::min(enc_ms, ms_since(t0));
        }
        const int n_res = (int)cfg.enc_n_resblocks + 2;
        for (int i = 0; i < n_res; ++i) {
            const std::string k = "enc.convs." + std::to_string(i);
            if (bl.has("enc." + k)) check("encoder." + k, taps[k], bl.vec("enc." + k), true);
        }
        for (int l = 0; l < L; ++l) {
            const std::string k = "feat." + std::to_string(l);
            check("encoder." + k, taps[k], bl.vec("enc." + k), true);
        }
        check("encoder.s_r", mine.s_r, bl.vec("enc.s_r"), true);
        check("encoder.alpha", mine.alpha, bl.vec("enc.alpha"), true);
        check("encoder.r_s", mine.r_s, bl.vec("enc.r_s"), true);
    }
    std::fprintf(stderr, "encoder done, peak RSS %.0f MB\n", peak_rss_mb());

    // ---- decoder, fed the reference s_r and features ----
    std::vector<float> latent = bl.vec("dec.latent");
    const int n_frames = (int)(latent.size() / cfg.style_dim);
    std::vector<float> ref_pixels = bl.vec("dec.pixels");
    std::vector<float> pixels;
    double dec_ms = INFINITY, frame_ms = INFINITY;
    {
        af::Portrait ref;
        ref.s_r = bl.vec("enc.s_r");
        for (int l = 0; l < L; ++l) ref.feats.push_back(bl.vec("enc.feat." + std::to_string(l)));
        if (!model.set_portrait(ref)) return 1;
        af::MotionTaps taps;
        if (!model.decode(latent.data(), 1, pixels, &taps)) return 1;   // frame 0 with taps
        if (const char* dd = std::getenv("AVATARFORCING_TEST_DUMP_DIR")) {
            // Raw float32 files of every frame-0 tap and the pixels, for
            // offline comparison (e.g. with a float64 run of the reference).
            taps["pixels"] = pixels;
            for (const auto& kv : taps) {
                const std::string path = std::string(dd) + "/" + kv.first + ".f32";
                if (FILE* f = std::fopen(path.c_str(), "wb")) {
                    std::fwrite(kv.second.data(), sizeof(float), kv.second.size(), f);
                    std::fclose(f);
                }
            }
            taps.erase("pixels");
        }
        check("decoder.const", taps["const"], bl.vec("dec.const"), false);
        check("decoder.conv1.style", taps["conv1.style"], bl.vec("dec.conv1.style"), false);
        check("decoder.conv1", taps["conv1"], bl.vec("dec.conv1"), false);
        static const char* keys[] = {"up.style", "up", "conv.style", "conv", "toflow.style", "flow_raw", "flow",
                                     "warp", "out", "skip"};
        for (int l = 0; l < L; ++l)
            for (const char* k : keys) {
                const std::string name = "level." + std::to_string(l) + "." + k;
                const bool last_warp = l == L - 1 && (std::strcmp(k, "warp") == 0 || std::strcmp(k, "out") == 0);
                check("decoder." + name, taps[name], bl.vec("dec." + name), false,
                      last_warp ? kDecGateLastWarp : kDecGate);
            }
        taps.clear();

        for (int r = 0; r < runs; ++r) {
            std::vector<float> one;
            auto t1 = std::chrono::steady_clock::now();
            if (!model.decode(latent.data(), 1, one)) return 1;
            frame_ms = std::min(frame_ms, ms_since(t1));
            auto t0 = std::chrono::steady_clock::now();
            if (!model.decode(latent.data(), n_frames, pixels)) return 1;
            dec_ms = std::min(dec_ms, ms_since(t0));
        }
        check("decoder.pixels", pixels, ref_pixels, false);
        check_ssim("decoder.pixels", pixels, ref_pixels, n_frames, S);

        std::vector<uint8_t> u8;
        af::MotionModel::to_u8(pixels.data(), n_frames, S, S, u8);
        std::vector<float> ref_u8 = bl.vec("dec.frames_u8");
        size_t n_diff = 0;
        int max_diff = 0;
        for (size_t i = 0; i < u8.size(); ++i) {
            const int d = std::abs((int)u8[i] - (int)ref_u8[i]);
            if (d) ++n_diff;
            max_diff = std::max(max_diff, d);
        }
        g_rows.push_back({"decoder.frames_u8 (differing values: " + std::to_string(n_diff) + " of " +
                              std::to_string(u8.size()) + ")",
                          (double)max_diff, 0.0, "report", max_diff <= 1});
    }
    std::fprintf(stderr, "decoder done, peak RSS %.0f MB\n", peak_rss_mb());

    // ---- grid_sample alone: reference flow and features -> reference warp ----
    for (int l = 0; l < L; ++l) {
        const std::string p = "level." + std::to_string(l) + ".";
        std::vector<float> feat = bl.vec("enc.feat." + std::to_string(l));
        std::vector<float> flow = bl.vec("dec." + p + "flow");          // [H, W, 2]
        std::vector<float> raw = bl.vec("dec." + p + "flow_raw");       // [3, H, W]
        std::vector<float> ref = bl.vec("dec." + p + "warp");           // [C, H, W]
        const int n = 8 << l;
        const size_t hw = (size_t)n * n;
        const int c = (int)(feat.size() / hw);
        std::vector<float> grid(2 * hw), out(feat.size());
        for (size_t i = 0; i < hw; ++i) { grid[i] = flow[2 * i]; grid[hw + i] = flow[2 * i + 1]; }
        af::grid_sample_bilinear_zeros_host(feat.data(), n, n, c, 1, grid.data(), n, n, 1, out.data());
        {
            // The ggml operator on the test backend must give the same bits
            // as the host reference. Two grids (the reference one and a
            // shifted copy that samples outside the map) against one shared
            // feature map, which also covers the batch broadcast.
            std::vector<float> grid2(4 * hw), op_out;
            for (size_t i = 0; i < 2 * hw; ++i) { grid2[i] = grid[i]; grid2[2 * hw + i] = grid[i] * 1.37f + 0.11f; }
            std::vector<float> host2(2 * feat.size());
            af::grid_sample_bilinear_zeros_host(feat.data(), n, n, c, 1, grid2.data(), n, n, 2, host2.data());
            ggml_tensor* res = nullptr;
            const bool ok = backend.compute(
                [&](af::Backend::Graph& g) {
                    ggml_tensor* f = g.input(feat.data(), n, n, c, 1);
                    ggml_tensor* gr = g.input(grid2.data(), n, n, 2, 2);
                    res = af::grid_sample_bilinear_zeros(g.ctx, f, gr);
                    ggml_set_output(res);
                    ggml_build_forward_expand(g.gf, res);
                    return true;
                },
                [&](af::Backend::Graph&) {
                    op_out.resize(ggml_nelements(res));
                    ggml_backend_tensor_get(res, op_out.data(), 0, ggml_nbytes(res));
                });
            size_t n_diff = 0;
            double md = 0;
            if (ok)
                for (size_t i = 0; i < host2.size(); ++i)
                    if (std::memcmp(&op_out[i], &host2[i], sizeof(float)) != 0) {
                        ++n_diff;
                        md = std::max(md, (double)std::fabs(op_out[i] - host2[i]));
                    }
            g_rows.push_back({"grid_sample." + p + "op_vs_host (differing values: " + std::to_string(n_diff) + ")",
                              md, 0.0, "bit-exact", ok && n_diff == 0});
        }
        for (int ch = 0; ch < c; ++ch)
            for (size_t i = 0; i < hw; ++i)
                out[ch * hw + i] *= 1.0f / (1.0f + std::exp(-raw[2 * hw + i]));
        check("grid_sample." + p + "warp", out, ref, false, kGsGate);
    }

    // ---- end to end: decoder on the C++ encoder outputs ----
    {
        if (!model.set_portrait(mine)) return 1;
        std::vector<float> e2e;
        if (!model.decode(latent.data(), n_frames, e2e)) return 1;
        check_info("end_to_end.pixels", e2e, ref_pixels);
        check_ssim("end_to_end.pixels", e2e, ref_pixels, n_frames, S);
    }

    bool all = true;
    std::printf("%-58s %12s %12s  %s\n", "module", "max_abs", "max_rel", "gate");
    for (const Row& r : g_rows) {
        std::printf("%-58s %12.3e %12.3e  %s %s\n", r.name.c_str(), r.max_abs, r.max_rel, r.gate.c_str(),
                    r.pass ? "pass" : "FAIL");
        all = all && r.pass;
    }
    std::printf("timing (%d threads, best of %d): encoder %.1f ms; decoder 1 frame %.1f ms; decoder block of %d "
                "frames, %d frame(s) per graph, %.1f ms (%.1f ms/frame); peak RSS %.0f MB\n",
                n_threads, runs, enc_ms, frame_ms, n_frames, chunk > 0 ? chunk : n_frames, dec_ms, dec_ms / n_frames,
                peak_rss_mb());
    std::printf("%s: %zu checks\n", all ? "PASS" : "FAIL", g_rows.size());
    return all ? 0 : 1;
}
