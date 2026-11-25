// avatarforcing-cli: portrait + speech WAV -> lip-synced talking-head video.
//
// The pipeline follows the offline streaming path of the upstream code
// (inference.py run_stream and AvatarForcing.inference_stream); see
// docs/cli.md for the details and the differences from the live app.

#include "audio_io.hpp"

#include "backend.hpp"
#include "common.hpp"
#include "flow.hpp"
#include "motion.hpp"
#include "sampler.hpp"
#include "wav2vec2.hpp"

#include "ggml.h"
#include "gguf.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_ONLY_BMP
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <spawn.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// ---------------------------------------------------------------------------
// arguments

struct Args {
    std::string model, portrait, audio, output, frames_dir, noise_file;
    int nfe = 0;               // 0: avatarforcing.sampler.default_nfe
    float cfg_scale = -1.f;    // < 0: avatarforcing.cfg.audio_scale
    uint64_t seed = 42;
    bool normalized = true;
    int threads = 0;
    int max_seconds = 30;
};

void usage(const char* prog) {
    std::fprintf(stderr,
        "usage:\n"
        "  %s --model MODEL.gguf --portrait IMG --audio WAV [-o OUT.mp4] [options]\n"
        "  %s info --model MODEL.gguf\n"
        "\n"
        "inputs:\n"
        "  --model PATH          avatarforcing GGUF\n"
        "  --portrait PATH       pre-cropped square portrait at the model size (512x512),\n"
        "                        PNG, JPEG or BMP; no face detection is done\n"
        "  --audio PATH          speech WAV, any rate and channel count (mixed to mono,\n"
        "                        resampled to 16 kHz)\n"
        "outputs (at least one):\n"
        "  -o, --output PATH     mp4 (H.264 + AAC) written by ffmpeg, which must be on PATH\n"
        "  --frames-dir DIR      also write every frame as DIR/frame_NNNNN.png\n"
        "options:\n"
        "  --nfe N               sampler steps + 1 (default: from the GGUF, 6)\n"
        "  --cfg-scale S         audio CFG scale (default: from the GGUF, 2.0)\n"
        "  --seed N              seed of the noise generator (default 42)\n"
        "  --noise PATH          read the noise from a raw float32 file instead (primer\n"
        "                        [clip, dim], then each steady block [block, dim])\n"
        "  --audio-mode M        normalized (default, offline path) or raw (live app)\n"
        "  --max-seconds N       keep the first N s of audio (default 30, as upstream;\n"
        "                        0: no limit)\n"
        "  --threads N           CPU threads (default: all cores)\n",
        prog, prog);
}

bool parse_args(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto val = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "error: %s needs a value\n", name);
                return nullptr;
            }
            return argv[++i];
        };
        const char* v = nullptr;
        if (k == "-h" || k == "--help") { usage(argv[0]); std::exit(0); }
        else if (k == "--model" || k == "-m") { if (!(v = val("--model"))) return false; a.model = v; }
        else if (k == "--portrait") { if (!(v = val("--portrait"))) return false; a.portrait = v; }
        else if (k == "--audio") { if (!(v = val("--audio"))) return false; a.audio = v; }
        else if (k == "-o" || k == "--output") { if (!(v = val("--output"))) return false; a.output = v; }
        else if (k == "--frames-dir") { if (!(v = val("--frames-dir"))) return false; a.frames_dir = v; }
        else if (k == "--noise") { if (!(v = val("--noise"))) return false; a.noise_file = v; }
        else if (k == "--nfe") { if (!(v = val("--nfe"))) return false; a.nfe = std::atoi(v); }
        else if (k == "--cfg-scale") { if (!(v = val("--cfg-scale"))) return false; a.cfg_scale = (float)std::atof(v); }
        else if (k == "--seed") { if (!(v = val("--seed"))) return false; a.seed = std::strtoull(v, nullptr, 10); }
        else if (k == "--threads" || k == "-t") { if (!(v = val("--threads"))) return false; a.threads = std::atoi(v); }
        else if (k == "--max-seconds") { if (!(v = val("--max-seconds"))) return false; a.max_seconds = std::atoi(v); }
        else if (k == "--audio-mode") {
            if (!(v = val("--audio-mode"))) return false;
            const std::string m = v;
            if (m == "normalized") a.normalized = true;
            else if (m == "raw") a.normalized = false;
            else { std::fprintf(stderr, "error: --audio-mode must be normalized or raw\n"); return false; }
        } else {
            std::fprintf(stderr, "error: unknown argument %s\n", k.c_str());
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// info mode

int run_info(const std::string& path) {
    gguf_init_params p = {/*no_alloc=*/true, /*ctx=*/nullptr};
    gguf_context* g = gguf_init_from_file(path.c_str(), p);
    if (!g) { std::fprintf(stderr, "error: cannot read GGUF %s\n", path.c_str()); return 1; }
    std::printf("file: %s\nGGUF version %u, alignment %zu\n", path.c_str(), gguf_get_version(g),
                gguf_get_alignment(g));
    const int64_t n_kv = gguf_get_n_kv(g);
    std::printf("\nmetadata (%lld keys):\n", (long long)n_kv);
    for (int64_t i = 0; i < n_kv; ++i) {
        const char* key = gguf_get_key(g, i);
        const gguf_type t = gguf_get_kv_type(g, i);
        std::printf("  %-48s ", key);
        switch (t) {
            case GGUF_TYPE_UINT8: std::printf("%u", gguf_get_val_u8(g, i)); break;
            case GGUF_TYPE_INT8: std::printf("%d", gguf_get_val_i8(g, i)); break;
            case GGUF_TYPE_UINT16: std::printf("%u", gguf_get_val_u16(g, i)); break;
            case GGUF_TYPE_INT16: std::printf("%d", gguf_get_val_i16(g, i)); break;
            case GGUF_TYPE_UINT32: std::printf("%u", gguf_get_val_u32(g, i)); break;
            case GGUF_TYPE_INT32: std::printf("%d", gguf_get_val_i32(g, i)); break;
            case GGUF_TYPE_UINT64: std::printf("%llu", (unsigned long long)gguf_get_val_u64(g, i)); break;
            case GGUF_TYPE_INT64: std::printf("%lld", (long long)gguf_get_val_i64(g, i)); break;
            case GGUF_TYPE_FLOAT32: std::printf("%g", gguf_get_val_f32(g, i)); break;
            case GGUF_TYPE_FLOAT64: std::printf("%g", gguf_get_val_f64(g, i)); break;
            case GGUF_TYPE_BOOL: std::printf("%s", gguf_get_val_bool(g, i) ? "true" : "false"); break;
            case GGUF_TYPE_STRING: std::printf("\"%s\"", gguf_get_val_str(g, i)); break;
            case GGUF_TYPE_ARRAY: {
                const gguf_type at = gguf_get_arr_type(g, i);
                const size_t n = gguf_get_arr_n(g, i);
                const size_t show = std::min<size_t>(n, 8);
                std::printf("[%s x %zu] ", gguf_type_name(at), n);
                for (size_t j = 0; j < show; ++j) {
                    if (j) std::printf(", ");
                    if (at == GGUF_TYPE_STRING) { std::printf("\"%s\"", gguf_get_arr_str(g, i, j)); continue; }
                    const void* d = gguf_get_arr_data(g, i);
                    switch (at) {
                        case GGUF_TYPE_UINT32: std::printf("%u", ((const uint32_t*)d)[j]); break;
                        case GGUF_TYPE_INT32: std::printf("%d", ((const int32_t*)d)[j]); break;
                        case GGUF_TYPE_FLOAT32: std::printf("%g", ((const float*)d)[j]); break;
                        case GGUF_TYPE_INT64: std::printf("%lld", (long long)((const int64_t*)d)[j]); break;
                        case GGUF_TYPE_UINT64: std::printf("%llu", (unsigned long long)((const uint64_t*)d)[j]); break;
                        case GGUF_TYPE_FLOAT64: std::printf("%g", ((const double*)d)[j]); break;
                        case GGUF_TYPE_INT8: std::printf("%d", ((const int8_t*)d)[j]); break;
                        case GGUF_TYPE_UINT8: std::printf("%u", ((const uint8_t*)d)[j]); break;
                        case GGUF_TYPE_BOOL: std::printf("%s", ((const int8_t*)d)[j] ? "true" : "false"); break;
                        default: std::printf("?"); break;
                    }
                }
                if (n > show) std::printf(", ...");
                break;
            }
            default: std::printf("(type %d)", (int)t); break;
        }
        std::printf("\n");
    }

    // Tensors, grouped by the first name component.
    const int64_t n_t = gguf_get_n_tensors(g);
    struct Group { std::string name; int64_t n = 0; size_t bytes = 0; };
    std::vector<Group> groups;
    size_t total = 0;
    for (int64_t i = 0; i < n_t; ++i) {
        std::string name = gguf_get_tensor_name(g, i);
        const size_t dot = name.find('.');
        const std::string pre = dot == std::string::npos ? name : name.substr(0, dot);
        const size_t sz = gguf_get_tensor_size(g, i);
        total += sz;
        auto it = std::find_if(groups.begin(), groups.end(), [&](const Group& x) { return x.name == pre; });
        if (it == groups.end()) { groups.push_back({pre, 0, 0}); it = groups.end() - 1; }
        it->n++;
        it->bytes += sz;
    }
    std::printf("\ntensors: %lld, %.1f MB\n", (long long)n_t, total / 1e6);
    for (const auto& gr : groups)
        std::printf("  %-24s %6lld tensors %10.1f MB\n", (gr.name + ".*").c_str(), (long long)gr.n, gr.bytes / 1e6);
    gguf_free(g);
    return 0;
}

uint32_t gguf_u32(const std::string& path, const char* key, uint32_t fallback) {
    gguf_init_params p = {/*no_alloc=*/true, /*ctx=*/nullptr};
    gguf_context* g = gguf_init_from_file(path.c_str(), p);
    if (!g) return fallback;
    const int64_t id = gguf_find_key(g, key);
    uint32_t v = fallback;
    if (id >= 0 && gguf_get_kv_type(g, id) == GGUF_TYPE_UINT32) v = gguf_get_val_u32(g, id);
    gguf_free(g);
    return v;
}

// ---------------------------------------------------------------------------
// noise

// Seeded standard-normal noise: xoshiro256** seeded through splitmix64, Box-
// Muller in double precision. It does not reproduce torch.randn.
class NoiseSource {
public:
    bool open_file(const std::string& path, size_t n_needed) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) { std::fprintf(stderr, "error: cannot open noise file %s\n", path.c_str()); return false; }
        std::fseek(f, 0, SEEK_END);
        const long sz = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        if (sz < 0 || (size_t)sz < n_needed * sizeof(float)) {
            std::fprintf(stderr, "error: noise file %s has %ld bytes, the run needs %zu float32 values (%zu bytes)\n",
                         path.c_str(), sz, n_needed, n_needed * sizeof(float));
            std::fclose(f);
            return false;
        }
        file_.resize(n_needed);
        const size_t got = std::fread(file_.data(), sizeof(float), n_needed, f);
        std::fclose(f);
        if (got != n_needed) { std::fprintf(stderr, "error: short read on %s\n", path.c_str()); return false; }
        from_file_ = true;
        return true;
    }
    void seed(uint64_t s) {
        for (auto& w : st_) w = splitmix64(s);
    }
    void next(size_t n, std::vector<float>& out) {
        out.resize(n);
        if (from_file_) {
            std::memcpy(out.data(), file_.data() + pos_, n * sizeof(float));
            pos_ += n;
            return;
        }
        for (size_t i = 0; i < n; ++i) out[i] = (float)gauss();
    }

private:
    static uint64_t splitmix64(uint64_t& x) {
        uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    static uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    uint64_t next_u64() {
        const uint64_t r = rotl(st_[1] * 5, 7) * 9, t = st_[1] << 17;
        st_[2] ^= st_[0]; st_[3] ^= st_[1]; st_[1] ^= st_[2]; st_[0] ^= st_[3];
        st_[2] ^= t; st_[3] = rotl(st_[3], 45);
        return r;
    }
    double uniform() { return ((next_u64() >> 11) + 0.5) * (1.0 / 9007199254740992.0); }   // (0, 1)
    double gauss() {
        if (have_spare_) { have_spare_ = false; return spare_; }
        const double u1 = uniform(), u2 = uniform();
        const double r = std::sqrt(-2.0 * std::log(u1)), th = 6.283185307179586 * u2;
        spare_ = r * std::sin(th);
        have_spare_ = true;
        return r * std::cos(th);
    }

    uint64_t st_[4] = {0, 0, 0, 0};
    bool have_spare_ = false;
    double spare_ = 0.0;
    bool from_file_ = false;
    std::vector<float> file_;
    size_t pos_ = 0;
};

// ---------------------------------------------------------------------------
// frame output

class FrameSink {
public:
    ~FrameSink() { close(); }

    bool open(const Args& a, int w, int h, int fps) {
        w_ = w;
        h_ = h;
        if (!a.frames_dir.empty()) {
            if (mkdir(a.frames_dir.c_str(), 0755) != 0 && errno != EEXIST) {
                std::fprintf(stderr, "error: cannot create %s: %s\n", a.frames_dir.c_str(), std::strerror(errno));
                return false;
            }
            dir_ = a.frames_dir;
        }
        if (a.output.empty()) return true;
        // ffmpeg reads raw RGB frames on stdin and muxes the input audio, as
        // the upstream run_stream does (it encodes first, then muxes with
        // -shortest; one ffmpeg call gives the same result).
        const std::string size = std::to_string(w) + "x" + std::to_string(h), rate = std::to_string(fps);
        std::vector<std::string> argv_s = {
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", size, "-r", rate, "-i", "-",
            "-i", a.audio,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
            a.output};
        std::vector<char*> argv;
        for (auto& s : argv_s) argv.push_back(const_cast<char*>(s.c_str()));
        argv.push_back(nullptr);
        int fds[2];
        if (pipe(fds) != 0) { std::perror("pipe"); return false; }
        posix_spawn_file_actions_t fa;
        posix_spawn_file_actions_init(&fa);
        posix_spawn_file_actions_adddup2(&fa, fds[0], STDIN_FILENO);
        posix_spawn_file_actions_addclose(&fa, fds[0]);
        posix_spawn_file_actions_addclose(&fa, fds[1]);
        const int rc = posix_spawnp(&pid_, "ffmpeg", &fa, nullptr, argv.data(), environ);
        posix_spawn_file_actions_destroy(&fa);
        ::close(fds[0]);
        if (rc != 0) {
            ::close(fds[1]);
            std::fprintf(stderr, "error: cannot start ffmpeg: %s\n", std::strerror(rc));
            pid_ = -1;
            return false;
        }
        std::signal(SIGPIPE, SIG_IGN);
        fd_ = fds[1];
        return true;
    }

    // frames: [n, h, w, 3] uint8
    bool write(const uint8_t* frames, int n) {
        const size_t fb = (size_t)w_ * h_ * 3;
        for (int f = 0; f < n; ++f) {
            const uint8_t* p = frames + f * fb;
            if (!dir_.empty()) {
                char name[64];
                std::snprintf(name, sizeof(name), "/frame_%05d.png", count_);
                if (!stbi_write_png((dir_ + name).c_str(), w_, h_, 3, p, w_ * 3)) {
                    std::fprintf(stderr, "error: cannot write %s%s\n", dir_.c_str(), name);
                    return false;
                }
            }
            if (fd_ >= 0) {
                size_t off = 0;
                while (off < fb) {
                    const ssize_t k = ::write(fd_, p + off, fb - off);
                    if (k < 0) {
                        if (errno == EINTR) continue;
                        std::fprintf(stderr, "error: writing to ffmpeg failed: %s\n", std::strerror(errno));
                        return false;
                    }
                    off += (size_t)k;
                }
            }
            ++count_;
        }
        return true;
    }

    // Closes the pipe and waits for ffmpeg. Returns false when ffmpeg failed.
    bool close() {
        bool ok = true;
        if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
        if (pid_ > 0) {
            int st = 0;
            while (waitpid(pid_, &st, 0) < 0 && errno == EINTR) {}
            ok = WIFEXITED(st) && WEXITSTATUS(st) == 0;
            if (!ok) std::fprintf(stderr, "error: ffmpeg failed (status %d)\n", st);
            pid_ = -1;
        }
        return ok;
    }

    int count() const { return count_; }

private:
    int w_ = 0, h_ = 0, count_ = 0;
    std::string dir_;
    int fd_ = -1;
    pid_t pid_ = -1;
};

double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// ---------------------------------------------------------------------------
// generation

int run_generate(const Args& a) {
    if (a.model.empty() || a.portrait.empty() || a.audio.empty()) {
        std::fprintf(stderr, "error: --model, --portrait and --audio are required\n");
        return 2;
    }
    if (a.output.empty() && a.frames_dir.empty()) {
        std::fprintf(stderr, "error: give -o OUT.mp4 and/or --frames-dir DIR\n");
        return 2;
    }
    const int n_threads = a.threads > 0 ? a.threads : std::max(1, (int)sysconf(_SC_NPROCESSORS_ONLN));
    const auto t_all = Clock::now();

    // ---- load ----
    auto t0 = Clock::now();
    af::Backend be(n_threads);
    af::FlowModel flow(be);
    if (!flow.load(a.model)) { std::fprintf(stderr, "error: flow transformer load failed\n"); return 1; }
    af::AudioEncoder audio(be, flow);
    if (!audio.load(a.model)) { std::fprintf(stderr, "error: wav2vec2 load failed\n"); return 1; }
    af::MotionModel motion(be);
    if (!motion.load(a.model)) { std::fprintf(stderr, "error: motion autoencoder load failed\n"); return 1; }
    motion.set_chunk_frames(1);
    const double ms_load = ms_since(t0);
    const af::FlowConfig& fc = flow.config();
    const int fps = (int)gguf_u32(a.model, "avatarforcing.fps", 25);
    const int S = (int)motion.config().size;
    const int D = (int)fc.dim_w, clip = (int)fc.clip_frames, nb = (int)fc.block_frames;
    const int ctx = (int)fc.context_frames;
    const int sr = (int)audio.config().sample_rate;

    // ---- inputs ----
    int iw = 0, ih = 0, ic = 0;
    uint8_t* img = stbi_load(a.portrait.c_str(), &iw, &ih, &ic, 3);
    if (!img) { std::fprintf(stderr, "error: cannot read %s: %s\n", a.portrait.c_str(), stbi_failure_reason()); return 1; }
    if (iw != S || ih != S) {
        std::fprintf(stderr, "error: the portrait must be a pre-cropped %dx%d face image, got %dx%d\n", S, S, iw, ih);
        stbi_image_free(img);
        return 1;
    }
    // [3, S, S] in [-1, 1], (u8 / 255 - 0.5) / 0.5 in float32.
    std::vector<float> image((size_t)3 * S * S);
    for (int y = 0; y < S; ++y)
        for (int x = 0; x < S; ++x)
            for (int c = 0; c < 3; ++c)
                image[((size_t)c * S + y) * S + x] = ((float)img[((size_t)y * S + x) * 3 + c] / 255.0f - 0.5f) / 0.5f;
    stbi_image_free(img);

    af_cli::Wav wav;
    std::string err;
    if (!af_cli::read_wav_mono(a.audio, wav, err)) { std::fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
    std::vector<float> pcm = af_cli::resample(wav.pcm, wav.sample_rate, sr);
    if (pcm.empty()) { std::fprintf(stderr, "error: %s has no samples\n", a.audio.c_str()); return 1; }
    std::fprintf(stderr, "audio: %s, %d Hz, %d ch, %.2f s%s\n", a.audio.c_str(), wav.sample_rate, wav.channels,
                 wav.pcm.size() / (double)wav.sample_rate,
                 wav.sample_rate != sr ? " (resampled to 16 kHz)" : "");

    FrameSink sink;
    if (!sink.open(a, S, S, fps)) return 1;

    const auto t_gen = Clock::now();

    // ---- portrait encoder ----
    t0 = Clock::now();
    af::Portrait portrait;
    if (!motion.encode(image.data(), portrait)) { std::fprintf(stderr, "error: motion encoder failed\n"); return 1; }
    if (!motion.set_portrait(portrait)) return 1;
    const double ms_encode = ms_since(t0);

    // ---- audio features ----
    t0 = Clock::now();
    const size_t cap = a.max_seconds > 0 ? (size_t)a.max_seconds * sr : pcm.size();
    std::vector<float> feats;
    int T = 0;
    if (a.normalized) {
        // Upstream normalizes the whole file, then keeps the first 30 s.
        // A whole number of seconds is a whole number of frames (640
        // samples), so the kept part needs no padding.
        std::vector<float> x;
        int n_full = 0;
        audio.model_input(pcm.data(), pcm.size(), af::AudioInputMode::Normalized, x, n_full);
        if (pcm.size() > cap) {
            x.resize(cap);
            n_full = (int)(cap / audio.config().samples_per_frame);
        }
        std::vector<float> stacked;
        if (!audio.hidden_states(x, n_full, stacked)) { std::fprintf(stderr, "error: wav2vec2 failed\n"); return 1; }
        if (!flow.audio_projection(n_full, stacked.data(), feats)) return 1;
        T = n_full;
    } else {
        if (!audio.features(pcm.data(), std::min(cap, pcm.size()), af::AudioInputMode::Raw, feats, T)) {
            std::fprintf(stderr, "error: wav2vec2 failed\n");
            return 1;
        }
    }
    if (pcm.size() > cap)
        std::fprintf(stderr, "audio: longer than %d s, only the first %d s drive the video\n", a.max_seconds, a.max_seconds);
    const double ms_audio = ms_since(t0);
    const int E = (int)fc.audio_proj_out;   // feature width, the flow transformer dim_a

    // ---- sampler setup ----
    af::Sampler sampler(flow);
    if (!sampler.init(a.nfe, a.cfg_scale)) return 1;
    sampler.set_identity(portrait.r_s.data());
    const int n_steady = T > clip ? (T - clip + nb - 1) / nb : 0;
    NoiseSource noise;
    if (!a.noise_file.empty()) {
        if (!noise.open_file(a.noise_file, (size_t)clip * D + (size_t)n_steady * nb * D)) return 1;
    } else {
        noise.seed(a.seed);
    }
    const int n_frames_out = clip + std::max(0, T - clip);
    std::fprintf(stderr, "run: %d audio frames (%.2f s), primer %d + %d steady blocks -> %d frames, nfe %d, cfg %.2f, "
                 "%s audio, %d threads, noise %s\n",
                 T, T / (double)fps, clip, n_steady, n_frames_out, sampler.nfe(), sampler.cfg_scale(),
                 a.normalized ? "normalized" : "raw", n_threads,
                 a.noise_file.empty() ? ("seed " + std::to_string(a.seed)).c_str() : a.noise_file.c_str());

    // Audio feature rows [lo, lo + n), padded by repeating the last row of the
    // track (F.pad replicate on the slice).
    auto window = [&](int lo, int n, std::vector<float>& w) {
        w.resize((size_t)n * E);
        for (int r = 0; r < n; ++r) {
            const int src = std::min(lo + r, T - 1);
            std::memcpy(w.data() + (size_t)r * E, feats.data() + (size_t)src * E, E * sizeof(float));
        }
    };

    double ms_sampler = 0, ms_decode = 0, ms_out = 0, ms_primer = 0;
    std::vector<double> steady_ms, block_ms;
    std::vector<float> wa, nz, x, pixels;
    std::vector<uint8_t> u8;
    int decoded = 0;

    auto emit = [&](const float* latent, int n) -> bool {
        auto td = Clock::now();
        if (!motion.decode(latent, n, pixels)) { std::fprintf(stderr, "error: decoder failed\n"); return false; }
        ms_decode += ms_since(td);
        td = Clock::now();
        af::MotionModel::to_u8(pixels.data(), n, S, S, u8);
        const bool ok = sink.write(u8.data(), n);
        ms_out += ms_since(td);
        decoded += n;
        return ok;
    };

    // ---- primer: clip frames, no KV read; decoded in blocks of nb ----
    auto tb = Clock::now();
    window(0, clip, wa);
    noise.next((size_t)clip * D, nz);
    t0 = Clock::now();
    if (!sampler.primer(wa.data(), nz.data(), x)) { std::fprintf(stderr, "error: primer failed\n"); return 1; }
    ms_primer = ms_since(t0);
    ms_sampler += ms_primer;
    for (int b = 0; b < clip; b += nb) {
        if (!emit(x.data() + (size_t)b * D, std::min(nb, clip - b))) return 1;
        block_ms.push_back(ms_since(tb));
        tb = Clock::now();
    }
    const double ms_first_block = block_ms.front();

    // ---- steady blocks ----
    std::vector<float> latent;
    for (int t = clip; t < T; t += nb) {
        tb = Clock::now();
        window(t - ctx, ctx + nb, wa);
        noise.next((size_t)nb * D, nz);
        t0 = Clock::now();
        if (!sampler.steady(wa.data(), nz.data(), latent)) { std::fprintf(stderr, "error: steady block failed\n"); return 1; }
        const double ms = ms_since(t0);
        ms_sampler += ms;
        steady_ms.push_back(ms);
        // The last block keeps only the frames the audio covers.
        if (!emit(latent.data(), std::min(nb, T - t))) return 1;
        block_ms.push_back(ms_since(tb));
        std::fprintf(stderr, "\rframes %d / %d", decoded, n_frames_out);
    }
    std::fprintf(stderr, "\n");

    t0 = Clock::now();
    if (!sink.close()) return 1;
    ms_out += ms_since(t0);
    const double ms_gen = ms_since(t_gen);
    const double ms_total = ms_since(t_all);

    const double clip_s = decoded / (double)fps;
    std::printf("frames: %d (%.2f s at %d fps)\n", decoded, clip_s, fps);
    if (!a.output.empty()) std::printf("output: %s\n", a.output.c_str());
    if (!a.frames_dir.empty()) std::printf("frames dir: %s\n", a.frames_dir.c_str());
    std::printf("timings (%d threads):\n", n_threads);
    std::printf("  load            %9.1f ms\n", ms_load);
    std::printf("  portrait encode %9.1f ms\n", ms_encode);
    std::printf("  audio features  %9.1f ms  (%d frames)\n", ms_audio, T);
    std::printf("  sampler         %9.1f ms  (primer %.1f ms; steady %zu blocks, median %.1f ms/block)\n",
                ms_sampler, ms_primer, steady_ms.size(), median(steady_ms));
    std::printf("  decoder         %9.1f ms  (%.1f ms/frame)\n", ms_decode, decoded ? ms_decode / decoded : 0.0);
    std::printf("  uint8 + output  %9.1f ms\n", ms_out);
    std::printf("  generate        %9.1f ms  (encode + audio + sampler + decoder + output)\n", ms_gen);
    std::printf("  total           %9.1f ms  (with load)\n", ms_total);
    std::printf("first block latency %.1f ms, per block median %.1f ms\n", ms_first_block, median(block_ms));
    std::printf("realtime factor: %.2fx (generate / clip; < 1 is faster than real time), %.2fx with load\n",
                ms_gen / 1000.0 / clip_s, ms_total / 1000.0 / clip_s);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc >= 2 && std::strcmp(argv[1], "info") == 0) {
        Args a;
        if (argc == 3 && argv[2][0] != '-') a.model = argv[2];
        else if (!parse_args(argc - 1, argv + 1, a)) return 2;
        if (a.model.empty()) { usage(argv[0]); return 2; }
        return run_info(a.model);
    }
    Args a;
    if (argc < 2 || !parse_args(argc, argv, a)) { usage(argv[0]); return 2; }
    return run_generate(a);
}
