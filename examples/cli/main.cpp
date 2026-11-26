// avatarforcing-cli: portrait + speech WAV -> lip-synced talking-head video.
//
// The pipeline follows the offline streaming path of the upstream code
// (inference.py run_stream and AvatarForcing.inference_stream); see
// docs/cli.md for the details and the differences from the live app. The
// pipeline itself is in src/pipeline.cpp, shared with the C API.

#include "audio_io.hpp"
#include "backend.hpp"
#include "flow.hpp"
#include "live.hpp"
#include "pipeline.hpp"

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <sys/resource.h>
#include <sys/stat.h>

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
    std::string backend = "cpu";
    int decode_batch = -1;     // < 0: 1 on the CPU, a whole block elsewhere
    bool profile = false;
    int overlap = -1;          // < 0: on for a GPU backend, off on the CPU
    bool normalized_set = false;
    // live mode
    std::string live_script;
    bool realtime = false;     // pace the steps on the wall clock
    int queue_blocks = 3;
};

void usage(const char* prog) {
    std::fprintf(stderr,
        "usage:\n"
        "  %s --model MODEL.gguf --portrait IMG --audio WAV [-o OUT.mp4] [options]\n"
        "  %s info --model MODEL.gguf\n"
        "  %s --model MODEL.gguf --portrait IMG --live SCRIPT [-o OUT.mp4] [options]\n"
        "\n"
        "inputs:\n"
        "  --model PATH          avatarforcing GGUF\n"
        "  --portrait PATH       pre-cropped square portrait at the model size (512x512),\n"
        "                        PNG, JPEG or BMP; no face detection is done\n"
        "  --audio PATH          speech WAV, any rate and channel count (mixed to mono,\n"
        "                        resampled to 16 kHz)\n"
        "  --live SCRIPT         live mode instead of --audio: simulate a live call from\n"
        "                        a script of timed events (docs/cli.md, \"Live mode\")\n"
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
        "  --threads N           CPU threads (default: all cores)\n"
        "  --backend B           compute device: cpu (default), cuda, cuda:N; --device is\n"
        "                        an alias\n"
        "  --decode-batch N      frames per decoder graph (default: 1 on the CPU, a whole\n"
        "                        block of 10 on a GPU)\n"
        "  --profile             print a per-op time profile of the decoder (runs the\n"
        "                        graphs node by node; slow)\n"
        "  --overlap, --no-overlap\n"
        "                        decode block N while the sampler computes block N + 1\n"
        "                        (default: on for a GPU backend)\n"
        "live mode:\n"
        "  --realtime            step on the wall clock, one block every 400 ms, and count\n"
        "                        the late blocks (default: step as fast as possible)\n"
        "  --queue-blocks N      blocks that may wait for the writer (default 3)\n",
        prog, prog, prog);
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
        else if (k == "--backend" || k == "--device") { if (!(v = val("--backend"))) return false; a.backend = v; }
        else if (k == "--decode-batch") { if (!(v = val("--decode-batch"))) return false; a.decode_batch = std::atoi(v); }
        else if (k == "--profile") a.profile = true;
        else if (k == "--overlap") a.overlap = 1;
        else if (k == "--no-overlap") a.overlap = 0;
        else if (k == "--max-seconds") { if (!(v = val("--max-seconds"))) return false; a.max_seconds = std::atoi(v); }
        else if (k == "--live") { if (!(v = val("--live"))) return false; a.live_script = v; }
        else if (k == "--realtime") a.realtime = true;
        else if (k == "--queue-blocks") { if (!(v = val("--queue-blocks"))) return false; a.queue_blocks = std::atoi(v); }
        else if (k == "--audio-mode") {
            if (!(v = val("--audio-mode"))) return false;
            const std::string m = v;
            a.normalized_set = true;
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
    struct Group { std::string name; int64_t n = 0; size_t bytes = 0; std::vector<std::pair<std::string, int>> types; };
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
        const std::string tn = ggml_type_name(gguf_get_tensor_type(g, i));
        auto ty = std::find_if(it->types.begin(), it->types.end(), [&](const auto& x) { return x.first == tn; });
        if (ty == it->types.end()) it->types.emplace_back(tn, 1); else ty->second++;
    }
    std::printf("\ntensors: %lld, %.1f MB\n", (long long)n_t, total / 1e6);
    for (const auto& gr : groups) {
        std::printf("  %-24s %6lld tensors %10.1f MB  ", (gr.name + ".*").c_str(), (long long)gr.n, gr.bytes / 1e6);
        for (size_t j = 0; j < gr.types.size(); ++j)
            std::printf("%s%s x %d", j ? ", " : "", gr.types[j].first.c_str(), gr.types[j].second);
        std::printf("\n");
    }
    gguf_free(g);
    return 0;
}

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
    const auto t_all = Clock::now();
    std::string err;

    // ---- load ----
    af::EngineOptions eo;
    eo.backend = a.backend;
    eo.threads = a.threads;
    eo.decode_batch = a.decode_batch;
    eo.overlap = a.overlap;
    af::Engine engine;
    if (!engine.load(a.model, eo, err)) { std::fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
    af::Backend& be = engine.backend();
    const int n_threads = engine.n_threads();
    const bool overlap = engine.overlap();
    const int fps = engine.fps(), S = engine.size(), sr = engine.sample_rate();
    const af::FlowConfig& fc = engine.flow().config();
    const int clip = (int)fc.clip_frames, nb = (int)fc.block_frames;

    // ---- inputs ----
    std::vector<float> image;
    if (!af::load_portrait(a.portrait, S, image, err)) { std::fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
    af::Wav wav;
    if (!af::read_wav_mono(a.audio, wav, err)) { std::fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
    std::vector<float> pcm = af::resample(wav.pcm, wav.sample_rate, sr);
    if (pcm.empty()) { std::fprintf(stderr, "error: %s has no samples\n", a.audio.c_str()); return 1; }
    std::fprintf(stderr, "audio: %s, %d Hz, %d ch, %.2f s%s\n", a.audio.c_str(), wav.sample_rate, wav.channels,
                 wav.pcm.size() / (double)wav.sample_rate,
                 wav.sample_rate != sr ? " (resampled to 16 kHz)" : "");

    if (!a.frames_dir.empty() && mkdir(a.frames_dir.c_str(), 0755) != 0 && errno != EEXIST) {
        std::fprintf(stderr, "error: cannot create %s: %s\n", a.frames_dir.c_str(), std::strerror(errno));
        return 1;
    }
    af::VideoWriter video;
    // The video has at least the clip frames of the primer; pad shorter
    // speech with silence so that ffmpeg reads them all.
    if (!a.output.empty() && !video.open("ffmpeg", a.output, a.audio, S, S, fps, err, clip)) {
        std::fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }
    int written = 0, n_frames_out = 0;
    auto sink = [&](const uint8_t* frames, int n, std::string& e) -> bool {
        const size_t fb = (size_t)S * S * 3;
        for (int f = 0; f < n; ++f) {
            if (a.frames_dir.empty()) break;
            char name[64];
            std::snprintf(name, sizeof(name), "/frame_%05d.png", written + f);
            if (!stbi_write_png((a.frames_dir + name).c_str(), S, S, 3, frames + f * fb, S * 3)) {
                e = "cannot write " + a.frames_dir + name;
                return false;
            }
        }
        if (video.is_open() && !video.write(frames, n, e)) return false;
        written += n;
        if (!overlap) std::fprintf(stderr, "\rframes %d / %d", written, n_frames_out);
        return true;
    };

    af::GenOptions go;
    go.nfe = a.nfe;
    go.cfg_scale = a.cfg_scale;
    go.seed = a.seed;
    go.noise_file = a.noise_file;
    go.normalized = a.normalized;
    go.max_seconds = a.max_seconds;
    go.profile = a.profile;
    af::Generator gen(engine, go, sink);

    const auto t_gen = Clock::now();

    // ---- portrait encoder ----
    if (!gen.encode_portrait(image.data(), err)) { std::fprintf(stderr, "error: %s\n", err.c_str()); return 1; }

    // ---- audio features ----
    auto t0 = Clock::now();
    std::vector<float> feats;
    int T = 0;
    bool capped = false;
    if (!af::clip_features(engine, pcm, a.normalized, a.max_seconds, feats, T, capped, err)) {
        std::fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }
    if (capped)
        std::fprintf(stderr, "audio: longer than %d s, only the first %d s drive the video\n", a.max_seconds, a.max_seconds);
    const double ms_audio = ms_since(t0);

    // ---- sampler setup ----
    if (!gen.init_sampler(err) || !gen.check_noise(T, err)) { std::fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
    const int n_steady = T > clip ? (T - clip + nb - 1) / nb : 0;
    n_frames_out = af::Generator::frames_for(T, clip);
    std::fprintf(stderr, "run: %d audio frames (%.2f s), primer %d + %d steady blocks -> %d frames, nfe %d, cfg %.2f, "
                 "%s audio, %d threads, noise %s\n",
                 T, T / (double)fps, clip, n_steady, n_frames_out, gen.nfe(), gen.cfg_scale(),
                 a.normalized ? "normalized" : "raw", n_threads,
                 a.noise_file.empty() ? ("seed " + std::to_string(a.seed)).c_str() : a.noise_file.c_str());

    // ---- primer and steady blocks, decoder, output ----
    gen.add_features(feats.data(), T);
    if (!gen.finish(err)) { std::fprintf(stderr, "\nerror: %s\n", err.c_str()); return 1; }
    std::fprintf(stderr, "\n");

    const af::GenStats& st = gen.stats();
    double ms_out = st.ms_out;
    t0 = Clock::now();
    if (!video.close(err)) { std::fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
    ms_out += ms_since(t0);
    const double ms_gen = ms_since(t_gen);
    const double ms_total = ms_since(t_all);
    const int decoded = st.frames;

    size_t mem_free1 = 0, mem_total = 0;
    ggml_backend_dev_memory(ggml_backend_get_device(be.handle()), &mem_free1, &mem_total);
    const size_t mem_free0 = engine.mem_free_before_load();
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    const double clip_s = decoded / (double)fps;
    std::printf("frames: %d (%.2f s at %d fps)\n", decoded, clip_s, fps);
    if (!a.output.empty()) std::printf("output: %s\n", a.output.c_str());
    if (!a.frames_dir.empty()) std::printf("frames dir: %s\n", a.frames_dir.c_str());
    std::printf("timings (%s, %d threads):\n", be.device().c_str(), n_threads);
    std::printf("  load            %9.1f ms\n", engine.load_ms());
    std::printf("  portrait encode %9.1f ms\n", st.ms_encode);
    std::printf("  audio features  %9.1f ms  (%d frames)\n", ms_audio, T);
    std::printf("  sampler         %9.1f ms  (primer %.1f ms; steady %zu blocks, median %.1f ms/block)\n",
                st.ms_sampler, st.ms_primer, st.steady_ms.size(), median(st.steady_ms));
    std::printf("  decoder         %9.1f ms  (%.1f ms/frame; full %d-frame block median %.1f ms, %d frame(s) per graph)\n",
                st.ms_decode, decoded ? st.ms_decode / decoded : 0.0, nb, median(st.decode_ms),
                engine.decode_batch() > 0 ? engine.decode_batch() : nb);
    std::printf("  uint8 + output  %9.1f ms\n", ms_out);
    std::printf("  generate        %9.1f ms  (encode + audio + sampler + decoder + output)\n", ms_gen);
    std::printf("  total           %9.1f ms  (with load)\n", ms_total);
    std::printf("first block latency %.1f ms (primer + first decoded block)\n", st.ms_first_block);
    std::printf("steady blocks (%d frames each, %.0f ms of video): sampler median %.1f ms + decoder median %.1f ms "
                "= %.1f ms; period between finished blocks median %.1f ms (%s)\n",
                nb, 1000.0 * nb / fps, median(st.steady_ms), median(st.decode_ms),
                median(st.steady_ms) + median(st.decode_ms), median(st.period_ms),
                overlap ? "decoder overlapped with the sampler" : "no overlap");
    if (a.profile) engine.decoder_backend().print_profile("decoder (all blocks)");
    std::printf("memory: peak RSS %.0f MB; device memory in use at the end %.0f MB (free %.0f -> %.0f MB of %.0f MB)\n",
                ru.ru_maxrss / 1024.0, ((double)mem_free0 - (double)mem_free1) / 1048576.0, mem_free0 / 1048576.0,
                mem_free1 / 1048576.0, mem_total / 1048576.0);
    std::printf("realtime factor: %.2fx (generate / clip; < 1 is faster than real time), %.2fx with load\n",
                ms_gen / 1000.0 / clip_s, ms_total / 1000.0 / clip_s);
    return 0;
}


// ---------------------------------------------------------------------------
// live mode: a simulated call

struct LiveEvent {
    double t = 0;          // seconds after frame 0
    std::string kind;      // say, interrupt, end
    std::string path;      // say: the WAV file
    int line = 0;
};

// Script lines: "<seconds> say <wav>", "<seconds> interrupt", "<seconds> end".
// '#' starts a comment. A relative WAV path is relative to the script.
bool read_script(const std::string& path, std::vector<LiveEvent>& ev, std::string& err) {
    std::ifstream f(path);
    if (!f) { err = "cannot open " + path; return false; }
    std::string dir;
    const size_t slash = path.rfind('/');
    if (slash != std::string::npos) dir = path.substr(0, slash + 1);
    std::string line;
    int no = 0;
    while (std::getline(f, line)) {
        ++no;
        const size_t hash = line.find('#');
        if (hash != std::string::npos) line.resize(hash);
        std::istringstream in(line);
        LiveEvent e;
        if (!(in >> e.t)) {
            std::string rest;
            if (std::istringstream(line) >> rest) { err = path + ":" + std::to_string(no) + ": expected a time"; return false; }
            continue;
        }
        e.line = no;
        if (!(in >> e.kind) || (e.kind != "say" && e.kind != "interrupt" && e.kind != "end")) {
            err = path + ":" + std::to_string(no) + ": expected say, interrupt or end";
            return false;
        }
        if (e.kind == "say") {
            if (!(in >> e.path)) { err = path + ":" + std::to_string(no) + ": say needs a WAV file"; return false; }
            if (e.path[0] != '/') e.path = dir + e.path;
        }
        if (e.t < 0) { err = path + ":" + std::to_string(no) + ": negative time"; return false; }
        ev.push_back(e);
    }
    std::stable_sort(ev.begin(), ev.end(), [](const LiveEvent& x, const LiveEvent& y) { return x.t < y.t; });
    return true;
}

int run_live(const Args& a) {
    if (a.model.empty() || a.portrait.empty()) {
        std::fprintf(stderr, "error: --model, --portrait and --live are required\n");
        return 2;
    }
    if (a.output.empty() && a.frames_dir.empty()) {
        std::fprintf(stderr, "error: give -o OUT.mp4 and/or --frames-dir DIR\n");
        return 2;
    }
    std::string err;
    std::vector<LiveEvent> events;
    if (!read_script(a.live_script, events, err)) { std::fprintf(stderr, "error: %s\n", err.c_str()); return 1; }

    af::EngineOptions eo;
    eo.backend = a.backend;
    eo.threads = a.threads;
    eo.decode_batch = a.decode_batch;
    eo.overlap = a.overlap;
    af::Engine engine;
    if (!engine.load(a.model, eo, err)) { std::fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
    const int fps = engine.fps(), S = engine.size(), sr = engine.sample_rate();
    const int nb = (int)engine.flow().config().block_frames;
    const int spf = sr / fps;   // samples per frame (640)

    // The call ends at the "end" event, or 2 s after the last event.
    double t_end = events.empty() ? 4.0 : events.back().t + 2.0;
    for (const LiveEvent& e : events)
        if (e.kind == "end") { t_end = e.t; break; }
    const int64_t end_frame = (int64_t)std::llround(t_end * fps);

    std::vector<float> image;
    if (!af::load_portrait(a.portrait, S, image, err)) { std::fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
    if (!a.frames_dir.empty() && mkdir(a.frames_dir.c_str(), 0755) != 0 && errno != EEXIST) {
        std::fprintf(stderr, "error: cannot create %s: %s\n", a.frames_dir.c_str(), std::strerror(errno));
        return 1;
    }
    af::VideoWriter video;
    const std::string video_tmp = a.output.empty() ? "" : a.output + ".video.mp4";
    if (!a.output.empty() && !video.open("ffmpeg", video_tmp, "", S, S, fps, err)) {
        std::fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }

    af::GenOptions go;
    go.nfe = a.nfe;
    go.cfg_scale = a.cfg_scale;
    go.seed = a.seed;
    go.noise_file = a.noise_file;
    go.normalized = a.normalized_set ? a.normalized : false;   // raw, as the live app
    go.profile = a.profile;
    af::LiveOptions lo;
    lo.queue_blocks = a.queue_blocks;
    af::LiveSession live(engine, go, lo);
    auto t0 = Clock::now();
    const auto t_open = t0;
    double ms_first_frame = -1;
    if (!live.open(image.data(), err)) { std::fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
    const double ms_open = ms_since(t0);
    std::fprintf(stderr, "live: %lld frames (%.2f s), nfe %d, %s audio, open (portrait + hidden primer) %.0f ms\n",
                 (long long)end_frame, t_end, live.generator().nfe(), go.normalized ? "normalized" : "raw", ms_open);

    // The audio track: each phrase at its start frame, cut at interrupts.
    std::vector<float> mix((size_t)end_frame * spf, 0.f);
    struct Placed { int64_t start, end; std::vector<float> pcm; };
    std::vector<Placed> placed;

    const size_t fb = live.frame_bytes();
    std::vector<uint8_t> buf(fb * (size_t)nb);
    int64_t written = 0;
    auto write_frames = [&](const uint8_t* p, int n) -> bool {
        for (int f = 0; f < n && !a.frames_dir.empty(); ++f) {
            char name[64];
            std::snprintf(name, sizeof(name), "/frame_%05lld.png", (long long)(written + f));
            if (!stbi_write_png((a.frames_dir + name).c_str(), S, S, 3, p + f * fb, S * 3)) {
                err = "cannot write " + a.frames_dir + name;
                return false;
            }
        }
        if (written == 0 && n > 0) ms_first_frame = ms_since(t_open);
        if (video.is_open() && !video.write(p, n, err)) return false;
        written += n;
        return true;
    };
    auto drain = [&](bool wait_for_one) -> bool {
        for (;;) {
            const int n = live.pull(buf.data(), nb);
            if (n < 0) { live.failed(&err); return false; }
            if (n > 0) {
                if (!write_frames(buf.data(), n)) return false;
                wait_for_one = false;
                continue;
            }
            if (!wait_for_one) return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    };

    std::vector<double> idle_ms, speech_ms;
    size_t next_ev = 0;
    int late = 0, queue_full = 0;
    const auto t_call = Clock::now();
    int64_t next = 0;
    while (next < end_frame) {
        // Events whose frame is in the next block happen before its step: the
        // simulated caller runs at least one block ahead of playback.
        while (next_ev < events.size() && (int64_t)std::llround(events[next_ev].t * fps) < next + nb) {
            const LiveEvent& e = events[next_ev++];
            const int64_t f = (int64_t)std::llround(e.t * fps);
            if (e.kind == "say") {
                std::vector<float> pcm;
                if (!af::load_audio(e.path, sr, pcm, err)) { std::fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
                const auto tp = Clock::now();
                const int64_t st = live.push_phrase(pcm.data(), pcm.size(), f, live.epoch(), err);
                if (st < 0) { std::fprintf(stderr, "error: line %d: %s\n", e.line, err.c_str()); return 1; }
                const int64_t n_fr = ((int64_t)pcm.size() + spf - 1) / spf;
                std::fprintf(stderr, "  %6.2f s  say %s: %.2f s, frames %lld..%lld (asked %lld), wav2vec2 %.0f ms\n",
                             e.t, e.path.c_str(), pcm.size() / (double)sr, (long long)st, (long long)(st + n_fr),
                             (long long)f, ms_since(tp));
                placed.push_back({st, st + n_fr, std::move(pcm)});
            } else if (e.kind == "interrupt") {
                const int64_t c = live.interrupt(f);
                for (Placed& p : placed) p.end = std::min(p.end, std::max(p.start, c));
                std::fprintf(stderr, "  %6.2f s  interrupt at frame %lld: idle from frame %lld\n", e.t, (long long)f,
                             (long long)c);
            }
        }
        // Realtime: block k is due at k * 400 ms; step one block ahead.
        if (a.realtime) {
            const auto due = t_call + std::chrono::microseconds((int64_t)((next - nb) * 1e6 / fps));
            std::this_thread::sleep_until(due);
        }
        const auto ts = Clock::now();
        const int64_t r = live.step(err);
        if (r == af::LiveSession::kQueueFull) {
            ++queue_full;
            if (!drain(true)) { std::fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
            continue;
        }
        if (r < 0) { std::fprintf(stderr, "error: step: %s\n", err.c_str()); return 1; }
        const double ms = ms_since(ts);
        bool speech = false;
        for (const Placed& p : placed) speech = speech || (p.start < r + nb && p.end > r);
        (speech ? speech_ms : idle_ms).push_back(ms);
        if (a.realtime && ts + std::chrono::microseconds((int64_t)(ms * 1000)) >
                              t_call + std::chrono::microseconds((int64_t)(r * 1e6 / fps)))
            ++late;
        next = r + nb;
        if (!drain(false)) { std::fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
        std::fprintf(stderr, "\rframes %lld / %lld", (long long)next, (long long)end_frame);
    }
    while (written < next)
        if (!drain(true)) { std::fprintf(stderr, "\nerror: %s\n", err.c_str()); return 1; }
    std::fprintf(stderr, "\n");

    for (const Placed& p : placed)
        for (int64_t i = 0; i < (p.end - p.start) * spf && i < (int64_t)p.pcm.size(); ++i) {
            const int64_t k = p.start * spf + i;
            if (k < (int64_t)mix.size()) mix[k] += p.pcm[i];
        }
    if (video.is_open()) {
        if (!video.close(err)) { std::fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
        const std::string wav_tmp = a.output + ".audio.wav";
        if (!af::write_wav_f32(wav_tmp, mix.data(), mix.size(), sr, err) ||
            !af::mux_audio("ffmpeg", video_tmp, wav_tmp, a.output, err)) {
            std::fprintf(stderr, "error: %s\n", err.c_str());
            return 1;
        }
        std::remove(video_tmp.c_str());
        std::remove(wav_tmp.c_str());
    }
    auto stat = [](std::vector<double> v, const char* name) {
        if (v.empty()) { std::printf("  %-7s   0 blocks\n", name); return; }
        std::sort(v.begin(), v.end());
        double sum = 0;
        for (double x : v) sum += x;
        std::printf("  %-7s %3zu blocks: median %.1f ms, mean %.1f ms, p90 %.1f ms, max %.1f ms\n", name, v.size(),
                    v[v.size() / 2], sum / v.size(), v[(v.size() * 9) / 10], v.back());
    };
    std::printf("frames: %lld (%.2f s at %d fps)\n", (long long)written, written / (double)fps, fps);
    if (!a.output.empty()) std::printf("output: %s\n", a.output.c_str());
    if (!a.frames_dir.empty()) std::printf("frames dir: %s\n", a.frames_dir.c_str());
    std::printf("open: %.1f ms (portrait encoder and hidden primer, primer %.1f ms)\n", ms_open,
                live.generator().stats().ms_primer);
    std::printf("first frame: %.1f ms after the start of open (open, first step, first decoded block)\n",
                ms_first_frame);
    std::printf("step time per block (%s, %d threads, %s; budget %.0f ms):\n", engine.backend().device().c_str(),
                engine.n_threads(), engine.overlap() ? "decoder overlapped" : "no overlap", 1000.0 * nb / fps);
    stat(idle_ms, "idle");
    stat(speech_ms, "speech");
    std::printf("decoder: median %.1f ms per block\n", median(live.generator().stats().decode_ms));
    if (a.realtime) std::printf("realtime: %d late block(s) of %zu\n", late, idle_ms.size() + speech_ms.size());
    std::printf("queue full: %d\nstats: %s\n", queue_full, live.stats_json().c_str());
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
    if (!a.live_script.empty()) return run_live(a);
    return run_generate(a);
}
