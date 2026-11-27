#include "pipeline.hpp"

#include "audio_io.hpp"
#include "common.hpp"

#include "ggml-backend.h"
#include "gguf.h"

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_ONLY_BMP
#include "stb_image.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;

namespace af {

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
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

// "what failed" plus the last log message of this thread, when there is one.
std::string with_cause(const std::string& what) {
    const std::string& l = last_log();
    return l.empty() ? what : what + ": " + l;
}

} // namespace

// ---------------------------------------------------------------------------
// Engine

Engine::~Engine() {
    motion_.reset();
    audio_.reset();
    flow_.reset();
    be_dec_.reset();
    be_.reset();
}

bool Engine::load(const std::string& path, const EngineOptions& o, std::string& err) {
    const auto t0 = Clock::now();
    clear_last_log();
    {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) { err = "cannot open model " + path + ": " + std::strerror(errno); return false; }
        std::fclose(f);
        gguf_init_params p = {/*no_alloc=*/true, /*ctx=*/nullptr};
        gguf_context* g = gguf_init_from_file(path.c_str(), p);
        if (!g) { err = "not a GGUF file: " + path; return false; }
        gguf_free(g);
    }
    n_threads_ = o.threads > 0 ? o.threads : std::max(1, (int)sysconf(_SC_NPROCESSORS_ONLN));
    be_.reset(new Backend(n_threads_, o.backend));
    if (!be_->ok()) { err = "backend " + o.backend + " is not available"; return false; }
    size_t total = 0;
    ggml_backend_dev_memory(ggml_backend_get_device(be_->handle()), &mem_free0_, &total);
    flow_.reset(new FlowModel(*be_));
    if (!flow_->load(path)) { err = with_cause("flow transformer load failed"); return false; }
    audio_.reset(new AudioEncoder(*be_, *flow_));
    if (!audio_->load(path)) { err = with_cause("wav2vec2 load failed"); return false; }
    // With overlap, the decoder gets its own backend instance (on a GPU: its
    // own stream, graph allocator and buffers) and runs in a second thread.
    overlap_ = o.overlap < 0 ? !be_->is_cpu() : o.overlap > 0;
    if (overlap_) {
        be_dec_.reset(new Backend(n_threads_, o.backend));
        if (!be_dec_->ok()) { err = "backend " + o.backend + " is not available for the decoder"; return false; }
        // The sampler is the latency-bound stage: its many small kernels get
        // the free SMs first while the decoder fills the rest of the GPU.
        // AVATARFORCING_SAMPLER_PRIORITY=0 turns this off.
        const char* sp = std::getenv("AVATARFORCING_SAMPLER_PRIORITY");
        if (!(sp && std::strcmp(sp, "0") == 0)) be_->set_high_priority(true);
    }
    Backend& bed = decoder_backend();
    motion_.reset(new MotionModel(bed));
    if (!motion_->load(path)) { err = with_cause("motion autoencoder load failed"); return false; }
    decode_batch_ = o.decode_batch >= 0 ? o.decode_batch : (bed.is_cpu() ? 1 : 0);
    motion_->set_chunk_frames(decode_batch_);
    fps_ = (int)gguf_u32(path, "avatarforcing.fps", 25);
    size_ = (int)motion_->config().size;
    sample_rate_ = (int)audio_->config().sample_rate;
    load_ms_ = ms_since(t0);
    return true;
}

// ---------------------------------------------------------------------------
// inputs

bool load_portrait(const std::string& path, int S, std::vector<float>& image, std::string& err) {
    int iw = 0, ih = 0, ic = 0;
    uint8_t* img = stbi_load(path.c_str(), &iw, &ih, &ic, 3);
    if (!img) { err = "cannot read " + path + ": " + stbi_failure_reason(); return false; }
    if (iw != S || ih != S) {
        err = "the portrait must be a pre-cropped " + std::to_string(S) + "x" + std::to_string(S) +
              " face image, got " + std::to_string(iw) + "x" + std::to_string(ih);
        stbi_image_free(img);
        return false;
    }
    // [3, S, S] in [-1, 1], (u8 / 255 - 0.5) / 0.5 in float32.
    image.resize((size_t)3 * S * S);
    for (int y = 0; y < S; ++y)
        for (int x = 0; x < S; ++x)
            for (int c = 0; c < 3; ++c)
                image[((size_t)c * S + y) * S + x] = ((float)img[((size_t)y * S + x) * 3 + c] / 255.0f - 0.5f) / 0.5f;
    stbi_image_free(img);
    return true;
}

bool load_audio(const std::string& path, int sr, std::vector<float>& pcm, std::string& err) {
    Wav wav;
    if (!read_wav_mono(path, wav, err)) return false;
    pcm = resample(wav.pcm, wav.sample_rate, sr);
    if (pcm.empty()) { err = path + " has no samples"; return false; }
    return true;
}

bool clip_features(Engine& e, const std::vector<float>& pcm, bool normalized, int max_seconds,
                   std::vector<float>& feats, int& T, bool& capped, std::string& err) {
    AudioEncoder& audio = e.audio();
    const int sr = (int)audio.config().sample_rate;
    const size_t cap = max_seconds > 0 ? (size_t)max_seconds * sr : pcm.size();
    capped = pcm.size() > cap;
    clear_last_log();
    if (normalized) {
        // Upstream normalizes the whole file, then keeps the first 30 s.
        // A whole number of seconds is a whole number of frames (640
        // samples), so the kept part needs no padding.
        std::vector<float> x;
        int n_full = 0;
        audio.model_input(pcm.data(), pcm.size(), AudioInputMode::Normalized, x, n_full);
        if (pcm.size() > cap) {
            x.resize(cap);
            n_full = (int)(cap / audio.config().samples_per_frame);
        }
        std::vector<float> stacked;
        if (!audio.hidden_states(x, n_full, stacked)) { err = with_cause("wav2vec2 failed"); return false; }
        if (!e.flow().audio_projection(n_full, stacked.data(), feats)) { err = with_cause("audio projection failed"); return false; }
        T = n_full;
    } else {
        if (!audio.features(pcm.data(), std::min(cap, pcm.size()), AudioInputMode::Raw, feats, T)) {
            err = with_cause("wav2vec2 failed");
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// noise

bool NoiseSource::open_file(const std::string& path, std::string& err) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { err = "cannot open noise file " + path; return false; }
    std::fseek(f, 0, SEEK_END);
    const long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (sz < 0) { std::fclose(f); err = "cannot read noise file " + path; return false; }
    file_.resize((size_t)sz / sizeof(float));
    const size_t got = std::fread(file_.data(), sizeof(float), file_.size(), f);
    std::fclose(f);
    if (got != file_.size()) { err = "short read on " + path; return false; }
    from_file_ = true;
    path_ = path;
    pos_ = 0;
    return true;
}

bool NoiseSource::require(size_t n, std::string& err) const {
    if (!from_file_ || file_.size() >= n) return true;
    char buf[512];
    std::snprintf(buf, sizeof(buf), "noise file %s has %zu bytes, the run needs %zu float32 values (%zu bytes)",
                  path_.c_str(), file_.size() * sizeof(float), n, n * sizeof(float));
    err = buf;
    return false;
}

void NoiseSource::seed(uint64_t s) {
    for (auto& w : st_) w = splitmix64(s);
    have_spare_ = false;
    from_file_ = false;
}

bool NoiseSource::next(size_t n, std::vector<float>& out, std::string& err) {
    out.resize(n);
    if (from_file_) {
        if (pos_ + n > file_.size()) { err = "noise file " + path_ + " is too short for this run"; return false; }
        std::memcpy(out.data(), file_.data() + pos_, n * sizeof(float));
        pos_ += n;
        return true;
    }
    for (size_t i = 0; i < n; ++i) out[i] = (float)gauss();
    return true;
}

uint64_t NoiseSource::splitmix64(uint64_t& x) {
    uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

uint64_t NoiseSource::next_u64() {
    const uint64_t r = rotl(st_[1] * 5, 7) * 9, t = st_[1] << 17;
    st_[2] ^= st_[0]; st_[3] ^= st_[1]; st_[1] ^= st_[2]; st_[0] ^= st_[3];
    st_[2] ^= t; st_[3] = rotl(st_[3], 45);
    return r;
}

double NoiseSource::uniform() { return ((next_u64() >> 11) + 0.5) * (1.0 / 9007199254740992.0); }   // (0, 1)

double NoiseSource::gauss() {
    if (have_spare_) { have_spare_ = false; return spare_; }
    const double u1 = uniform(), u2 = uniform();
    const double r = std::sqrt(-2.0 * std::log(u1)), th = 6.283185307179586 * u2;
    spare_ = r * std::sin(th);
    have_spare_ = true;
    return r * std::cos(th);
}

// ---------------------------------------------------------------------------
// Generator

bool Generator::Stage::push(Job&& j, const std::atomic<bool>& failed) {
    std::unique_lock<std::mutex> lk(mu);
    cv.wait(lk, [&] { return q.size() < 2 || failed; });
    if (failed) return false;
    q.push_back(std::move(j));
    cv.notify_all();
    return true;
}

bool Generator::Stage::pop(Job& j) {
    std::unique_lock<std::mutex> lk(mu);
    cv.wait(lk, [&] { return closing || !q.empty(); });
    if (q.empty()) return false;
    j = std::move(q.front());
    q.pop_front();
    cv.notify_all();
    return true;
}

void Generator::Stage::close() {
    std::lock_guard<std::mutex> lk(mu);
    closing = true;
    cv.notify_all();
}

Generator::Generator(Engine& e, const GenOptions& o, FrameSink sink, const std::atomic<bool>* cancel)
    : e_(e), o_(o), sink_(std::move(sink)), cancel_(cancel) {
    const FlowConfig& fc = e_.flow().config();
    S_ = e_.size();
    D_ = (int)fc.dim_w;
    E_ = (int)fc.audio_proj_out;   // feature width, the flow transformer dim_a
    clip_ = (int)fc.clip_frames;
    nb_ = (int)fc.block_frames;
    ctx_ = (int)fc.context_frames;
    n_primer_blocks_ = (clip_ + nb_ - 1) / nb_;
}

Generator::~Generator() {
    // Closed before the end: the threads drop the queued blocks.
    if (threads_) fail("closed");
    stop_threads();
}

void Generator::stop_threads() {
    if (!threads_) return;
    to_decode_.close();
    if (dec_thread_.joinable()) dec_thread_.join();
    to_output_.close();
    if (out_thread_.joinable()) out_thread_.join();
    threads_ = false;
}

void Generator::fail(const std::string& msg) {
    {
        std::lock_guard<std::mutex> lk(err_mu_);
        if (thread_err_.empty()) thread_err_ = msg;
    }
    failed_ = true;
    // Wake both stages under their locks, so no waiter misses the flag.
    { std::lock_guard<std::mutex> lk(to_decode_.mu); to_decode_.cv.notify_all(); }
    { std::lock_guard<std::mutex> lk(to_output_.mu); to_output_.cv.notify_all(); }
}

bool Generator::failed(std::string* err) const {
    if (!failed_) return false;
    if (err) {
        std::lock_guard<std::mutex> lk(err_mu_);
        *err = thread_err_.empty() ? "generation failed" : thread_err_;
    }
    return true;
}

bool Generator::failed_error(std::string& err) {
    std::lock_guard<std::mutex> lk(err_mu_);
    err = thread_err_.empty() ? "generation failed" : thread_err_;
    return false;
}

bool Generator::encode_portrait(const float* image, std::string& err) {
    const auto t0 = Clock::now();
    clear_last_log();
    MotionModel& motion = e_.motion();
    if (!motion.encode(image, portrait_)) { err = with_cause("motion encoder failed"); return false; }
    if (!motion.set_portrait(portrait_)) { err = with_cause("cannot upload the portrait features"); return false; }
    stats_.ms_encode = ms_since(t0);
    return true;
}

bool Generator::init_sampler(std::string& err) {
    clear_last_log();
    sampler_.reset(new Sampler(e_.flow()));
    if (!sampler_->init(o_.nfe, o_.cfg_scale)) { err = with_cause("sampler setup failed"); return false; }
    sampler_->set_identity(portrait_.r_s.data());
    if (!o_.noise_file.empty()) {
        if (!noise_.open_file(o_.noise_file, err)) return false;
    } else {
        noise_.seed(o_.seed);
    }
    if (e_.overlap()) {
        threads_ = true;
        dec_thread_ = std::thread([this] {
            clear_last_log();
            Job job;
            std::string err;
            while (to_decode_.pop(job)) {
                if (failed_) continue;
                if (job.data.empty()) {   // a held block: nothing to decode
                    if (!to_output_.push(std::move(job), failed_)) fail("generation failed");
                    continue;
                }
                Job done{{}, job.n};
                if (!decode(job.data.data(), job.n, done.data, err)) { fail(err); continue; }
                if (!to_output_.push(std::move(done), failed_)) fail("generation failed");
            }
            to_output_.close();
        });
        out_thread_ = std::thread([this] {
            clear_last_log();
            Job job;
            std::string err;
            while (to_output_.pop(job)) {
                if (failed_) continue;
                const bool ok = job.data.empty() ? sink_(nullptr, job.n, err) : output(job.data, job.n, err);
                if (!ok) fail(err.empty() ? "writing frames failed" : err);
            }
        });
    }
    return true;
}

bool Generator::check_noise(int T, std::string& err) const {
    const int n_steady = T > clip_ ? (T - clip_ + nb_ - 1) / nb_ : 0;
    return noise_.require((size_t)clip_ * D_ + (size_t)n_steady * nb_ * D_, err);
}

void Generator::add_features(const float* rows, int n) {
    if (n <= 0) return;
    feats_.insert(feats_.end(), rows, rows + (size_t)n * E_);
    n_rows_ += n;
}

// Audio feature rows [lo, lo + n), padded by repeating the last row of the
// track (F.pad replicate on the slice).
void Generator::window(int lo, int n, std::vector<float>& w) const {
    const int T = n_rows_;
    w.resize((size_t)n * E_);
    for (int r = 0; r < n; ++r) {
        const int src = std::min(lo + r, T - 1);
        std::memcpy(w.data() + (size_t)r * E_, feats_.data() + (size_t)src * E_, E_ * sizeof(float));
    }
}

bool Generator::decode(const float* latent, int n, std::vector<float>& px, std::string& err) {
    const auto td = Clock::now();
    Backend& bed = e_.decoder_backend();
    if (o_.profile) bed.set_profile(true);
    const bool ok = e_.motion().decode(latent, n, px);
    if (o_.profile) bed.set_profile(false);
    if (!ok) { err = with_cause("decoder failed"); return false; }
    const double dms = ms_since(td);
    stats_.ms_decode += dms;
    if (n == nb_) stats_.decode_ms.push_back(dms);
    return true;
}

// Convert and hand over one decoded block. period_ms: time between two
// finished steady blocks, the pipeline throughput.
bool Generator::output(const std::vector<float>& px, int n, std::string& err) {
    const auto td = Clock::now();
    MotionModel::to_u8(px.data(), n, S_, S_, u8_);
    const bool ok = sink_(u8_.data(), n, err);
    stats_.ms_out += ms_since(td);
    stats_.frames += n;
    const auto now = Clock::now();
    if (blocks_done_ == 0) stats_.ms_first_block = std::chrono::duration<double, std::milli>(now - t_start_).count();
    if (blocks_done_ >= n_primer_blocks_)
        stats_.period_ms.push_back(std::chrono::duration<double, std::milli>(now - t_last_done_).count());
    t_last_done_ = now;
    ++blocks_done_;
    if (!ok && err.empty()) err = "writing frames failed";
    return ok;
}

bool Generator::submit(const float* latent, int n, std::string& err) {
    if (n > 0) last_latent_.assign(latent + (size_t)(n - 1) * D_, latent + (size_t)n * D_);
    if (!threads_) return decode(latent, n, pixels_, err) && output(pixels_, n, err);
    if (!to_decode_.push(Job{std::vector<float>(latent, latent + (size_t)n * D_), n}, failed_))
        return failed_error(err);
    return true;
}

bool Generator::submit_hold(int n, std::string& err) {
    if (!threads_) {
        if (sink_(nullptr, n, err)) return true;
        if (err.empty()) err = "writing frames failed";
        return false;
    }
    if (!to_decode_.push(Job{{}, n}, failed_)) return failed_error(err);
    return true;
}

float Generator::latent_fade_weight(int k, int total) {
    return (float)(0.5 * (1.0 + std::cos(3.14159265358979323846 * (k + 1) / (double)(total + 1))));
}

void Generator::apply_fade(float* latent, int n, bool primer_rows) {
    for (int j = 0; j < n && fade_i_ < fade_total_; ++j, ++fade_i_) {
        const float w = latent_fade_weight(fade_i_, fade_total_);
        float* row = latent + (size_t)j * D_;
        for (int d = 0; d < D_; ++d) {
            const float ref = primer_rows ? row[d] : fade_ref_[d];
            row[d] += w * (fade_old_[d] - ref);
        }
    }
}

bool Generator::run(bool final, std::string& err) {
    if (!sampler_) { err = "the sampler is not set up"; return false; }
    if (failed_) return failed_error(err);
    if (done_) { err = "the audio has already ended"; return false; }
    const int T = n_rows_;
    auto cancelled = [&] {
        if (cancel_ && cancel_->load()) { err = "cancelled"; return true; }
        return false;
    };

    // ---- primer: clip frames, no KV read; decoded in blocks of nb ----
    if (!primed_) {
        if (T == 0 || (!final && T < clip_)) {
            if (final) done_ = true;
            return true;
        }
        if (cancelled()) return false;
        t_start_ = Clock::now();
        t_last_done_ = t_start_;
        window(0, clip_, wa_);
        if (!noise_.next((size_t)clip_ * D_, nz_, err)) return false;
        clear_last_log();
        const auto t0 = Clock::now();
        if (!sampler_->primer(wa_.data(), nz_.data(), x_)) { err = with_cause("primer failed"); return false; }
        stats_.ms_primer = ms_since(t0);
        stats_.ms_sampler += stats_.ms_primer;
        primed_ = true;
        t_ = clip_;
        for (int b = 0; b < clip_; b += nb_)
            if (!submit(x_.data() + (size_t)b * D_, std::min(nb_, clip_ - b), err)) return false;
    }

    // ---- steady blocks ----
    while (final ? t_ < T : t_ + nb_ <= T) {
        if (cancelled()) return false;
        if (failed_) return failed_error(err);
        window(t_ - ctx_, ctx_ + nb_, wa_);
        if (!noise_.next((size_t)nb_ * D_, nz_, err)) return false;
        clear_last_log();
        const auto t0 = Clock::now();
        if (!sampler_->steady(wa_.data(), nz_.data(), latent_)) { err = with_cause("steady block failed"); return false; }
        const double ms = ms_since(t0);
        stats_.ms_sampler += ms;
        stats_.steady_ms.push_back(ms);
        // The last block keeps only the frames the audio covers.
        if (!submit(latent_.data(), std::min(nb_, T - t_), err)) return false;
        t_ += nb_;
    }
    if (final) done_ = true;
    return true;
}

bool Generator::primer_hidden(const float* wa, std::string& err) {
    if (!sampler_) { err = "the sampler is not set up"; return false; }
    if (primed_) { err = "the primer has already run"; return false; }
    if (cancel_ && cancel_->load()) { err = "cancelled"; return false; }
    t_start_ = Clock::now();
    t_last_done_ = t_start_;
    n_primer_blocks_ = 0;   // no primer frames: every output block is a steady block
    if (!noise_.next((size_t)clip_ * D_, nz_, err)) return false;
    clear_last_log();
    const auto t0 = Clock::now();
    if (!sampler_->primer(wa, nz_.data(), x_)) { err = with_cause("primer failed"); return false; }
    stats_.ms_primer = ms_since(t0);
    stats_.ms_sampler += stats_.ms_primer;
    primed_ = true;
    t_ = clip_;
    return true;
}

bool Generator::steady_block(const float* wa, std::string& err) {
    if (!primed_) { err = "steady block before the primer"; return false; }
    if (failed_) return failed_error(err);
    if (cancel_ && cancel_->load()) { err = "cancelled"; return false; }
    if (!noise_.next((size_t)nb_ * D_, nz_, err)) return false;
    clear_last_log();
    const auto t0 = Clock::now();
    if (!sampler_->steady(wa, nz_.data(), latent_)) { err = with_cause("steady block failed"); return false; }
    const double ms = ms_since(t0);
    stats_.ms_sampler += ms;
    stats_.steady_ms.push_back(ms);
    const float* lat = latent_.data();
    if (fade_i_ < fade_total_) {   // a motion fade after a reanchor
        fade_lat_ = latent_;
        apply_fade(fade_lat_.data(), nb_, false);
        lat = fade_lat_.data();
    }
    if (!submit(lat, nb_, err)) return false;
    t_ += nb_;
    return true;
}

bool Generator::reanchor_block(const float* wa, bool emit, int latent_fade, std::string& err) {
    if (!primed_) { err = "reanchor before the primer"; return false; }
    if (failed_) return failed_error(err);
    if (cancel_ && cancel_->load()) { err = "cancelled"; return false; }
    if (!noise_.next((size_t)clip_ * D_, nz_, err)) return false;
    clear_last_log();
    const auto t0 = Clock::now();
    if (!sampler_->reanchor(wa, nz_.data(), x_)) { err = with_cause("reanchor failed"); return false; }
    const double ms = ms_since(t0);
    stats_.ms_sampler += ms;
    stats_.reanchor_ms.push_back(ms);
    // The fade starts at the first frame from the new caches: this block,
    // or the block after a held one (latent_fade_weight).
    fade_i_ = fade_total_ = 0;
    if (latent_fade > 0 && (int)last_latent_.size() == D_) {
        fade_old_ = last_latent_;
        fade_ref_.assign(x_.begin() + (size_t)(clip_ - 1) * D_, x_.begin() + (size_t)clip_ * D_);
        fade_total_ = latent_fade;
    }
    bool ok;
    if (emit) {
        fade_lat_.assign(x_.begin() + (size_t)(clip_ - nb_) * D_, x_.begin() + (size_t)clip_ * D_);
        apply_fade(fade_lat_.data(), nb_, true);
        ok = submit(fade_lat_.data(), nb_, err);
    } else {
        ok = submit_hold(nb_, err);
    }
    if (!ok) return false;
    t_ += nb_;
    return true;
}

bool Generator::finish(std::string& err) {
    bool ok = run(true, err);
    stop_threads();
    if (ok && failed_) ok = failed_error(err);
    return ok;
}

// ---------------------------------------------------------------------------
// VideoWriter

namespace {

// write() on a pipe whose reader is gone raises SIGPIPE, which kills the
// process by default. Block it on this thread for the write and discard a
// SIGPIPE the write raised, so the caller gets EPIPE instead; the process
// signal disposition stays as it is.
bool write_all(int fd, const uint8_t* p, size_t n, std::string& err, int& write_errno) {
    sigset_t pipe_set, old;
    sigemptyset(&pipe_set);
    sigaddset(&pipe_set, SIGPIPE);
    sigset_t pending;
    sigemptyset(&pending);
    sigpending(&pending);
    const bool was_pending = sigismember(&pending, SIGPIPE) == 1;
    pthread_sigmask(SIG_BLOCK, &pipe_set, &old);
    bool ok = true;
    size_t off = 0;
    while (off < n) {
        const ssize_t k = ::write(fd, p + off, n - off);
        if (k < 0) {
            if (errno == EINTR) continue;
            write_errno = errno;
            err = std::string("writing to ffmpeg failed: ") + std::strerror(errno);
#if defined(__linux__)
            if (write_errno == EPIPE && !was_pending) {
                const struct timespec zero = {0, 0};
                while (sigtimedwait(&pipe_set, nullptr, &zero) < 0 && errno == EINTR) {}
            }
#else
            (void)was_pending;
#endif
            ok = false;
            break;
        }
        off += (size_t)k;
    }
    pthread_sigmask(SIG_SETMASK, &old, nullptr);
    return ok;
}

} // namespace

VideoWriter::~VideoWriter() {
    std::string err;
    close(err);
}

bool VideoWriter::open(const std::string& ffmpeg, const std::string& out_path, const std::string& audio_path, int w,
                       int h, int fps, std::string& err, int min_frames) {
    w_ = w;
    h_ = h;
    done_ = false;
    stderr_.clear();
    // ffmpeg reads raw RGB frames on stdin and muxes the input audio, as
    // the upstream run_stream does (it encodes first, then muxes with
    // -shortest; one ffmpeg call gives the same result).
    const std::string size = std::to_string(w) + "x" + std::to_string(h), rate = std::to_string(fps);
    std::vector<std::string> argv_s = {
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", size, "-r", rate, "-i", "-",
        "-i", audio_path,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac"};
    if (min_frames > 0) {
        // Audio shorter than the video (speech shorter than the primer):
        // -shortest would stop ffmpeg at the end of the audio, before it
        // reads all frames. Pad the audio with silence to the video length.
        // apad does not change audio that is already long enough.
        const long long us = (long long)min_frames * 1000000LL / fps;
        argv_s.insert(argv_s.end(), {"-af", "apad=whole_dur=" + std::to_string(us) + "us"});
    }
    argv_s.insert(argv_s.end(), {"-shortest", out_path});
    if (audio_path.empty())
        argv_s = {ffmpeg, "-y", "-loglevel", "error",
                  "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", size, "-r", rate, "-i", "-",
                  "-c:v", "libx264", "-pix_fmt", "yuv420p", out_path};
    std::vector<char*> argv;
    for (auto& s : argv_s) argv.push_back(const_cast<char*>(s.c_str()));
    argv.push_back(nullptr);
    // O_CLOEXEC: a second ffmpeg started from another thread must not
    // inherit this write end, or this ffmpeg never sees the end of input.
    // dup2 onto stdin clears the flag for the child's read end. The same
    // for the error output pipe.
    int fds[2], efds[2];
#if defined(__linux__)
    if (pipe2(fds, O_CLOEXEC) != 0) { err = std::string("pipe: ") + std::strerror(errno); return false; }
    if (pipe2(efds, O_CLOEXEC) != 0) {
        err = std::string("pipe: ") + std::strerror(errno);
        ::close(fds[0]);
        ::close(fds[1]);
        return false;
    }
#else
    if (pipe(fds) != 0) { err = std::string("pipe: ") + std::strerror(errno); return false; }
    if (pipe(efds) != 0) {
        err = std::string("pipe: ") + std::strerror(errno);
        ::close(fds[0]);
        ::close(fds[1]);
        return false;
    }
    for (int fd : {fds[0], fds[1], efds[0], efds[1]}) fcntl(fd, F_SETFD, FD_CLOEXEC);
#endif
#if defined(F_SETNOSIGPIPE)
    fcntl(fds[1], F_SETNOSIGPIPE, 1);
#endif
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_adddup2(&fa, fds[0], STDIN_FILENO);
    posix_spawn_file_actions_adddup2(&fa, efds[1], STDERR_FILENO);
    const int rc = posix_spawnp(&pid_, ffmpeg.c_str(), &fa, nullptr, argv.data(), environ);
    posix_spawn_file_actions_destroy(&fa);
    ::close(fds[0]);
    ::close(efds[1]);
    if (rc != 0) {
        ::close(fds[1]);
        ::close(efds[0]);
        pid_ = -1;
        if (rc == ENOENT)
            err = "cannot start " + ffmpeg + ": not found (mp4 output needs ffmpeg on PATH)";
        else
            err = "cannot start " + ffmpeg + ": " + std::strerror(rc);
        return false;
    }
    fd_ = fds[1];
    // Keep the error output of ffmpeg for the error message. Read it all,
    // so that ffmpeg never blocks on a full pipe; keep the last 4 KiB.
    const int efd = efds[0];
    stderr_thread_ = std::thread([this, efd] {
        char buf[1024];
        for (;;) {
            const ssize_t k = ::read(efd, buf, sizeof(buf));
            if (k < 0 && errno == EINTR) continue;
            if (k <= 0) break;
            stderr_.append(buf, (size_t)k);
            if (stderr_.size() > 8192) stderr_.erase(0, stderr_.size() - 4096);
        }
        ::close(efd);
    });
    return true;
}

bool VideoWriter::write(const uint8_t* frames, int n, std::string& err) {
    if (done_) return true;
    if (fd_ < 0) { err = "the video writer is not open"; return false; }
    int write_errno = 0;
    if (write_all(fd_, frames, (size_t)n * w_ * h_ * 3, err, write_errno)) return true;
    // ffmpeg stopped reading. Report its own error when it failed.
    const std::string write_err = err;
    ::close(fd_);
    fd_ = -1;
    if (!wait_ffmpeg(err)) {
        err = "ffmpeg stopped reading frames: " + err;
        return false;
    }
    if (write_errno != EPIPE) {
        err = write_err;
        return false;
    }
    done_ = true;   // ffmpeg finished the file; drop the frames that are left
    return true;
}

bool VideoWriter::wait_ffmpeg(std::string& err) {
    if (pid_ <= 0) return true;
    int st = 0;
    while (waitpid(pid_, &st, 0) < 0 && errno == EINTR) {}
    pid_ = -1;
    if (stderr_thread_.joinable()) stderr_thread_.join();
    if (WIFEXITED(st) && WEXITSTATUS(st) == 0) return true;
    if (WIFEXITED(st))
        err = "ffmpeg failed (exit status " + std::to_string(WEXITSTATUS(st)) + ")";
    else if (WIFSIGNALED(st))
        err = "ffmpeg failed (killed by signal " + std::to_string(WTERMSIG(st)) + ")";
    else
        err = "ffmpeg failed (status " + std::to_string(st) + ")";
    // The error output of ffmpeg, on one line.
    std::string msg;
    bool line_end = false;
    for (char c : stderr_) {
        if (c == '\r' || c == '\n') { line_end = true; continue; }
        if (line_end && !msg.empty()) msg += "; ";
        line_end = false;
        msg += c;
    }
    if (!msg.empty()) err += ": " + msg;
    return false;
}

bool VideoWriter::close(std::string& err) {
    done_ = false;
    if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
    return wait_ffmpeg(err);
}

bool mux_audio(const std::string& ffmpeg, const std::string& video_path, const std::string& audio_path,
               const std::string& out_path, std::string& err) {
    std::vector<std::string> argv_s = {ffmpeg, "-y", "-loglevel", "error", "-i", video_path, "-i", audio_path,
                                       "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
                                       "-shortest", out_path};
    std::vector<char*> argv;
    for (auto& s : argv_s) argv.push_back(const_cast<char*>(s.c_str()));
    argv.push_back(nullptr);
    pid_t pid = -1;
    const int rc = posix_spawnp(&pid, ffmpeg.c_str(), nullptr, nullptr, argv.data(), environ);
    if (rc != 0) { err = "cannot start " + ffmpeg + ": " + std::strerror(rc); return false; }
    int st = 0;
    while (waitpid(pid, &st, 0) < 0 && errno == EINTR) {}
    if (!(WIFEXITED(st) && WEXITSTATUS(st) == 0)) {
        err = "ffmpeg failed to mux " + audio_path + " (status " + std::to_string(st) + ")";
        return false;
    }
    return true;
}

} // namespace af
