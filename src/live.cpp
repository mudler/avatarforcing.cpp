#include "live.hpp"

#include "common.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>

namespace af {

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

} // namespace

// ---------------------------------------------------------------------------
// Reanchor cross-fade

void crossfade(const uint8_t* anchor, uint8_t* frame, size_t n, int remaining, int total) {
    // numpy: alpha * anchor.astype(float32) + (1.0 - alpha) * frame.astype(float32)
    // with a Python float alpha: both weights round to float32, each product
    // and the sum are float32. volatile keeps the compiler from fusing them.
    const double alpha = (double)remaining / (double)total;
    const float a = (float)alpha, b = (float)(1.0 - alpha);
    for (size_t i = 0; i < n; ++i) {
        volatile float pa = a * (float)anchor[i];
        volatile float pb = b * (float)frame[i];
        float v = pa + pb;
        v = std::min(255.0f, std::max(0.0f, v));
        frame[i] = (uint8_t)(int)v;   // astype(uint8) truncates
    }
}

// ---------------------------------------------------------------------------
// LiveSchedule

void LiveSchedule::init(const std::vector<float>& idle, int idle_rows, int dim, int origin, int history) {
    idle_ = idle;
    idle_rows_ = idle_rows;
    dim_ = dim;
    origin_ = origin;
    history_ = history;
    next_ = -origin;
    phrases_.clear();
}

int64_t LiveSchedule::speech_end() const {
    return phrases_.empty() ? next_ : std::max(next_, phrases_.back().end());
}

int LiveSchedule::pending_phrases() const {
    int n = 0;
    for (const Phrase& p : phrases_)
        if (p.end() > next_) ++n;
    return n;
}

int64_t LiveSchedule::place(const float* rows, int n, int64_t at, int64_t max_ahead, std::string& err) {
    if (n <= 0) { err = "the phrase has no audio frames"; return -1; }
    const int64_t end_last = speech_end();
    const int64_t start = at < 0 ? end_last : std::max(at, end_last);
    if (start + n - next_ > max_ahead) {
        char buf[160];
        std::snprintf(buf, sizeof(buf),
                      "the phrase would end at frame %lld, more than %lld frames after the next block (frame %lld)",
                      (long long)(start + n), (long long)max_ahead, (long long)next_);
        err = buf;
        return -1;
    }
    Phrase p;
    p.start = start;
    p.n = n;
    p.rows.assign(rows, rows + (size_t)n * dim_);
    phrases_.push_back(std::move(p));
    return start;
}

int64_t LiveSchedule::cut(int64_t at) {
    const int64_t c = at < 0 ? next_ : std::max(at, next_);
    while (!phrases_.empty() && phrases_.back().start >= c) phrases_.pop_back();
    if (!phrases_.empty() && phrases_.back().end() > c) {
        Phrase& p = phrases_.back();
        p.n = (int)(c - p.start);
        p.rows.resize((size_t)p.n * dim_);
    }
    return c;
}

void LiveSchedule::window(int64_t lo, int n, float* out, int* speech_rows) const {
    int speech = 0;
    size_t k = 0;   // the first phrase that can cover the frame; frames go up
    for (int r = 0; r < n; ++r) {
        const int64_t f = lo + r;
        while (k < phrases_.size() && phrases_[k].end() <= f) ++k;
        const float* src;
        if (k < phrases_.size() && phrases_[k].start <= f) {
            src = phrases_[k].rows.data() + (size_t)(f - phrases_[k].start) * dim_;
            ++speech;
        } else {
            int64_t i = (f + origin_) % idle_rows_;
            if (i < 0) i += idle_rows_;
            src = idle_.data() + (size_t)i * dim_;
        }
        std::memcpy(out + (size_t)r * dim_, src, (size_t)dim_ * sizeof(float));
    }
    if (speech_rows) *speech_rows = speech;
}

void LiveSchedule::commit(int64_t end) {
    next_ = std::max(next_, end);
    while (!phrases_.empty() && phrases_.front().end() + history_ <= next_) phrases_.pop_front();
}

// ---------------------------------------------------------------------------
// LiveSession

LiveSession::LiveSession(Engine& e, const GenOptions& go, const LiveOptions& lo, const std::atomic<bool>* cancel)
    : e_(e), go_(go), lo_(lo), cancel_(cancel) {
    const FlowConfig& fc = e_.flow().config();
    clip_ = (int)fc.clip_frames;
    nb_ = (int)fc.block_frames;
    ctx_ = (int)fc.context_frames;
    E_ = (int)fc.audio_proj_out;
    queue_cap_ = std::max(1, lo_.queue_blocks) * nb_;
    max_ahead_ = (int64_t)std::max(0, lo_.max_ahead_seconds) * e_.fps();
    frame_bytes_ = (size_t)e_.size() * e_.size() * 3;
    policy_.every = lo_.reanchor_every < 0 ? (int)fc.reanchor_every : lo_.reanchor_every;
    policy_.max_defer = lo_.reanchor_max_defer < 0 ? (int)fc.reanchor_max_defer : lo_.reanchor_max_defer;
    fade_frames_ = lo_.reanchor_fade_frames < 0 ? (int)fc.reanchor_fade_frames : lo_.reanchor_fade_frames;
    if (lo_.reanchor_fade == ReanchorFade::None) fade_frames_ = 0;
}

LiveSession::~LiveSession() {
    gen_.reset();   // stops the decoder and output threads
}

void LiveSession::set_failed(const std::string& err) {
    {
        std::lock_guard<std::mutex> lk(err_mu_);
        if (err_.empty()) err_ = err;
    }
    failed_ = true;
}

bool LiveSession::failed(std::string* err) const {
    std::string gerr;
    if (failed_) {
        if (err) {
            std::lock_guard<std::mutex> lk(err_mu_);
            *err = err_;
        }
        return true;
    }
    if (gen_ && gen_->failed(err ? &gerr : nullptr)) {
        if (err) *err = gerr;
        return true;
    }
    return false;
}

bool LiveSession::open(const float* image, std::string& err) {
    std::lock_guard<std::mutex> lk(compute_mu_);
    AudioEncoder& audio = e_.audio();
    if (audio.idle_frames() <= 0) { err = "the model has no idle audio features (audio.idle_pcm)"; return false; }
    {
        std::lock_guard<std::mutex> slk(sched_mu_);
        sched_.init(audio.idle_features(), audio.idle_frames(), E_, clip_, clip_);
    }
    auto sink = [this](const uint8_t* rgb, int n, std::string&) -> bool { return this->sink(rgb, n); };
    gen_.reset(new Generator(e_, go_, sink, cancel_));
    if (!gen_->encode_portrait(image, err) || !gen_->init_sampler(err)) return false;
    // Hidden primer: clip frames of idle features (frames -clip .. -1),
    // written to the caches only, as the upstream live app does.
    wa_.resize((size_t)clip_ * E_);
    {
        std::lock_guard<std::mutex> slk(sched_mu_);
        sched_.window(-clip_, clip_, wa_.data());
        sched_.commit(0);
    }
    return gen_->primer_hidden(wa_.data(), err);
}

// The frames of one block, in order. rgb nullptr: a held block, n copies of
// the last frame. A cross-fade that starts at a frame takes the frame before
// it as its anchor and blends the next fade_frames_ frames.
bool LiveSession::sink(const uint8_t* rgb, int n) {
    std::lock_guard<std::mutex> qlk(q_mu_);
    for (int f = 0; f < n; ++f) {
        const uint8_t* prev = f > 0 ? ready_.back().data() : (last_.empty() ? nullptr : last_.data());
        std::vector<uint8_t> fr;
        if (rgb) fr.assign(rgb + (size_t)f * frame_bytes_, rgb + (size_t)(f + 1) * frame_bytes_);
        else if (prev) fr.assign(prev, prev + frame_bytes_);
        else fr.assign(frame_bytes_, 0);
        while (!fades_.empty() && fades_.front() <= out_frame_) {
            if (fades_.front() == out_frame_ && prev) {
                anchor_.assign(prev, prev + frame_bytes_);
                fade_left_ = fade_frames_;
            }
            fades_.pop_front();
        }
        if (fade_left_ > 0) {
            crossfade(anchor_.data(), fr.data(), frame_bytes_, fade_left_, fade_frames_);
            if (--fade_left_ == 0) anchor_.clear();
        }
        ready_.push_back(std::move(fr));
        ++out_frame_;
    }
    if (n > 0) last_ = ready_.back();
    return true;
}

int64_t LiveSession::push_phrase(const float* pcm, size_t n, int64_t at, uint64_t epoch, std::string& err,
                                 bool* bad_input) {
    if (bad_input) *bad_input = false;
    auto bad = [&](const std::string& m) {
        err = m;
        if (bad_input) *bad_input = true;
        return (int64_t)-1;
    };
    if (!pcm || n == 0) return bad("the phrase has no samples");
    std::lock_guard<std::mutex> lk(compute_mu_);
    if (!gen_) { err = "the session is not open"; return -1; }
    if (failed(&err)) return -1;
    if (epoch != epoch_.load()) {
        std::lock_guard<std::mutex> slk(sched_mu_);
        ++dropped_;
        return kDropped;
    }
    // wav2vec2 on the whole phrase, as the live app does.
    std::vector<float> feats;
    int T = 0;
    bool capped = false;
    std::vector<float> x(pcm, pcm + n);
    if (!clip_features(e_, x, go_.normalized, 0, feats, T, capped, err)) {
        set_failed(err);
        return -1;
    }
    std::lock_guard<std::mutex> slk(sched_mu_);
    if (epoch != epoch_.load()) {
        ++dropped_;
        return kDropped;
    }
    const int64_t start = sched_.place(feats.data(), T, at, max_ahead_, err);
    if (start < 0) return bad(err);
    ++phrases_;
    return start;
}

int64_t LiveSession::step(std::string& err) {
    std::lock_guard<std::mutex> lk(compute_mu_);
    if (!gen_) { err = "the session is not open"; return -1; }
    if (failed(&err)) return -1;
    {
        std::lock_guard<std::mutex> qlk(q_mu_);
        if (produced_ - pulled_ + nb_ > queue_cap_) {
            std::lock_guard<std::mutex> slk(sched_mu_);
            ++queue_full_;
            return kQueueFull;
        }
    }
    const auto t0 = Clock::now();
    // A steady block reads ctx frames of context (already computed) and nb
    // new frames. A reanchor block (ReanchorPolicy) is a primer on the clip
    // frames that end with the block, as in the live app.
    int64_t t = 0;
    int speech = 0;
    bool reanchor = false, hold = false;
    {
        std::lock_guard<std::mutex> slk(sched_mu_);
        t = sched_.next();
        reanchor = policy_.due(sched_.speech_end() <= t, reanchor_req_.load());
        if (reanchor) {
            reanchor_req_ = false;
            // The live app holds the last frame; before any frame, emit.
            hold = lo_.reanchor_hold && blocks_ > 0;
            wa_.resize((size_t)(clip_ + nb_) * E_);
            sched_.window(t + nb_ - clip_, clip_, wa_.data());
            sched_.window(t, nb_, wa_.data() + (size_t)clip_ * E_, &speech);   // to count speech
        } else {
            wa_.resize((size_t)(ctx_ + nb_) * E_);
            sched_.window(t - ctx_, ctx_ + nb_, wa_.data());
            sched_.window(t, nb_, wa_.data() + (size_t)ctx_ * E_, &speech);   // the new frames, to count speech
        }
        sched_.commit(t + nb_);
    }
    {
        std::lock_guard<std::mutex> qlk(q_mu_);
        produced_ += nb_;
        // The fade starts with the first frame that comes from the new caches.
        if (reanchor && fade_frames_ > 0 && lo_.reanchor_fade == ReanchorFade::Pixel)
            fades_.push_back(hold ? t + nb_ : t);
    }
    const int latent_fade = lo_.reanchor_fade == ReanchorFade::Latent ? fade_frames_ : 0;
    const bool ok = reanchor ? gen_->reanchor_block(wa_.data(), !hold, latent_fade, err)
                             : gen_->steady_block(wa_.data(), err);
    if (!ok) {
        set_failed(err);
        return -1;
    }
    const double ms = ms_since(t0);
    std::lock_guard<std::mutex> slk(sched_mu_);
    policy_.done(reanchor);
    ++blocks_;
    if (speech > 0) ++speech_blocks_;
    last_step_ms_ = ms;
    max_step_ms_ = std::max(max_step_ms_, ms);
    if (reanchor) {
        ++reanchors_;
        last_reanchor_ = t;
        last_reanchor_ms_ = ms;
        max_reanchor_ms_ = std::max(max_reanchor_ms_, ms);
    }
    return t;
}

int64_t LiveSession::request_reanchor() {
    std::lock_guard<std::mutex> slk(sched_mu_);
    reanchor_req_ = true;
    return sched_.next();
}

int64_t LiveSession::last_reanchor_frame() const {
    std::lock_guard<std::mutex> slk(sched_mu_);
    return last_reanchor_;
}

int64_t LiveSession::interrupt(int64_t at) {
    std::lock_guard<std::mutex> slk(sched_mu_);
    ++epoch_;
    ++interrupts_;
    return sched_.cut(at);
}

int LiveSession::frames_ready() {
    std::lock_guard<std::mutex> qlk(q_mu_);
    return (int)ready_.size();
}

int LiveSession::pull(uint8_t* dst, int max) {
    int n = 0;
    {
        std::lock_guard<std::mutex> qlk(q_mu_);
        // Copy under the lock: a frame is 768 KB and the sink waits at most
        // for one pull.
        while (n < max && !ready_.empty()) {
            std::memcpy(dst + (size_t)n * frame_bytes_, ready_.front().data(), frame_bytes_);
            ready_.pop_front();
            ++n;
        }
        pulled_ += n;
    }
    if (n == 0 && failed()) return -1;
    return n;
}

std::string LiveSession::stats_json() {
    int64_t queued = 0;
    int ready = 0;
    {
        std::lock_guard<std::mutex> qlk(q_mu_);
        queued = produced_ - pulled_;
        ready = (int)ready_.size();
    }
    std::lock_guard<std::mutex> slk(sched_mu_);
    char buf[1024];
    std::snprintf(buf, sizeof(buf),
                  "{\"next_frame\":%lld,\"speech_end\":%lld,\"pending_phrases\":%d,\"queued_frames\":%lld,"
                  "\"ready_frames\":%d,\"queue_frames\":%d,\"blocks\":%lld,\"speech_blocks\":%lld,"
                  "\"queue_full\":%lld,\"phrases\":%lld,\"dropped_phrases\":%lld,\"interrupts\":%lld,"
                  "\"last_step_ms\":%.1f,\"max_step_ms\":%.1f,\"reanchors\":%lld,\"last_reanchor_frame\":%lld,"
                  "\"blocks_since_reanchor\":%d,\"reanchor_every\":%d,\"reanchor_max_defer\":%d,"
                  "\"reanchor_fade_frames\":%d,\"reanchor_fade\":\"%s\",\"last_reanchor_ms\":%.1f,"
                  "\"max_reanchor_ms\":%.1f}",
                  (long long)sched_.next(), (long long)sched_.speech_end(), sched_.pending_phrases(),
                  (long long)queued, ready, queue_cap_, (long long)blocks_, (long long)speech_blocks_,
                  (long long)queue_full_, (long long)phrases_, (long long)dropped_, (long long)interrupts_,
                  last_step_ms_, max_step_ms_, (long long)reanchors_, (long long)last_reanchor_, policy_.since,
                  policy_.every, policy_.max_defer, fade_frames_,
                  fade_frames_ == 0 ? "none" : lo_.reanchor_fade == ReanchorFade::Pixel ? "pixel" : "latent",
                  last_reanchor_ms_, max_reanchor_ms_);
    return buf;
}

} // namespace af
