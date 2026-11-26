#pragma once
// Live mode: a session that drives the face for as long as a call lasts.
// The caller owns the clock and calls step() once per block (10 frames,
// 400 ms of video). Each step computes exactly one block: lip-sync where a
// phrase covers the block, the idle features of the GGUF elsewhere. The
// primer runs at open() on idle features and is not output; frame 0 is the
// first frame of the first step. A reanchor (a fresh primer in place of a
// block) resets the caches from time to time, as the live app does, so that
// a long call does not drift away from the portrait. See docs/capi.md, "Live
// mode".

#include "pipeline.hpp"

#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace af {

// The fade after a reanchor. Latent: the decoder input moves from the old
// motion to the new one (Generator::latent_fade_weight), so every frame is
// one face. Pixel: the live app's RGB cross-fade from the last frame
// (crossfade()), which shows two faces at once when the pose changed.
enum class ReanchorFade { Latent, Pixel, None };

struct LiveOptions {
    int queue_blocks = 3;        // blocks that may wait for pull_frames, >= 1
    int max_ahead_seconds = 30;  // a phrase must end at most this far after next()
    // Reanchor policy (ReanchorPolicy). < 0: the GGUF value, which is the live
    // app value (112, 162, 25).
    int reanchor_every = -1;        // steady blocks before an idle reanchor; 0: none
    int reanchor_max_defer = -1;    // steady blocks before a forced reanchor; 0: none
    int reanchor_fade_frames = -1;  // fade frames after a reanchor; 0: no fade
    ReanchorFade reanchor_fade = ReanchorFade::Latent;
    // false: the reanchor block is the decoded end of the new primer. true:
    // it repeats the last frame, as the live app does (it skips the output of
    // the reanchor because its primer takes longer than a block there).
    bool reanchor_hold = false;
};

// When the live app reanchors. `since` counts the steady blocks after the
// primer or the last reanchor. Before a block, a reanchor is due when it is
// asked for, when since >= every and the face is idle (no phrase covers the
// block or comes after it), or when since >= max_defer, also in speech.
struct ReanchorPolicy {
    int every = 0, max_defer = 0;
    int since = 0;

    bool due(bool idle, bool requested) const {
        return requested || (every > 0 && since >= every && idle) || (max_defer > 0 && since >= max_defer);
    }
    // After each block: reanchored or a steady block.
    void done(bool reanchored) { since = reanchored ? 0 : since + 1; }
};

// The cross-fade of the live app after a reanchor: frame k (0-based) of the
// fade is (r / total) * anchor + (1 - r / total) * frame for r = total - k,
// in float32, clipped to [0, 255] and truncated, as numpy does it. The
// anchor is the last frame before the fade. remaining: r, from total to 1.
// frame: n bytes, blended in place.
void crossfade(const uint8_t* anchor, uint8_t* frame, size_t n, int remaining, int total);

// The audio features of a live session, indexed by frame. Frame 0 is the
// first output frame; the hidden primer covers frames -clip .. -1. A frame
// that no phrase covers takes the idle features, cyclically: frame f takes
// idle row (f + origin) mod idle_rows, so the primer starts at idle row 0
// and every frame keeps the same features in every window that reads it.
//
// Frames before next() are committed: a step has read them, so a phrase or a
// cut cannot change them any more. Phrases never overlap. Not thread-safe:
// LiveSession guards it with a lock.
class LiveSchedule {
public:
    // idle: [idle_rows, dim]. origin: the primer length. history: committed
    // frames to keep (a reanchor reads the last clip frames).
    void init(const std::vector<float>& idle, int idle_rows, int dim, int origin, int history);

    int64_t next() const { return next_; }
    // End of the last phrase, or next() when no phrase ends after next().
    int64_t speech_end() const;
    int pending_phrases() const;

    // Places a phrase of n feature rows. at < 0: right after the last phrase,
    // or at next() when none is pending. at >= 0: at frame at, moved later
    // when it is before next() or before the end of the last phrase. Returns
    // the start frame, or -1 (err) when the phrase would end more than
    // max_ahead frames after next().
    int64_t place(const float* rows, int n, int64_t at, int64_t max_ahead, std::string& err);

    // From frame max(at, next()) on, no phrase: drops the phrases that start
    // there or later and shortens the phrase that runs over it. at < 0 means
    // next(). Returns the first frame that is idle for sure.
    int64_t cut(int64_t at);

    // Feature rows [lo, lo + n) into out [n, dim]. speech_rows (optional)
    // receives the number of rows that come from a phrase.
    void window(int64_t lo, int n, float* out, int* speech_rows = nullptr) const;

    // Marks the frames before end as read by a step. Phrases that end more
    // than `history` frames before end are released.
    void commit(int64_t end);

private:
    struct Phrase {
        int64_t start = 0;
        int n = 0;
        std::vector<float> rows;
        int64_t end() const { return start + n; }
    };
    std::vector<float> idle_;
    int idle_rows_ = 0, dim_ = 0, origin_ = 0, history_ = 0;
    int64_t next_ = 0;
    std::deque<Phrase> phrases_;   // sorted by start, not overlapping
};

// One live session on an engine. Calls that compute (open, push_phrase, step)
// run one at a time: a lock in the session serializes them. interrupt,
// request_reanchor, epoch, frames_ready, pull and stats_json do not wait for
// that lock.
class LiveSession {
public:
    static constexpr int64_t kQueueFull = -2;   // step: the frame queue is full
    static constexpr int64_t kDropped = -3;     // push_phrase: an interrupt dropped the phrase

    LiveSession(Engine& e, const GenOptions& go, const LiveOptions& lo, const std::atomic<bool>* cancel = nullptr);
    ~LiveSession();
    LiveSession(const LiveSession&) = delete;
    LiveSession& operator=(const LiveSession&) = delete;

    // image: [3, S, S] in [-1, 1]. Encodes the portrait, sets up the sampler
    // and runs the primer on idle features without output.
    bool open(const float* image, std::string& err);

    // The interrupt count. A caller reads it when a phrase arrives and passes
    // it to push_phrase; an interrupt in between drops the phrase.
    uint64_t epoch() const { return epoch_.load(); }

    // pcm: one phrase, mono at the engine sample rate. Runs wav2vec2 on it,
    // then places it (LiveSchedule::place). Returns the start frame, kDropped
    // when an interrupt came after `epoch`, or -1 (err). bad_input: the error
    // is in the call (too far ahead, empty PCM) and the session is fine.
    int64_t push_phrase(const float* pcm, size_t n, int64_t at, uint64_t epoch, std::string& err,
                        bool* bad_input = nullptr);

    // Computes the next block. Returns its first frame, kQueueFull when the
    // frames that wait for pull (ready or still in the decoder) and this
    // block would exceed the queue (nothing is computed), or -1 (err; the
    // session failed).
    int64_t step(std::string& err);

    // Barge-in: from frame max(at, next()) on the face is idle. Returns that
    // frame. Phrases that a push_phrase computes at the same time are dropped.
    int64_t interrupt(int64_t at);

    int frames_ready();
    // Copies up to max ready frames to dst. Returns the count; -1 when no
    // frame is ready and the session failed.
    int pull(uint8_t* dst, int max);

    bool failed(std::string* err = nullptr) const;
    std::string stats_json();

    // Asks for a reanchor at the next step, whatever the policy. Returns the
    // first frame of that step. Does not wait for a running step.
    int64_t request_reanchor();
    // The first frame of the last reanchor block, -1 before the first.
    int64_t last_reanchor_frame() const;

    size_t frame_bytes() const { return frame_bytes_; }
    int block_frames() const { return nb_; }
    int queue_frames() const { return queue_cap_; }
    Generator& generator() { return *gen_; }

private:
    void set_failed(const std::string& err);
    bool sink(const uint8_t* rgb, int n);

    Engine& e_;
    GenOptions go_;
    LiveOptions lo_;
    const std::atomic<bool>* cancel_;
    std::unique_ptr<Generator> gen_;
    int clip_ = 0, nb_ = 0, ctx_ = 0, E_ = 0, queue_cap_ = 0;
    int64_t max_ahead_ = 0;
    size_t frame_bytes_ = 0;
    std::vector<float> wa_;

    std::mutex compute_mu_;   // open, push_phrase, step

    mutable std::mutex sched_mu_;   // sched_, the counters below
    LiveSchedule sched_;
    std::atomic<uint64_t> epoch_{0};
    int64_t blocks_ = 0, speech_blocks_ = 0, queue_full_ = 0, interrupts_ = 0, phrases_ = 0, dropped_ = 0;
    double last_step_ms_ = 0, max_step_ms_ = 0;
    ReanchorPolicy policy_;
    int fade_frames_ = 0;
    std::atomic<bool> reanchor_req_{false};
    int64_t reanchors_ = 0, last_reanchor_ = -1;
    double last_reanchor_ms_ = 0, max_reanchor_ms_ = 0;

    std::mutex q_mu_;   // ready_, produced_, pulled_, the output state below
    std::deque<std::vector<uint8_t>> ready_;
    int64_t produced_ = 0, pulled_ = 0;   // frames handed to the decoder, frames pulled
    // Output (the frame sink): the index of the next frame, the last frame,
    // and the cross-fades that start at a frame.
    int64_t out_frame_ = 0;
    std::vector<uint8_t> last_, anchor_;
    std::deque<int64_t> fades_;
    int fade_left_ = 0;

    std::atomic<bool> failed_{false};
    mutable std::mutex err_mu_;
    std::string err_;
};

} // namespace af
