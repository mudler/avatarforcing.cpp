// Reanchor test of live mode, on a model.
//
// Short calls with a short reanchor period (every 3 steady blocks, forced
// after 5) check:
//   - the reanchors come at the expected frames: idle, deferred in speech and
//     forced at max_defer, and on request;
//   - the frames before the first reanchor are the frames of a session with
//     no reanchor (bit-identical);
//   - the cross-fade is continuous: no step between frames in a fade is
//     larger than 1.25 times the largest step elsewhere in the call; a held
//     reanchor block repeats the last frame;
//   - determinism: the same call twice gives the same frames.
// With AVATARFORCING_TEST_DRIFT_SECONDS=N (for example 120), a long idle call
// of N s runs with the default policy and with no reanchor, and the test
// prints the difference to the portrait every 10 s (mean absolute difference
// and SSIM). Off by default: on a CPU a 120 s call takes hours.
//
// Needs AVATARFORCING_TEST_GGUF; exits 77 (skip) when it is unset. Optional:
// AVATARFORCING_TEST_PORTRAIT (a 512x512 face image; else a synthetic
// image), AVATARFORCING_TEST_WAV (speech of at least 2.4 s; else a synthetic
// signal), AVATARFORCING_TEST_THREADS, AVATARFORCING_TEST_BACKEND,
// AVATARFORCING_TEST_OVERLAP (1 or 0) and AVATARFORCING_TEST_DRIFT_SECONDS.

#include "live.hpp"
#include "pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>

static int g_failures = 0;

#define CHECK(cond, ...)                                              \
    do {                                                              \
        if (!(cond)) {                                                \
            std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
            std::fprintf(stderr, __VA_ARGS__);                        \
            std::fprintf(stderr, "\n");                               \
            ++g_failures;                                             \
        }                                                             \
    } while (0)

using Clock = std::chrono::steady_clock;
using Frame = std::vector<uint8_t>;

static const char* env(const char* k) {
    const char* v = std::getenv(k);
    return v && *v ? v : nullptr;
}

static double mean_abs(const uint8_t* a, const uint8_t* b, size_t n) {
    double s = 0;
    for (size_t i = 0; i < n; ++i) s += std::abs((int)a[i] - (int)b[i]);
    return s / (double)n;
}

// Mean SSIM of the luma over 8x8 windows (no overlap), constants of the
// SSIM paper for 8-bit data.
static double ssim(const uint8_t* a, const uint8_t* b, int S) {
    const double c1 = 6.5025, c2 = 58.5225;
    double sum = 0;
    int n = 0;
    for (int y0 = 0; y0 + 8 <= S; y0 += 8)
        for (int x0 = 0; x0 + 8 <= S; x0 += 8) {
            double ma = 0, mb = 0, va = 0, vb = 0, cov = 0;
            double la[64], lb[64];
            for (int k = 0; k < 64; ++k) {
                const size_t p = ((size_t)(y0 + k / 8) * S + x0 + k % 8) * 3;
                la[k] = 0.299 * a[p] + 0.587 * a[p + 1] + 0.114 * a[p + 2];
                lb[k] = 0.299 * b[p] + 0.587 * b[p + 1] + 0.114 * b[p + 2];
                ma += la[k];
                mb += lb[k];
            }
            ma /= 64;
            mb /= 64;
            for (int k = 0; k < 64; ++k) {
                va += (la[k] - ma) * (la[k] - ma);
                vb += (lb[k] - mb) * (lb[k] - mb);
                cov += (la[k] - ma) * (lb[k] - mb);
            }
            va /= 63;
            vb /= 63;
            cov /= 63;
            sum += ((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2));
            ++n;
        }
    return sum / n;
}

static uint64_t fnv1a(const std::vector<Frame>& fs) {
    uint64_t h = 1469598103934665603ULL;
    for (const Frame& f : fs)
        for (uint8_t c : f) { h ^= c; h *= 1099511628211ULL; }
    return h;
}

struct Phrase {
    int64_t at;
    const float* pcm;
    size_t n;
};

struct CallResult {
    std::vector<Frame> frames;      // empty when a callback takes them
    std::vector<int64_t> reanchors; // the first frame of each reanchor block
    double max_step_ms = 0, max_reanchor_ms = 0;
    bool ok = false;
};

// Runs `blocks` steps of a live call. Phrases are pushed before the step
// that computes their first frame. request_at: a step at that frame is asked
// to reanchor (-1: none). on_frame: when set, receives each frame and the
// frames are not kept. Never blocks for long: a frame that does not come
// within the deadline fails the call.
static CallResult run_call(af::Engine& e, const af::GenOptions& go, const af::LiveOptions& lo,
                           const std::vector<float>& image, int blocks, const std::vector<Phrase>& phrases,
                           int64_t request_at = -1,
                           const std::function<void(int64_t, const uint8_t*)>& on_frame = nullptr) {
    CallResult res;
    af::LiveSession live(e, go, lo);
    std::string err;
    if (!live.open(image.data(), err)) {
        CHECK(false, "open: %s", err.c_str());
        return res;
    }
    const int nb = live.block_frames();
    const size_t fb = live.frame_bytes();
    std::vector<uint8_t> buf(fb * (size_t)nb);
    int64_t got = 0;
    auto take = [&]() -> bool {
        for (;;) {
            const int n = live.pull(buf.data(), nb);
            if (n < 0) {
                live.failed(&err);
                CHECK(false, "pull: %s", err.c_str());
                return false;
            }
            if (n == 0) return true;
            for (int f = 0; f < n; ++f) {
                const uint8_t* p = buf.data() + (size_t)f * fb;
                if (on_frame) on_frame(got + f, p);
                else res.frames.emplace_back(p, p + fb);
            }
            got += n;
        }
    };
    const auto deadline = Clock::now() + std::chrono::minutes(30);
    size_t next_phrase = 0;
    for (int b = 0; b < blocks; ++b) {
        const int64_t t = (int64_t)b * nb;
        while (next_phrase < phrases.size() && phrases[next_phrase].at < t + nb) {
            const Phrase& p = phrases[next_phrase++];
            const int64_t st = live.push_phrase(p.pcm, p.n, p.at, live.epoch(), err);
            CHECK(st == std::max(p.at, t), "phrase at %lld placed at %lld: %s", (long long)p.at, (long long)st,
                  err.c_str());
        }
        if (t == request_at) CHECK(live.request_reanchor() == t, "request_reanchor");
        int64_t r;
        while ((r = live.step(err)) == af::LiveSession::kQueueFull) {
            if (!take()) return res;
            if (Clock::now() > deadline) {
                CHECK(false, "the frame queue stays full");
                return res;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        if (r < 0) {
            CHECK(false, "step %d: %s", b, err.c_str());
            return res;
        }
        CHECK(r == t, "step %d returned %lld", b, (long long)r);
        if (live.last_reanchor_frame() == t) res.reanchors.push_back(t);
        if (!take()) return res;
    }
    while (got < (int64_t)blocks * nb) {
        if (Clock::now() > deadline) {
            CHECK(false, "%lld of %d frames after the deadline", (long long)got, blocks * nb);
            return res;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        if (!take()) return res;
    }
    const std::string js = live.stats_json();
    auto num = [&](const char* k) {
        const size_t p = js.find(std::string("\"") + k + "\":");
        return p == std::string::npos ? -1.0 : std::atof(js.c_str() + p + std::strlen(k) + 3);
    };
    res.max_step_ms = num("max_step_ms");
    res.max_reanchor_ms = num("max_reanchor_ms");
    CHECK((size_t)num("reanchors") == res.reanchors.size(), "stats reanchors: %s", js.c_str());
    res.ok = true;
    return res;
}

// Largest step between consecutive frames in [lo, hi) (frame i - 1 to i).
static double max_step(const std::vector<Frame>& f, int lo, int hi) {
    double m = 0;
    for (int i = std::max(1, lo); i < hi && i < (int)f.size(); ++i)
        m = std::max(m, mean_abs(f[i].data(), f[i - 1].data(), f[i].size()));
    return m;
}

// A fade step may be this much larger than the largest step elsewhere: the
// motion of the face goes on under the fade, and the per-block 8-bit
// scaling makes small steps at block edges.
static const double kMotionSlack = 1.25;

static std::string list(const std::vector<int64_t>& v) {
    std::string s;
    for (int64_t x : v) s += (s.empty() ? "" : " ") + std::to_string(x);
    return s.empty() ? "none" : s;
}

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const char* gguf = env("AVATARFORCING_TEST_GGUF");
    if (!gguf) {
        std::fprintf(stderr, "skip: set AVATARFORCING_TEST_GGUF\n");
        return 77;
    }
    af::EngineOptions eo;
    if (const char* v = env("AVATARFORCING_TEST_BACKEND")) eo.backend = v;
    if (const char* v = env("AVATARFORCING_TEST_THREADS")) eo.threads = std::atoi(v);
    if (const char* v = env("AVATARFORCING_TEST_OVERLAP")) eo.overlap = std::atoi(v) ? 1 : 0;
    af::Engine e;
    std::string err;
    if (!e.load(gguf, eo, err)) {
        std::fprintf(stderr, "FAIL: load: %s\n", err.c_str());
        return 1;
    }
    const int S = e.size(), sr = e.sample_rate();
    const size_t fb = (size_t)S * S * 3;

    // Portrait: [3, S, S] in [-1, 1], and as RGB bytes for the metrics.
    std::vector<float> image;
    if (const char* p = env("AVATARFORCING_TEST_PORTRAIT")) {
        if (!af::load_portrait(p, S, image, err)) {
            std::fprintf(stderr, "FAIL: %s\n", err.c_str());
            return 1;
        }
    } else {
        image.resize((size_t)3 * S * S);
        for (int c = 0; c < 3; ++c)
            for (int y = 0; y < S; ++y)
                for (int x = 0; x < S; ++x) {
                    const double dx = (x - S / 2) / (double)S, dy = (y - S / 2) / (double)S;
                    const double v = 0.8 * std::exp(-8.0 * (dx * dx + dy * dy)) - 0.4 + 0.1 * c;
                    image[((size_t)c * S + y) * S + x] = (float)v;
                }
    }
    Frame portrait(fb);
    for (int y = 0; y < S; ++y)
        for (int x = 0; x < S; ++x)
            for (int c = 0; c < 3; ++c) {
                const float v = image[((size_t)c * S + y) * S + x] * 0.5f + 0.5f;
                portrait[((size_t)y * S + x) * 3 + c] = (uint8_t)std::lround(std::min(1.f, std::max(0.f, v)) * 255.f);
            }

    // Speech: 2.4 s.
    std::vector<float> speech;
    if (const char* w = env("AVATARFORCING_TEST_WAV")) {
        if (!af::load_audio(w, sr, speech, err)) {
            std::fprintf(stderr, "FAIL: %s\n", err.c_str());
            return 1;
        }
    }
    if (speech.size() < (size_t)(2.4 * sr)) {
        speech.resize((size_t)(2.4 * sr));
        for (size_t i = 0; i < speech.size(); ++i) {
            const double t = i / (double)sr;
            const double env3 = 0.5 + 0.5 * std::sin(2.0 * M_PI * 3.0 * t);
            speech[i] = (float)(0.3 * env3 * (std::sin(2.0 * M_PI * 180.0 * t) + 0.5 * std::sin(2.0 * M_PI * 720.0 * t)));
        }
    }
    speech.resize((size_t)(2.4 * sr));

    af::GenOptions go;
    go.seed = 11;
    go.normalized = false;   // raw, the live default
    af::LiveOptions fast;    // short period: reanchors in a few blocks
    fast.reanchor_every = 3;
    fast.reanchor_max_defer = 5;

    // ---- A: idle, every 3 blocks: reanchors at frames 30 and 70 ----
    const CallResult a = run_call(e, go, fast, image, 9, {});
    CHECK(a.ok && a.reanchors == std::vector<int64_t>({30, 70}), "idle reanchors: %s", list(a.reanchors).c_str());
    std::printf("A idle: reanchors at %s; max step %.0f ms, max reanchor step %.0f ms\n", list(a.reanchors).c_str(),
                a.max_step_ms, a.max_reanchor_ms);

    // ---- determinism ----
    {
        // The second run keeps only a hash and the frame count (memory).
        uint64_t h2 = 1469598103934665603ULL;
        int64_t n2 = 0;
        const CallResult a2 = run_call(e, go, fast, image, 9, {}, -1, [&](int64_t, const uint8_t* p) {
            for (size_t i = 0; i < fb; ++i) { h2 ^= p[i]; h2 *= 1099511628211ULL; }
            ++n2;
        });
        const uint64_t h1 = fnv1a(a.frames);
        CHECK(a2.ok && a.frames.size() == 90 && n2 == 90 && h1 == h2, "a second run gives other frames");
        std::printf("determinism: fnv1a %016llx and %016llx\n", (unsigned long long)h1, (unsigned long long)h2);
    }

    // ---- no reanchor: the same frames up to the first reanchor ----
    af::LiveOptions off;
    off.reanchor_every = 0;
    off.reanchor_max_defer = 0;
    const CallResult b = run_call(e, go, off, image, 4, {});
    CHECK(b.ok && b.reanchors.empty(), "reanchors with the policy off: %s", list(b.reanchors).c_str());
    bool same_before = b.frames.size() == 40 && a.frames.size() >= 40;
    for (int f = 0; same_before && f < 30; ++f) same_before = a.frames[f] == b.frames[f];
    const bool differ_after = same_before && a.frames[35] != b.frames[35];
    CHECK(same_before, "frames 0..29 differ from a call with no reanchor");
    CHECK(differ_after, "frame 35 does not change with a reanchor at 30");
    std::printf("no reanchor: frames 0..29 %s, frames 30..39 %s\n", same_before ? "identical" : "DIFFER",
                differ_after ? "differ" : "SAME");

    // ---- continuity of the cross-fade ----
    if (a.frames.size() == 90) {
        double fade = 0, rest = 0;
        for (int i = 1; i < 90; ++i) {
            const double d = mean_abs(a.frames[i].data(), a.frames[i - 1].data(), fb);
            const bool in_fade = (i >= 30 && i < 30 + 25) || (i >= 70 && i < 70 + 25);
            (in_fade ? fade : rest) = std::max(in_fade ? fade : rest, d);
        }
        std::printf("continuity: largest step between frames %.3f in the fades, %.3f elsewhere (0..255)\n", fade, rest);
        CHECK(fade <= kMotionSlack * rest, "a fade step of %.3f is larger than the motion elsewhere (%.3f)", fade, rest);
    }
    // The jump that the fade hides: the same call with no fade.
    {
    af::LiveOptions nofade = fast;
    nofade.reanchor_fade_frames = 0;
    const CallResult c = run_call(e, go, nofade, image, 4, {});
    if (c.ok && c.frames.size() == 40) {
        CHECK(c.reanchors == std::vector<int64_t>({30}), "no fade reanchors: %s", list(c.reanchors).c_str());
        const double jump = mean_abs(c.frames[30].data(), c.frames[29].data(), fb);
        std::printf("no fade: step at the reanchor (frame 29 to 30) %.3f, largest step in frames 1..29 %.3f\n", jump,
                    max_step(c.frames, 1, 30));
        // The fade starts with the frame before the reanchor, and the frames
        // before the reanchor do not depend on the fade.
        CHECK(a.frames.size() == 90 && a.frames[30] == a.frames[29], "fade frame 30 is not frame 29");
        bool same = true;
        for (int f = 0; f < 30; ++f) same = same && a.frames[f] == c.frames[f];
        CHECK(same, "the fade changed the frames before the reanchor");
    }
    }

    // ---- speech: deferred, then forced in the phrase; hold mode ----
    // The phrase covers frames 20..79. Blocks 3 and 4 (frames 30, 40) are in
    // speech: no reanchor. Block 5 (frame 50) is at max_defer: forced, in the
    // phrase. The next idle reanchor is block 9 (frame 90): 3 blocks after
    // block 5, and the phrase has ended.
    {
    af::LiveOptions hold = fast;
    hold.reanchor_hold = true;
    const CallResult d = run_call(e, go, hold, image, 11, {{20, speech.data(), speech.size()}});
    CHECK(d.ok && d.reanchors == std::vector<int64_t>({50, 90}), "speech reanchors: %s", list(d.reanchors).c_str());
    std::printf("D speech (hold): reanchors at %s\n", list(d.reanchors).c_str());
    if (d.ok && d.frames.size() == 110) {
        // Held block: frames 50..59 repeat frame 49; the fade starts at 60
        // with frame 49 again (alpha 1).
        bool held = true;
        for (int f = 50; f <= 60; ++f) held = held && d.frames[f] == d.frames[49];
        CHECK(held, "frames 50..60 do not repeat frame 49");
        const double fade = std::max(max_step(d.frames, 61, 85), max_step(d.frames, 101, 110));
        const double rest = std::max(max_step(d.frames, 1, 50), max_step(d.frames, 86, 90));
        std::printf("D continuity: largest step %.3f in the fades, %.3f elsewhere\n", fade, rest);
        CHECK(fade <= kMotionSlack * rest, "a fade step of %.3f is larger than the motion elsewhere (%.3f)", fade, rest);
    }
    }

    // ---- request: a reanchor at frame 20 with the policy off ----
    const CallResult r = run_call(e, go, off, image, 3, {}, 20);
    CHECK(r.ok && r.reanchors == std::vector<int64_t>({20}), "requested reanchor: %s", list(r.reanchors).c_str());
    if (r.ok && b.ok) {
        bool same = r.frames.size() == 30;
        for (int f = 0; same && f < 20; ++f) same = r.frames[f] == b.frames[f];
        CHECK(same, "frames before a requested reanchor changed");
    }
    std::printf("request: reanchor at %s\n", list(r.reanchors).c_str());

    // ---- drift over a long idle call (optional) ----
    const int drift_s = env("AVATARFORCING_TEST_DRIFT_SECONDS") ? std::atoi(env("AVATARFORCING_TEST_DRIFT_SECONDS")) : 0;
    if (drift_s >= 20) {
        const int fps = e.fps(), win = fps;   // one second of frames per point
        const int blocks = drift_s * fps / 10;
        struct Curve { std::vector<double> mad, ss; };
        auto measure = [&](const af::LiveOptions& lo, Curve& cv, CallResult& res) {
            const int points = drift_s / 10;
            cv.mad.assign(points + 1, 0);
            cv.ss.assign(points + 1, 0);
            res = run_call(e, go, lo, image, blocks, {}, -1, [&](int64_t f, const uint8_t* p) {
                // Point k: the second that ends at k * 10 s (point 0: the first second).
                const int64_t k = (f + win) / (10 * fps);
                const int64_t end = k == 0 ? win : k * 10 * fps;
                if (f >= end - win && f < end && k <= points) {
                    cv.mad[k] += mean_abs(p, portrait.data(), fb) / win;
                    cv.ss[k] += ssim(p, portrait.data(), S) / win;
                }
            });
        };
        Curve with, without;
        CallResult rw, rn;
        af::LiveOptions def;   // the live app policy
        const auto t0 = Clock::now();
        measure(def, with, rw);
        measure(off, without, rn);
        const double s = std::chrono::duration<double>(Clock::now() - t0).count();
        std::printf("drift over %d s of idle (difference to the portrait, mean of 1 s; %.0f s to compute):\n",
                    drift_s, s);
        std::printf("  reanchors at %s\n", list(rw.reanchors).c_str());
        std::printf("  %6s  %22s  %22s\n", "time", "MAD with / without", "SSIM with / without");
        for (size_t k = 0; k < with.mad.size(); ++k)
            std::printf("  %5zus  %10.3f / %-9.3f  %10.4f / %-9.4f\n", k * 10, with.mad[k], without.mad[k],
                        with.ss[k], without.ss[k]);
        std::printf("  max step %.0f ms, max reanchor step %.0f ms\n", rw.max_step_ms, rw.max_reanchor_ms);
        CHECK(rw.ok && rn.ok, "drift calls failed");
        const size_t last = with.mad.size() - 1;
        // With the reanchor, the end of the call is no further from the
        // portrait than without it (0.5 of slack for the noise of one second).
        CHECK(with.mad[last] <= without.mad[last] + 0.5, "with reanchor %.3f, without %.3f at %d s",
              with.mad[last], without.mad[last], drift_s);
    } else {
        std::printf("drift: skipped (set AVATARFORCING_TEST_DRIFT_SECONDS, for example 120)\n");
    }

    if (g_failures) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("ok\n");
    return 0;
}
