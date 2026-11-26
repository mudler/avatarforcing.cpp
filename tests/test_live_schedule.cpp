// LiveSchedule test: the frame-indexed features of live mode, with no model.
// Idle features are read cyclically by frame, phrases are placed without
// overlap, a cut drops the phrase features from a frame on, and committed
// frames do not change. Also the reanchor policy and cross-fade of the live
// app, with no model.

#include "live.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

static int g_failures = 0;

#define CHECK(cond, ...)                                          \
    do {                                                          \
        if (!(cond)) {                                            \
            std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
            std::fprintf(stderr, __VA_ARGS__);                    \
            std::fprintf(stderr, "\n");                           \
            ++g_failures;                                         \
        }                                                         \
    } while (0)

// Feature rows of width 2: idle row i is {i, 0}; phrase p row j is {j, p}.
static std::vector<float> phrase(int p, int n) {
    std::vector<float> r;
    for (int j = 0; j < n; ++j) { r.push_back((float)j); r.push_back((float)p); }
    return r;
}

int main() {
    const int idle_rows = 7, dim = 2, origin = 5, history = 5;
    std::vector<float> idle;
    for (int i = 0; i < idle_rows; ++i) { idle.push_back((float)i); idle.push_back(0.f); }
    af::LiveSchedule s;
    s.init(idle, idle_rows, dim, origin, history);
    CHECK(s.next() == -origin, "next %lld", (long long)s.next());

    // Idle: frame f takes idle row (f + origin) mod idle_rows, also for the
    // primer frames before 0.
    std::vector<float> w(12 * dim);
    int sp = -1;
    s.window(-origin, 12, w.data(), &sp);
    CHECK(sp == 0, "speech rows %d", sp);
    for (int r = 0; r < 12; ++r) CHECK(w[r * dim] == (float)(r % idle_rows) && w[r * dim + 1] == 0.f, "idle row %d", r);
    s.commit(0);
    CHECK(s.next() == 0 && s.speech_end() == 0, "next %lld end %lld", (long long)s.next(), (long long)s.speech_end());

    std::string err;
    // Append: at the next frame when nothing is pending, then after the last phrase.
    std::vector<float> p1 = phrase(1, 4), p2 = phrase(2, 3), p3 = phrase(3, 2);
    CHECK(s.place(p1.data(), 4, -1, 100, err) == 0, "p1");
    CHECK(s.place(p2.data(), 3, -1, 100, err) == 4, "p2");
    // An explicit start inside an earlier phrase moves after it; a later one stays.
    CHECK(s.place(p3.data(), 2, 5, 100, err) == 7, "p3");
    std::vector<float> p4 = phrase(4, 2);
    CHECK(s.place(p4.data(), 2, 12, 100, err) == 12, "p4");
    CHECK(s.speech_end() == 14 && s.pending_phrases() == 4, "end %lld", (long long)s.speech_end());

    // Too far ahead: the end must be at most max_ahead frames after next().
    std::vector<float> p5 = phrase(5, 3);
    CHECK(s.place(p5.data(), 3, -1, 16, err) == -1 && !err.empty(), "max ahead");
    CHECK(s.speech_end() == 14, "a refused phrase was placed");

    // Window 0..15: p1 0..3, p2 4..6, p3 7..8, idle 9..11, p4 12..13, idle 14, 15.
    w.assign(16 * dim, -1.f);
    s.window(0, 16, w.data(), &sp);
    CHECK(sp == 11, "speech rows %d", sp);
    const float want[16][2] = {{0, 1}, {1, 1}, {2, 1}, {3, 1}, {0, 2}, {1, 2}, {2, 2}, {0, 3}, {1, 3},
                               {(9 + origin) % idle_rows, 0}, {(10 + origin) % idle_rows, 0},
                               {(11 + origin) % idle_rows, 0}, {0, 4}, {1, 4},
                               {(14 + origin) % idle_rows, 0}, {(15 + origin) % idle_rows, 0}};
    for (int r = 0; r < 16; ++r)
        CHECK(w[r * dim] == want[r][0] && w[r * dim + 1] == want[r][1], "frame %d: %g %g", r, w[r * dim], w[r * dim + 1]);

    // A step reads frames 0..4; a cut before next() cuts at next().
    s.commit(5);
    CHECK(s.cut(2) == 5, "cut before next");
    CHECK(s.speech_end() == 5 && s.pending_phrases() == 0, "end %lld", (long long)s.speech_end());
    // The committed frames keep their features.
    s.window(0, 6, w.data(), &sp);
    CHECK(sp == 5 && w[4 * dim + 1] == 2.f && w[5 * dim + 1] == 0.f, "committed frames changed");

    // A cut in the middle of a phrase shortens it; a cut at -1 is at next().
    std::vector<float> p6 = phrase(6, 6);
    CHECK(s.place(p6.data(), 6, 8, 100, err) == 8, "p6");
    CHECK(s.cut(10) == 10 && s.speech_end() == 10, "cut in a phrase");
    s.window(8, 3, w.data(), &sp);
    CHECK(sp == 2 && w[1 * dim + 1] == 6.f && w[2 * dim + 1] == 0.f, "cut rows");
    CHECK(s.cut(-1) == 5 && s.speech_end() == 5, "cut at next");

    // Phrases end more than `history` frames before next() are released, and
    // the frames read as idle again.
    std::vector<float> p7 = phrase(7, 2);
    CHECK(s.place(p7.data(), 2, -1, 100, err) == 5, "p7");
    s.commit(20);
    s.window(5, 2, w.data(), &sp);
    CHECK(sp == 0, "released phrase still read: %d rows", sp);

    // ---- reanchor policy: the block indices of the reanchors ----
    // Idle all the time: every `every` steady blocks, so blocks 4, 9, 14.
    {
        af::ReanchorPolicy rp;
        rp.every = 4;
        rp.max_defer = 6;
        std::vector<int> got;
        for (int b = 0; b < 16; ++b) {
            const bool r = rp.due(true, false);
            if (r) got.push_back(b);
            rp.done(r);
        }
        CHECK(got == std::vector<int>({4, 9, 14}), "idle reanchors: %zu", got.size());
    }
    // Speech in blocks 3..7: the reanchor waits, then is forced at
    // max_defer (block 6, in speech); idle again from block 8.
    {
        af::ReanchorPolicy rp;
        rp.every = 4;
        rp.max_defer = 6;
        std::vector<int> got;
        for (int b = 0; b < 20; ++b) {
            const bool idle = b < 3 || b > 7;
            const bool r = rp.due(idle, false);
            if (r) got.push_back(b);
            rp.done(r);
        }
        CHECK(got == std::vector<int>({6, 11, 16}), "speech reanchors: %zu: %d", got.size(), got.empty() ? -1 : got[0]);
    }
    // The live app values: the first reanchor of an idle call is block 112
    // (frame 1120, 44.8 s); in endless speech, block 162.
    {
        af::ReanchorPolicy idle_rp{112, 162}, speech_rp{112, 162};
        int first_idle = -1, first_speech = -1;
        for (int b = 0; b < 400 && (first_idle < 0 || first_speech < 0); ++b) {
            if (first_idle < 0 && idle_rp.due(true, false)) first_idle = b;
            idle_rp.done(false);
            if (first_speech < 0 && speech_rp.due(false, false)) first_speech = b;
            speech_rp.done(false);
        }
        CHECK(first_idle == 112 && first_speech == 162, "live app policy: %d %d", first_idle, first_speech);
    }
    // Off (0, 0): only a request reanchors.
    {
        af::ReanchorPolicy rp;
        int n = 0;
        for (int b = 0; b < 500; ++b) {
            const bool r = rp.due(true, b == 300);
            n += r;
            rp.done(r);
        }
        CHECK(n == 1 && rp.since == 199, "policy off: %d reanchors", n);
    }

    // ---- cross-fade: the live app ramp, alpha = r / total for r = total..1 ----
    {
        const int total = 25;
        std::vector<uint8_t> anchor = {0, 255, 100, 7}, frame = {255, 0, 100, 200};
        std::vector<std::vector<uint8_t>> seq;
        for (int r = total; r >= 1; --r) {
            std::vector<uint8_t> f = frame;
            af::crossfade(anchor.data(), f.data(), f.size(), r, total);
            seq.push_back(f);
        }
        // The first frame is the anchor, the steps are linear (at most
        // 255 / total + 1 apart), and the frame after the fade is the frame.
        CHECK(seq[0] == anchor, "fade frame 0 is not the anchor");
        int worst = 0;
        std::vector<uint8_t> prev = anchor;
        seq.push_back(frame);
        for (const auto& f : seq) {
            for (size_t i = 0; i < f.size(); ++i) worst = std::max(worst, std::abs((int)f[i] - (int)prev[i]));
            prev = f;
        }
        CHECK(worst <= 255 / total + 1, "fade step %d", worst);
        // The values of the live app expression in numpy for r = 12:
        // (0.48 * anchor + 0.52 * frame).clip(0, 255).astype(uint8).
        std::vector<uint8_t> f = frame;
        af::crossfade(anchor.data(), f.data(), f.size(), 12, total);
        CHECK(f[0] == 132 && f[1] == 122 && f[2] == 100 && f[3] == 107, "fade r=12: %d %d %d %d", f[0], f[1], f[2],
              f[3]);
    }

    if (g_failures) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("ok\n");
    return 0;
}
