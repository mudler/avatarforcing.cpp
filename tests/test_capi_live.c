/* C API test of live mode, in plain C.
 *
 * Runs one simulated call twice on a live session and checks:
 *   - the frame queue bound (AVATARFORCING_LIVE_QUEUE_FULL, nothing lost);
 *   - phrase placement: the returned start frames, appends, a start inside
 *     an earlier phrase, a phrase too far ahead;
 *   - interrupt: the cut frame, and a push that waits during an interrupt
 *     is dropped (AVATARFORCING_LIVE_DROPPED);
 *   - idle frames are stable, and the mouth moves more in the phrase than in
 *     the idle parts (needs AVATARFORCING_TEST_PORTRAIT, a real face);
 *   - determinism: the two runs give the same frames;
 *   - a live session leaves the offline path unchanged: the same clip before
 *     and after it gives the same frames;
 *   - the reanchor options and stats (test_live_reanchor tests the reanchor).
 *
 * Needs AVATARFORCING_TEST_GGUF; exits 77 (skip) when it is unset.
 * Optional: AVATARFORCING_TEST_PORTRAIT (a 512x512 face image; else a
 * synthetic image and no mouth check), AVATARFORCING_TEST_WAV (speech, 16-bit
 * or float WAV of at least 3.6 s; else a synthetic signal),
 * AVATARFORCING_TEST_THREADS, AVATARFORCING_TEST_BACKEND and
 * AVATARFORCING_TEST_OVERLAP (1: decoder thread, 0: none; default: the
 * context default). */

#define _POSIX_C_SOURCE 200809L

#include "avatarforcing.h"

#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int g_failures = 0;

#define CHECK(cond, ...)                                       \
    do {                                                       \
        if (!(cond)) {                                         \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
            fprintf(stderr, __VA_ARGS__);                      \
            fprintf(stderr, "\n");                             \
            ++g_failures;                                      \
        }                                                      \
    } while (0)

static void put_u16(FILE* f, unsigned v) { fputc(v & 0xFF, f); fputc((v >> 8) & 0xFF, f); }
static void put_u32(FILE* f, uint32_t v) { put_u16(f, v & 0xFFFF); put_u16(f, v >> 16); }

/* A smooth synthetic 24-bit BMP, as in test_capi.c. */
static int write_bmp(const char* path, int s) {
    FILE* f = fopen(path, "wb");
    if (!f) return 0;
    const uint32_t row = (uint32_t)s * 3, size = row * (uint32_t)s;
    fputc('B', f); fputc('M', f);
    put_u32(f, 54 + size); put_u32(f, 0); put_u32(f, 54);
    put_u32(f, 40); put_u32(f, (uint32_t)s); put_u32(f, (uint32_t)s);
    put_u16(f, 1); put_u16(f, 24); put_u32(f, 0); put_u32(f, size);
    put_u32(f, 2835); put_u32(f, 2835); put_u32(f, 0); put_u32(f, 0);
    for (int y = s - 1; y >= 0; --y)
        for (int x = 0; x < s; ++x) {
            const double dx = (x - s / 2) / (double)s, dy = (y - s / 2) / (double)s;
            const double r = sqrt(dx * dx + dy * dy);
            const int v = (int)(200.0 * exp(-8.0 * r * r));
            fputc(40 + v / 2, f);
            fputc(60 + (v * 3) / 4, f);
            fputc(50 + v, f);
        }
    return fclose(f) == 0;
}

static uint32_t rd_u32(const uint8_t* p) { return p[0] | (p[1] << 8) | (p[2] << 16) | ((uint32_t)p[3] << 24); }
static unsigned rd_u16(const uint8_t* p) { return p[0] | (p[1] << 8); }

/* Reads a 16-bit PCM or 32-bit float WAV file as mono float. */
static float* read_wav(const char* path, int* n_out, int* sr_out) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    const long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t* b = (uint8_t*)malloc((size_t)sz);
    const size_t got = fread(b, 1, (size_t)sz, f);
    fclose(f);
    float* out = NULL;
    unsigned fmt = 0, ch = 0, bits = 0;
    uint32_t sr = 0;
    if (got != (size_t)sz || sz < 12 || memcmp(b, "RIFF", 4) || memcmp(b + 8, "WAVE", 4)) goto done;
    for (long p = 12; p + 8 <= sz;) {
        const uint32_t len = rd_u32(b + p + 4);
        if (!memcmp(b + p, "fmt ", 4) && len >= 16) {
            fmt = rd_u16(b + p + 8);
            ch = rd_u16(b + p + 10);
            sr = rd_u32(b + p + 12);
            bits = rd_u16(b + p + 22);
            if (fmt == 0xFFFE && len >= 26) fmt = rd_u16(b + p + 32);
        } else if (!memcmp(b + p, "data", 4) && ch > 0) {
            const int bps = (int)bits / 8;
            if (!((fmt == 1 && bits == 16) || (fmt == 3 && bits == 32))) goto done;
            uint32_t avail = (uint32_t)(sz - (p + 8));
            if (len < avail) avail = len;
            const int n = (int)(avail / (uint32_t)(bps * (int)ch));
            out = (float*)malloc(sizeof(float) * (size_t)(n > 0 ? n : 1));
            for (int i = 0; i < n; ++i) {
                double acc = 0;
                for (unsigned c = 0; c < ch; ++c) {
                    const uint8_t* q = b + p + 8 + ((size_t)i * ch + c) * (size_t)bps;
                    if (fmt == 1) acc += (int16_t)rd_u16(q) / 32768.0;
                    else { float v; memcpy(&v, q, 4); acc += v; }
                }
                out[i] = (float)(acc / ch);
            }
            *n_out = n;
            *sr_out = (int)sr;
            goto done;
        }
        p += 8 + len + (len & 1);
    }
done:
    free(b);
    return out;
}

static uint64_t fnv1a(uint64_t h, const uint8_t* p, size_t n) {
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}

/* Mean absolute difference of two frames over a box (fractions of the size). */
static double box_diff(const uint8_t* a, const uint8_t* b, int s, double x0, double x1, double y0, double y1) {
    double sum = 0;
    long n = 0;
    for (int y = (int)(y0 * s); y < (int)(y1 * s); ++y)
        for (int x = (int)(x0 * s); x < (int)(x1 * s); ++x)
            for (int c = 0; c < 3; ++c) {
                const size_t k = ((size_t)y * s + x) * 3 + c;
                sum += fabs((double)a[k] - (double)b[k]);
                ++n;
            }
    return n ? sum / n : 0;
}

static long long json_int(const char* js, const char* key) {
    char pat[64];
    snprintf(pat, sizeof(pat), "\"%s\":", key);
    const char* p = js ? strstr(js, pat) : NULL;
    return p ? atoll(p + strlen(pat)) : -999;
}

static long long stat_int(avatarforcing_session* s, const char* key) {
    char* js = avatarforcing_session_stats_json(s);
    const long long v = json_int(js, key);
    avatarforcing_free_string(js);
    return v;
}

/* ---- the call ------------------------------------------------------------ */

#define N_FRAMES 150   /* 15 blocks */

typedef struct {
    avatarforcing_session* s;
    const float* pcm;
    int n, sr;
    int64_t step_rc, push_rc;
} Race;

static void* race_step(void* p) {
    Race* r = (Race*)p;
    r->step_rc = avatarforcing_session_step(r->s);
    return NULL;
}

static void* race_push(void* p) {
    Race* r = (Race*)p;
    r->push_rc = avatarforcing_session_push_phrase(r->s, r->pcm, r->n, r->sr, -1);
    return NULL;
}

static void sleep_ms(int ms) {
    struct timespec ts = {ms / 1000, (long)(ms % 1000) * 1000000L};
    nanosleep(&ts, NULL);
}

/* Pulls every ready frame into frames (N_FRAMES max); returns the count. */
static int pull_all(avatarforcing_session* s, uint8_t* frames, int have, size_t fb) {
    int k;
    while (have < N_FRAMES && (k = avatarforcing_session_pull_frames(s, frames + (size_t)have * fb, N_FRAMES - have)) > 0)
        have += k;
    return have;
}

/* One step. With a decoder thread the frames of earlier steps can still be
 * in the queue: on AVATARFORCING_LIVE_QUEUE_FULL pull what is ready and step
 * again, as a live caller does. */
static int64_t step_pull(avatarforcing_session* s, uint8_t* frames, int* have, size_t fb) {
    for (int tries = 0; tries < 120000; ++tries) {
        const int64_t r = avatarforcing_session_step(s);
        if (r != AVATARFORCING_LIVE_QUEUE_FULL) return r;
        sleep_ms(5);
        *have = pull_all(s, frames, *have, fb);
    }
    return AVATARFORCING_LIVE_QUEUE_FULL;
}

/* Phrases: three of 1.2 s (30 frames) from the speech. The script:
 *   frames   0..39  idle (with the queue test)
 *   frames  45..74  phrase 1 (asked at 45)
 *   frames  75..104 phrase 2 (appended)
 *   frames 105..109 phrase 3 (asked at 50, moved after phrase 2), cut at 110
 *   frames 110..149 idle */
static int run_call(avatarforcing_ctx* ctx, const char* portrait, const float* speech, int sr, int cpu,
                    uint8_t* frames, int verbose) {
    const size_t fb = 512 * 512 * 3;
    const int ph = (int)(1.2 * sr);
    avatarforcing_session* s = avatarforcing_session_open(ctx, portrait, "{\"live\": true, \"seed\": 7}");
    CHECK(s != NULL, "live session_open: %s", avatarforcing_last_error(ctx));
    if (!s) return 0;
    char* js = avatarforcing_session_stats_json(s);
    if (verbose) printf("open: %s\n", js ? js : "(null)");
    CHECK(json_int(js, "next_frame") == 0 && json_int(js, "queue_frames") == 30, "stats after open: %s", js);
    CHECK(json_int(js, "reanchors") == 0 && json_int(js, "last_reanchor_frame") == -1 &&
              json_int(js, "reanchor_every") == 112 && json_int(js, "reanchor_max_defer") == 162 &&
              json_int(js, "reanchor_fade_frames") == 25,
          "reanchor stats after open: %s", js);
    avatarforcing_free_string(js);

    /* Not for a live session. */
    CHECK(avatarforcing_session_push_pcm(s, speech, ph, sr) == -1, "push_pcm on a live session");
    CHECK(avatarforcing_session_flush(s) == -1, "flush on a live session");

    /* Queue bound: 3 blocks without a pull, the 4th does not fit. */
    int have = 0;
    for (int b = 0; b < 3; ++b) {
        const int64_t r = avatarforcing_session_step(s);
        CHECK(r == b * 10, "step %d returned %lld: %s", b, (long long)r, avatarforcing_last_error(ctx));
    }
    CHECK(avatarforcing_session_step(s) == AVATARFORCING_LIVE_QUEUE_FULL, "step with a full queue");
    CHECK(stat_int(s, "next_frame") == 30 && stat_int(s, "queue_full") == 1 && stat_int(s, "queued_frames") == 30,
          "stats with a full queue");
    /* Without overlap the frames are ready when step returns. */
    while (avatarforcing_session_frames_ready(s) < 30) sleep_ms(5);
    CHECK(avatarforcing_session_pull_frames(s, frames, 10) == 10, "pull 10");
    have = 10;
    CHECK(avatarforcing_session_step(s) == 30, "step after a pull: %s", avatarforcing_last_error(ctx));
    while (avatarforcing_session_frames_ready(s) < 30) sleep_ms(5);
    have = pull_all(s, frames, have, fb);
    CHECK(have == 40, "frames after the queue test: %d", have);

    /* Phrase placement. */
    int64_t st = avatarforcing_session_push_phrase(s, speech, ph, sr, 45);
    CHECK(st == 45, "phrase 1 at %lld: %s", (long long)st, avatarforcing_last_error(ctx));
    st = avatarforcing_session_push_phrase(s, speech + ph, ph, sr, -1);
    CHECK(st == 75, "phrase 2 at %lld: %s", (long long)st, avatarforcing_last_error(ctx));
    st = avatarforcing_session_push_phrase(s, speech + 2 * ph, ph, sr, 50);
    CHECK(st == 105, "phrase 3 at %lld: %s", (long long)st, avatarforcing_last_error(ctx));
    CHECK(stat_int(s, "speech_end") == 135 && stat_int(s, "pending_phrases") == 3, "speech_end after 3 phrases");
    /* Too far ahead: the error leaves the session as it was. */
    st = avatarforcing_session_push_phrase(s, speech, ph, sr, 40 + 30 * 25);
    CHECK(st == -1 && strstr(avatarforcing_last_error(ctx), "frames after"), "far phrase: %lld %s", (long long)st,
          avatarforcing_last_error(ctx));
    CHECK(avatarforcing_session_push_phrase(s, speech, 0, sr, -1) == -1, "empty phrase");

    /* Steps to frame 100. */
    for (int64_t f = 40; f < 100; f += 10) {
        const int64_t r = step_pull(s, frames, &have, fb);
        CHECK(r == f, "step at %lld returned %lld: %s", (long long)f, (long long)r, avatarforcing_last_error(ctx));
        have = pull_all(s, frames, have, fb);
    }
    /* Interrupt (barge-in) at 110: phrase 3 keeps 105..109. An interrupt at a
     * computed frame cuts at the next block. */
    CHECK(avatarforcing_session_interrupt(s, 110) == 110, "interrupt at 110");
    CHECK(stat_int(s, "speech_end") == 110, "speech_end after the interrupt");

    /* A push that waits for the context lock while a step runs, and an
     * interrupt in that time: the push is dropped. The step computes frames
     * 100..109; the interrupt at -1 cuts at 110, which is where the cut is
     * already, so the frames do not change. */
    while (avatarforcing_session_frames_ready(s) + have < 100) sleep_ms(5);   /* room in the queue */
    have = pull_all(s, frames, have, fb);
    Race race = {s, speech, ph, sr, -9, -9};
    pthread_t t1, t2;
    pthread_create(&t1, NULL, race_step, &race);
    sleep_ms(cpu ? 300 : 20);
    pthread_create(&t2, NULL, race_push, &race);
    sleep_ms(cpu ? 300 : 20);
    const int64_t cut = avatarforcing_session_interrupt(s, -1);
    pthread_join(t1, NULL);
    pthread_join(t2, NULL);
    CHECK(race.step_rc == 100, "step in the race: %lld", (long long)race.step_rc);
    CHECK(cut == 110, "interrupt during a step: %lld", (long long)cut);
    /* The push read the epoch before the interrupt. It is dropped, or, when
     * the step and the push both ended before the interrupt (a fast device),
     * it landed at 110 and the interrupt cut it. The frames are the same. */
    CHECK(race.push_rc == AVATARFORCING_LIVE_DROPPED || race.push_rc == 110, "push during an interrupt: %lld",
          (long long)race.push_rc);
    if (verbose)
        printf("race: push %s\n", race.push_rc == AVATARFORCING_LIVE_DROPPED ? "dropped" : "landed, then cut");
    CHECK(stat_int(s, "speech_end") == 110 && stat_int(s, "interrupts") >= 2, "stats after the race");
    have = pull_all(s, frames, have, fb);

    for (int64_t f = 110; f < N_FRAMES; f += 10) {
        const int64_t r = step_pull(s, frames, &have, fb);
        CHECK(r == f, "step at %lld returned %lld: %s", (long long)f, (long long)r, avatarforcing_last_error(ctx));
        have = pull_all(s, frames, have, fb);
    }
    for (int waited = 0; have < N_FRAMES && waited < 120000; waited += 5) {   /* at most 10 min */
        sleep_ms(5);
        have = pull_all(s, frames, have, fb);
    }
    js = avatarforcing_session_stats_json(s);
    if (verbose) printf("end: %s\n", js ? js : "(null)");
    CHECK(json_int(js, "blocks") == 15 && json_int(js, "speech_blocks") == 7 && json_int(js, "queued_frames") == 0,
          "stats at the end: %s", js);
    /* 15 blocks: no reanchor with the default policy (the first is at block 112). */
    CHECK(json_int(js, "reanchors") == 0 && json_int(js, "blocks_since_reanchor") == 15, "reanchors at the end: %s", js);
    avatarforcing_free_string(js);
    avatarforcing_session_close(s);
    return have;
}

int main(void) {
    const char* gguf = getenv("AVATARFORCING_TEST_GGUF");
    if (!gguf || !*gguf) {
        fprintf(stderr, "skip: set AVATARFORCING_TEST_GGUF\n");
        return 77;
    }
    const char* th = getenv("AVATARFORCING_TEST_THREADS");
    const char* be = getenv("AVATARFORCING_TEST_BACKEND");
    const char* portrait_env = getenv("AVATARFORCING_TEST_PORTRAIT");
    const char* wav_env = getenv("AVATARFORCING_TEST_WAV");
    const int cpu = !(be && *be) || strcmp(be, "cpu") == 0;
    const char* ov = getenv("AVATARFORCING_TEST_OVERLAP");
    char opts[256];
    snprintf(opts, sizeof(opts), "{\"backend\": \"%s\", \"threads\": %d%s}", be && *be ? be : "cpu",
             th && *th ? atoi(th) : 0,
             ov && *ov ? (atoi(ov) ? ", \"overlap\": true" : ", \"overlap\": false") : "");

    char* err = NULL;
    avatarforcing_ctx* ctx = avatarforcing_load(gguf, opts, &err);
    if (!ctx) {
        fprintf(stderr, "FAIL: load: %s\n", err ? err : "(null)");
        avatarforcing_free_string(err);
        return 1;
    }
    const int real_face = portrait_env && *portrait_env;
    const char* portrait = real_face ? portrait_env : "test_capi_live_portrait.bmp";
    if (!real_face) CHECK(write_bmp(portrait, 512), "cannot write %s", portrait);

    /* Speech: 3.6 s for the three phrases. */
    int sr = 16000, n = 0;
    float* speech = NULL;
    if (wav_env && *wav_env) {
        speech = read_wav(wav_env, &n, &sr);
        CHECK(speech != NULL, "cannot read %s (16-bit PCM or float WAV)", wav_env);
        CHECK(n >= (int)(3.6 * sr), "%s is shorter than 3.6 s", wav_env);
        if (!speech || n < (int)(3.6 * sr)) return 1;
    } else {
        n = (int)(3.6 * sr);
        speech = (float*)malloc(sizeof(float) * (size_t)n);
        for (int i = 0; i < n; ++i) {
            const double t = i / (double)sr;
            const double env = 0.5 + 0.5 * sin(2.0 * M_PI * 3.0 * t);
            speech[i] = (float)(0.3 * env * (sin(2.0 * M_PI * 180.0 * t) + 0.5 * sin(2.0 * M_PI * 720.0 * t)));
        }
    }

    /* ---- option errors ---- */
    CHECK(avatarforcing_session_open(ctx, portrait, "{\"live_queue_blocks\": 2}") == NULL &&
              strstr(avatarforcing_last_error(ctx), "live"),
          "live_* without live: %s", avatarforcing_last_error(ctx));
    CHECK(avatarforcing_session_open(ctx, portrait, "{\"live\": true, \"live_queue_blocks\": 0}") == NULL,
          "live_queue_blocks 0 opens");
    CHECK(avatarforcing_session_open(ctx, portrait, "{\"live_reanchor_every\": 5}") == NULL &&
              strstr(avatarforcing_last_error(ctx), "live"),
          "live_reanchor_every without live: %s", avatarforcing_last_error(ctx));
    CHECK(avatarforcing_session_open(ctx, portrait, "{\"live\": true, \"live_reanchor_max_defer\": -1}") == NULL,
          "live_reanchor_max_defer -1 opens");
    CHECK(avatarforcing_session_open(ctx, portrait, "{\"live\": true, \"live_reanchor_hold\": 1}") == NULL,
          "live_reanchor_hold takes a number");
    uint8_t* none = NULL;
    CHECK(avatarforcing_generate_frames_pcm(ctx, portrait, speech, sr, sr, "{\"live\": true}", &none, NULL, NULL, NULL,
                                            NULL) == -1 &&
              strstr(avatarforcing_last_error(ctx), "unknown request option"),
          "live in a generate request: %s", avatarforcing_last_error(ctx));

    /* ---- offline before ---- */
    const size_t fb = 512 * 512 * 3;
    uint8_t* off1 = NULL;
    uint8_t* off2 = NULL;
    int nf1 = 0, nf2 = 0;
    const char* req = "{\"seed\": 3}";
    CHECK(avatarforcing_generate_frames_pcm(ctx, portrait, speech, 2 * sr, sr, req, &off1, &nf1, NULL, NULL, NULL) == 0,
          "offline: %s", avatarforcing_last_error(ctx));

    /* A session that is not live has no live functions. */
    avatarforcing_session* plain = avatarforcing_session_open(ctx, portrait, NULL);
    CHECK(plain != NULL, "plain session: %s", avatarforcing_last_error(ctx));
    if (plain) {
        CHECK(avatarforcing_session_step(plain) == -1 && strstr(avatarforcing_last_error(ctx), "live"), "step on a plain session");
        CHECK(avatarforcing_session_interrupt(plain, 0) == -1, "interrupt on a plain session");
        CHECK(avatarforcing_session_push_phrase(plain, speech, sr, sr, -1) == -1, "push_phrase on a plain session");
        CHECK(avatarforcing_session_stats_json(plain) == NULL, "stats on a plain session");
        CHECK(avatarforcing_session_reanchor(plain) == -1, "reanchor on a plain session");
        avatarforcing_session_close(plain);
    }

    /* ---- the call, twice ---- */
    uint8_t* a = (uint8_t*)malloc(fb * N_FRAMES);
    uint8_t* b = (uint8_t*)malloc(fb * N_FRAMES);
    const int na = run_call(ctx, portrait, speech, sr, cpu, a, 1);
    const int nb = run_call(ctx, portrait, speech, sr, cpu, b, 0);
    CHECK(na == N_FRAMES && nb == N_FRAMES, "frames %d and %d", na, nb);
    const uint64_t ha = fnv1a(1469598103934665603ULL, a, fb * (size_t)na);
    const uint64_t hb = fnv1a(1469598103934665603ULL, b, fb * (size_t)nb);
    printf("live: %d frames, fnv1a %016llx; second run %016llx\n", na, (unsigned long long)ha, (unsigned long long)hb);
    CHECK(na == nb && memcmp(a, b, fb * (size_t)na) == 0, "the two runs differ");

    /* ---- idle and speech motion ----
     * Per frame: the change from the previous frame, over the whole frame and
     * over the mouth box (a face-centred crop: x 0.38..0.64, y 0.55..0.70).
     * Idle parts: frames 1..39 and 125..149 (the model needs a few frames to
     * settle after the cut at 110). Speech: frames 50..109. */
    double idle_all = 0, idle_mouth = 0, speech_mouth = 0, drift = 0;
    int n_idle = 0, n_speech = 0;
    for (int f = 1; f < na; ++f) {
        const uint8_t* p = a + (size_t)(f - 1) * fb;
        const uint8_t* q = a + (size_t)f * fb;
        const double m = box_diff(p, q, 512, 0.38, 0.64, 0.55, 0.70);
        if (f < 40 || f >= 125) {
            idle_all += box_diff(p, q, 512, 0, 1, 0, 1);
            idle_mouth += m;
            ++n_idle;
            if (f < 40) {   /* the first idle part, against frame 0 */
                const double d = box_diff(a, q, 512, 0, 1, 0, 1);
                if (d > drift) drift = d;
            }
        } else if (f >= 50 && f < 110) {
            speech_mouth += m;
            ++n_speech;
        }
    }
    idle_all /= n_idle;
    idle_mouth /= n_idle;
    speech_mouth /= n_speech;
    printf("motion per frame (mean abs diff, 0..255): idle frame %.3f, idle mouth %.3f, speech mouth %.3f; "
           "drift of frames 1..39 from frame 0: max %.3f\n",
           idle_all, idle_mouth, speech_mouth, drift);
    /* Stable idle: small changes between frames, and the first 40 frames stay
     * close to frame 0 (after speech the head can rest in another pose). */
    CHECK(idle_all < 2.0, "idle frames change by %.3f per frame", idle_all);
    CHECK(drift < 12.0, "idle frames drift by %.3f from frame 0", drift);
    if (real_face)
        CHECK(speech_mouth > 1.5 * idle_mouth, "the mouth moves %.3f in the phrase, %.3f when idle", speech_mouth,
              idle_mouth);
    else
        printf("note: synthetic portrait, no mouth check (set AVATARFORCING_TEST_PORTRAIT)\n");

    /* ---- offline after: the live sessions left no state ---- */
    CHECK(avatarforcing_generate_frames_pcm(ctx, portrait, speech, 2 * sr, sr, req, &off2, &nf2, NULL, NULL, NULL) == 0,
          "offline after: %s", avatarforcing_last_error(ctx));
    CHECK(nf1 == nf2 && off1 && off2 && memcmp(off1, off2, fb * (size_t)nf1) == 0, "offline frames changed");
    printf("offline: %d frames, fnv1a %016llx before and %016llx after\n", nf1,
           (unsigned long long)(off1 ? fnv1a(1469598103934665603ULL, off1, fb * (size_t)nf1) : 0),
           (unsigned long long)(off2 ? fnv1a(1469598103934665603ULL, off2, fb * (size_t)nf2) : 0));

    avatarforcing_free_frames(off1);
    avatarforcing_free_frames(off2);
    free(a);
    free(b);
    free(speech);
    avatarforcing_free(ctx);

    if (g_failures) {
        fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    printf("ok\n");
    return 0;
}
