/* C API test, in plain C: loads the model through libavatarforcing,
 * generates a short clip as RGB frames from PCM, from a WAV file and through
 * a session, checks that the three give the same frames, and checks the
 * error paths (bad GGUF, bad options, missing image, ffmpeg missing for mp4,
 * a second session, generation during a session).
 *
 * Needs AVATARFORCING_TEST_GGUF; exits 77 (skip) when it is unset.
 * AVATARFORCING_TEST_THREADS and AVATARFORCING_TEST_BACKEND are optional.
 * The test writes its inputs (a synthetic portrait and a WAV file) to the
 * current directory. */

#define _POSIX_C_SOURCE 200809L

#include "avatarforcing.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

/* A smooth synthetic 24-bit BMP; the model runs on any image. */
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
            fputc(40 + v / 2, f);          /* B */
            fputc(60 + (v * 3) / 4, f);    /* G */
            fputc(50 + v, f);              /* R */
        }
    return fclose(f) == 0;
}

/* Mono float32 WAV. */
static int write_wav(const char* path, const float* pcm, int n, int sr) {
    FILE* f = fopen(path, "wb");
    if (!f) return 0;
    const uint32_t data = (uint32_t)n * 4;
    fwrite("RIFF", 1, 4, f); put_u32(f, 36 + data); fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f); put_u32(f, 16); put_u16(f, 3); put_u16(f, 1);
    put_u32(f, (uint32_t)sr); put_u32(f, (uint32_t)sr * 4); put_u16(f, 4); put_u16(f, 32);
    fwrite("data", 1, 4, f); put_u32(f, data);
    fwrite(pcm, 4, (size_t)n, f);
    return fclose(f) == 0;
}

static uint64_t fnv1a(const uint8_t* p, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}

int main(void) {
    const char* gguf = getenv("AVATARFORCING_TEST_GGUF");
    if (!gguf || !*gguf) {
        fprintf(stderr, "skip: set AVATARFORCING_TEST_GGUF\n");
        return 77;
    }
    const char* th = getenv("AVATARFORCING_TEST_THREADS");
    const char* be = getenv("AVATARFORCING_TEST_BACKEND");
    char opts[256];
    snprintf(opts, sizeof(opts), "{\"backend\": \"%s\", \"threads\": %d}", be && *be ? be : "cpu",
             th && *th ? atoi(th) : 0);

    CHECK(avatarforcing_abi_version() == AVATARFORCING_ABI_VERSION, "abi version %d", avatarforcing_abi_version());

    /* ---- load errors ---- */
    char* err = NULL;
    const char* bad_gguf = "test_capi_bad.gguf";
    FILE* f = fopen(bad_gguf, "wb");
    if (f) { fputs("this is not a GGUF file", f); fclose(f); }
    avatarforcing_ctx* bad = avatarforcing_load(bad_gguf, NULL, &err);
    CHECK(bad == NULL, "a bad GGUF loads");
    CHECK(err && strstr(err, "GGUF"), "bad GGUF error: %s", err ? err : "(null)");
    printf("bad GGUF: %s\n", err ? err : "(null)");
    avatarforcing_free_string(err);
    err = NULL;
    bad = avatarforcing_load("test_capi_missing.gguf", NULL, &err);
    CHECK(bad == NULL && err && strstr(err, "cannot open"), "missing GGUF error: %s", err ? err : "(null)");
    avatarforcing_free_string(err);
    err = NULL;
    bad = avatarforcing_load(gguf, "{\"thread\": 4}", &err);
    CHECK(bad == NULL && err && strstr(err, "unknown option"), "unknown option error: %s", err ? err : "(null)");
    avatarforcing_free_string(err);
    err = NULL;
    bad = avatarforcing_load(gguf, "{\"threads\": 4", &err);
    CHECK(bad == NULL && err && strstr(err, "invalid options"), "bad JSON error: %s", err ? err : "(null)");
    avatarforcing_free_string(err);
    err = NULL;

    /* ---- load ---- */
    avatarforcing_ctx* ctx = avatarforcing_load(gguf, opts, &err);
    if (!ctx) {
        fprintf(stderr, "FAIL: load: %s\n", err ? err : "(null)");
        avatarforcing_free_string(err);
        return 1;
    }
    CHECK(err == NULL, "error_out is set on success");
    char* info = avatarforcing_info_json(ctx);
    CHECK(info && strstr(info, "\"width\":512") && strstr(info, "\"fps\":25"), "info: %s", info ? info : "(null)");
    printf("info: %s\n", info ? info : "(null)");
    avatarforcing_free_string(info);
    CHECK(strcmp(avatarforcing_last_error(ctx), "") == 0, "fresh context has an error");

    /* ---- inputs: 2.5 s at 16 kHz, T = 63 audio frames: the primer (50),
     * one full steady block and a last block of 3 frames ---- */
    const char* portrait = "test_capi_portrait.bmp";
    const char* wav = "test_capi_speech.wav";
    const int sr = 16000, n = 40000;
    float* pcm = (float*)malloc(sizeof(float) * n);
    for (int i = 0; i < n; ++i) {
        const double t = i / (double)sr;
        const double env = 0.5 + 0.5 * sin(2.0 * M_PI * 3.0 * t);
        pcm[i] = (float)(0.3 * env * (sin(2.0 * M_PI * 180.0 * t) + 0.5 * sin(2.0 * M_PI * 720.0 * t) +
                                      0.25 * sin(2.0 * M_PI * 1300.0 * t)));
    }
    CHECK(write_bmp(portrait, 512), "cannot write %s", portrait);
    CHECK(write_wav(wav, pcm, n, sr), "cannot write %s", wav);

    uint8_t* frames = NULL;
    int nf = 0, w = 0, h = 0, fps = 0;

    /* ---- request errors ---- */
    CHECK(avatarforcing_generate_frames(ctx, "test_capi_no_such_image.png", wav, NULL, &frames, &nf, &w, &h, &fps) == -1,
          "a missing image succeeds");
    CHECK(strstr(avatarforcing_last_error(ctx), "cannot read"), "missing image error: %s", avatarforcing_last_error(ctx));
    printf("missing image: %s\n", avatarforcing_last_error(ctx));
    CHECK(avatarforcing_generate_frames(ctx, portrait, wav, "{\"nfe\": 2}", &frames, &nf, &w, &h, &fps) == -1,
          "nfe 2 succeeds");
    CHECK(strstr(avatarforcing_last_error(ctx), "nfe"), "nfe error: %s", avatarforcing_last_error(ctx));
    {
        /* ffmpeg missing: posix_spawnp finds nothing on this PATH. */
        const char* old = getenv("PATH");
        char* saved = old ? strdup(old) : NULL;
        setenv("PATH", "/nonexistent-avatarforcing-test", 1);
        const int rc = avatarforcing_generate_mp4(ctx, portrait, wav, "test_capi_out.mp4", NULL);
        if (saved) { setenv("PATH", saved, 1); free(saved); } else unsetenv("PATH");
        CHECK(rc == -1, "mp4 without ffmpeg succeeds");
        CHECK(strstr(avatarforcing_last_error(ctx), "ffmpeg"), "ffmpeg error: %s", avatarforcing_last_error(ctx));
        printf("ffmpeg missing: %s\n", avatarforcing_last_error(ctx));
        const int rc2 = avatarforcing_generate_mp4_pcm(ctx, portrait, pcm, n, sr, "test_capi_out.mp4", "{\"ffmpeg\": 1}");
        CHECK(rc2 == -1 && strstr(avatarforcing_last_error(ctx), "unknown request option"),
              "context option in a request: %s", avatarforcing_last_error(ctx));
    }

    /* ---- offline generation from PCM ---- */
    const char* req = "{\"nfe\": 6, \"seed\": 7}";
    CHECK(avatarforcing_generate_frames_pcm(ctx, portrait, pcm, n, sr, req, &frames, &nf, &w, &h, &fps) == 0,
          "generate_frames_pcm: %s", avatarforcing_last_error(ctx));
    CHECK(nf == 63 && w == 512 && h == 512 && fps == 25, "frames %d, %dx%d at %d fps", nf, w, h, fps);
    const size_t fb = (size_t)w * h * 3, total = (size_t)nf * fb;
    uint64_t h_pcm = frames ? fnv1a(frames, total) : 0;
    printf("pcm: %d frames, fnv1a %016llx\n", nf, (unsigned long long)h_pcm);
    {
        /* The frames are not constant. */
        int lo = 255, hi = 0;
        for (size_t i = 0; frames && i < total; i += 997) {
            if (frames[i] < lo) lo = frames[i];
            if (frames[i] > hi) hi = frames[i];
        }
        CHECK(hi - lo > 50, "frame range %d..%d", lo, hi);
    }

    /* ---- the same audio from the WAV file ---- */
    uint8_t* frames2 = NULL;
    int nf2 = 0;
    CHECK(avatarforcing_generate_frames(ctx, portrait, wav, req, &frames2, &nf2, NULL, NULL, NULL) == 0,
          "generate_frames: %s", avatarforcing_last_error(ctx));
    CHECK(nf2 == nf && frames && frames2 && memcmp(frames, frames2, total) == 0, "WAV and PCM frames differ");
    printf("wav: %d frames, fnv1a %016llx\n", nf2, (unsigned long long)(frames2 ? fnv1a(frames2, (size_t)nf2 * fb) : 0));
    avatarforcing_free_frames(frames2);

    /* ---- session: one phrase, then flush ---- */
    avatarforcing_session* s = avatarforcing_session_open(ctx, portrait, req);
    CHECK(s != NULL, "session_open: %s", avatarforcing_last_error(ctx));
    if (s) {
        CHECK(avatarforcing_session_open(ctx, portrait, req) == NULL, "a second session opens");
        CHECK(strstr(avatarforcing_last_error(ctx), "session"), "second session error: %s", avatarforcing_last_error(ctx));
        CHECK(avatarforcing_generate_frames_pcm(ctx, portrait, pcm, n, sr, req, &frames2, &nf2, NULL, NULL, NULL) == -1,
              "generation runs during a session");
        uint8_t* got = (uint8_t*)malloc(total + fb);
        int have = 0;
        CHECK(avatarforcing_session_push_pcm(s, pcm, n, sr) == 0, "push: %s", avatarforcing_last_error(ctx));
        for (int k; (k = avatarforcing_session_pull_frames(s, got + (size_t)have * fb, nf + 1 - have)) > 0;) have += k;
        printf("session: %d frames ready after the push\n", have);
        CHECK(avatarforcing_session_flush(s) == 0, "flush: %s", avatarforcing_last_error(ctx));
        CHECK(avatarforcing_session_push_pcm(s, pcm, 640, sr) == -1, "push after flush succeeds");
        for (int k; (k = avatarforcing_session_pull_frames(s, got + (size_t)have * fb, nf + 1 - have)) > 0;) have += k;
        CHECK(have == nf, "session frames %d, offline %d", have, nf);
        CHECK(frames && memcmp(frames, got, total) == 0, "session and offline frames differ");
        printf("session: %d frames, fnv1a %016llx\n", have, (unsigned long long)fnv1a(got, (size_t)have * fb));
        CHECK(avatarforcing_session_frames_ready(s) == 0, "frames left");
        free(got);
        avatarforcing_session_close(s);
    }

    avatarforcing_free_frames(frames);
    free(pcm);
    avatarforcing_free(ctx);
    avatarforcing_free(NULL);
    avatarforcing_session_close(NULL);

    if (g_failures) {
        fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    printf("ok\n");
    return 0;
}
