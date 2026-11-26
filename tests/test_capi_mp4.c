/* C API mp4 test, in plain C: speech shorter than the 2 s primer.
 *
 * The video always has the 50 primer frames (2 s at 25 fps). The test
 * writes an mp4 from a 0.8 s WAV file and from 1.5 s of PCM, and checks
 * that both calls succeed. When ffprobe is on PATH, it also checks that each
 * mp4 has 50 frames and that the audio stream is as long as the video
 * stream. Last, it checks that an ffmpeg failure gives ffmpeg's own error
 * message.
 *
 * Needs AVATARFORCING_TEST_GGUF and ffmpeg on PATH; exits 77 (skip) when
 * either is missing. AVATARFORCING_TEST_THREADS and
 * AVATARFORCING_TEST_BACKEND are optional. The test writes its inputs and
 * outputs to the current directory. */

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
            fputc(40 + v / 2, f);
            fputc(60 + (v * 3) / 4, f);
            fputc(50 + v, f);
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

static void speech(float* pcm, int n, int sr) {
    for (int i = 0; i < n; ++i) {
        const double t = i / (double)sr;
        const double env = 0.5 + 0.5 * sin(2.0 * M_PI * 3.0 * t);
        pcm[i] = (float)(0.3 * env * (sin(2.0 * M_PI * 180.0 * t) + 0.5 * sin(2.0 * M_PI * 720.0 * t) +
                                      0.25 * sin(2.0 * M_PI * 1300.0 * t)));
    }
}

/* Reads "frames duration" of the first stream of one type with ffprobe.
 * Returns 0 when ffprobe gives no answer. */
static int probe(const char* path, const char* stream, long* frames, double* duration) {
    char cmd[512];
    snprintf(cmd, sizeof(cmd),
             "ffprobe -v error -count_frames -select_streams %s:0 "
             "-show_entries stream=nb_read_frames,duration -of csv=p=0 '%s' 2>/dev/null",
             stream, path);
    FILE* p = popen(cmd, "r");
    if (!p) return 0;
    char line[256] = {0};
    const int got = fgets(line, sizeof(line), p) != NULL;
    pclose(p);
    if (!got) return 0;
    /* csv order follows the stream fields: duration,nb_read_frames */
    double d = 0;
    long f = 0;
    if (sscanf(line, "%lf,%ld", &d, &f) != 2) return 0;
    *frames = f;
    *duration = d;
    return 1;
}

static void check_mp4(const char* path, int have_ffprobe) {
    if (!have_ffprobe) return;
    long vf = 0, af = 0;
    double vd = 0, ad = 0;
    CHECK(probe(path, "v", &vf, &vd), "ffprobe gives no video stream for %s", path);
    CHECK(probe(path, "a", &af, &ad), "ffprobe gives no audio stream for %s", path);
    printf("%s: video %ld frames %.3f s, audio %.3f s\n", path, vf, vd, ad);
    CHECK(vf == 50, "%s: %ld video frames, expected 50", path, vf);
    CHECK(fabs(vd - 2.0) < 1e-3, "%s: video %.3f s, expected 2 s", path, vd);
    CHECK(fabs(ad - vd) < 1e-3, "%s: audio %.3f s, video %.3f s", path, ad, vd);
}

int main(void) {
    const char* gguf = getenv("AVATARFORCING_TEST_GGUF");
    if (!gguf || !*gguf) {
        fprintf(stderr, "skip: set AVATARFORCING_TEST_GGUF\n");
        return 77;
    }
    if (system("ffmpeg -version > /dev/null 2>&1") != 0) {
        fprintf(stderr, "skip: ffmpeg is not on PATH\n");
        return 77;
    }
    const int have_ffprobe = system("ffprobe -version > /dev/null 2>&1") == 0;
    if (!have_ffprobe) printf("ffprobe is not on PATH: the mp4 streams are not checked\n");

    const char* th = getenv("AVATARFORCING_TEST_THREADS");
    const char* be = getenv("AVATARFORCING_TEST_BACKEND");
    char opts[256];
    snprintf(opts, sizeof(opts), "{\"backend\": \"%s\", \"threads\": %d}", be && *be ? be : "cpu",
             th && *th ? atoi(th) : 0);
    char* err = NULL;
    avatarforcing_ctx* ctx = avatarforcing_load(gguf, opts, &err);
    if (!ctx) {
        fprintf(stderr, "FAIL: load: %s\n", err ? err : "(null)");
        avatarforcing_free_string(err);
        return 1;
    }

    const char* portrait = "test_capi_mp4_portrait.bmp";
    const char* wav = "test_capi_mp4_speech.wav";
    const int sr = 16000, n_short = 12800, n_pcm = 24000;   /* 0.8 s and 1.5 s */
    float* pcm = (float*)malloc(sizeof(float) * n_pcm);
    speech(pcm, n_pcm, sr);
    CHECK(write_bmp(portrait, 512), "cannot write %s", portrait);
    CHECK(write_wav(wav, pcm, n_short, sr), "cannot write %s", wav);
    const char* req = "{\"nfe\": 6, \"seed\": 7}";

    /* ---- 0.8 s from a WAV file ---- */
    const char* out_wav = "test_capi_mp4_short_wav.mp4";
    remove(out_wav);
    CHECK(avatarforcing_generate_mp4(ctx, portrait, wav, out_wav, req) == 0, "generate_mp4, 0.8 s: %s",
          avatarforcing_last_error(ctx));
    check_mp4(out_wav, have_ffprobe);

    /* ---- 1.5 s from PCM ---- */
    const char* out_pcm = "test_capi_mp4_short_pcm.mp4";
    remove(out_pcm);
    CHECK(avatarforcing_generate_mp4_pcm(ctx, portrait, pcm, n_pcm, sr, out_pcm, req) == 0,
          "generate_mp4_pcm, 1.5 s: %s", avatarforcing_last_error(ctx));
    check_mp4(out_pcm, have_ffprobe);

    /* ---- ffmpeg fails: its own message is in the error ---- */
    const char* out_bad = "test-capi-mp4-no-such-dir/out.mp4";
    CHECK(avatarforcing_generate_mp4(ctx, portrait, wav, out_bad, req) == -1, "mp4 to a missing directory succeeds");
    const char* e = avatarforcing_last_error(ctx);
    printf("ffmpeg failure: %s\n", e);
    CHECK(strstr(e, "ffmpeg failed (exit status "), "ffmpeg failure error without the exit status: %s", e);
    CHECK(strstr(e, "No such file or directory"), "ffmpeg failure error without ffmpeg's message: %s", e);

    free(pcm);
    avatarforcing_free(ctx);

    if (g_failures) {
        fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    printf("ok\n");
    return 0;
}
