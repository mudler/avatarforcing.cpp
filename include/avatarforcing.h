#ifndef AVATARFORCING_H
#define AVATARFORCING_H

/*
 * avatarforcing.cpp C API.
 *
 * A flat C interface to libavatarforcing for dlopen-style loaders (for
 * example Go with purego, Python ctypes) and for C programs. It uses only
 * plain C types: opaque handles, pointers, int, float and NUL-terminated
 * UTF-8 strings. No function takes or returns a struct by value, and no
 * function takes a callback.
 *
 * Errors. Functions that return int return 0 on success and -1 on error.
 * Functions that return a pointer return NULL on error. After an error,
 * avatarforcing_last_error(ctx) gives the message. avatarforcing_load has
 * no context yet, so it returns its error through `error_out`.
 *
 * Memory. Strings and frame buffers that a function returns are allocated
 * with malloc. Release strings with avatarforcing_free_string and frame
 * buffers with avatarforcing_free_frames.
 *
 * Options. Options are flat JSON objects: string, number and boolean values,
 * no nesting. NULL or "" means no options. An unknown key is an error.
 *
 *   Context options (avatarforcing_load):
 *     "backend"      string  "cpu" (default), "cuda", "cuda:N" or "gpu"
 *     "threads"      int     CPU threads; 0 (default): all cores
 *     "decode_batch" int     frames per decoder graph; default 1 on the CPU,
 *                            a whole block (10) on a GPU; 0: a whole block
 *     "overlap"      bool    run the decoder in a second thread while the
 *                            sampler computes the next block; default true
 *                            on a GPU, false on the CPU
 *     "ffmpeg"       string  ffmpeg program for mp4 output (default
 *                            "ffmpeg", looked up on PATH)
 *     and any request option below, as the default of every request.
 *
 *   Request options (generate and session functions):
 *     "nfe"          int     sampler time points, >= 3; 0 (default): the
 *                            GGUF value (6). nfe - 1 denoising steps per block
 *     "cfg_scale"    number  audio classifier-free guidance scale; < 0
 *                            (default): the GGUF value (2.0)
 *     "seed"         int     seed of the built-in noise generator (default 42)
 *     "noise_file"   string  read the noise from a raw float32 file instead:
 *                            the primer [clip_frames, dim_w], then each
 *                            steady block [block_frames, dim_w]
 *     "audio_mode"   string  "normalized" (default; the upstream offline
 *                            path) or "raw" (the upstream live app)
 *     "max_seconds"  int     offline only: use the first N s of audio
 *                            (default 30); 0: no limit
 *
 * Video. Frames are 8-bit RGB, row-major [n_frames, height, width, 3], at
 * `fps` frames per second (the GGUF value, 25). The video has
 * max(clip_frames, T) frames for T = ceil(seconds * fps) audio frames: the
 * whole primer (50 frames, 2 s) is always output.
 *
 * Audio. PCM is mono float in [-1, 1]. A sample rate other than 16000 is
 * resampled to 16 kHz.
 *
 * Threads. Contexts are independent: different contexts can run at the same
 * time in different threads. The calls on one context are serialized by a
 * lock in the context, so they are safe from any thread but run one at a
 * time; a context holds one model and runs one generation at a time.
 * avatarforcing_cancel and the session functions frames_ready and
 * pull_frames do not wait for that lock (see each function).
 * avatarforcing_last_error returns a pointer into the context that the next
 * failing call on the context overwrites; copy it before the next call.
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#  if defined(AVATARFORCING_BUILD)
#    define AVATARFORCING_API __declspec(dllexport)
#  else
#    define AVATARFORCING_API __declspec(dllimport)
#  endif
#else
#  define AVATARFORCING_API __attribute__((visibility("default")))
#endif

/* ABI version of this header. A change to a signature or to the meaning of
 * a function increments it. v1: the first version. */
#define AVATARFORCING_ABI_VERSION 1

typedef struct avatarforcing_ctx avatarforcing_ctx;
typedef struct avatarforcing_session avatarforcing_session;

/* ---- context ------------------------------------------------------------ */

/* The ABI version of the library (AVATARFORCING_ABI_VERSION when it was
 * built). Compare it with the version the caller was written for. */
AVATARFORCING_API int avatarforcing_abi_version(void);

/* Loads a model GGUF. options_json: context options (see above).
 * Returns a context, or NULL on error. On error, when error_out is not NULL,
 * *error_out receives the message (free it with avatarforcing_free_string);
 * on success *error_out is set to NULL. */
AVATARFORCING_API avatarforcing_ctx* avatarforcing_load(const char* gguf_path, const char* options_json,
                                                        char** error_out);

/* Releases a context. NULL is allowed. An open session of the context is
 * closed and released too; do not use it afterwards. */
AVATARFORCING_API void avatarforcing_free(avatarforcing_ctx* ctx);

/* The message of the last error on the context, "" when there was none.
 * The pointer stays valid until the next failing call on the context or
 * avatarforcing_free. ctx NULL gives "". */
AVATARFORCING_API const char* avatarforcing_last_error(avatarforcing_ctx* ctx);

/* Releases a string from this library. NULL is allowed. */
AVATARFORCING_API void avatarforcing_free_string(char* s);

/* A JSON object that describes the loaded model and the context:
 *   {"abi":1,"width":512,"height":512,"fps":25,"sample_rate":16000,
 *    "clip_frames":50,"block_frames":10,"nfe":6,"cfg_scale":2,
 *    "device":"CPU","threads":16,"overlap":false,"decode_batch":1}
 * "nfe" and "cfg_scale" are the request defaults of the context. Free the
 * string with avatarforcing_free_string. NULL on error. */
AVATARFORCING_API char* avatarforcing_info_json(avatarforcing_ctx* ctx);

/* Asks the generation that runs on the context now to stop. It stops before
 * its next block and fails with the error "cancelled". Callable from any
 * thread at any time; it does not wait for the context lock. A generation
 * that starts after the call is not affected. On an open session, the
 * running or the next push or flush fails and the session stays failed;
 * close it. */
AVATARFORCING_API void avatarforcing_cancel(avatarforcing_ctx* ctx);

/* ---- offline generation -------------------------------------------------- */

/* Generates a video from a portrait image and speech, and returns its frames.
 *
 * portrait_path: a face-centred square image at the model size (512x512),
 * PNG, JPEG or BMP; no face detection is done. wav_path: a WAV file, any
 * sample rate and channel count (mixed down to mono). request_json: request
 * options.
 *
 * On success *frames_out receives a malloc'd buffer of
 * n_frames * height * width * 3 bytes (free it with
 * avatarforcing_free_frames), and the other out-parameters, when not NULL,
 * receive the frame count, the size and the frame rate. A 30 s clip at
 * 512x512 is 590 MB; use a session to receive the frames block by block. */
AVATARFORCING_API int avatarforcing_generate_frames(avatarforcing_ctx* ctx, const char* portrait_path,
                                                    const char* wav_path, const char* request_json,
                                                    uint8_t** frames_out, int* n_frames, int* width, int* height,
                                                    int* fps);

/* As avatarforcing_generate_frames, with the speech given as mono float PCM
 * (n_samples values at sample_rate Hz). */
AVATARFORCING_API int avatarforcing_generate_frames_pcm(avatarforcing_ctx* ctx, const char* portrait_path,
                                                        const float* pcm, int n_samples, int sample_rate,
                                                        const char* request_json, uint8_t** frames_out,
                                                        int* n_frames, int* width, int* height, int* fps);

/* Releases a frame buffer from this library. NULL is allowed. */
AVATARFORCING_API void avatarforcing_free_frames(uint8_t* frames);

/* Generates a video and writes it as an mp4 file (H.264 video, AAC audio)
 * with ffmpeg, as avatarforcing-cli -o does. The audio of the mp4 is the
 * input audio; the video stops at the end of the audio (-shortest). Fails
 * before any work when ffmpeg cannot be started. */
AVATARFORCING_API int avatarforcing_generate_mp4(avatarforcing_ctx* ctx, const char* portrait_path,
                                                 const char* wav_path, const char* output_path,
                                                 const char* request_json);

/* As avatarforcing_generate_mp4, with the speech given as mono float PCM.
 * The PCM is written to a temporary float WAV file next to output_path for
 * ffmpeg, and the file is removed afterwards. */
AVATARFORCING_API int avatarforcing_generate_mp4_pcm(avatarforcing_ctx* ctx, const char* portrait_path,
                                                     const float* pcm, int n_samples, int sample_rate,
                                                     const char* output_path, const char* request_json);

/* ---- streaming session ---------------------------------------------------- */

/* A session generates one video while its audio arrives in parts. Each
 * avatarforcing_session_push_pcm call gives one phrase. wav2vec2 runs on each
 * phrase on its own (normalized per phrase in "normalized" mode), so a phrase
 * should be a whole utterance, for example one TTS sentence. The session
 * computes every block whose audio is present: the primer (50 frames) once
 * 2 s of audio are present, then one block of 10 frames per 0.4 s of audio.
 * avatarforcing_session_flush ends the audio and computes the rest; the last
 * block keeps only the frames the audio covers.
 *
 * One phrase of the whole clip, then flush, gives the same frames as
 * avatarforcing_generate_frames_pcm when the clip is not longer than
 * max_seconds ("max_seconds" does not apply to a session).
 *
 * With overlap, the decoder works in its own thread: push returns when the
 * sampler has computed the ready blocks, and their frames become ready
 * later. Without overlap, push returns when the frames are ready.
 *
 * Not implemented yet (the real-time milestone): idle frames when no audio
 * arrives (the upstream live app feeds pink-noise idle features), the
 * periodic re-run of the primer (reanchor) for unlimited sessions, and a
 * hidden primer. A session now follows the audio only.
 *
 * A context has at most one open session. While it is open, the offline
 * generate functions on the context fail. */

/* Opens a session with a portrait (as in avatarforcing_generate_frames) and
 * request options. Returns NULL on error (avatarforcing_last_error(ctx)). */
AVATARFORCING_API avatarforcing_session* avatarforcing_session_open(avatarforcing_ctx* ctx,
                                                                    const char* portrait_path,
                                                                    const char* request_json);

/* Adds one phrase of mono float PCM and computes the blocks it completes.
 * n_samples 0 does nothing. Fails after avatarforcing_session_flush.
 * Errors go to avatarforcing_last_error of the session's context. */
AVATARFORCING_API int avatarforcing_session_push_pcm(avatarforcing_session* s, const float* pcm, int n_samples,
                                                     int sample_rate);

/* Ends the audio: computes the remaining blocks and waits until all frames
 * are ready. */
AVATARFORCING_API int avatarforcing_session_flush(avatarforcing_session* s);

/* The number of frames that are ready to pull. Callable from another thread
 * while push or flush run. -1 when s is NULL. */
AVATARFORCING_API int avatarforcing_session_frames_ready(avatarforcing_session* s);

/* Copies up to max_frames ready frames, in order, to dst (max_frames *
 * height * width * 3 bytes) and removes them from the session. Returns the
 * number of frames copied, 0 when no frame is ready. Returns -1 when an
 * argument is invalid, or when no frame is ready and the session failed
 * (see avatarforcing_last_error): frames that were ready before a failure
 * can still be pulled. Callable from another thread while push or flush
 * run, but not concurrently with another pull. */
AVATARFORCING_API int avatarforcing_session_pull_frames(avatarforcing_session* s, uint8_t* dst, int max_frames);

/* Closes the session and releases it. NULL is allowed. Frames that were not
 * pulled are dropped. Do not call it while another thread uses the
 * session. */
AVATARFORCING_API void avatarforcing_session_close(avatarforcing_session* s);

#ifdef __cplusplus
}
#endif

#endif /* AVATARFORCING_H */
