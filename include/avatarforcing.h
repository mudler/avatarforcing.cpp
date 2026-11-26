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
 * whole primer (50 frames, 2 s) is always output, except in live mode.
 *
 * Audio. PCM is mono float in [-1, 1]. A sample rate other than 16000 is
 * resampled to 16 kHz.
 *
 * Threads. Contexts are independent: different contexts can run at the same
 * time in different threads. The calls on one context are serialized by a
 * lock in the context, so they are safe from any thread but run one at a
 * time; a context holds one model and runs one generation at a time.
 * avatarforcing_cancel and the session functions frames_ready,
 * pull_frames, interrupt and stats_json do not wait for that lock (see each
 * function).
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
 * input audio; the video stops at the end of the audio (-shortest). Audio
 * shorter than the 2 s primer is padded with silence to 2 s, so the mp4
 * has all the primer frames. Fails before any work when ffmpeg cannot be
 * started. When ffmpeg fails, the error holds its exit status and its error
 * output. */
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
 * A session opened with "live": true works differently: see "live mode"
 * below.
 *
 * A context has at most one open session. While it is open, the offline
 * generate functions on the context fail. */

/* Opens a session with a portrait (as in avatarforcing_generate_frames) and
 * request options. Returns NULL on error (avatarforcing_last_error(ctx)).
 * Besides the request options, request_json can hold the session options of
 * live mode (see below): "live", "live_queue_blocks" and
 * "live_max_ahead_seconds". A live session runs the hidden primer here. */
AVATARFORCING_API avatarforcing_session* avatarforcing_session_open(avatarforcing_ctx* ctx,
                                                                    const char* portrait_path,
                                                                    const char* request_json);

/* Adds one phrase of mono float PCM and computes the blocks it completes.
 * n_samples 0 does nothing. Fails after avatarforcing_session_flush and on
 * a live session.
 * Errors go to avatarforcing_last_error of the session's context. */
AVATARFORCING_API int avatarforcing_session_push_pcm(avatarforcing_session* s, const float* pcm, int n_samples,
                                                     int sample_rate);

/* Ends the audio: computes the remaining blocks and waits until all frames
 * are ready. Fails on a live session. */
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

/* ---- live mode ------------------------------------------------------------ */

/* A live session drives the face for as long as a call lasts, as the
 * upstream live app does. The library has no timer: the caller calls
 * avatarforcing_session_step once per block (block_frames = 10 frames,
 * 400 ms at 25 fps), a little before the block is due. Each step computes
 * exactly one block. Where a phrase covers a frame, the frame gets the
 * features of the phrase (lip-sync); elsewhere it gets the idle features
 * (wav2vec2 on the pink-noise room tone in the GGUF, read cyclically), so the
 * face moves slightly and the mouth stays closed.
 *
 * Session options (avatarforcing_session_open):
 *   "live"                    bool  true: open a live session
 *   "live_queue_blocks"       int   blocks that may wait for pull, 1 to 1000
 *                                   (default 3)
 *   "live_max_ahead_seconds"  int   a phrase must end at most this far after
 *                                   the next block (default 30)
 * In a live session "audio_mode" defaults to "raw", as in the live app.
 *
 * Frames. Frame indices are int64_t and count from 0, the first frame of the
 * first step. The 50 primer frames are not output: session_open runs the
 * primer on idle features and only writes the caches. Frames come out in
 * order, block_frames per step, with no gap.
 *
 * Phrases. avatarforcing_session_push_phrase runs wav2vec2 on the whole
 * phrase and returns the frame where it starts; play its audio from that
 * frame on (frame f plays at f / fps seconds after frame 0). A phrase can
 * start only at a frame that no step has computed yet, and phrases do not
 * overlap.
 *
 * Interrupt. avatarforcing_session_interrupt (barge-in) drops the phrase
 * audio from a frame on; from there the frames are idle. Blocks that are
 * already computed do not change; drop their frames in the caller for a
 * hard cut. There is no cross-fade, as in the live app: the model moves the
 * face from speech to idle within the next block.
 *
 * Backpressure. At most live_queue_blocks blocks of frames wait for pull
 * (ready, or still in the decoder thread). When the next block does not fit,
 * step computes nothing and returns AVATARFORCING_LIVE_QUEUE_FULL; the clock
 * of the session does not move. Pull frames, then step again. Nothing is
 * dropped.
 *
 * Threads. step and push_phrase take the context lock: they run one at a
 * time, from any thread. push_phrase computes wav2vec2, so a step waits for
 * a push that runs. interrupt, frames_ready, pull_frames and stats_json do
 * not take the context lock; call them from any thread at any time, but
 * not pull_frames from two threads at once. avatarforcing_cancel makes the
 * next step fail; the session stays failed. A push_phrase that has started
 * (also one that waits for the lock) when an interrupt comes returns
 * AVATARFORCING_LIVE_DROPPED.
 *
 * Not in live mode yet: the reanchor (a fresh primer every 112 blocks),
 * which the live app needs for calls of more than about a minute. */

/* step: the frame queue is full; nothing was computed. */
#define AVATARFORCING_LIVE_QUEUE_FULL (-2)
/* push_phrase: an interrupt came after the call started; the phrase is
 * dropped. */
#define AVATARFORCING_LIVE_DROPPED (-3)

/* Adds one phrase of mono float PCM (any sample rate). at_frame < 0: the
 * phrase starts right after the last phrase, or at the next block when no
 * phrase is pending. at_frame >= 0: the phrase starts at that frame, or
 * later when that frame is computed already or is inside an earlier phrase.
 * Returns the start frame, AVATARFORCING_LIVE_DROPPED, or -1 on error. An
 * error in the call (no samples, a phrase that ends after
 * live_max_ahead_seconds) leaves the session as it was; a failed wav2vec2
 * fails the session. */
AVATARFORCING_API int64_t avatarforcing_session_push_phrase(avatarforcing_session* s, const float* pcm,
                                                            int n_samples, int sample_rate, int64_t at_frame);

/* Computes the next block of block_frames frames. Returns the index of its
 * first frame, AVATARFORCING_LIVE_QUEUE_FULL, or -1 on error (the session
 * failed). With overlap the frames are ready later; without it they are
 * ready when step returns. */
AVATARFORCING_API int64_t avatarforcing_session_step(avatarforcing_session* s);

/* Barge-in: drops the phrase audio from frame at_frame on (at_frame < 0:
 * from the next block on). Frames that a step has computed already keep
 * their features, so the cut is at max(at_frame, next block). Returns the
 * first frame that is idle, or -1 when s is not a live session. Does not
 * wait for a running step. */
AVATARFORCING_API int64_t avatarforcing_session_interrupt(avatarforcing_session* s, int64_t at_frame);

/* A JSON object with the state of a live session:
 *   {"next_frame":40,"speech_end":115,"pending_phrases":1,
 *    "queued_frames":10,"ready_frames":10,"queue_frames":30,"blocks":4,
 *    "speech_blocks":1,"queue_full":0,"phrases":1,"dropped_phrases":0,
 *    "interrupts":0,"last_step_ms":350.2,"max_step_ms":410.0}
 * next_frame: the first frame of the next step. speech_end: the end of the
 * last phrase (next_frame when none is pending). queued_frames: frames that
 * wait for pull, ready or in the decoder. Free the string with
 * avatarforcing_free_string. NULL on error. */
AVATARFORCING_API char* avatarforcing_session_stats_json(avatarforcing_session* s);

#ifdef __cplusplus
}
#endif

#endif /* AVATARFORCING_H */
