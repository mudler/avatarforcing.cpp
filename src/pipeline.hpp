#pragma once
// The generation pipeline shared by avatarforcing-cli and the C API
// (include/avatarforcing.h): model loading, portrait and audio input, the
// noise source, the sampler and decoder loop with its optional decoder
// thread, and the ffmpeg mp4 writer.

#include "backend.hpp"
#include "flow.hpp"
#include "motion.hpp"
#include "sampler.hpp"
#include "wav2vec2.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <sys/types.h>

namespace af {

struct EngineOptions {
    std::string backend = "cpu";   // cpu, cuda, cuda:N, gpu
    int threads = 0;               // <= 0: all cores
    int decode_batch = -1;         // frames per decoder graph; < 0: 1 on the CPU, a block on a GPU; 0: a block
    int overlap = -1;              // decoder thread; < 0: on for a GPU backend
};

// A loaded model: the compute backend (two with overlap: one for the
// sampler, one for the decoder), the flow transformer, wav2vec2 and the
// motion autoencoder. One generation at a time may use an engine.
class Engine {
public:
    Engine() = default;
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    bool load(const std::string& gguf_path, const EngineOptions& opts, std::string& err);

    Backend& backend() { return *be_; }
    Backend& decoder_backend() { return be_dec_ ? *be_dec_ : *be_; }
    FlowModel& flow() { return *flow_; }
    AudioEncoder& audio() { return *audio_; }
    MotionModel& motion() { return *motion_; }

    bool overlap() const { return overlap_; }
    int n_threads() const { return n_threads_; }
    int decode_batch() const { return decode_batch_; }   // as set on the decoder; 0: a whole block
    int fps() const { return fps_; }
    int size() const { return size_; }                   // frame width and height
    int sample_rate() const { return sample_rate_; }     // wav2vec2 input rate (16000)
    size_t mem_free_before_load() const { return mem_free0_; }
    double load_ms() const { return load_ms_; }

private:
    std::unique_ptr<Backend> be_, be_dec_;
    std::unique_ptr<FlowModel> flow_;
    std::unique_ptr<AudioEncoder> audio_;
    std::unique_ptr<MotionModel> motion_;
    bool overlap_ = false;
    int n_threads_ = 1, decode_batch_ = 1, fps_ = 25, size_ = 0, sample_rate_ = 16000;
    size_t mem_free0_ = 0;
    double load_ms_ = 0;
};

struct GenOptions {
    int nfe = 0;               // <= 0: avatarforcing.sampler.default_nfe
    float cfg_scale = -1.f;    // < 0: avatarforcing.cfg.audio_scale
    uint64_t seed = 42;
    std::string noise_file;    // raw float32 noise instead of the seeded generator
    bool normalized = true;    // wav2vec2 input: normalized (offline path) or raw (live app)
    int max_seconds = 30;      // offline only: keep the first N s of audio; 0: no limit
    bool profile = false;      // per-op decoder profile (slow)
};

// Reads a portrait image (PNG, JPEG or BMP) that must be size x size, as
// [3, size, size] float32 in [-1, 1]: (u8 / 255 - 0.5) / 0.5.
bool load_portrait(const std::string& path, int size, std::vector<float>& image, std::string& err);

// Reads a WAV file as mono PCM at `sample_rate` (mixed down and resampled).
bool load_audio(const std::string& path, int sample_rate, std::vector<float>& pcm, std::string& err);

// Audio features of a whole clip, as the offline path computes them.
// pcm: mono at the wav2vec2 rate. normalized: zero mean and unit variance
// over the whole clip, then the first max_seconds s (0: all). feats: [T, E].
// capped: the clip was longer than max_seconds.
bool clip_features(Engine& e, const std::vector<float>& pcm, bool normalized, int max_seconds,
                   std::vector<float>& feats, int& T, bool& capped, std::string& err);

// Standard-normal noise: xoshiro256** seeded through splitmix64, Box-Muller
// in double precision (it does not reproduce torch.randn), or raw float32
// values read from a file.
class NoiseSource {
public:
    bool open_file(const std::string& path, std::string& err);
    void seed(uint64_t s);
    // A file must hold at least n values.
    bool require(size_t n, std::string& err) const;
    bool next(size_t n, std::vector<float>& out, std::string& err);

private:
    static uint64_t splitmix64(uint64_t& x);
    static uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    uint64_t next_u64();
    double uniform();
    double gauss();

    uint64_t st_[4] = {0, 0, 0, 0};
    bool have_spare_ = false;
    double spare_ = 0.0;
    bool from_file_ = false;
    std::string path_;
    std::vector<float> file_;
    size_t pos_ = 0;
};

// Receives finished uint8 frames, [n, H, W, 3]. With overlap it runs in the
// output thread. Returns false to stop the generation (set err). rgb is
// nullptr only for the held block of a reanchor (Generator::reanchor_block):
// n frames that repeat the last frame.
using FrameSink = std::function<bool(const uint8_t* rgb, int n, std::string& err)>;

struct GenStats {
    double ms_encode = 0, ms_sampler = 0, ms_primer = 0, ms_decode = 0, ms_out = 0;
    double ms_first_block = 0;            // primer start to the first block of frames
    std::vector<double> steady_ms;        // sampler time per steady block
    std::vector<double> reanchor_ms;      // sampler time per reanchor (live mode)
    std::vector<double> decode_ms;        // decoder time per full block
    std::vector<double> period_ms;        // time between finished steady blocks
    int frames = 0;                       // frames given to the sink
};

// One video: the streaming loop of the upstream offline path. A primer of
// clip_frames frames, then steady blocks of block_frames frames, each decoded
// and converted to uint8 per block. With overlap a decoder thread and an
// output thread (each fed by a queue of at most two blocks) run while the
// sampler computes the next block.
//
// The audio features arrive with add_features(); run() computes every block
// whose features are all present. The offline path adds all features, then
// calls finish(). A stream adds them phrase by phrase.
class Generator {
public:
    Generator(Engine& e, const GenOptions& o, FrameSink sink, const std::atomic<bool>* cancel = nullptr);
    ~Generator();
    Generator(const Generator&) = delete;
    Generator& operator=(const Generator&) = delete;

    // image: [3, S, S] in [-1, 1].
    bool encode_portrait(const float* image, std::string& err);
    // Sampler, noise and (with overlap) the decoder and output threads.
    bool init_sampler(std::string& err);
    // Fails when a noise file is too short for a clip of T audio frames.
    bool check_noise(int T, std::string& err) const;

    void add_features(const float* rows, int n);   // [n, E]
    int n_features() const { return n_rows_; }

    // Runs every block whose features are all present. final: the audio ends
    // here; blocks pad with the last feature row and the last block keeps
    // only the frames the audio covers.
    bool run(bool final, std::string& err);
    // run(true), then waits for the decoder and output threads.
    bool finish(std::string& err);

    // Live mode (src/live.hpp). The caller gives the features of each block
    // and does not use add_features, run or finish.
    // A primer that writes the caches and outputs no frames. wa [clip, E].
    bool primer_hidden(const float* wa, std::string& err);
    // One steady block of block_frames output frames. wa [context + block, E]:
    // the features of the context frames, then of the new frames.
    bool steady_block(const float* wa, std::string& err);
    // A reanchor in place of one steady block: a primer on wa [clip, E], the
    // features of the clip frames that end with the block, which resets the
    // caches. emit: the block is the last block_frames tokens of the primer,
    // decoded. Otherwise the block repeats the last frame (the sink receives
    // rgb == nullptr), as the live app does.
    bool reanchor_block(const float* wa, bool emit, std::string& err);

    // A decoder or output thread failed; err receives its message.
    bool failed(std::string* err = nullptr) const;

    const GenStats& stats() const { return stats_; }
    int nfe() const { return sampler_ ? sampler_->nfe() : 0; }
    float cfg_scale() const { return sampler_ ? sampler_->cfg_scale() : 0.f; }

    // Frames of a video for T audio frames: the whole primer, then one per
    // audio frame.
    static int frames_for(int T, int clip_frames) { return clip_frames + (T > clip_frames ? T - clip_frames : 0); }

private:
    struct Job { std::vector<float> data; int n = 0; };
    struct Stage {
        std::mutex mu;
        std::condition_variable cv;
        std::deque<Job> q;
        bool closing = false;
        bool push(Job&& j, const std::atomic<bool>& failed);
        bool pop(Job& j);
        void close();
    };

    void window(int lo, int n, std::vector<float>& w) const;
    bool decode(const float* latent, int n, std::vector<float>& px, std::string& err);
    bool output(const std::vector<float>& px, int n, std::string& err);
    bool submit(const float* latent, int n, std::string& err);
    bool submit_hold(int n, std::string& err);
    void fail(const std::string& msg);
    bool failed_error(std::string& err);
    void stop_threads();

    Engine& e_;
    GenOptions o_;
    FrameSink sink_;
    const std::atomic<bool>* cancel_;
    std::unique_ptr<Sampler> sampler_;
    Portrait portrait_;
    NoiseSource noise_;
    int S_ = 0, D_ = 0, E_ = 0, clip_ = 0, nb_ = 0, ctx_ = 0;
    std::vector<float> feats_;
    int n_rows_ = 0;
    bool primed_ = false, done_ = false;
    int t_ = 0;   // first frame of the next steady block
    int n_primer_blocks_ = 0, blocks_done_ = 0;
    std::vector<float> wa_, nz_, x_, latent_, pixels_;
    std::vector<uint8_t> u8_;
    GenStats stats_;
    std::chrono::steady_clock::time_point t_start_, t_last_done_;

    bool threads_ = false;
    Stage to_decode_, to_output_;
    std::atomic<bool> failed_{false};
    mutable std::mutex err_mu_;
    std::string thread_err_;
    std::thread dec_thread_, out_thread_;
};

// mp4 output: raw RGB frames go to ffmpeg on a pipe, encoded with libx264
// (yuv420p); the audio file is muxed as AAC with -shortest.
class VideoWriter {
public:
    ~VideoWriter();
    // ffmpeg: program name (looked up on PATH) or path. audio_path "": an
    // mp4 with video only. The audio is muxed with -shortest; min_frames > 0
    // pads audio that is shorter than min_frames / fps with silence, so
    // that ffmpeg reads at least that many frames.
    bool open(const std::string& ffmpeg, const std::string& out_path, const std::string& audio_path, int w, int h,
              int fps, std::string& err, int min_frames = 0);
    // [n, h, w, 3]. When ffmpeg stops reading and exits with status 0 (with
    // -shortest it stops at the end of the audio), the frames are dropped
    // and the call succeeds. When ffmpeg fails, err holds its exit status
    // and its error output.
    bool write(const uint8_t* frames, int n, std::string& err);
    // Closes the pipe and waits for ffmpeg; fails when ffmpeg failed.
    bool close(std::string& err);
    bool is_open() const { return fd_ >= 0 || done_; }

private:
    // Waits for ffmpeg and its error output. Returns false and sets err
    // when ffmpeg failed.
    bool wait_ffmpeg(std::string& err);

    int w_ = 0, h_ = 0;
    int fd_ = -1;
    pid_t pid_ = -1;
    bool done_ = false;          // ffmpeg exited with status 0 before the end of the frames
    std::thread stderr_thread_;  // reads the error output of ffmpeg into stderr_
    std::string stderr_;
};

// Copies the video of video_path and adds audio_path as AAC (-shortest).
bool mux_audio(const std::string& ffmpeg, const std::string& video_path, const std::string& audio_path,
               const std::string& out_path, std::string& err);

} // namespace af
