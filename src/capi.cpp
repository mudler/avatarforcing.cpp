// The C API (include/avatarforcing.h) on top of the pipeline in pipeline.hpp.
// No C++ exception crosses the API: every entry point catches them and
// reports an error.

#include "avatarforcing.h"

#include "audio_io.hpp"
#include "common.hpp"
#include "pipeline.hpp"

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

struct avatarforcing_ctx {
    std::mutex mu;   // one call at a time on the engine
    af::Engine engine;
    af::GenOptions defaults;
    std::string ffmpeg = "ffmpeg";
    std::mutex err_mu;
    std::string last_error;
    std::atomic<bool> cancel{false};
    avatarforcing_session* session = nullptr;   // guarded by mu
};

struct avatarforcing_session {
    avatarforcing_ctx* ctx = nullptr;
    std::unique_ptr<af::Generator> gen;
    size_t frame_bytes = 0;
    std::mutex qmu;
    std::deque<std::vector<uint8_t>> ready;   // guarded by qmu
    std::atomic<bool> failed{false};
    bool flushed = false;
    bool normalized = true;
};

namespace {

void set_error(avatarforcing_ctx* ctx, const std::string& msg) {
    if (!ctx) return;
    std::lock_guard<std::mutex> lk(ctx->err_mu);
    ctx->last_error = msg;
}

char* dup_string(const std::string& s) {
    char* p = (char*)std::malloc(s.size() + 1);
    if (p) std::memcpy(p, s.c_str(), s.size() + 1);
    return p;
}

// ---------------------------------------------------------------------------
// flat JSON options

struct JsonValue {
    enum Kind { String, Number, Bool, Null } kind = Null;
    std::string text;   // the string, or the number as written
    bool b = false;
};
using JsonObject = std::vector<std::pair<std::string, JsonValue>>;

class JsonParser {
public:
    explicit JsonParser(const char* s) : p_(s) {}

    bool parse(JsonObject& out, std::string& err) {
        ws();
        if (*p_ != '{') return fail("options must be a JSON object", err);
        ++p_;
        ws();
        if (*p_ == '}') { ++p_; return end(err); }
        for (;;) {
            ws();
            std::string key;
            if (*p_ != '"' || !string(key, err)) return err.empty() ? fail("expected a key string", err) : false;
            ws();
            if (*p_ != ':') return fail("expected ':' after \"" + key + "\"", err);
            ++p_;
            ws();
            JsonValue v;
            if (*p_ == '"') {
                v.kind = JsonValue::String;
                if (!string(v.text, err)) return false;
            } else if (*p_ == '{' || *p_ == '[') {
                return fail("option \"" + key + "\": nested values are not supported", err);
            } else if (word("true")) {
                v.kind = JsonValue::Bool;
                v.b = true;
            } else if (word("false")) {
                v.kind = JsonValue::Bool;
            } else if (word("null")) {
                v.kind = JsonValue::Null;
            } else {
                const char* q = p_;
                while (*p_ && std::strchr("+-.0123456789eE", *p_)) ++p_;
                if (q == p_) return fail("option \"" + key + "\": invalid value", err);
                v.kind = JsonValue::Number;
                v.text.assign(q, p_);
            }
            for (auto& kv : out)
                if (kv.first == key) return fail("option \"" + key + "\" appears twice", err);
            out.emplace_back(key, v);
            ws();
            if (*p_ == ',') { ++p_; continue; }
            if (*p_ == '}') { ++p_; return end(err); }
            return fail("expected ',' or '}' in the options", err);
        }
    }

private:
    void ws() { while (*p_ == ' ' || *p_ == '\t' || *p_ == '\n' || *p_ == '\r') ++p_; }
    bool fail(const std::string& m, std::string& err) { err = m; return false; }
    bool end(std::string& err) {
        ws();
        return *p_ ? fail("unexpected text after the options object", err) : true;
    }
    bool word(const char* w) {
        const size_t n = std::strlen(w);
        if (std::strncmp(p_, w, n) != 0) return false;
        p_ += n;
        return true;
    }
    static void put_utf8(std::string& s, unsigned cp) {
        if (cp < 0x80) {
            s += (char)cp;
        } else if (cp < 0x800) {
            s += (char)(0xC0 | (cp >> 6));
            s += (char)(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            s += (char)(0xE0 | (cp >> 12));
            s += (char)(0x80 | ((cp >> 6) & 0x3F));
            s += (char)(0x80 | (cp & 0x3F));
        } else {
            s += (char)(0xF0 | (cp >> 18));
            s += (char)(0x80 | ((cp >> 12) & 0x3F));
            s += (char)(0x80 | ((cp >> 6) & 0x3F));
            s += (char)(0x80 | (cp & 0x3F));
        }
    }
    bool hex4(unsigned& v) {
        v = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = p_[i];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= (unsigned)(c - '0');
            else if (c >= 'a' && c <= 'f') v |= (unsigned)(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= (unsigned)(c - 'A' + 10);
            else return false;
        }
        p_ += 4;
        return true;
    }
    bool string(std::string& s, std::string& err) {
        ++p_;   // opening quote
        for (;;) {
            const char c = *p_;
            if (c == '\0') return fail("unterminated string in the options", err);
            ++p_;
            if (c == '"') return true;
            if ((unsigned char)c < 0x20) return fail("control character in an options string", err);
            if (c != '\\') { s += c; continue; }
            const char e = *p_++;
            switch (e) {
                case '"': s += '"'; break;
                case '\\': s += '\\'; break;
                case '/': s += '/'; break;
                case 'b': s += '\b'; break;
                case 'f': s += '\f'; break;
                case 'n': s += '\n'; break;
                case 'r': s += '\r'; break;
                case 't': s += '\t'; break;
                case 'u': {
                    unsigned cp = 0;
                    if (!hex4(cp)) return fail("invalid \\u escape in the options", err);
                    if (cp >= 0xD800 && cp < 0xDC00 && p_[0] == '\\' && p_[1] == 'u') {
                        p_ += 2;
                        unsigned lo = 0;
                        if (!hex4(lo) || lo < 0xDC00 || lo >= 0xE000) return fail("invalid surrogate pair in the options", err);
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    put_utf8(s, cp);
                    break;
                }
                default: return fail("invalid escape in the options", err);
            }
        }
    }

    const char* p_;
};

bool parse_options(const char* json, JsonObject& out, std::string& err) {
    out.clear();
    if (!json) return true;
    const char* p = json;
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') ++p;
    if (!*p) return true;
    JsonParser parser(json);
    if (!parser.parse(out, err)) { err = "invalid options: " + err; return false; }
    return true;
}

bool as_int(const std::string& key, const JsonValue& v, long long& out, std::string& err) {
    if (v.kind == JsonValue::Number) {
        char* end = nullptr;
        errno = 0;
        const long long x = std::strtoll(v.text.c_str(), &end, 10);
        if (errno == 0 && end && *end == '\0') { out = x; return true; }
    }
    err = "option \"" + key + "\" must be an integer";
    return false;
}

bool as_float(const std::string& key, const JsonValue& v, double& out, std::string& err) {
    if (v.kind == JsonValue::Number) {
        char* end = nullptr;
        const double x = std::strtod(v.text.c_str(), &end);
        if (end && *end == '\0') { out = x; return true; }
    }
    err = "option \"" + key + "\" must be a number";
    return false;
}

bool as_bool(const std::string& key, const JsonValue& v, bool& out, std::string& err) {
    if (v.kind == JsonValue::Bool) { out = v.b; return true; }
    err = "option \"" + key + "\" must be true or false";
    return false;
}

bool as_string(const std::string& key, const JsonValue& v, std::string& out, std::string& err) {
    if (v.kind == JsonValue::String) { out = v.text; return true; }
    err = "option \"" + key + "\" must be a string";
    return false;
}

// Applies one request option. handled: the key is a request option.
bool apply_request_option(const std::string& k, const JsonValue& v, af::GenOptions& g, bool& handled,
                          std::string& err) {
    handled = true;
    long long i = 0;
    double f = 0;
    std::string s;
    if (k == "nfe") {
        if (!as_int(k, v, i, err)) return false;
        if (i < 0 || i == 1 || i == 2 || i > 1000) { err = "option \"nfe\" must be 0 (model default) or 3 to 1000"; return false; }
        g.nfe = (int)i;
    } else if (k == "cfg_scale") {
        if (!as_float(k, v, f, err)) return false;
        g.cfg_scale = (float)f;
    } else if (k == "seed") {
        if (v.kind != JsonValue::Number || v.text.empty() || v.text[0] == '-') {
            err = "option \"seed\" must be a non-negative integer";
            return false;
        }
        char* end = nullptr;
        errno = 0;
        const unsigned long long x = std::strtoull(v.text.c_str(), &end, 10);
        if (errno != 0 || !end || *end != '\0') { err = "option \"seed\" must be a non-negative integer"; return false; }
        g.seed = (uint64_t)x;
    } else if (k == "noise_file") {
        if (!as_string(k, v, g.noise_file, err)) return false;
    } else if (k == "audio_mode") {
        if (!as_string(k, v, s, err)) return false;
        if (s == "normalized") g.normalized = true;
        else if (s == "raw") g.normalized = false;
        else { err = "option \"audio_mode\" must be \"normalized\" or \"raw\""; return false; }
    } else if (k == "max_seconds") {
        if (!as_int(k, v, i, err)) return false;
        if (i < 0 || i > 86400) { err = "option \"max_seconds\" must be 0 to 86400"; return false; }
        g.max_seconds = (int)i;
    } else {
        handled = false;
    }
    return true;
}

bool request_options(avatarforcing_ctx* ctx, const char* json, af::GenOptions& g, std::string& err) {
    g = ctx->defaults;
    JsonObject obj;
    if (!parse_options(json, obj, err)) return false;
    for (auto& kv : obj) {
        bool handled = false;
        if (!apply_request_option(kv.first, kv.second, g, handled, err)) return false;
        if (!handled) { err = "unknown request option \"" + kv.first + "\""; return false; }
    }
    return true;
}

std::string json_escape(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if ((unsigned char)c < 0x20) { char b[8]; std::snprintf(b, sizeof(b), "\\u%04x", c); o += b; }
        else o += c;
    }
    return o;
}

// ---------------------------------------------------------------------------
// offline generation

struct AudioIn {
    const char* path = nullptr;   // WAV file, or
    const float* pcm = nullptr;   // mono PCM
    int n = 0, sr = 0;
};

bool read_audio(avatarforcing_ctx* ctx, const AudioIn& in, std::vector<float>& pcm, std::string& err) {
    const int sr = ctx->engine.sample_rate();
    if (in.path) return af::load_audio(in.path, sr, pcm, err);
    if (!in.pcm || in.n <= 0) { err = "no audio samples"; return false; }
    if (in.sr <= 0) { err = "invalid sample rate " + std::to_string(in.sr); return false; }
    pcm = af::resample(std::vector<float>(in.pcm, in.pcm + in.n), in.sr, sr);
    if (pcm.empty()) { err = "no audio samples"; return false; }
    return true;
}

// Removes a file when it goes out of scope.
struct TempFile {
    std::string path;
    ~TempFile() { if (!path.empty()) ::unlink(path.c_str()); }
};

// One offline video. mp4_path set: the frames go to ffmpeg, else to a
// malloc'd buffer in *frames_out.
int run_offline(avatarforcing_ctx* ctx, const char* portrait_path, const AudioIn& audio_in, const char* mp4_path,
                const char* request_json, uint8_t** frames_out, int* n_frames, int* width, int* height, int* fps) {
    if (!ctx) return -1;
    std::lock_guard<std::mutex> lk(ctx->mu);
    ctx->cancel = false;
    af::clear_last_log();
    std::string err;
    auto failed = [&](const std::string& m) { set_error(ctx, m); return -1; };
    if (ctx->session) return failed("a session is open on this context");
    if (!portrait_path) return failed("no portrait path");
    if (!mp4_path && !frames_out) return failed("frames_out is NULL");
    if (frames_out) *frames_out = nullptr;

    af::GenOptions go;
    if (!request_options(ctx, request_json, go, err)) return failed(err);
    af::Engine& e = ctx->engine;
    const int S = e.size();
    const int clip = (int)e.flow().config().clip_frames;

    std::vector<float> image;
    if (!af::load_portrait(portrait_path, S, image, err)) return failed(err);
    std::vector<float> pcm;
    if (!read_audio(ctx, audio_in, pcm, err)) return failed(err);

    af::VideoWriter video;
    TempFile tmp_wav;
    if (mp4_path) {
        std::string mux = audio_in.path ? audio_in.path : "";
        if (!audio_in.path) {
            // ffmpeg muxes the audio from a file: the PCM as given, as float.
            std::string tmpl = std::string(mp4_path) + ".audio-XXXXXX.wav";
            std::vector<char> buf(tmpl.begin(), tmpl.end());
            buf.push_back('\0');
            const int fd = mkstemps(buf.data(), 4);
            if (fd < 0) return failed("cannot create a temporary file next to " + std::string(mp4_path) + ": " + std::strerror(errno));
            ::close(fd);
            tmp_wav.path = buf.data();
            if (!af::write_wav_f32(tmp_wav.path, audio_in.pcm, (size_t)audio_in.n, audio_in.sr, err)) return failed(err);
            mux = tmp_wav.path;
        }
        if (!video.open(ctx->ffmpeg, mp4_path, mux, S, S, e.fps(), err)) return failed(err);
    }

    uint8_t* buf = nullptr;
    size_t cap_frames = 0, written = 0;
    const size_t fb = (size_t)S * S * 3;
    auto sink = [&](const uint8_t* rgb, int n, std::string& serr) -> bool {
        if (video.is_open()) return video.write(rgb, n, serr);
        if (written + (size_t)n > cap_frames) { serr = "more frames than expected"; return false; }
        std::memcpy(buf + written * fb, rgb, (size_t)n * fb);
        written += (size_t)n;
        return true;
    };
    af::Generator gen(e, go, sink, &ctx->cancel);
    if (!gen.encode_portrait(image.data(), err)) return failed(err);
    std::vector<float> feats;
    int T = 0;
    bool capped = false;
    if (!af::clip_features(e, pcm, go.normalized, go.max_seconds, feats, T, capped, err)) return failed(err);
    if (!gen.init_sampler(err) || !gen.check_noise(T, err)) return failed(err);
    const int n_out = af::Generator::frames_for(T, clip);
    if (!mp4_path) {
        cap_frames = (size_t)n_out;
        buf = (uint8_t*)std::malloc(cap_frames * fb);
        if (!buf) return failed("out of memory for " + std::to_string(n_out) + " frames");
    }
    gen.add_features(feats.data(), T);
    const bool ok = gen.finish(err);
    if (!ok) {
        std::free(buf);
        std::string cerr;
        video.close(cerr);
        return failed(err);
    }
    if (mp4_path && !video.close(err)) return failed(err);
    if (!mp4_path && written != cap_frames) {
        std::free(buf);
        return failed("expected " + std::to_string(cap_frames) + " frames, got " + std::to_string(written));
    }
    if (frames_out) *frames_out = buf;
    if (n_frames) *n_frames = gen.stats().frames;
    if (width) *width = S;
    if (height) *height = S;
    if (fps) *fps = e.fps();
    return 0;
}

template <typename F>
int guarded(avatarforcing_ctx* ctx, F&& f) {
    try {
        return f();
    } catch (const std::bad_alloc&) {
        set_error(ctx, "out of memory");
    } catch (const std::exception& ex) {
        set_error(ctx, std::string("internal error: ") + ex.what());
    } catch (...) {
        set_error(ctx, "internal error");
    }
    return -1;
}

} // namespace

// ---------------------------------------------------------------------------
// context

extern "C" {

int avatarforcing_abi_version(void) { return AVATARFORCING_ABI_VERSION; }

avatarforcing_ctx* avatarforcing_load(const char* gguf_path, const char* options_json, char** error_out) {
    if (error_out) *error_out = nullptr;
    std::string err;
    avatarforcing_ctx* ctx = nullptr;
    try {
        if (!gguf_path) {
            err = "no model path";
        } else {
            JsonObject obj;
            af::EngineOptions eo;
            af::GenOptions defaults;
            std::string ffmpeg = "ffmpeg";
            bool ok = parse_options(options_json, obj, err);
            for (size_t i = 0; ok && i < obj.size(); ++i) {
                const std::string& k = obj[i].first;
                const JsonValue& v = obj[i].second;
                long long n = 0;
                bool handled = false;
                if (k == "backend") {
                    ok = as_string(k, v, eo.backend, err);
                } else if (k == "threads") {
                    ok = as_int(k, v, n, err);
                    if (ok && (n < 0 || n > 4096)) { err = "option \"threads\" must be 0 to 4096"; ok = false; }
                    eo.threads = (int)n;
                } else if (k == "decode_batch") {
                    ok = as_int(k, v, n, err);
                    if (ok && (n < 0 || n > 1000)) { err = "option \"decode_batch\" must be 0 to 1000"; ok = false; }
                    eo.decode_batch = (int)n;
                } else if (k == "overlap") {
                    bool b = false;
                    ok = as_bool(k, v, b, err);
                    eo.overlap = b ? 1 : 0;
                } else if (k == "ffmpeg") {
                    ok = as_string(k, v, ffmpeg, err);
                    if (ok && ffmpeg.empty()) { err = "option \"ffmpeg\" is empty"; ok = false; }
                } else {
                    ok = apply_request_option(k, v, defaults, handled, err);
                    if (ok && !handled) { err = "unknown option \"" + k + "\""; ok = false; }
                }
            }
            if (ok) {
                std::unique_ptr<avatarforcing_ctx> c(new avatarforcing_ctx());
                c->defaults = defaults;
                c->ffmpeg = ffmpeg;
                if (c->engine.load(gguf_path, eo, err)) ctx = c.release();
            }
        }
    } catch (const std::bad_alloc&) {
        err = "out of memory";
    } catch (const std::exception& ex) {
        err = std::string("internal error: ") + ex.what();
    } catch (...) {
        err = "internal error";
    }
    if (!ctx && error_out) *error_out = dup_string(err.empty() ? "load failed" : err);
    return ctx;
}

void avatarforcing_free(avatarforcing_ctx* ctx) {
    if (!ctx) return;
    avatarforcing_session* s = nullptr;
    {
        std::lock_guard<std::mutex> lk(ctx->mu);
        s = ctx->session;
        if (s) s->gen.reset();
        ctx->session = nullptr;
    }
    delete s;
    delete ctx;
}

const char* avatarforcing_last_error(avatarforcing_ctx* ctx) {
    if (!ctx) return "";
    std::lock_guard<std::mutex> lk(ctx->err_mu);
    return ctx->last_error.c_str();
}

void avatarforcing_free_string(char* s) { std::free(s); }

char* avatarforcing_info_json(avatarforcing_ctx* ctx) {
    if (!ctx) return nullptr;
    char* out = nullptr;
    guarded(ctx, [&] {
        std::lock_guard<std::mutex> lk(ctx->mu);
        af::Engine& e = ctx->engine;
        const af::FlowConfig& fc = e.flow().config();
        const int nfe = ctx->defaults.nfe > 0 ? ctx->defaults.nfe : (int)fc.default_nfe;
        const float cfg = ctx->defaults.cfg_scale >= 0 ? ctx->defaults.cfg_scale : fc.cfg_audio_scale;
        char buf[1024];
        std::snprintf(buf, sizeof(buf),
                      "{\"abi\":%d,\"width\":%d,\"height\":%d,\"fps\":%d,\"sample_rate\":%d,"
                      "\"clip_frames\":%u,\"block_frames\":%u,\"nfe\":%d,\"cfg_scale\":%.6g,"
                      "\"device\":\"%s\",\"threads\":%d,\"overlap\":%s,\"decode_batch\":%d}",
                      AVATARFORCING_ABI_VERSION, e.size(), e.size(), e.fps(), e.sample_rate(), fc.clip_frames,
                      fc.block_frames, nfe, (double)cfg, json_escape(e.backend().device()).c_str(), e.n_threads(),
                      e.overlap() ? "true" : "false", e.decode_batch());
        out = dup_string(buf);
        return out ? 0 : -1;
    });
    return out;
}

void avatarforcing_cancel(avatarforcing_ctx* ctx) {
    if (ctx) ctx->cancel = true;
}

// ---------------------------------------------------------------------------
// offline generation

int avatarforcing_generate_frames(avatarforcing_ctx* ctx, const char* portrait_path, const char* wav_path,
                                  const char* request_json, uint8_t** frames_out, int* n_frames, int* width,
                                  int* height, int* fps) {
    return guarded(ctx, [&] {
        if (!wav_path) { set_error(ctx, "no audio path"); return -1; }
        AudioIn in;
        in.path = wav_path;
        return run_offline(ctx, portrait_path, in, nullptr, request_json, frames_out, n_frames, width, height, fps);
    });
}

int avatarforcing_generate_frames_pcm(avatarforcing_ctx* ctx, const char* portrait_path, const float* pcm,
                                      int n_samples, int sample_rate, const char* request_json,
                                      uint8_t** frames_out, int* n_frames, int* width, int* height, int* fps) {
    return guarded(ctx, [&] {
        AudioIn in;
        in.pcm = pcm;
        in.n = n_samples;
        in.sr = sample_rate;
        return run_offline(ctx, portrait_path, in, nullptr, request_json, frames_out, n_frames, width, height, fps);
    });
}

void avatarforcing_free_frames(uint8_t* frames) { std::free(frames); }

int avatarforcing_generate_mp4(avatarforcing_ctx* ctx, const char* portrait_path, const char* wav_path,
                               const char* output_path, const char* request_json) {
    return guarded(ctx, [&] {
        if (!wav_path) { set_error(ctx, "no audio path"); return -1; }
        if (!output_path) { set_error(ctx, "no output path"); return -1; }
        AudioIn in;
        in.path = wav_path;
        return run_offline(ctx, portrait_path, in, output_path, request_json, nullptr, nullptr, nullptr, nullptr,
                           nullptr);
    });
}

int avatarforcing_generate_mp4_pcm(avatarforcing_ctx* ctx, const char* portrait_path, const float* pcm,
                                   int n_samples, int sample_rate, const char* output_path,
                                   const char* request_json) {
    return guarded(ctx, [&] {
        if (!output_path) { set_error(ctx, "no output path"); return -1; }
        AudioIn in;
        in.pcm = pcm;
        in.n = n_samples;
        in.sr = sample_rate;
        return run_offline(ctx, portrait_path, in, output_path, request_json, nullptr, nullptr, nullptr, nullptr,
                           nullptr);
    });
}

// ---------------------------------------------------------------------------
// streaming session

avatarforcing_session* avatarforcing_session_open(avatarforcing_ctx* ctx, const char* portrait_path,
                                                  const char* request_json) {
    if (!ctx) return nullptr;
    avatarforcing_session* out = nullptr;
    guarded(ctx, [&] {
        std::lock_guard<std::mutex> lk(ctx->mu);
        ctx->cancel = false;
        af::clear_last_log();
        std::string err;
        if (ctx->session) { set_error(ctx, "a session is already open on this context"); return -1; }
        if (!portrait_path) { set_error(ctx, "no portrait path"); return -1; }
        af::GenOptions go;
        if (!request_options(ctx, request_json, go, err)) { set_error(ctx, err); return -1; }
        af::Engine& e = ctx->engine;
        std::vector<float> image;
        if (!af::load_portrait(portrait_path, e.size(), image, err)) { set_error(ctx, err); return -1; }
        std::unique_ptr<avatarforcing_session> s(new avatarforcing_session());
        s->ctx = ctx;
        s->frame_bytes = (size_t)e.size() * e.size() * 3;
        s->normalized = go.normalized;
        avatarforcing_session* sp = s.get();
        auto sink = [sp](const uint8_t* rgb, int n, std::string&) -> bool {
            std::lock_guard<std::mutex> qlk(sp->qmu);
            for (int f = 0; f < n; ++f)
                sp->ready.emplace_back(rgb + (size_t)f * sp->frame_bytes, rgb + (size_t)(f + 1) * sp->frame_bytes);
            return true;
        };
        s->gen.reset(new af::Generator(e, go, sink, &ctx->cancel));
        if (!s->gen->encode_portrait(image.data(), err) || !s->gen->init_sampler(err)) {
            s->gen.reset();
            set_error(ctx, err);
            return -1;
        }
        ctx->session = s.get();
        out = s.release();
        return 0;
    });
    return out;
}

int avatarforcing_session_push_pcm(avatarforcing_session* s, const float* pcm, int n_samples, int sample_rate) {
    if (!s) return -1;
    avatarforcing_ctx* ctx = s->ctx;
    return guarded(ctx, [&] {
        std::lock_guard<std::mutex> lk(ctx->mu);
        af::clear_last_log();
        if (!s->gen) { set_error(ctx, "the session is closed"); return -1; }
        if (s->failed) { set_error(ctx, "the session failed earlier"); return -1; }
        if (s->flushed) { set_error(ctx, "the session audio has ended (flush was called)"); return -1; }
        if (n_samples < 0 || (n_samples > 0 && !pcm)) { set_error(ctx, "invalid PCM buffer"); return -1; }
        if (n_samples == 0) return 0;
        if (sample_rate <= 0) { set_error(ctx, "invalid sample rate " + std::to_string(sample_rate)); return -1; }
        af::Engine& e = ctx->engine;
        std::vector<float> x(pcm, pcm + n_samples);
        if (sample_rate != e.sample_rate()) x = af::resample(x, sample_rate, e.sample_rate());
        std::string err;
        std::vector<float> feats;
        int T = 0;
        // One phrase; a session has no max_seconds cap.
        bool capped = false;
        if (!af::clip_features(e, x, s->normalized, 0, feats, T, capped, err)) {
            s->failed = true;
            set_error(ctx, err);
            return -1;
        }
        s->gen->add_features(feats.data(), T);
        if (!s->gen->run(false, err)) {
            s->failed = true;
            set_error(ctx, err);
            return -1;
        }
        return 0;
    });
}

int avatarforcing_session_flush(avatarforcing_session* s) {
    if (!s) return -1;
    avatarforcing_ctx* ctx = s->ctx;
    return guarded(ctx, [&] {
        std::lock_guard<std::mutex> lk(ctx->mu);
        af::clear_last_log();
        if (!s->gen) { set_error(ctx, "the session is closed"); return -1; }
        if (s->failed) { set_error(ctx, "the session failed earlier"); return -1; }
        if (s->flushed) return 0;
        std::string err;
        s->flushed = true;
        if (!s->gen->finish(err)) {
            s->failed = true;
            set_error(ctx, err);
            return -1;
        }
        return 0;
    });
}

int avatarforcing_session_frames_ready(avatarforcing_session* s) {
    if (!s) return -1;
    std::lock_guard<std::mutex> qlk(s->qmu);
    return (int)s->ready.size();
}

int avatarforcing_session_pull_frames(avatarforcing_session* s, uint8_t* dst, int max_frames) {
    if (!s) return -1;
    avatarforcing_ctx* ctx = s->ctx;
    if (!dst || max_frames < 0) { set_error(ctx, "invalid frame buffer"); return -1; }
    std::deque<std::vector<uint8_t>> got;
    {
        std::lock_guard<std::mutex> qlk(s->qmu);
        while ((int)got.size() < max_frames && !s->ready.empty()) {
            got.push_back(std::move(s->ready.front()));
            s->ready.pop_front();
        }
    }
    for (size_t i = 0; i < got.size(); ++i) std::memcpy(dst + i * s->frame_bytes, got[i].data(), s->frame_bytes);
    if (got.empty()) {
        std::string err;
        // A decoder or output thread failure shows here before the next push.
        af::Generator* g = s->gen.get();
        if (s->failed || (g && g->failed(&err))) {
            if (!err.empty()) set_error(ctx, err);
            return -1;
        }
    }
    return (int)got.size();
}

void avatarforcing_session_close(avatarforcing_session* s) {
    if (!s) return;
    avatarforcing_ctx* ctx = s->ctx;
    {
        std::lock_guard<std::mutex> lk(ctx->mu);
        try {
            s->gen.reset();   // stops the decoder and output threads
        } catch (...) {
        }
        if (ctx->session == s) ctx->session = nullptr;
    }
    delete s;
}

} // extern "C"
