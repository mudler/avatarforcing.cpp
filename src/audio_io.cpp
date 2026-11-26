#include "audio_io.hpp"

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#include <algorithm>
#include <cmath>

namespace af {

bool read_wav_mono(const std::string& path, Wav& out, std::string& err) {
    drwav wav;
    if (!drwav_init_file(&wav, path.c_str(), nullptr)) {
        err = "cannot open or parse WAV file: " + path;
        return false;
    }
    const int ch = (int)wav.channels;
    const drwav_uint64 n = wav.totalPCMFrameCount;
    std::vector<float> inter((size_t)n * ch);
    const drwav_uint64 got = drwav_read_pcm_frames_f32(&wav, n, inter.data());
    out.sample_rate = (int)wav.sampleRate;
    out.channels = ch;
    drwav_uninit(&wav);
    if (ch <= 0 || out.sample_rate <= 0) {
        err = "invalid WAV header: " + path;
        return false;
    }
    out.pcm.resize((size_t)got);
    if (ch == 1) {
        std::copy(inter.begin(), inter.begin() + (size_t)got, out.pcm.begin());
    } else {
        for (size_t i = 0; i < (size_t)got; ++i) {
            float s = 0.f;
            for (int c = 0; c < ch; ++c) s += inter[i * ch + c];
            out.pcm[i] = s / (float)ch;
        }
    }
    return true;
}

bool write_wav_f32(const std::string& path, const float* pcm, size_t n, int sample_rate, std::string& err) {
    drwav_data_format fmt;
    fmt.container = drwav_container_riff;
    fmt.format = DR_WAVE_FORMAT_IEEE_FLOAT;
    fmt.channels = 1;
    fmt.sampleRate = (drwav_uint32)sample_rate;
    fmt.bitsPerSample = 32;
    drwav wav;
    if (!drwav_init_file_write(&wav, path.c_str(), &fmt, nullptr)) {
        err = "cannot create WAV file: " + path;
        return false;
    }
    const drwav_uint64 put = drwav_write_pcm_frames(&wav, n, pcm);
    const bool ok = drwav_uninit(&wav) == DRWAV_SUCCESS && put == n;
    if (!ok) err = "cannot write WAV file: " + path;
    return ok;
}

namespace {

double bessel_i0(double x) {
    double sum = 1.0, term = 1.0;
    const double q = x * x / 4.0;
    for (int k = 1; k < 64; ++k) {
        term *= q / ((double)k * k);
        sum += term;
        if (term < 1e-12 * sum) break;
    }
    return sum;
}

} // namespace

std::vector<float> resample(const std::vector<float>& in, int sr_in, int sr_out) {
    if (sr_in == sr_out || in.empty()) return in;
    const double ratio = (double)sr_out / sr_in;
    const size_t n_out = (size_t)std::ceil((double)in.size() * ratio);
    // Cutoff in cycles per input sample.
    const double fc = 0.5 * std::min(1.0, ratio) * 0.97;
    const int zeros = 16;
    const double half = zeros / (2.0 * fc);   // kernel half width in input samples
    const double beta = 8.6, i0b = bessel_i0(beta);
    const double pi = 3.14159265358979323846;
    const long n_in = (long)in.size();
    // Kaiser window sampled on |u| in [0, 1], linearly interpolated.
    const int n_tab = 4096;
    std::vector<double> tab(n_tab + 2);
    for (int k = 0; k <= n_tab + 1; ++k) {
        const double u = std::min(1.0, (double)k / n_tab);
        tab[k] = bessel_i0(beta * std::sqrt(1.0 - u * u)) / i0b;
    }

    std::vector<float> out(n_out);
    for (size_t j = 0; j < n_out; ++j) {
        const double t = (double)j / ratio;   // output sample position in input samples
        const long lo = (long)std::ceil(t - half), hi = (long)std::floor(t + half);
        double acc = 0.0;
        for (long i = std::max(0L, lo); i <= std::min(n_in - 1, hi); ++i) {
            const double x = (double)i - t;
            const double pos = std::min(1.0, std::fabs(x) / half) * n_tab;
            const int k = (int)pos;
            const double win = tab[k] + (pos - k) * (tab[k + 1] - tab[k]);
            const double arg = 2.0 * fc * x;
            const double sinc = std::fabs(arg) < 1e-12 ? 1.0 : std::sin(pi * arg) / (pi * arg);
            acc += in[i] * 2.0 * fc * sinc * win;
        }
        out[j] = (float)acc;
    }
    return out;
}

} // namespace af
