#pragma once
#include <string>
#include <vector>

namespace af {

// Decoded WAV file: mono float PCM in [-1, 1) at the file's own rate.
struct Wav {
    std::vector<float> pcm;
    int sample_rate = 0;
    int channels = 0;   // channels in the file (the PCM is already mono)
};

// Reads a WAV file (any PCM or float format dr_wav supports). Multi-channel
// audio is mixed down to mono by averaging the channels, as librosa does.
// 16-bit samples convert as int16 / 32768.
bool read_wav_mono(const std::string& path, Wav& out, std::string& err);

// Band-limited resampling with a Kaiser-windowed sinc kernel (16 zero
// crossings, beta 8.6, cutoff 0.97 of the lower Nyquist frequency).
// Output length is ceil(n * sr_out / sr_in). Returns the input unchanged
// when the rates are equal.
std::vector<float> resample(const std::vector<float>& in, int sr_in, int sr_out);

// Writes mono float PCM as a 32-bit float WAV file.
bool write_wav_f32(const std::string& path, const float* pcm, size_t n, int sample_rate, std::string& err);

} // namespace af
