// avatarforcing-quantize: F32 avatarforcing GGUF -> F16 or quantized GGUF.
//
//   avatarforcing-quantize IN-f32.gguf OUT.gguf TYPE [--tensor-type REGEX=T]... [--exclude REGEX]...
//                          [--threads N]
//
// TYPE is f16, q8_0, q6_k, q5_k or q4_k. The tensors that the policy of
// src/quant_policy.hpp allows get TYPE (F16 for the sensitive ones when TYPE
// is a block type); all other tensors are copied as F32. --tensor-type gives
// the allowed tensors whose name matches REGEX (std::regex, searched in the
// name; the first matching option wins) the type T instead; --exclude REGEX
// is --tensor-type REGEX=f32. The output has the KV of the input plus the
// avatarforcing.quant.* keys described in docs/quantization.md.

#include "quant_policy.hpp"

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <regex>
#include <string>
#include <thread>
#include <vector>

namespace {

struct TypeInfo {
    const char* name;
    ggml_type type;
    uint32_t file_type;   // general.file_type (llama.cpp numbering)
};

const TypeInfo kTypes[] = {
    {"f32", GGML_TYPE_F32, 0},   // --tensor-type only
    {"f16", GGML_TYPE_F16, 1},
    {"q8_0", GGML_TYPE_Q8_0, 7},
    {"q6_k", GGML_TYPE_Q6_K, 18},
    {"q5_k", GGML_TYPE_Q5_K, 16},   // Q5_K_S: one K type for all quantized tensors
    {"q4_k", GGML_TYPE_Q4_K, 14},   // Q4_K_S
};

void usage(const char* argv0) {
    std::fprintf(stderr,
                 "usage: %s IN-f32.gguf OUT.gguf TYPE [options]\n"
                 "  TYPE: f16, q8_0, q6_k, q5_k, q4_k\n"
                 "  --tensor-type REGEX=T  give the tensors whose name matches REGEX the type T\n"
                 "                         (f32, f16, q8_0, q6_k, q5_k, q4_k); repeatable, first match wins\n"
                 "  --exclude REGEX        same as --tensor-type REGEX=f32\n"
                 "  --threads N            quantization threads (default: all cores)\n",
                 argv0);
}

// Quantizes nrows rows of n_per_row values from src to dst, split over threads.
void quantize_rows(ggml_type type, const float* src, void* dst, int64_t nrows, int64_t n_per_row, int n_threads) {
    const int nt = (int)std::max<int64_t>(1, std::min<int64_t>(n_threads, nrows));
    std::vector<std::thread> pool;
    const int64_t per = (nrows + nt - 1) / nt;
    for (int i = 0; i < nt; ++i) {
        const int64_t r0 = i * per, r1 = std::min(nrows, r0 + per);
        if (r0 >= r1) break;
        pool.emplace_back([=] {
            ggml_quantize_chunk(type, src, dst, r0 * n_per_row, r1 - r0, n_per_row, nullptr);
        });
    }
    for (auto& t : pool) t.join();
}

const TypeInfo* find_type(const std::string& name) {
    for (const TypeInfo& t : kTypes)
        if (name == t.name) return &t;
    return nullptr;
}

struct Override {
    std::string spec;   // REGEX=T, as given
    std::regex re;
    const TypeInfo* type;
};

} // namespace

int main(int argc, char** argv) {
    std::vector<std::string> pos;
    std::vector<std::string> specs;   // REGEX=T
    int n_threads = (int)std::max(1u, std::thread::hardware_concurrency());
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--exclude" && i + 1 < argc) specs.push_back(std::string(argv[++i]) + "=f32");
        else if (a == "--tensor-type" && i + 1 < argc) specs.push_back(argv[++i]);
        else if (a == "--threads" && i + 1 < argc) n_threads = std::max(1, std::atoi(argv[++i]));
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else if (!a.empty() && a[0] == '-') { usage(argv[0]); return 2; }
        else pos.push_back(a);
    }
    if (pos.size() != 3) { usage(argv[0]); return 2; }
    const std::string in_path = pos[0], out_path = pos[1], type_name = pos[2];

    const TypeInfo* ti = find_type(type_name);
    if (!ti || ti->type == GGML_TYPE_F32) {
        std::fprintf(stderr, "error: unknown type %s\n", type_name.c_str());
        usage(argv[0]);
        return 2;
    }
    std::vector<Override> overrides;
    for (const auto& sp : specs) {
        const size_t eq = sp.rfind('=');
        const TypeInfo* ot = eq == std::string::npos ? nullptr : find_type(sp.substr(eq + 1));
        if (!ot) { std::fprintf(stderr, "error: bad --tensor-type %s\n", sp.c_str()); return 2; }
        try { overrides.push_back({sp, std::regex(sp.substr(0, eq)), ot}); }
        catch (const std::regex_error&) { std::fprintf(stderr, "error: bad regex in %s\n", sp.c_str()); return 2; }
    }

    ggml_context* meta = nullptr;
    gguf_init_params gp = {/*no_alloc=*/true, /*ctx=*/&meta};
    gguf_context* in = gguf_init_from_file(in_path.c_str(), gp);
    if (!in) { std::fprintf(stderr, "error: cannot read %s\n", in_path.c_str()); return 1; }
    const int64_t arch = gguf_find_key(in, "general.architecture");
    if (arch < 0 || std::string(gguf_get_val_str(in, arch)) != "avatarforcing") {
        std::fprintf(stderr, "error: %s is not an avatarforcing GGUF\n", in_path.c_str());
        return 1;
    }
    if (gguf_find_key(in, "avatarforcing.quant.type") >= 0) {
        std::fprintf(stderr, "error: %s is already quantized; start from the F32 file\n", in_path.c_str());
        return 1;
    }

    // Output: the input KV, the quant keys, then the tensors with their new types.
    gguf_context* out = gguf_init_empty();
    gguf_set_kv(out, in);
    gguf_set_val_u32(out, "general.file_type", ti->file_type);
    gguf_set_val_u32(out, "general.quantization_version", GGML_QNT_VERSION);
    gguf_set_val_str(out, "avatarforcing.quant.type", ti->name);
    gguf_set_val_str(out, "avatarforcing.quant.policy", af::quant_policy_text());
    if (!overrides.empty()) {
        std::vector<const char*> p;
        for (const auto& o : overrides) p.push_back(o.spec.c_str());
        gguf_set_arr_str(out, "avatarforcing.quant.overrides", p.data(), p.size());
    }

    const int64_t n_t = gguf_get_n_tensors(in);
    std::vector<ggml_type> new_type(n_t, GGML_TYPE_F32);
    uint32_t n_quant = 0;
    std::map<std::string, int> type_count;
    for (int64_t i = 0; i < n_t; ++i) {
        const char* name = gguf_get_tensor_name(in, i);
        ggml_tensor* t = ggml_get_tensor(meta, name);
        if (t->type != GGML_TYPE_F32) {
            std::fprintf(stderr, "error: %s is %s; the input must be all F32\n", name, ggml_type_name(t->type));
            return 1;
        }
        gguf_add_tensor(out, t);
        if (!af::quant_allowed(name) || ggml_n_dims(t) != 2) continue;
        // The file type; F16 for the sensitive tensors when it is a block
        // type (src/quant_policy.hpp); the first matching --tensor-type wins.
        ggml_type ty = ggml_is_quantized(ti->type) && af::quant_sensitive(name) ? GGML_TYPE_F16 : ti->type;
        for (const auto& o : overrides)
            if (std::regex_search(name, o.re)) { ty = o.type->type; break; }
        if (t->ne[0] % ggml_blck_size(ty) != 0) {
            std::fprintf(stderr, "note: %s: row size %lld is not a multiple of %lld, kept F32\n", name,
                         (long long)t->ne[0], (long long)ggml_blck_size(ty));
            ty = GGML_TYPE_F32;
        }
        if (ty == GGML_TYPE_F32) continue;
        gguf_set_tensor_type(out, name, ty);
        new_type[i] = ty;
        ++n_quant;
        ++type_count[ggml_type_name(ty)];
    }
    gguf_set_val_u32(out, "avatarforcing.quant.n_tensors", n_quant);   // tensors not in F32

    FILE* fi = std::fopen(in_path.c_str(), "rb");
    FILE* fo = std::fopen(out_path.c_str(), "wb");
    if (!fi || !fo) { std::fprintf(stderr, "error: cannot open the input or output file\n"); return 1; }
    std::vector<uint8_t> hdr(gguf_get_meta_size(out));
    gguf_get_meta_data(out, hdr.data());
    if (std::fwrite(hdr.data(), 1, hdr.size(), fo) != hdr.size()) { std::fprintf(stderr, "error: write failed\n"); return 1; }

    const size_t align = gguf_get_alignment(out);
    const size_t in_data = gguf_get_data_offset(in);
    std::vector<float> src;
    std::vector<uint8_t> dst;
    size_t in_bytes = 0, out_bytes = 0;
    for (int64_t i = 0; i < n_t; ++i) {
        const char* name = gguf_get_tensor_name(in, i);
        ggml_tensor* t = ggml_get_tensor(meta, name);
        const int64_t n = ggml_nelements(t);
        src.resize(n);
        if (std::fseek(fi, (long)(in_data + gguf_get_tensor_offset(in, i)), SEEK_SET) != 0 ||
            std::fread(src.data(), sizeof(float), n, fi) != (size_t)n) {
            std::fprintf(stderr, "error: read failed for %s\n", name);
            return 1;
        }
        const uint8_t* data = (const uint8_t*)src.data();
        size_t nb = (size_t)n * sizeof(float);
        if (new_type[i] != GGML_TYPE_F32) {
            const int64_t n_per_row = t->ne[0], nrows = n / n_per_row;
            nb = ggml_row_size(new_type[i], n_per_row) * nrows;
            dst.resize(nb);
            quantize_rows(new_type[i], src.data(), dst.data(), nrows, n_per_row, n_threads);
            data = dst.data();
        }
        // The offsets gguf_add_tensor assigned: each tensor padded to the alignment.
        const size_t padded = GGML_PAD(nb, align);
        bool w = std::fwrite(data, 1, nb, fo) == nb;
        static const uint8_t zeros[64] = {0};
        for (size_t p = nb; w && p < padded; p += std::min<size_t>(sizeof(zeros), padded - p))
            w = std::fwrite(zeros, 1, std::min<size_t>(sizeof(zeros), padded - p), fo) ==
                std::min<size_t>(sizeof(zeros), padded - p);
        if (!w) { std::fprintf(stderr, "error: write failed for %s\n", name); return 1; }
        in_bytes += (size_t)n * sizeof(float);
        out_bytes += nb;
        if (new_type[i] != GGML_TYPE_F32)
            std::printf("%-64s %-5s -> %-5s %8.2f MB -> %8.2f MB\n", name, "f32", ggml_type_name(new_type[i]),
                        n * 4 / 1e6, nb / 1e6);
    }
    std::fclose(fi);
    if (std::fclose(fo) != 0) { std::fprintf(stderr, "error: closing %s failed\n", out_path.c_str()); return 1; }
    std::string counts;
    for (const auto& kv : type_count)
        counts += (counts.empty() ? "" : ", ") + kv.first + " x " + std::to_string(kv.second);
    std::printf("%s: %s, %u of %lld tensors not F32 (%s), tensor data %.1f MB -> %.1f MB\n", out_path.c_str(),
                ti->name, n_quant, (long long)n_t, counts.c_str(), in_bytes / 1e6, out_bytes / 1e6);
    gguf_free(out);
    gguf_free(in);
    ggml_free(meta);
    return 0;
}
