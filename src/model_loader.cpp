#include "model_loader.hpp"
#include "common.hpp"
#include "quant_policy.hpp"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>

namespace af {

namespace {

struct KvReader {
    gguf_context* g;
    bool ok = true;

    int64_t find(const char* key) {
        int64_t id = gguf_find_key(g, key);
        if (id < 0) { AF_LOG("missing KV: %s", key); ok = false; }
        return id;
    }
    uint32_t u32(const char* key) {
        int64_t id = find(key);
        if (id < 0) return 0;
        if (gguf_get_kv_type(g, id) != GGUF_TYPE_UINT32) { AF_LOG("KV %s: not u32", key); ok = false; return 0; }
        return gguf_get_val_u32(g, id);
    }
    float f32(const char* key) {
        int64_t id = find(key);
        if (id < 0) return 0.f;
        if (gguf_get_kv_type(g, id) != GGUF_TYPE_FLOAT32) { AF_LOG("KV %s: not f32", key); ok = false; return 0.f; }
        return gguf_get_val_f32(g, id);
    }
    std::string str(const char* key) {
        int64_t id = find(key);
        if (id < 0) return {};
        if (gguf_get_kv_type(g, id) != GGUF_TYPE_STRING) { AF_LOG("KV %s: not str", key); ok = false; return {}; }
        return gguf_get_val_str(g, id);
    }
    std::vector<std::string> str_arr(const char* key) {
        std::vector<std::string> out;
        int64_t id = find(key);
        if (id < 0) return out;
        if (gguf_get_kv_type(g, id) != GGUF_TYPE_ARRAY || gguf_get_arr_type(g, id) != GGUF_TYPE_STRING) {
            AF_LOG("KV %s: not [str]", key); ok = false; return out;
        }
        size_t n = gguf_get_arr_n(g, id);
        for (size_t i = 0; i < n; ++i) out.emplace_back(gguf_get_arr_str(g, id, i));
        return out;
    }
};

bool read_config(gguf_context* g, FlowConfig& c) {
    KvReader r{g};
    c.clip_frames            = r.u32("avatarforcing.clip_frames");
    c.block_frames           = r.u32("avatarforcing.block_frames");
    c.context_frames         = r.u32("avatarforcing.context_frames");
    c.self_cache_len         = r.u32("avatarforcing.stream.self_cache_len");
    c.cross_cache_len        = r.u32("avatarforcing.stream.cross_cache_len");
    c.kv_shift               = r.u32("avatarforcing.stream.kv_shift");
    c.first_steady_start_pos = r.u32("avatarforcing.stream.first_steady_start_pos");
    c.dim_w                  = r.u32("avatarforcing.flow.dim_w");
    c.dim_h                  = r.u32("avatarforcing.flow.dim_h");
    c.dim_a                  = r.u32("avatarforcing.flow.dim_a");
    c.depth                  = r.u32("avatarforcing.flow.depth");
    c.n_heads                = r.u32("avatarforcing.flow.n_heads");
    c.head_dim               = r.u32("avatarforcing.flow.head_dim");
    c.mlp_hidden             = r.u32("avatarforcing.flow.mlp_hidden");
    c.mlp_act                = r.str("avatarforcing.flow.mlp_act");
    c.ln_eps                 = r.f32("avatarforcing.flow.ln_eps");
    c.freq_embed_dim         = r.u32("avatarforcing.flow.freq_embed_dim");
    c.rope_theta             = r.f32("avatarforcing.flow.rope.theta");
    c.rope_dim               = r.u32("avatarforcing.flow.rope.dim");
    c.rope_layout            = r.str("avatarforcing.flow.rope.layout");
    c.self_mask_block        = r.u32("avatarforcing.flow.self_mask.block");
    c.self_mask_lookahead    = r.u32("avatarforcing.flow.self_mask.lookahead");
    c.cross_mask_window      = r.u32("avatarforcing.flow.cross_mask.window");
    c.adaln_order            = r.str_arr("avatarforcing.flow.adaln.order");
    c.final_adaln_order      = r.str_arr("avatarforcing.flow.final_adaln.order");
    c.cond_n_heads           = r.u32("avatarforcing.cond.n_heads");
    c.cond_ln_eps            = r.f32("avatarforcing.cond.ln_eps");
    c.audio_proj_in          = r.u32("avatarforcing.audio.proj_in");
    c.audio_proj_out         = r.u32("avatarforcing.audio.proj_out");
    c.audio_proj_ln_eps      = r.f32("avatarforcing.audio.proj_ln_eps");
    c.default_nfe            = r.u32("avatarforcing.sampler.default_nfe");
    c.num_train_timesteps    = r.u32("avatarforcing.sampler.num_train_timesteps");
    c.sigma_table_len        = r.u32("avatarforcing.sampler.sigma_table_len");
    const std::string solver = r.str("avatarforcing.sampler.solver");
    c.cfg_audio_scale        = r.f32("avatarforcing.cfg.audio_scale");
    if (!r.ok) return false;

    // Checks of the invariants the graph relies on.
    bool ok = true;
    auto check = [&](bool cond, const char* what) {
        if (!cond) { AF_LOG("unsupported config: %s", what); ok = false; }
    };
    check(c.n_heads * c.head_dim == c.dim_h, "n_heads * head_dim != dim_h");
    check(c.rope_dim == c.head_dim, "rope.dim != head_dim");
    check(c.rope_layout == "interleaved", "rope.layout != interleaved");
    check(c.mlp_act == "gelu_tanh", "mlp_act != gelu_tanh");
    check(c.self_cache_len + c.block_frames + c.context_frames == c.clip_frames,
          "self_cache_len + block + context != clip");
    check(c.kv_shift == c.block_frames, "kv_shift != block_frames");
    check(c.adaln_order.size() == 6 && c.adaln_order[0] == "shift_msa" && c.adaln_order[1] == "scale_msa" &&
          c.adaln_order[2] == "gate_msa" && c.adaln_order[3] == "shift_mlp" &&
          c.adaln_order[4] == "scale_mlp" && c.adaln_order[5] == "gate_mlp", "adaln.order");
    check(c.final_adaln_order.size() == 2 && c.final_adaln_order[0] == "shift" &&
          c.final_adaln_order[1] == "scale", "final_adaln.order");
    check(c.dim_a % c.cond_n_heads == 0, "dim_a % cond.n_heads");
    check(solver == "closed_form_argmin_f64", "sampler.solver != closed_form_argmin_f64");
    check(c.default_nfe >= 3, "sampler.default_nfe < 3");
    return ok;
}

bool has_prefix(const char* name, const std::vector<std::string>& prefixes) {
    for (const auto& p : prefixes)
        if (std::strncmp(name, p.c_str(), p.size()) == 0) return true;
    return false;
}

} // namespace

ModelLoader::~ModelLoader() {
    if (buf_) ggml_backend_buffer_free(buf_);
    if (ctx_) ggml_free(ctx_);
}

bool ModelLoader::load(const std::string& path, ggml_backend_t backend,
                       const std::vector<std::string>& prefixes) {
    ggml_context* meta = nullptr;
    gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ &meta };
    gguf_context* g = gguf_init_from_file(path.c_str(), gp);
    if (!g) { AF_LOG("cannot open GGUF: %s", path.c_str()); return false; }

    bool ok = true;
    int64_t arch_id = gguf_find_key(g, "general.architecture");
    if (arch_id < 0 || std::string(gguf_get_val_str(g, arch_id)) != "avatarforcing") {
        AF_LOG("%s: general.architecture is not avatarforcing", path.c_str());
        ok = false;
    }
    ok = ok && read_config(g, cfg_);

    // Select the tensors and create them in one no_alloc context.
    std::vector<int64_t> ids;
    const int64_t nt = gguf_get_n_tensors(g);
    for (int64_t i = 0; ok && i < nt; ++i)
        if (has_prefix(gguf_get_tensor_name(g, i), prefixes)) ids.push_back(i);
    if (ok && ids.empty()) { AF_LOG("no tensors match the requested prefixes"); ok = false; }

    // Types other than F32 only where the quantization policy allows them
    // (src/quant_policy.hpp): F16 or a block type, on a 2-D weight matrix.
    for (size_t k = 0; ok && k < ids.size(); ++k) {
        const char* name = gguf_get_tensor_name(g, ids[k]);
        const ggml_tensor* t = ggml_get_tensor(meta, name);
        if (t->type == GGML_TYPE_F32) continue;
        const bool type_ok = t->type == GGML_TYPE_F16 || ggml_is_quantized(t->type);
        if (!type_ok || !quant_allowed(name) || ggml_n_dims(t) != 2) {
            AF_LOG("tensor %s: type %s is not allowed for this tensor (docs/quantization.md)", name,
                   ggml_type_name(t->type));
            ok = false;
        }
    }

    if (ok) {
        ggml_init_params p = { ggml_tensor_overhead() * (ids.size() + 1), nullptr, true };
        ctx_ = ggml_init(p);
        for (int64_t i : ids) {
            const char* name = gguf_get_tensor_name(g, i);
            ggml_tensor* src = ggml_get_tensor(meta, name);
            ggml_tensor* t = ggml_dup_tensor(ctx_, src);
            ggml_set_name(t, name);
            tensors_[name] = t;
        }
        buf_ = ggml_backend_alloc_ctx_tensors(ctx_, backend);
        if (!buf_) { AF_LOG("weight buffer allocation failed"); ok = false; }
        else ggml_backend_buffer_set_usage(buf_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }

    // Read the tensor data from the file.
    if (ok) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) { AF_LOG("cannot reopen %s", path.c_str()); ok = false; }
        std::vector<uint8_t> staging;
        const size_t data_off = gguf_get_data_offset(g);
        for (size_t k = 0; ok && k < ids.size(); ++k) {
            const char* name = gguf_get_tensor_name(g, ids[k]);
            ggml_tensor* t = tensors_[name];
            const size_t nb = ggml_nbytes(t);
            staging.resize(nb);
            const size_t off = data_off + gguf_get_tensor_offset(g, ids[k]);
            if (std::fseek(f, (long)off, SEEK_SET) != 0 || std::fread(staging.data(), 1, nb, f) != nb) {
                AF_LOG("read failed for %s", name);
                ok = false;
                break;
            }
            ggml_backend_tensor_set(t, staging.data(), 0, nb);
            n_bytes_ += nb;
        }
        if (f) std::fclose(f);
    }

    gguf_free(g);
    if (meta) ggml_free(meta);
    return ok;
}

ggml_tensor* ModelLoader::tensor(const std::string& name) const {
    auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : it->second;
}

ggml_tensor* ModelLoader::require(const std::string& name) {
    ggml_tensor* t = tensor(name);
    if (!t) { AF_LOG("missing tensor: %s", name.c_str()); missing_ = true; }
    return t;
}

} // namespace af
