#include "backend.hpp"
#include "common.hpp"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

namespace af {

static constexpr size_t kMaxNodes = 8192;

Backend::Backend(int n_threads) {
    backend_ = ggml_backend_cpu_init();
    galloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    set_n_threads(n_threads);
}

Backend::~Backend() {
    if (galloc_) ggml_gallocr_free(galloc_);
    if (backend_) ggml_backend_free(backend_);
}

void Backend::set_n_threads(int n) {
    n_threads_ = n > 0 ? n : 1;
    ggml_backend_cpu_set_n_threads(backend_, n_threads_);
}

ggml_tensor* Backend::Graph::input(const void* data, int64_t ne0, int64_t ne1,
                                   int64_t ne2, int64_t ne3) {
    ggml_tensor* t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne0, ne1, ne2, ne3);
    ggml_set_input(t);
    inputs.emplace_back(t, data);
    return t;
}

ggml_tensor* Backend::Graph::input_i32(const int32_t* data, int64_t ne0) {
    ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, ne0);
    ggml_set_input(t);
    inputs.emplace_back(t, data);
    return t;
}

bool Backend::compute(const std::function<bool(Graph&)>& build,
                      const std::function<void(Graph&)>& read) {
    ggml_init_params p = {
        /*.mem_size   =*/ ggml_tensor_overhead() * kMaxNodes + ggml_graph_overhead_custom(kMaxNodes, false),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    Graph g;
    g.ctx = ggml_init(p);
    if (!g.ctx) { AF_LOG("ggml_init failed"); return false; }
    g.gf = ggml_new_graph_custom(g.ctx, kMaxNodes, false);
    bool ok = build(g);
    if (ok && !ggml_gallocr_alloc_graph(galloc_, g.gf)) {
        AF_LOG("gallocr_alloc_graph failed");
        ok = false;
    }
    if (ok) {
        for (auto& in : g.inputs)
            ggml_backend_tensor_set(in.first, in.second, 0, ggml_nbytes(in.first));
        if (ggml_backend_graph_compute(backend_, g.gf) != GGML_STATUS_SUCCESS) {
            AF_LOG("graph compute failed");
            ok = false;
        }
    }
    if (ok && read) read(g);
    ggml_free(g.ctx);
    return ok;
}

} // namespace af
