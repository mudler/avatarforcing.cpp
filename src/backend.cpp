#include "backend.hpp"
#include "common.hpp"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <map>

namespace af {

static constexpr size_t kMaxNodes = 8192;

static std::string lower(std::string s) {
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

std::string Backend::device_from_env(const char* var) {
    const char* v = std::getenv(var);
    return v && *v ? std::string(v) : std::string("cpu");
}

Backend::Backend(int n_threads, const std::string& device) {
    const std::string d = lower(device);
    ggml_backend_dev_t dev = nullptr;
    if (d.empty() || d == "cpu") {
        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    } else if (d == "gpu") {
        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    } else if (d.rfind("cuda", 0) == 0) {
        int idx = 0;
        if (d.size() > 4) idx = std::atoi(d.c_str() + (d[4] == ':' ? 5 : 4));
        const std::string name = "CUDA" + std::to_string(idx);
        dev = ggml_backend_dev_by_name(name.c_str());
    }
    if (!dev) {
        AF_LOG("backend: device '%s' is not available in this build (available:", device.c_str());
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i)
            AF_LOG("  %s (%s)", ggml_backend_dev_name(ggml_backend_dev_get(i)),
                   ggml_backend_dev_description(ggml_backend_dev_get(i)));
        AF_LOG(")");
        return;
    }
    backend_ = ggml_backend_dev_init(dev, nullptr);
    if (!backend_) { AF_LOG("backend: cannot initialize %s", ggml_backend_dev_name(dev)); return; }
    device_ = ggml_backend_dev_name(dev);
    is_cpu_ = ggml_backend_is_cpu(backend_);
    galloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    set_n_threads(n_threads);
    const char* p = std::getenv("AVATARFORCING_PROFILE");
    profile_ = p && *p && std::strcmp(p, "0") != 0;
    const char* lo = std::getenv("AVATARFORCING_LOG_OPS");
    log_ops_ = lo && *lo && std::strcmp(lo, "0") != 0;
    if (!is_cpu_) AF_LOG("backend: %s (%s)", device_.c_str(), ggml_backend_dev_description(dev));
}

Backend::~Backend() {
    if (galloc_) ggml_gallocr_free(galloc_);
    if (backend_) ggml_backend_free(backend_);
}

void Backend::set_n_threads(int n) {
    n_threads_ = n > 0 ? n : 1;
    if (backend_ && is_cpu_) ggml_backend_cpu_set_n_threads(backend_, n_threads_);
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

// Every node must run on this backend. On the CPU backend this always holds.
bool Backend::check_ops(ggml_cgraph* gf) const {
    if (is_cpu_ && !log_ops_) return true;
    std::map<std::string, int> bad, all;
    const int n = ggml_graph_n_nodes(gf);
    for (int i = 0; i < n; ++i) {
        ggml_tensor* t = ggml_graph_node(gf, i);
        const std::string op = ggml_op_desc(t);
        all[op]++;
        if (!ggml_backend_supports_op(backend_, t)) bad[op]++;
    }
    if (log_ops_) {
        std::string s;
        for (auto& kv : all) s += " " + kv.first + "=" + std::to_string(kv.second);
        AF_LOG("graph on %s: %d nodes:%s", device_.c_str(), n, s.c_str());
    }
    if (bad.empty()) return true;
    for (auto& kv : bad) AF_LOG("op %s (%d nodes) is not supported on %s", kv.first.c_str(), kv.second, device_.c_str());
    return false;
}

bool Backend::compute_profiled(ggml_cgraph* gf) {
    using Clock = std::chrono::steady_clock;
    ggml_init_params p = { ggml_graph_overhead_custom(1, false) + 1024, nullptr, true };
    ggml_context* pctx = ggml_init(p);
    if (!pctx) return false;
    const int n = ggml_graph_n_nodes(gf);
    bool ok = true;
    for (int i = 0; i < n && ok; ++i) {
        ggml_tensor* t = ggml_graph_node(gf, i);
        if (t->op == GGML_OP_NONE || t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE ||
            t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE)
            continue;
        ggml_cgraph* one = ggml_new_graph_custom(pctx, 1, false);
        ggml_graph_add_node(one, t);
        const auto t0 = Clock::now();
        ok = ggml_backend_graph_compute(backend_, one) == GGML_STATUS_SUCCESS;
        ggml_backend_synchronize(backend_);
        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        char key[160];
        std::snprintf(key, sizeof(key), "%-14s w=%-5lld", ggml_op_desc(t), (long long)t->ne[0]);
        auto it = std::find_if(prof_.begin(), prof_.end(), [&](const ProfEntry& e) { return e.key == key; });
        if (it == prof_.end()) { prof_.push_back({key, 0, 0}); it = prof_.end() - 1; }
        it->ms += ms;
        it->count++;
        // Graph-local scratch: one graph object per node, freed with pctx.
        ggml_free(pctx);
        pctx = ggml_init(p);
    }
    ggml_free(pctx);
    return ok;
}

void Backend::print_profile(const char* title) const {
    std::vector<ProfEntry> v = prof_;
    std::sort(v.begin(), v.end(), [](const ProfEntry& a, const ProfEntry& b) { return a.ms > b.ms; });
    double total = 0;
    std::map<std::string, double> by_op;
    for (auto& e : v) {
        total += e.ms;
        by_op[e.key.substr(0, e.key.find(' '))] += e.ms;
    }
    std::printf("profile %s on %s: %.1f ms in profiled nodes\n", title, device_.c_str(), total);
    std::vector<std::pair<std::string, double>> ops(by_op.begin(), by_op.end());
    std::sort(ops.begin(), ops.end(), [](auto& a, auto& b) { return a.second > b.second; });
    std::printf("  by op:\n");
    for (auto& o : ops) std::printf("    %-16s %9.2f ms  %5.1f %%\n", o.first.c_str(), o.second, 100.0 * o.second / total);
    std::printf("  by op and output width (top 30):\n");
    for (size_t i = 0; i < v.size() && i < 30; ++i)
        std::printf("    %s %9.2f ms  %5ld nodes  %5.1f %%\n", v[i].key.c_str(), v[i].ms, v[i].count,
                    100.0 * v[i].ms / total);
}

bool Backend::compute(const std::function<bool(Graph&)>& build,
                      const std::function<void(Graph&)>& read) {
    if (!backend_) { AF_LOG("backend not initialized"); return false; }
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
    if (ok && !check_ops(g.gf)) ok = false;
    if (ok && !ggml_gallocr_alloc_graph(galloc_, g.gf)) {
        AF_LOG("gallocr_alloc_graph failed");
        ok = false;
    }
    if (ok) {
        for (auto& in : g.inputs)
            ggml_backend_tensor_set(in.first, in.second, 0, ggml_nbytes(in.first));
        const bool run_ok = profile_ ? compute_profiled(g.gf)
                                     : ggml_backend_graph_compute(backend_, g.gf) == GGML_STATUS_SUCCESS;
        if (!run_ok) {
            AF_LOG("graph compute failed");
            ok = false;
        }
    }
    if (ok && read) read(g);
    ggml_free(g.ctx);
    return ok;
}

} // namespace af
