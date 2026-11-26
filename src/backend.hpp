#pragma once
#include <functional>
#include <string>
#include <utility>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct ggml_cgraph;
struct ggml_backend;
typedef struct ggml_backend* ggml_backend_t;
struct ggml_gallocr;
typedef struct ggml_gallocr* ggml_gallocr_t;

namespace af {

// Persistent compute backend plus a reusable graph allocator.
//
// Each call to compute() builds a graph in a no_alloc context (metadata only),
// allocates it with the persistent gallocr, uploads the inputs registered
// during the build, runs the graph and returns. Input data must be uploaded
// after the allocation, so the build function registers it with set_input()
// instead of writing tensor data directly.
//
// The device is "cpu" or "cuda" ("cuda:N" for device N; "gpu" takes the first
// GPU device of any ggml backend). Every node of a graph must be supported by
// the device: compute() checks this before it runs a graph on a non-CPU
// device and fails with the list of unsupported ops, so no op runs on the
// CPU behind the caller's back.
class Backend {
public:
    explicit Backend(int n_threads, const std::string& device = "cpu");
    ~Backend();
    Backend(const Backend&) = delete;
    Backend& operator=(const Backend&) = delete;

    bool ok() const { return backend_ != nullptr; }
    bool is_cpu() const { return is_cpu_; }
    const std::string& device() const { return device_; }   // ggml device name, e.g. "CPU", "CUDA0"
    ggml_backend_t handle() const { return backend_; }
    int n_threads() const { return n_threads_; }
    void set_n_threads(int n);

    // Device name from an environment variable (e.g. AVATARFORCING_TEST_BACKEND),
    // "cpu" when it is unset.
    static std::string device_from_env(const char* var);

    // Graph build context, valid during one compute() call.
    struct Graph {
        ggml_context* ctx = nullptr;
        ggml_cgraph* gf = nullptr;
        std::vector<std::pair<ggml_tensor*, const void*>> inputs;
        // Create an F32 input tensor and register its host data.
        ggml_tensor* input(const void* data, int64_t ne0, int64_t ne1 = 1,
                           int64_t ne2 = 1, int64_t ne3 = 1);
        ggml_tensor* input_i32(const int32_t* data, int64_t ne0);
    };

    // build() creates the graph and returns false on error. After the run,
    // read outputs with ggml_backend_tensor_get inside read().
    bool compute(const std::function<bool(Graph&)>& build,
                 const std::function<void(Graph&)>& read);

    // Profiling (AVATARFORCING_PROFILE=1 or set_profile(true)): compute()
    // accumulates the time per op and per output width. On the patched CUDA
    // backend every kernel launch (a node or a fused group of nodes) is
    // timed with CUDA events, with CUDA graphs off; on other backends the
    // graph runs one node at a time with a sync after each node (no fusion).
    void set_profile(bool on) { profile_ = on; }
    void reset_profile() { prof_.clear(); }
    void print_profile(const char* title) const;

private:
    bool check_ops(ggml_cgraph* gf) const;
    bool compute_profiled(ggml_cgraph* gf);
    bool compute_op_profiled(ggml_cgraph* gf, bool& ok);

    ggml_backend_t backend_ = nullptr;
    ggml_gallocr_t galloc_ = nullptr;
    std::string device_;
    bool is_cpu_ = true;
    int n_threads_ = 1;
    bool profile_ = false;
    bool log_ops_ = false;
    struct ProfEntry { std::string key; double ms = 0; long count = 0; };
    std::vector<ProfEntry> prof_;
};

} // namespace af
