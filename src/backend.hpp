#pragma once
#include <functional>
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
class Backend {
public:
    explicit Backend(int n_threads);
    ~Backend();
    Backend(const Backend&) = delete;
    Backend& operator=(const Backend&) = delete;

    ggml_backend_t handle() const { return backend_; }
    int n_threads() const { return n_threads_; }
    void set_n_threads(int n);

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

private:
    ggml_backend_t backend_ = nullptr;
    ggml_gallocr_t galloc_ = nullptr;
    int n_threads_ = 1;
};

} // namespace af
