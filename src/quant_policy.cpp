#include "quant_policy.hpp"

#include <regex>

namespace af {

namespace {

const std::regex& allowed_re() {
    static const std::regex re(
        // flow transformer blocks
        R"(^ft\.blocks\.[0-9]+\.(attn\.(qkv|proj)|cross_attn\.(q|kv|proj)|mlp\.(fc1|fc2))\.weight$)"
        // timestep embedder, adaLN tables, input and output projections
        R"(|^ft\.(t_embedder\.mlp\.[02]|adaLN_modulation\.1|final_layer\.adaLN_modulation\.1|)"
        R"(final_layer\.linear|x_embedder\.proj)\.weight$)"
        // condition embedder
        R"(|^ft\.c_embedder\.((dual_modal_attn|user_agent_attn)\.(q|k|v|out)_proj|)"
        R"(motion_embedder\.[03]|to_cond_user)\.weight$)"
        // audio projection (Linear 9216 -> 512)
        R"(|^audio_projection\.0\.weight$)"
        // wav2vec2 encoder layers and feature projection
        R"(|^w2v2\.encoder\.layers\.[0-9]+\.(attention\.(q|k|v|out)_proj|)"
        R"(feed_forward\.(intermediate|output)_dense)\.weight$)"
        R"(|^w2v2\.feature_projection\.projection\.weight$)",
        std::regex::optimize);
    return re;
}

const std::regex& sensitive_re() {
    static const std::regex re(
        R"(^ft\.(x_embedder\.proj|final_layer\.linear|t_embedder\.mlp\.[02]|adaLN_modulation\.1|)"
        R"(final_layer\.adaLN_modulation\.1)\.weight$)",
        std::regex::optimize);
    return re;
}

} // namespace

bool quant_sensitive(const std::string& name) {
    return std::regex_search(name, sensitive_re());
}

bool quant_allowed(const std::string& name) {
    return std::regex_search(name, allowed_re());
}

const char* quant_policy_text() {
    return "linear weights of ft.*, audio_projection.0 and the w2v2 encoder layers and feature projection "
           "in the file type, except ft x_embedder, final_layer.linear, t_embedder and adaLN weights in F16 "
           "when the file type is a block type; everything else (biases, norms, mae.*, w2v2 convolutions, "
           "tables) F32";
}

} // namespace af
