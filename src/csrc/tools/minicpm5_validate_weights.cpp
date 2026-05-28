#include "minicpmv/weights.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace minicpmv;

namespace {

struct MiniCPM5Config {
    int64_t hidden_size{1536};
    int64_t intermediate_size{4608};
    int64_t num_layers{24};
    int64_t num_q_heads{16};
    int64_t num_kv_heads{2};
    int64_t head_dim{128};
    int64_t vocab_size{130560};
    std::string model_prefix{"model"};
    std::string embed_weight_name{"model.embed_tokens.weight"};
    std::string final_norm_weight_name{"model.norm.weight"};
    std::string lm_head_weight_name{"lm_head.weight"};
};

void expect_shape(const WeightsIndex& index, const std::string& name, const std::vector<int64_t>& shape) {
    if (!index.contains(name)) {
        throw std::runtime_error("missing tensor: " + name);
    }
    const auto& actual = index.at(name).shape;
    if (actual != shape) {
        throw std::runtime_error("shape mismatch for " + name);
    }
}

void usage(const char* argv0) {
    std::cerr << "usage: " << argv0 << " <model.safetensors>\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        usage(argv[0]);
        return 2;
    }

    try {
        const MiniCPM5Config cfg;
        WeightsIndex index(argv[1]);

        expect_shape(index, cfg.embed_weight_name, {cfg.vocab_size, cfg.hidden_size});
        expect_shape(index, cfg.final_norm_weight_name, {cfg.hidden_size});
        expect_shape(index, cfg.lm_head_weight_name, {cfg.vocab_size, cfg.hidden_size});

        const int64_t q_dim = cfg.num_q_heads * cfg.head_dim;
        const int64_t kv_dim = cfg.num_kv_heads * cfg.head_dim;
        for (int64_t layer = 0; layer < cfg.num_layers; ++layer) {
            const std::string base = cfg.model_prefix + ".layers." + std::to_string(layer) + ".";
            expect_shape(index, base + "input_layernorm.weight", {cfg.hidden_size});
            expect_shape(index, base + "post_attention_layernorm.weight", {cfg.hidden_size});
            expect_shape(index, base + "self_attn.q_proj.weight", {q_dim, cfg.hidden_size});
            expect_shape(index, base + "self_attn.k_proj.weight", {kv_dim, cfg.hidden_size});
            expect_shape(index, base + "self_attn.v_proj.weight", {kv_dim, cfg.hidden_size});
            expect_shape(index, base + "self_attn.o_proj.weight", {cfg.hidden_size, q_dim});
            expect_shape(index, base + "mlp.gate_proj.weight", {cfg.intermediate_size, cfg.hidden_size});
            expect_shape(index, base + "mlp.up_proj.weight", {cfg.intermediate_size, cfg.hidden_size});
            expect_shape(index, base + "mlp.down_proj.weight", {cfg.hidden_size, cfg.intermediate_size});
        }

        std::cout << "MiniCPM5-1B weights validated: " << index.size() << " tensors\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
