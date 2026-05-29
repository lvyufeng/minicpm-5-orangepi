#include "minicpmv/acl_context.h"
#include "minicpmv/language_model.h"
#include "minicpmv/ops.h"
#include "minicpmv/weights.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace minicpmv;

namespace {

std::vector<int32_t> parse_ids(const std::string& s) {
    std::vector<int32_t> ids;
    size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == ',')) {
            ++i;
        }
        if (i >= s.size()) break;
        size_t j = i;
        while (j < s.size() && s[j] != ',') {
            ++j;
        }
        std::string part = s.substr(i, j - i);
        part.erase(std::remove_if(part.begin(), part.end(), [](unsigned char c) { return std::isspace(c); }), part.end());
        if (!part.empty()) {
            ids.push_back(static_cast<int32_t>(std::stol(part)));
        }
        i = j;
    }
    return ids;
}

bool profile_enabled() {
    const char* v = std::getenv("MINICPM_PROFILE");
    return v != nullptr && *v != '\0' && std::string(v) != "0" && std::string(v) != "false";
}

void print_profile(const char* name, std::chrono::steady_clock::time_point start) {
    if (!profile_enabled()) return;
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    std::cerr << "# profile " << name << " ms=" << ms << '\n';
}

void usage(const char* argv0) {
    std::cerr << "usage: " << argv0
              << " [--weights PATH] [--input-ids 1,2,3] [--max-new N] [--max-seq N] [--device-id N]\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string weights_path = default_safetensors_path();
    std::string input_ids_arg = "0";
    int64_t max_new_tokens = 16;
    int64_t max_seq_len = 4096;
    int device_id = 0;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--weights" && i + 1 < argc) {
            weights_path = argv[++i];
        } else if (arg == "--input-ids" && i + 1 < argc) {
            input_ids_arg = argv[++i];
        } else if (arg == "--max-new" && i + 1 < argc) {
            max_new_tokens = std::stoll(argv[++i]);
        } else if (arg == "--max-seq" && i + 1 < argc) {
            max_seq_len = std::stoll(argv[++i]);
        } else if (arg == "--device-id" && i + 1 < argc) {
            device_id = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    try {
        const auto total_start = std::chrono::steady_clock::now();
        auto stage_start = std::chrono::steady_clock::now();
        AclContext ctx(device_id);
        const LanguageModelConfig cfg = default_minicpm5_1b_lm_config();
        WeightsIndex index(weights_path);
        LanguageModelWeights w = load_language_model_weights(index, cfg);
        print_profile("load_weights", stage_start);

        std::vector<int32_t> input_ids = parse_ids(input_ids_arg);
        if (input_ids.empty()) {
            input_ids.push_back(0);
        }
        if (static_cast<int64_t>(input_ids.size()) + max_new_tokens > max_seq_len) {
            throw std::runtime_error("max_seq too small for prompt plus generation");
        }

        stage_start = std::chrono::steady_clock::now();
        Tensor cos_t, sin_t;
        build_rope_tables(max_seq_len, cfg, cos_t, sin_t);
        print_profile("build_rope_tables", stage_start);

        stage_start = std::chrono::steady_clock::now();
        Tensor prompt_hidden({static_cast<int64_t>(input_ids.size()), cfg.hidden_size}, DType::Float16);
        prompt_hidden.allocate();
        embedding_lookup(w.embed, input_ids, prompt_hidden, ctx.stream());
        DecodeState state = make_decode_state(
            max_seq_len, cfg.num_layers,
            AttentionDecoderLayerConfig{cfg.num_q_heads, cfg.num_kv_heads,
                                        cfg.head_dim, cfg.rotary_dim, cfg.rms_epsilon},
            ctx.stream());
        Tensor last_hidden = prefill_from_embeddings(prompt_hidden, w, cfg, cos_t, sin_t, state, ctx.stream());
        int64_t next_tok = lm_head_greedy(last_hidden, w, cfg, ctx.stream());
        print_profile("prefill_and_first_lm_head", stage_start);

        const std::vector<int64_t> eos_token_ids{1, 130073};
        int64_t generated = 0;
        while (generated < max_new_tokens) {
            std::cout << next_tok << '\n';
            std::cout.flush();
            ++generated;
            if (is_eos(next_tok, eos_token_ids)) {
                break;
            }
            if (generated >= max_new_tokens || state.seq_len >= state.max_seq_len) {
                break;
            }
            stage_start = std::chrono::steady_clock::now();
            next_tok = decode_step_greedy(static_cast<int32_t>(next_tok), w, cfg, cos_t, sin_t, state, ctx.stream());
            print_profile("decode_next_token", stage_start);
        }

        std::cerr << "# done generated=" << generated << "\n";
        print_profile("total", total_start);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "minicpm5_decode error: " << e.what() << '\n';
        return 1;
    }
}
