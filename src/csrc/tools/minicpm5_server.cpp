#include "minicpmv/acl_context.h"
#include "minicpmv/language_model.h"
#include "minicpmv/ops.h"
#include "minicpmv/weights.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <sstream>
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

struct Request {
    int64_t max_new{0};
    std::vector<int32_t> input_ids;
};

Request parse_request(const std::string& line) {
    std::istringstream iss(line);
    std::string command;
    std::string ids_arg;
    Request req;
    iss >> command >> req.max_new >> ids_arg;
    if (command != "REQUEST" || req.max_new <= 0 || ids_arg.empty()) {
        throw std::runtime_error("expected: REQUEST <max_new> <comma_separated_input_ids>");
    }
    req.input_ids = parse_ids(ids_arg);
    if (req.input_ids.empty()) {
        throw std::runtime_error("request input_ids must not be empty");
    }
    return req;
}

bool write_line(const std::string& line) {
    std::cout << line << '\n';
    std::cout.flush();
    return static_cast<bool>(std::cout);
}

void usage(const char* argv0) {
    std::cerr << "usage: " << argv0 << " [--weights PATH] [--max-seq N] [--device-id N]\n";
}

class MiniCpm5Server {
public:
    MiniCpm5Server(std::string weights_path, int64_t max_seq_len, int device_id)
        : max_seq_len_(max_seq_len),
          ctx_(device_id),
          cfg_(default_minicpm5_1b_lm_config()) {
        if (max_seq_len_ <= 0) {
            throw std::runtime_error("max_seq must be positive");
        }

        WeightsIndex index(weights_path);
        weights_ = load_language_model_weights(index, cfg_);
        build_rope_tables(max_seq_len_, cfg_, cos_table_, sin_table_);
        state_ = make_decode_state(
            max_seq_len_, cfg_.num_layers,
            AttentionDecoderLayerConfig{cfg_.num_q_heads, cfg_.num_kv_heads,
                                        cfg_.head_dim, cfg_.rotary_dim, cfg_.rms_epsilon},
            ctx_.stream());
    }

    void handle(const Request& req) {
        if (static_cast<int64_t>(req.input_ids.size()) + req.max_new > max_seq_len_) {
            throw std::runtime_error("max_seq too small for prompt plus generation");
        }

        state_.seq_len = 0;
        Tensor prompt_hidden({static_cast<int64_t>(req.input_ids.size()), cfg_.hidden_size}, DType::Float16);
        prompt_hidden.allocate();
        embedding_lookup(weights_.embed, req.input_ids, prompt_hidden, ctx_.stream());

        Tensor last_hidden = prefill_from_embeddings(prompt_hidden, weights_, cfg_, cos_table_, sin_table_, state_, ctx_.stream());
        int64_t next_tok = lm_head_greedy_with_state(last_hidden, weights_, cfg_, state_, ctx_.stream());

        const std::vector<int64_t> eos_token_ids{1, 130073};
        int64_t generated = 0;
        while (generated < req.max_new) {
            write_line(std::to_string(next_tok));
            ++generated;
            if (is_eos(next_tok, eos_token_ids)) {
                break;
            }
            if (generated >= req.max_new || state_.seq_len >= state_.max_seq_len) {
                break;
            }
            next_tok = decode_step_greedy(static_cast<int32_t>(next_tok), weights_, cfg_, cos_table_, sin_table_, state_, ctx_.stream());
        }
        write_line("DONE");
    }

    void reset_state() { state_.seq_len = 0; }

private:
    int64_t max_seq_len_{0};
    AclContext ctx_;
    LanguageModelConfig cfg_;
    LanguageModelWeights weights_;
    Tensor cos_table_;
    Tensor sin_table_;
    DecodeState state_;
};

}  // namespace

int main(int argc, char** argv) {
    std::string weights_path = default_safetensors_path();
    int64_t max_seq_len = 4096;
    int device_id = 0;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--weights" && i + 1 < argc) {
            weights_path = argv[++i];
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
        MiniCpm5Server server(weights_path, max_seq_len, device_id);
        std::cerr << "# minicpm5_server ready\n";

        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) {
                continue;
            }
            if (line == "QUIT") {
                break;
            }
            try {
                Request req = parse_request(line);
                server.handle(req);
            } catch (const std::exception& e) {
                server.reset_state();
                write_line(std::string("ERROR ") + e.what());
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "minicpm5_server error: " << e.what() << '\n';
        return 1;
    }
}
