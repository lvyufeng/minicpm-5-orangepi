#pragma once

#include "minicpmv/decoder_layer.h"
#include "minicpmv/quantized_weight.h"
#include "minicpmv/tensor.h"
#include "minicpmv/weights.h"

#include <acl/acl.h>

#include <cstdint>
#include <string>
#include <vector>

namespace minicpmv {

struct LanguageModelConfig {
    int64_t hidden_size{1536};
    int64_t intermediate_size{4608};
    int64_t num_q_heads{16};
    int64_t num_kv_heads{2};
    int64_t head_dim{128};
    int64_t rotary_dim{128};
    double rope_theta{5000000.0};
    double rms_epsilon{1e-6};
    int64_t num_layers{24};
    int64_t vocab_size{130560};
    std::string model_prefix{"model"};
    std::string embed_weight_name{"model.embed_tokens.weight"};
    std::string final_norm_weight_name{"model.norm.weight"};
    std::string lm_head_weight_name{"lm_head.weight"};
};

LanguageModelConfig default_minicpm5_1b_lm_config();

struct LanguageModelLayerWeights {
    Tensor input_norm_w;
    Tensor post_norm_w;
    Tensor gate_w;
    Tensor up_w;
    Tensor down_w;
    Tensor q_w;
    Tensor k_w;
    Tensor v_w;
    Tensor o_w;

    W4A16QuantizedWeight gate_q;
    W4A16QuantizedWeight up_q;
    W4A16QuantizedWeight down_q;
    W4A16QuantizedWeight q_q;
    W4A16QuantizedWeight k_q;
    W4A16QuantizedWeight v_q;
    W4A16QuantizedWeight o_q;

    W8A8QuantizedWeight gate_w8;
    W8A8QuantizedWeight up_w8;
    W8A8QuantizedWeight down_w8;
    W8A8QuantizedWeight q_w8;
    W8A8QuantizedWeight k_w8;
    W8A8QuantizedWeight v_w8;
    W8A8QuantizedWeight o_w8;
};

struct LmHeadChunk {
    int64_t start_vocab{0};
    Tensor weight_kn;
    W8A8QuantizedWeight weight_w8;
};

struct LanguageModelWeights {
    Tensor embed;
    Tensor final_norm_w;
    std::vector<LanguageModelLayerWeights> layers;
    std::vector<LmHeadChunk> lm_head_chunks;
};

LanguageModelWeights load_language_model_weights(WeightsIndex& index,
                                                 const LanguageModelConfig& cfg);

void build_rope_tables(int64_t T,
                       const LanguageModelConfig& cfg,
                       Tensor& cos_table,
                       Tensor& sin_table);

Tensor prefill_from_embeddings(const Tensor& prompt_hidden,
                               const LanguageModelWeights& w,
                               const LanguageModelConfig& cfg,
                               const Tensor& cos_table,
                               const Tensor& sin_table,
                               DecodeState& state,
                               aclrtStream stream);

int64_t lm_head_greedy(const Tensor& last_hidden_1xH,
                       const LanguageModelWeights& w,
                       const LanguageModelConfig& cfg,
                       aclrtStream stream);

int64_t decode_step_greedy(int32_t token_id,
                           const LanguageModelWeights& w,
                           const LanguageModelConfig& cfg,
                           const Tensor& cos_table,
                           const Tensor& sin_table,
                           DecodeState& state,
                           aclrtStream stream);

bool is_eos(int64_t token_id, const std::vector<int64_t>& eos_token_ids);

}  // namespace minicpmv
