#pragma once

#include "minicpmv/ops.h"
#include "minicpmv/quantized_weight.h"
#include "minicpmv/tensor.h"

#include <acl/acl.h>

#include <cstdint>
#include <vector>

namespace minicpmv {

struct AttentionDecoderLayerConfig {
    int64_t num_q_heads;
    int64_t num_kv_heads;
    int64_t head_dim;
    int64_t rotary_dim;
    double rms_epsilon;
};

struct AttentionDecoderLayerWeights {
    const Tensor* input_norm_weight;
    const Tensor* post_attention_norm_weight;
    const Tensor* q_proj_weight;
    const Tensor* k_proj_weight;
    const Tensor* v_proj_weight;
    const Tensor* o_proj_weight;
    const Tensor* gate_proj_weight;
    const Tensor* up_proj_weight;
    const Tensor* down_proj_weight;

    const W4A16QuantizedWeight* q_proj_q{nullptr};
    const W4A16QuantizedWeight* k_proj_q{nullptr};
    const W4A16QuantizedWeight* v_proj_q{nullptr};
    const W4A16QuantizedWeight* o_proj_q{nullptr};
    const W4A16QuantizedWeight* gate_proj_q{nullptr};
    const W4A16QuantizedWeight* up_proj_q{nullptr};
    const W4A16QuantizedWeight* down_proj_q{nullptr};

    const W8A8QuantizedWeight* q_proj_w8{nullptr};
    const W8A8QuantizedWeight* k_proj_w8{nullptr};
    const W8A8QuantizedWeight* v_proj_w8{nullptr};
    const W8A8QuantizedWeight* o_proj_w8{nullptr};
    const W8A8QuantizedWeight* gate_proj_w8{nullptr};
    const W8A8QuantizedWeight* up_proj_w8{nullptr};
    const W8A8QuantizedWeight* down_proj_w8{nullptr};
};

struct PrefillAttentionShared {
    Tensor scale;
    Tensor causal_mask;
    std::vector<int32_t> q_row_to_t;
    std::vector<int32_t> k_row_to_t;
};

struct PrefillLayerScratch {
    Tensor normed;
    Tensor q_full;
    Tensor k_full;
    Tensor v_full;
    Tensor q_heads;
    Tensor k_heads;
    Tensor q_rope;
    Tensor k_rope;
    RopeScratch q_rope_scratch;
    RopeScratch k_rope_scratch;
    Tensor attn_out;
    Tensor q_seq;
    Tensor k_seq;
    Tensor v_seq;
    Tensor scores;
    Tensor scaled_scores;
    Tensor masked_scores;
    Tensor probs;
    Tensor ctx_seq;
    Tensor attn_proj;
    Tensor after_attn;
    Tensor mlp_in;
    Tensor gate;
    Tensor up;
    Tensor gated;
    Tensor mlp_out;
};

struct AttentionLayerScratch {
    Tensor normed;
    Tensor q_full;
    Tensor k_full;
    Tensor v_full;
    Tensor normed_i8;
    Tensor normed_scale;
    Tensor q_heads;
    Tensor k_heads;
    Tensor q_rope;
    Tensor attn_out;
    Tensor attn_proj;
    Tensor after_attn;
    Tensor mlp_in;
    Tensor mlp_i8;
    Tensor mlp_scale;
    Tensor gate;
    Tensor up;
    Tensor gated;
    Tensor mlp_out;
};

struct AttentionLayerCache {
    Tensor k_cache;
    Tensor v_cache;
    AttentionLayerScratch scratch;
};

struct DecodeState {
    int64_t max_seq_len{0};
    int64_t seq_len{0};
    Tensor hidden_a;
    Tensor hidden_b;
    std::vector<AttentionLayerCache> layers;
};

DecodeState make_decode_state(int64_t max_seq_len,
                              int64_t num_layers,
                              const AttentionDecoderLayerConfig& config,
                              aclrtStream stream);

void build_prefill_attention_shared(const std::vector<int32_t>& row_to_t,
                                    const AttentionDecoderLayerConfig& config,
                                    PrefillAttentionShared& shared);

void attention_decoder_layer_prefill(const Tensor& hidden,
                                     const AttentionDecoderLayerWeights& weights,
                                     const Tensor& cos_table,
                                     const Tensor& sin_table,
                                     const std::vector<int32_t>& row_to_t,
                                     const PrefillAttentionShared& shared,
                                     const AttentionDecoderLayerConfig& config,
                                     AttentionLayerCache& cache,
                                     int64_t cache_offset,
                                     PrefillLayerScratch& scratch,
                                     int64_t layer_index,
                                     Tensor& out,
                                     aclrtStream stream);

void attention_decoder_layer_with_cache(const Tensor& hidden,
                                        const AttentionDecoderLayerWeights& weights,
                                        const Tensor& cos_table,
                                        const Tensor& sin_table,
                                        const std::vector<int32_t>& row_to_t,
                                        const AttentionDecoderLayerConfig& config,
                                        AttentionLayerCache& cache,
                                        Tensor& out,
                                        aclrtStream stream);

void attention_decoder_layer_step(const Tensor& hidden_1xH,
                                  const AttentionDecoderLayerWeights& weights,
                                  const Tensor& cos_table,
                                  const Tensor& sin_table,
                                  int32_t pos,
                                  int64_t cache_len,
                                  const AttentionDecoderLayerConfig& config,
                                  AttentionLayerCache& cache,
                                  Tensor& out_1xH,
                                  aclrtStream stream);

}  // namespace minicpmv
