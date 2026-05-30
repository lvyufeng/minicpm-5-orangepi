#pragma once

#include "minicpmv/tensor.h"

#include <acl/acl.h>

#include <cstdint>
#include <vector>

namespace minicpmv {

void embedding_lookup(const Tensor& weight,
                      const std::vector<int32_t>& host_ids,
                      Tensor& out,
                      aclrtStream stream);

void matmul(const Tensor& a, const Tensor& b, Tensor& out, aclrtStream stream);

void matmul_b_transposed(const Tensor& a, const Tensor& b, Tensor& out, aclrtStream stream);

void argmax_last_dim(const Tensor& self, Tensor& out, aclrtStream stream);

void incre_flash_attention(const Tensor& query,
                           const Tensor& k_cache,
                           const Tensor& v_cache,
                           int64_t context,
                           int64_t num_q_heads,
                           int64_t num_kv_heads,
                           int64_t head_dim,
                           float scale,
                           Tensor& out,
                           aclrtStream stream);

void silu_mul(const Tensor& gate, const Tensor& up, Tensor& out, aclrtStream stream);

void add(const Tensor& a, const Tensor& b, Tensor& out, aclrtStream stream);

void mul(const Tensor& a, const Tensor& b, Tensor& out, aclrtStream stream);

void silu(const Tensor& self, Tensor& out, aclrtStream stream);

void sigmoid(const Tensor& self, Tensor& out, aclrtStream stream);

void softmax_last_dim(const Tensor& self, Tensor& out, aclrtStream stream);

struct RmsNormScratch {
    Tensor x_f32;
    Tensor gamma_f32;
    Tensor x_sq;
    Tensor mean_x_sq;
    Tensor rstd;
    Tensor scaled;
    Tensor normed_f32;
};

void rms_norm(const Tensor& x, const Tensor& gamma, Tensor& out,
              double epsilon, aclrtStream stream);

void rms_norm_with_scratch(const Tensor& x, const Tensor& gamma, Tensor& out,
                           double epsilon, RmsNormScratch& scratch, aclrtStream stream);

void cast(const Tensor& self, Tensor& out, aclrtStream stream);

struct RopeScratch {
    Tensor x1;
    Tensor x2;
    Tensor cos_e;
    Tensor sin_e;
    Tensor a;
    Tensor b;
    Tensor y1;
    Tensor y2;
};

void apply_rope_partial(const Tensor& x,
                        const Tensor& cos_table,
                        const Tensor& sin_table,
                        const std::vector<int32_t>& row_to_t,
                        int64_t rot,
                        Tensor& out,
                        aclrtStream stream);

void apply_rope_partial_with_scratch(const Tensor& x,
                                     const Tensor& cos_table,
                                     const Tensor& sin_table,
                                     const std::vector<int32_t>& row_to_t,
                                     int64_t rot,
                                     RopeScratch& scratch,
                                     Tensor& out,
                                     aclrtStream stream);

// Prefill-only fused RoPE custom op. x is [N, head_dim] laid out token-major as
// row = t*heads + h, so the token index is t = row / heads (no index tensor needed).
// Applies the same partial RoPE as apply_rope_partial in a single kernel launch.
void apply_rope_prefill(const Tensor& x,
                        const Tensor& cos_table,
                        const Tensor& sin_table,
                        int64_t heads,
                        int64_t rotary_dim,
                        Tensor& out,
                        aclrtStream stream);

// Prefill causal self-attention custom op. Replaces the per-head host loop.
// q_rope: [T*num_q_heads, head_dim], k_rope: [T*num_kv_heads, head_dim],
// v_full: [T, kv_dim]. Output: [T, q_dim]. Causal via loop bound (tk<=tq).
// Max T supported: 256; caller must fall back to host path for longer sequences.
void prefill_attention_custom(const Tensor& q_rope,
                              const Tensor& k_rope,
                              const Tensor& v_full,
                              int64_t seq_len,
                              int64_t num_q_heads,
                              int64_t num_kv_heads,
                              int64_t head_dim,
                              float scale,
                              Tensor& out,
                              aclrtStream stream);

void logits_top1(const Tensor& logits,
                 int64_t valid,
                 Tensor& value,
                 Tensor& index,
                 aclrtStream stream);

void rope_cache_write(const Tensor& q,
                      const Tensor& k,
                      const Tensor& v,
                      Tensor& k_cache,
                      Tensor& v_cache,
                      const Tensor& cos_table,
                      const Tensor& sin_table,
                      int64_t pos,
                      int64_t cache_len,
                      int64_t num_q_heads,
                      int64_t num_kv_heads,
                      int64_t head_dim,
                      int64_t rotary_dim,
                      Tensor& q_rope,
                      aclrtStream stream);

void matmul_w4a16(const Tensor& x,
                  const Tensor& w_int8,
                  const Tensor& scales,
                  Tensor& out,
                  aclrtStream stream);

void matmul_w8a8_i32(const Tensor& x,
                     const Tensor& w_int8,
                     Tensor& out,
                     aclrtStream stream);

void w8a8_quantize(const Tensor& x,
                   Tensor& x_int8,
                   Tensor& x_scale,
                   aclrtStream stream);

void w8a8_dequant(const Tensor& acc,
                  const Tensor& x_scale,
                  const Tensor& w_scale,
                  Tensor& out,
                  aclrtStream stream);

}  // namespace minicpmv
