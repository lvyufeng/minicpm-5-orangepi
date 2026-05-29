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

void rms_norm(const Tensor& x, const Tensor& gamma, Tensor& out,
              double epsilon, aclrtStream stream);

void cast(const Tensor& self, Tensor& out, aclrtStream stream);

void apply_rope_partial(const Tensor& x,
                        const Tensor& cos_table,
                        const Tensor& sin_table,
                        const std::vector<int32_t>& row_to_t,
                        int64_t rot,
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
