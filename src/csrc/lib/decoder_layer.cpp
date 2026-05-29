#include "minicpmv/decoder_layer.h"

#include "minicpmv/acl_context.h"
#include "minicpmv/ops.h"

#include <acl/acl.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace minicpmv {
namespace {

bool w8a8_decode_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("MINICPM_W8A8_DECODE");
        return v != nullptr && std::string(v) != "0" && std::string(v) != "false";
    }();
    return enabled;
}

uint16_t f32_to_f16_bits(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;
    if (exp <= 0) return static_cast<uint16_t>(sign);
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

void check_ptr(const Tensor* t, const char* name) {
    if (t == nullptr) {
        throw std::runtime_error(std::string("missing decoder layer weight: ") + name);
    }
}

void copy_col_block(const Tensor& src, int64_t col_offset, Tensor& dst, aclrtStream stream) {
    const int64_t rows = src.shape()[0];
    const int64_t src_cols = src.shape()[1];
    const int64_t dst_cols = dst.shape()[1];
    const size_t elem = dtype_size(src.dtype());
    const size_t src_row_bytes = static_cast<size_t>(src_cols) * elem;
    const size_t dst_row_bytes = static_cast<size_t>(dst_cols) * elem;
    const size_t block_bytes = static_cast<size_t>(dst_cols) * elem;
    auto* s = static_cast<const uint8_t*>(src.data());
    auto* d = static_cast<uint8_t*>(dst.data());
    for (int64_t r = 0; r < rows; ++r) {
        check_acl(aclrtMemcpyAsync(d + static_cast<size_t>(r) * dst_row_bytes, block_bytes,
                                   s + static_cast<size_t>(r) * src_row_bytes + static_cast<size_t>(col_offset) * elem,
                                   block_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                  "copy_col_block");
    }
}

void copy_head_to_seq(const Tensor& src_heads, int64_t head, int64_t heads_per_token,
                      Tensor& dst_seq, aclrtStream stream) {
    const int64_t tokens = dst_seq.shape()[0];
    const int64_t dim = dst_seq.shape()[1];
    const size_t row_bytes = static_cast<size_t>(dim) * dtype_size(src_heads.dtype());
    auto* s = static_cast<const uint8_t*>(src_heads.data());
    auto* d = static_cast<uint8_t*>(dst_seq.data());
    for (int64_t t = 0; t < tokens; ++t) {
        const int64_t src_row = t * heads_per_token + head;
        check_acl(aclrtMemcpyAsync(d + static_cast<size_t>(t) * row_bytes, row_bytes,
                                   s + static_cast<size_t>(src_row) * row_bytes, row_bytes,
                                   ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                  "copy_head_to_seq");
    }
}

void copy_seq_to_head_block(const Tensor& src_seq, Tensor& dst, int64_t col_offset,
                            aclrtStream stream) {
    const int64_t rows = src_seq.shape()[0];
    const int64_t dst_cols = dst.shape()[1];
    const int64_t src_cols = src_seq.shape()[1];
    const size_t elem = dtype_size(src_seq.dtype());
    const size_t src_row_bytes = static_cast<size_t>(src_cols) * elem;
    const size_t dst_row_bytes = static_cast<size_t>(dst_cols) * elem;
    auto* s = static_cast<const uint8_t*>(src_seq.data());
    auto* d = static_cast<uint8_t*>(dst.data());
    for (int64_t r = 0; r < rows; ++r) {
        check_acl(aclrtMemcpyAsync(d + static_cast<size_t>(r) * dst_row_bytes + static_cast<size_t>(col_offset) * elem,
                                   src_row_bytes,
                                   s + static_cast<size_t>(r) * src_row_bytes,
                                   src_row_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                  "copy_seq_to_head_block");
    }
}

void copy_heads_from_cols(const Tensor& src, int64_t heads, int64_t head_dim,
                          Tensor& dst, aclrtStream stream) {
    const int64_t rows = src.shape()[0];
    const int64_t src_cols = src.shape()[1];
    const size_t elem = dtype_size(src.dtype());
    const size_t src_row_bytes = static_cast<size_t>(src_cols) * elem;
    const size_t head_bytes = static_cast<size_t>(head_dim) * elem;
    auto* s = static_cast<const uint8_t*>(src.data());
    auto* d = static_cast<uint8_t*>(dst.data());
    for (int64_t t = 0; t < rows; ++t) {
        for (int64_t h = 0; h < heads; ++h) {
            check_acl(aclrtMemcpyAsync(d + static_cast<size_t>(t * heads + h) * head_bytes, head_bytes,
                                       s + static_cast<size_t>(t) * src_row_bytes + static_cast<size_t>(h * head_dim) * elem,
                                       head_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                      "copy_heads_from_cols");
        }
    }
}

void pack_heads_to_row(const Tensor& src_heads, Tensor& dst_row, int64_t heads,
                       int64_t head_dim, aclrtStream stream) {
    const size_t elem = dtype_size(src_heads.dtype());
    const size_t head_bytes = static_cast<size_t>(head_dim) * elem;
    auto* s = static_cast<const uint8_t*>(src_heads.data());
    auto* d = static_cast<uint8_t*>(dst_row.data());
    for (int64_t h = 0; h < heads; ++h) {
        check_acl(aclrtMemcpyAsync(d + static_cast<size_t>(h) * head_bytes, head_bytes,
                                   s + static_cast<size_t>(h) * head_bytes, head_bytes,
                                   ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                  "pack_heads_to_row");
    }
}

void copy_tensor_to_cache_row(const Tensor& src, Tensor& cache, int64_t row, aclrtStream stream) {
    const size_t row_bytes = static_cast<size_t>(cache.shape()[1]) * dtype_size(cache.dtype());
    auto* d = static_cast<uint8_t*>(cache.data()) + static_cast<size_t>(row) * row_bytes;
    check_acl(aclrtMemcpyAsync(d, row_bytes, src.data(), row_bytes,
                               ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
              "copy_tensor_to_cache_row");
}

void copy_matrix_rows(const Tensor& src, int64_t src_row, Tensor& dst, int64_t dst_row,
                      int64_t rows, aclrtStream stream) {
    const size_t row_bytes = static_cast<size_t>(src.shape()[1]) * dtype_size(src.dtype());
    auto* s = static_cast<const uint8_t*>(src.data());
    auto* d = static_cast<uint8_t*>(dst.data());
    for (int64_t r = 0; r < rows; ++r) {
        check_acl(aclrtMemcpyAsync(d + static_cast<size_t>(dst_row + r) * row_bytes, row_bytes,
                                   s + static_cast<size_t>(src_row + r) * row_bytes, row_bytes,
                                   ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                  "copy_matrix_rows");
    }
}

bool matmul_shape_ok(const Tensor* t, int64_t N, int64_t K) {
    const auto& s = t->shape();
    return s == std::vector<int64_t>{N, K} || s == std::vector<int64_t>{K, N};
}

bool w8a8_weight_ready(const W8A8QuantizedWeight* w8_weight) {
    return w8a8_decode_enabled() && w8_weight != nullptr && w8_weight->w_int8.data() != nullptr;
}

void matmul_decode_w8a8_prequant(const Tensor& x_int8,
                                 const Tensor& x_scale,
                                 const W8A8QuantizedWeight& w8_weight,
                                 Tensor& out,
                                 aclrtStream stream) {
    Tensor acc({1, w8_weight.N}, DType::Int32); acc.allocate();
    matmul_w8a8_i32(x_int8, w8_weight.w_int8, acc, stream);
    w8a8_dequant(acc, x_scale, w8_weight.w_scale, out, stream);
}

void matmul_decode_dispatch(const Tensor& x,
                            const Tensor* dense_weight,
                            const W4A16QuantizedWeight* quant_weight,
                            const W8A8QuantizedWeight* w8_weight,
                            Tensor& out,
                            aclrtStream stream) {
    if (w8a8_weight_ready(w8_weight) && x.shape().size() == 2 && x.shape()[0] == 1) {
        Tensor x_int8(x.shape(), DType::Int8); x_int8.allocate();
        Tensor x_scale({1}, DType::Float16); x_scale.allocate();
        w8a8_quantize(x, x_int8, x_scale, stream);
        matmul_decode_w8a8_prequant(x_int8, x_scale, *w8_weight, out, stream);
        return;
    }
    if (quant_weight != nullptr && x.shape().size() == 2 && x.shape()[0] == 1) {
        matmul_w4a16(x, quant_weight->w_int8, quant_weight->scales, out, stream);
        return;
    }
    matmul_b_transposed(x, *dense_weight, out, stream);
}

int64_t infer_intermediate(const Tensor& gate_proj_weight, int64_t hidden) {
    const auto& s = gate_proj_weight.shape();
    if (s.size() != 2) {
        throw std::runtime_error("decoder layer MLP weight must be 2D");
    }
    if (s[1] == hidden) return s[0];
    if (s[0] == hidden) return s[1];
    throw std::runtime_error("decoder layer MLP weight hidden dim mismatch");
}

void validate_shapes(const Tensor& hidden,
                     const AttentionDecoderLayerWeights& w,
                     const AttentionDecoderLayerConfig& c,
                     const Tensor& out) {
    check_ptr(w.input_norm_weight, "input_norm_weight");
    check_ptr(w.post_attention_norm_weight, "post_attention_norm_weight");
    check_ptr(w.q_proj_weight, "q_proj_weight");
    check_ptr(w.k_proj_weight, "k_proj_weight");
    check_ptr(w.v_proj_weight, "v_proj_weight");
    check_ptr(w.o_proj_weight, "o_proj_weight");
    check_ptr(w.gate_proj_weight, "gate_proj_weight");
    check_ptr(w.up_proj_weight, "up_proj_weight");
    check_ptr(w.down_proj_weight, "down_proj_weight");

    if (hidden.shape().size() != 2 || out.shape() != hidden.shape()) {
        throw std::runtime_error("decoder layer hidden/out must be [T, H] and same shape");
    }
    if (hidden.dtype() != DType::Float16 || out.dtype() != DType::Float16) {
        throw std::runtime_error("decoder layer requires fp16 hidden/out");
    }
    if (c.num_q_heads <= 0 || c.num_kv_heads <= 0 || c.head_dim <= 0 ||
        c.rotary_dim <= 0 || c.rotary_dim > c.head_dim || (c.rotary_dim % 2) != 0) {
        throw std::runtime_error("decoder layer invalid config dims");
    }
    if (c.num_q_heads % c.num_kv_heads != 0) {
        throw std::runtime_error("decoder layer num_q_heads must be divisible by num_kv_heads");
    }

    const int64_t hidden_size = hidden.shape()[1];
    const int64_t q_dim = c.num_q_heads * c.head_dim;
    const int64_t kv_dim = c.num_kv_heads * c.head_dim;
    const int64_t intermediate = infer_intermediate(*w.gate_proj_weight, hidden_size);

    if (w.input_norm_weight->shape() != std::vector<int64_t>{hidden_size} ||
        w.post_attention_norm_weight->shape() != std::vector<int64_t>{hidden_size} ||
        !matmul_shape_ok(w.q_proj_weight, q_dim, hidden_size) ||
        !matmul_shape_ok(w.k_proj_weight, kv_dim, hidden_size) ||
        !matmul_shape_ok(w.v_proj_weight, kv_dim, hidden_size) ||
        !matmul_shape_ok(w.o_proj_weight, hidden_size, q_dim) ||
        !matmul_shape_ok(w.gate_proj_weight, intermediate, hidden_size) ||
        !matmul_shape_ok(w.up_proj_weight, intermediate, hidden_size) ||
        !matmul_shape_ok(w.down_proj_weight, hidden_size, intermediate)) {
        throw std::runtime_error("decoder layer weight shape mismatch");
    }
}

void run_attention_core(const Tensor& hidden,
                        const AttentionDecoderLayerWeights& weights,
                        const Tensor& cos_table,
                        const Tensor& sin_table,
                        const std::vector<int32_t>& row_to_t,
                        const AttentionDecoderLayerConfig& config,
                        AttentionLayerCache* cache,
                        int64_t cache_offset,
                        Tensor& out,
                        aclrtStream stream) {
    validate_shapes(hidden, weights, config, out);

    const int64_t T = hidden.shape()[0];
    const int64_t hidden_size = hidden.shape()[1];
    const int64_t num_q_heads = config.num_q_heads;
    const int64_t num_kv_heads = config.num_kv_heads;
    const int64_t q_per_kv = num_q_heads / num_kv_heads;
    const int64_t head_dim = config.head_dim;
    const int64_t q_dim = num_q_heads * head_dim;
    const int64_t kv_dim = num_kv_heads * head_dim;
    const int64_t intermediate = infer_intermediate(*weights.gate_proj_weight, hidden_size);

    if (static_cast<int64_t>(row_to_t.size()) != T) {
        throw std::runtime_error("decoder layer row_to_t size must match sequence length");
    }
    if (cos_table.shape().size() != 2 || sin_table.shape() != cos_table.shape() ||
        cos_table.shape()[1] != config.rotary_dim / 2) {
        throw std::runtime_error("decoder layer RoPE table shape mismatch");
    }
    if (cache != nullptr) {
        if (cache->k_cache.shape().size() != 2 || cache->v_cache.shape() != cache->k_cache.shape() ||
            cache->k_cache.shape()[1] != kv_dim) {
            throw std::runtime_error("attention cache KV dim mismatch");
        }
        if (cache_offset < 0 || cache_offset + T > cache->k_cache.shape()[0]) {
            throw std::runtime_error("attention cache overflow");
        }
    }

    Tensor normed({T, hidden_size}, DType::Float16); normed.allocate();
    rms_norm(hidden, *weights.input_norm_weight, normed, config.rms_epsilon, stream);

    Tensor q_full({T, q_dim}, DType::Float16); q_full.allocate();
    Tensor k_full({T, kv_dim}, DType::Float16); k_full.allocate();
    Tensor v_full({T, kv_dim}, DType::Float16); v_full.allocate();
    matmul_b_transposed(normed, *weights.q_proj_weight, q_full, stream);
    matmul_b_transposed(normed, *weights.k_proj_weight, k_full, stream);
    matmul_b_transposed(normed, *weights.v_proj_weight, v_full, stream);

    Tensor q_heads({T * num_q_heads, head_dim}, DType::Float16); q_heads.allocate();
    Tensor k_heads({T * num_kv_heads, head_dim}, DType::Float16); k_heads.allocate();
    copy_heads_from_cols(q_full, num_q_heads, head_dim, q_heads, stream);
    copy_heads_from_cols(k_full, num_kv_heads, head_dim, k_heads, stream);

    std::vector<int32_t> q_row_to_t(static_cast<size_t>(T * num_q_heads));
    std::vector<int32_t> k_row_to_t(static_cast<size_t>(T * num_kv_heads));
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t h = 0; h < num_q_heads; ++h) q_row_to_t[static_cast<size_t>(t * num_q_heads + h)] = row_to_t[t];
        for (int64_t h = 0; h < num_kv_heads; ++h) k_row_to_t[static_cast<size_t>(t * num_kv_heads + h)] = row_to_t[t];
    }

    Tensor q_rope({T * num_q_heads, head_dim}, DType::Float16); q_rope.allocate();
    Tensor k_rope({T * num_kv_heads, head_dim}, DType::Float16); k_rope.allocate();
    apply_rope_partial(q_heads, cos_table, sin_table, q_row_to_t, config.rotary_dim, q_rope, stream);
    apply_rope_partial(k_heads, cos_table, sin_table, k_row_to_t, config.rotary_dim, k_rope, stream);

    if (cache != nullptr) {
        const size_t elem = dtype_size(k_rope.dtype());
        const size_t head_bytes = static_cast<size_t>(head_dim) * elem;
        const size_t row_bytes = static_cast<size_t>(kv_dim) * elem;
        auto* src = static_cast<const uint8_t*>(k_rope.data());
        auto* dst = static_cast<uint8_t*>(cache->k_cache.data());
        for (int64_t t = 0; t < T; ++t) {
            for (int64_t h = 0; h < num_kv_heads; ++h) {
                check_acl(aclrtMemcpyAsync(dst + static_cast<size_t>(cache_offset + t) * row_bytes
                                               + static_cast<size_t>(h) * head_bytes,
                                           head_bytes,
                                           src + static_cast<size_t>(t * num_kv_heads + h) * head_bytes,
                                           head_bytes,
                                           ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                          "k_rope -> k_cache");
            }
        }
        auto* vs = static_cast<const uint8_t*>(v_full.data());
        auto* vd = static_cast<uint8_t*>(cache->v_cache.data());
        for (int64_t t = 0; t < T; ++t) {
            check_acl(aclrtMemcpyAsync(vd + static_cast<size_t>(cache_offset + t) * row_bytes,
                                       row_bytes,
                                       vs + static_cast<size_t>(t) * row_bytes,
                                       row_bytes,
                                       ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                      "v_full -> v_cache");
        }
        check_acl(aclrtSynchronizeStream(stream), "kv cache write sync");
    }

    Tensor scale({T, T}, DType::Float16);
    std::vector<uint16_t> scale_host(static_cast<size_t>(T * T),
                                     f32_to_f16_bits(1.0f / std::sqrt(static_cast<float>(head_dim))));
    scale.copy_from_host(scale_host.data(), scale_host.size() * sizeof(uint16_t));

    std::vector<uint16_t> mask_host(static_cast<size_t>(T * T));
    for (int64_t r = 0; r < T; ++r) {
        for (int64_t c = 0; c < T; ++c) {
            mask_host[static_cast<size_t>(r * T + c)] =
                f32_to_f16_bits(row_to_t[c] <= row_to_t[r] ? 0.0f : -65504.0f);
        }
    }
    Tensor causal_mask({T, T}, DType::Float16);
    causal_mask.copy_from_host(mask_host.data(), mask_host.size() * sizeof(uint16_t));

    Tensor attn_out({T, q_dim}, DType::Float16); attn_out.allocate();
    Tensor q_seq({T, head_dim}, DType::Float16); q_seq.allocate();
    Tensor k_seq({T, head_dim}, DType::Float16); k_seq.allocate();
    Tensor v_seq({T, head_dim}, DType::Float16); v_seq.allocate();
    Tensor scores({T, T}, DType::Float16); scores.allocate();
    Tensor scaled_scores({T, T}, DType::Float16); scaled_scores.allocate();
    Tensor masked_scores({T, T}, DType::Float16); masked_scores.allocate();
    Tensor probs({T, T}, DType::Float16); probs.allocate();
    Tensor ctx_seq({T, head_dim}, DType::Float16); ctx_seq.allocate();

    for (int64_t qh = 0; qh < num_q_heads; ++qh) {
        const int64_t kvh = qh / q_per_kv;
        copy_head_to_seq(q_rope, qh, num_q_heads, q_seq, stream);
        copy_head_to_seq(k_rope, kvh, num_kv_heads, k_seq, stream);
        copy_col_block(v_full, kvh * head_dim, v_seq, stream);
        matmul_b_transposed(q_seq, k_seq, scores, stream);
        mul(scores, scale, scaled_scores, stream);
        add(scaled_scores, causal_mask, masked_scores, stream);
        softmax_last_dim(masked_scores, probs, stream);
        matmul(probs, v_seq, ctx_seq, stream);
        copy_seq_to_head_block(ctx_seq, attn_out, qh * head_dim, stream);
    }

    Tensor attn_proj({T, hidden_size}, DType::Float16); attn_proj.allocate();
    matmul_b_transposed(attn_out, *weights.o_proj_weight, attn_proj, stream);

    Tensor after_attn({T, hidden_size}, DType::Float16); after_attn.allocate();
    add(hidden, attn_proj, after_attn, stream);

    Tensor mlp_in({T, hidden_size}, DType::Float16); mlp_in.allocate();
    rms_norm(after_attn, *weights.post_attention_norm_weight, mlp_in, config.rms_epsilon, stream);

    Tensor gate({T, intermediate}, DType::Float16); gate.allocate();
    Tensor up({T, intermediate}, DType::Float16); up.allocate();
    Tensor gated({T, intermediate}, DType::Float16); gated.allocate();
    Tensor mlp_out({T, hidden_size}, DType::Float16); mlp_out.allocate();

    matmul_b_transposed(mlp_in, *weights.gate_proj_weight, gate, stream);
    matmul_b_transposed(mlp_in, *weights.up_proj_weight, up, stream);
    silu_mul(gate, up, gated, stream);
    matmul_b_transposed(gated, *weights.down_proj_weight, mlp_out, stream);
    add(after_attn, mlp_out, out, stream);
}

}  // namespace

DecodeState make_decode_state(int64_t max_seq_len,
                              int64_t num_layers,
                              const AttentionDecoderLayerConfig& config,
                              aclrtStream stream) {
    if (max_seq_len <= 0) {
        throw std::runtime_error("decode state max_seq_len must be positive");
    }
    if (num_layers <= 0) {
        throw std::runtime_error("decode state num_layers must be positive");
    }
    if (config.num_kv_heads <= 0 || config.head_dim <= 0) {
        throw std::runtime_error("decode state invalid attention config");
    }

    DecodeState state;
    state.max_seq_len = max_seq_len;
    state.seq_len = 0;
    state.layers.reserve(static_cast<size_t>(num_layers));

    const int64_t kv_dim = config.num_kv_heads * config.head_dim;
    for (int64_t layer = 0; layer < num_layers; ++layer) {
        AttentionLayerCache cache;
        cache.k_cache = Tensor({max_seq_len, kv_dim}, DType::Float16);
        cache.v_cache = Tensor({max_seq_len, kv_dim}, DType::Float16);
        cache.k_cache.allocate();
        cache.v_cache.allocate();
        check_acl(aclrtMemsetAsync(cache.k_cache.data(), cache.k_cache.size_bytes(), 0,
                                   cache.k_cache.size_bytes(), stream), "memset attention k cache");
        check_acl(aclrtMemsetAsync(cache.v_cache.data(), cache.v_cache.size_bytes(), 0,
                                   cache.v_cache.size_bytes(), stream), "memset attention v cache");
        state.layers.push_back(std::move(cache));
    }
    check_acl(aclrtSynchronizeStream(stream), "make_decode_state sync");
    return state;
}

namespace {

void ensure_step_scratch(AttentionLayerScratch& s,
                         int64_t hidden_size,
                         int64_t q_dim,
                         int64_t kv_dim,
                         int64_t num_q_heads,
                         int64_t num_kv_heads,
                         int64_t head_dim,
                         int64_t intermediate) {
    if (s.normed.data() != nullptr) {
        return;
    }
    auto make = [](std::vector<int64_t> shape) {
        Tensor t(std::move(shape), DType::Float16);
        t.allocate();
        return t;
    };
    s.normed = make({1, hidden_size});
    s.q_heads = make({num_q_heads, head_dim});
    s.k_heads = make({num_kv_heads, head_dim});
    s.q_rope = make({num_q_heads, head_dim});
    s.attn_out = make({1, q_dim});
    s.attn_proj = make({1, hidden_size});
    s.after_attn = make({1, hidden_size});
    s.mlp_in = make({1, hidden_size});
    s.gate = make({1, intermediate});
    s.up = make({1, intermediate});
    s.gated = make({1, intermediate});
    s.mlp_out = make({1, hidden_size});
}

}  // namespace

void attention_decoder_layer_with_cache(const Tensor& hidden,
                                        const AttentionDecoderLayerWeights& weights,
                                        const Tensor& cos_table,
                                        const Tensor& sin_table,
                                        const std::vector<int32_t>& row_to_t,
                                        const AttentionDecoderLayerConfig& config,
                                        AttentionLayerCache& cache,
                                        Tensor& out,
                                        aclrtStream stream) {
    run_attention_core(hidden, weights, cos_table, sin_table, row_to_t, config,
                       &cache, 0, out, stream);
}

void attention_decoder_layer_step(const Tensor& hidden,
                                  const AttentionDecoderLayerWeights& weights,
                                  const Tensor& cos_table,
                                  const Tensor& sin_table,
                                  int32_t pos,
                                  int64_t cache_len,
                                  const AttentionDecoderLayerConfig& config,
                                  AttentionLayerCache& cache,
                                  Tensor& out,
                                  aclrtStream stream) {
    validate_shapes(hidden, weights, config, out);
    if (hidden.shape()[0] != 1) {
        throw std::runtime_error("attention_decoder_layer_step hidden must be [1, H]");
    }
    if (cache_len < 0 || cache_len >= cache.k_cache.shape()[0]) {
        throw std::runtime_error("attention_decoder_layer_step cache_len out of range");
    }

    const int64_t hidden_size = hidden.shape()[1];
    const int64_t num_q_heads = config.num_q_heads;
    const int64_t num_kv_heads = config.num_kv_heads;
    const int64_t head_dim = config.head_dim;
    const int64_t q_dim = num_q_heads * head_dim;
    const int64_t kv_dim = num_kv_heads * head_dim;
    const int64_t intermediate = infer_intermediate(*weights.gate_proj_weight, hidden_size);
    const int64_t context = cache_len + 1;

    if (cache.k_cache.shape().size() != 2 ||
        cache.k_cache.shape() != std::vector<int64_t>{cache.k_cache.shape()[0], kv_dim} ||
        cache.v_cache.shape() != cache.k_cache.shape()) {
        throw std::runtime_error("attention_decoder_layer_step cache shape mismatch");
    }
    if (cos_table.shape().size() != 2 || sin_table.shape() != cos_table.shape() ||
        cos_table.shape()[0] <= pos || cos_table.shape()[1] != config.rotary_dim / 2) {
        throw std::runtime_error("attention_decoder_layer_step RoPE table shape mismatch");
    }

    ensure_step_scratch(cache.scratch, hidden_size, q_dim, kv_dim,
                        num_q_heads, num_kv_heads, head_dim, intermediate);
    auto& s = cache.scratch;

    Tensor& normed = s.normed;
    rms_norm(hidden, *weights.input_norm_weight, normed, config.rms_epsilon, stream);

    Tensor q_full({1, q_dim}, DType::Float16); q_full.allocate();
    Tensor k_full({1, kv_dim}, DType::Float16); k_full.allocate();
    Tensor v_full({1, kv_dim}, DType::Float16); v_full.allocate();

    if (w8a8_weight_ready(weights.q_proj_w8) || w8a8_weight_ready(weights.k_proj_w8) ||
        w8a8_weight_ready(weights.v_proj_w8)) {
        Tensor normed_i8(normed.shape(), DType::Int8); normed_i8.allocate();
        Tensor normed_scale({1}, DType::Float16); normed_scale.allocate();
        w8a8_quantize(normed, normed_i8, normed_scale, stream);
        if (w8a8_weight_ready(weights.q_proj_w8)) {
            matmul_decode_w8a8_prequant(normed_i8, normed_scale, *weights.q_proj_w8, q_full, stream);
        } else {
            matmul_decode_dispatch(normed, weights.q_proj_weight, weights.q_proj_q, nullptr, q_full, stream);
        }
        if (w8a8_weight_ready(weights.k_proj_w8)) {
            matmul_decode_w8a8_prequant(normed_i8, normed_scale, *weights.k_proj_w8, k_full, stream);
        } else {
            matmul_decode_dispatch(normed, weights.k_proj_weight, weights.k_proj_q, nullptr, k_full, stream);
        }
        if (w8a8_weight_ready(weights.v_proj_w8)) {
            matmul_decode_w8a8_prequant(normed_i8, normed_scale, *weights.v_proj_w8, v_full, stream);
        } else {
            matmul_decode_dispatch(normed, weights.v_proj_weight, weights.v_proj_q, nullptr, v_full, stream);
        }
    } else {
        matmul_decode_dispatch(normed, weights.q_proj_weight, weights.q_proj_q, nullptr, q_full, stream);
        matmul_decode_dispatch(normed, weights.k_proj_weight, weights.k_proj_q, nullptr, k_full, stream);
        matmul_decode_dispatch(normed, weights.v_proj_weight, weights.v_proj_q, nullptr, v_full, stream);
    }

    Tensor& q_heads = s.q_heads;
    Tensor& k_heads = s.k_heads;
    copy_heads_from_cols(q_full, num_q_heads, head_dim, q_heads, stream);
    copy_heads_from_cols(k_full, num_kv_heads, head_dim, k_heads, stream);

    Tensor& q_rope = s.q_rope;
    rope_cache_write(q_heads, k_heads, v_full, cache.k_cache, cache.v_cache,
                     cos_table, sin_table, pos, cache_len,
                     num_q_heads, num_kv_heads, head_dim, config.rotary_dim,
                     q_rope, stream);

    Tensor& attn_out = s.attn_out;
    const float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    incre_flash_attention(q_rope, cache.k_cache, cache.v_cache,
                          context, num_q_heads, num_kv_heads, head_dim,
                          attn_scale, attn_out, stream);

    Tensor& attn_proj = s.attn_proj;
    matmul_decode_dispatch(attn_out, weights.o_proj_weight, weights.o_proj_q, weights.o_proj_w8, attn_proj, stream);

    Tensor& after_attn = s.after_attn;
    add(hidden, attn_proj, after_attn, stream);

    Tensor& mlp_in = s.mlp_in;
    rms_norm(after_attn, *weights.post_attention_norm_weight, mlp_in, config.rms_epsilon, stream);

    Tensor& gate = s.gate;
    Tensor& up = s.up;
    Tensor& gated = s.gated;
    Tensor& mlp_out = s.mlp_out;

    if (w8a8_weight_ready(weights.gate_proj_w8) || w8a8_weight_ready(weights.up_proj_w8)) {
        Tensor mlp_i8(mlp_in.shape(), DType::Int8); mlp_i8.allocate();
        Tensor mlp_scale({1}, DType::Float16); mlp_scale.allocate();
        w8a8_quantize(mlp_in, mlp_i8, mlp_scale, stream);
        if (w8a8_weight_ready(weights.gate_proj_w8)) {
            matmul_decode_w8a8_prequant(mlp_i8, mlp_scale, *weights.gate_proj_w8, gate, stream);
        } else {
            matmul_decode_dispatch(mlp_in, weights.gate_proj_weight, weights.gate_proj_q, nullptr, gate, stream);
        }
        if (w8a8_weight_ready(weights.up_proj_w8)) {
            matmul_decode_w8a8_prequant(mlp_i8, mlp_scale, *weights.up_proj_w8, up, stream);
        } else {
            matmul_decode_dispatch(mlp_in, weights.up_proj_weight, weights.up_proj_q, nullptr, up, stream);
        }
    } else {
        matmul_decode_dispatch(mlp_in, weights.gate_proj_weight, weights.gate_proj_q, nullptr, gate, stream);
        matmul_decode_dispatch(mlp_in, weights.up_proj_weight, weights.up_proj_q, nullptr, up, stream);
    }
    silu_mul(gate, up, gated, stream);
    matmul_decode_dispatch(gated, weights.down_proj_weight, weights.down_proj_q, weights.down_proj_w8, mlp_out, stream);
    add(after_attn, mlp_out, out, stream);
}

}  // namespace minicpmv
