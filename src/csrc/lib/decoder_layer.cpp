#include "minicpmv/decoder_layer.h"

#include "minicpmv/acl_context.h"
#include "minicpmv/ops.h"

#include <acl/acl.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace minicpmv {
namespace {

bool profile_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("MINICPM_PROFILE");
        return v != nullptr && *v != '\0' && std::string(v) != "0" && std::string(v) != "false";
    }();
    return enabled;
}

class ProfileScope {
public:
    ProfileScope(std::string name, aclrtStream stream) : name_(std::move(name)), stream_(stream), enabled_(profile_enabled()) {
        if (enabled_) {
            check_acl(aclrtSynchronizeStream(stream_), "profile sync start");
            start_ = Clock::now();
        }
    }

    ~ProfileScope() {
        if (!enabled_) return;
        auto ret = aclrtSynchronizeStream(stream_);
        const auto end = Clock::now();
        const double ms = std::chrono::duration<double, std::milli>(end - start_).count();
        if (ret == ACL_SUCCESS) {
            std::cerr << "# profile " << name_ << " ms=" << ms << '\n';
        } else {
            std::cerr << "# profile " << name_ << " sync_error=" << ret << " ms=" << ms << '\n';
        }
    }

private:
    using Clock = std::chrono::steady_clock;
    std::string name_;
    aclrtStream stream_;
    bool enabled_;
    Clock::time_point start_;
};

std::string prefill_profile_name(int64_t layer, const char* stage) {
    return "prefill.layer." + std::to_string(layer) + "." + stage;
}

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
    if (src.shape().size() != 2 || dst.shape().size() != 2 ||
        src.shape()[1] != heads * head_dim ||
        dst.shape() != std::vector<int64_t>{src.shape()[0] * heads, head_dim} ||
        src.dtype() != dst.dtype()) {
        throw std::runtime_error("copy_heads_from_cols shape mismatch");
    }
    check_acl(aclrtMemcpyAsync(dst.data(), dst.size_bytes(), src.data(), src.size_bytes(),
                               ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
              "copy_heads_from_cols");
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
                                 Tensor& acc_i32,
                                 Tensor& out,
                                 aclrtStream stream) {
    if (x_int8.shape() != std::vector<int64_t>{1, w8_weight.K} || x_int8.dtype() != DType::Int8) {
        throw std::runtime_error("W8A8 input shape mismatch");
    }
    if (acc_i32.shape() != std::vector<int64_t>{1, w8_weight.N} || acc_i32.dtype() != DType::Int32) {
        throw std::runtime_error("W8A8 accumulator shape mismatch");
    }
    if (out.shape() != std::vector<int64_t>{1, w8_weight.N} || out.dtype() != DType::Float16) {
        throw std::runtime_error("W8A8 output shape mismatch");
    }
    matmul_w8a8_i32(x_int8, w8_weight.w_int8, acc_i32, stream);
    w8a8_dequant(acc_i32, x_scale, w8_weight.w_scale, out, stream);
}

void matmul_decode_w8a8_prequant_allocating(const Tensor& x_int8,
                                            const Tensor& x_scale,
                                            const W8A8QuantizedWeight& w8_weight,
                                            Tensor& out,
                                            aclrtStream stream) {
    Tensor acc({1, w8_weight.N}, DType::Int32); acc.allocate();
    matmul_decode_w8a8_prequant(x_int8, x_scale, w8_weight, acc, out, stream);
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
        matmul_decode_w8a8_prequant_allocating(x_int8, x_scale, *w8_weight, out, stream);
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

void ensure_prefill_scratch(PrefillLayerScratch& s,
                            int64_t T,
                            int64_t hidden_size,
                            int64_t q_dim,
                            int64_t kv_dim,
                            int64_t num_q_heads,
                            int64_t num_kv_heads,
                            int64_t head_dim,
                            int64_t intermediate) {
    if (s.normed.shape() == std::vector<int64_t>{T, hidden_size} &&
        s.q_full.shape() == std::vector<int64_t>{T, q_dim} &&
        s.k_full.shape() == std::vector<int64_t>{T, kv_dim} &&
        s.v_full.shape() == std::vector<int64_t>{T, kv_dim} &&
        s.q_heads.shape() == std::vector<int64_t>{T * num_q_heads, head_dim} &&
        s.k_heads.shape() == std::vector<int64_t>{T * num_kv_heads, head_dim} &&
        s.gate.shape() == std::vector<int64_t>{T, intermediate}) {
        return;
    }
    auto make = [](std::vector<int64_t> shape) {
        Tensor t(std::move(shape), DType::Float16);
        t.allocate();
        return t;
    };
    s.normed = make({T, hidden_size});
    s.q_full = make({T, q_dim});
    s.k_full = make({T, kv_dim});
    s.v_full = make({T, kv_dim});
    s.q_heads = make({T * num_q_heads, head_dim});
    s.k_heads = make({T * num_kv_heads, head_dim});
    s.q_rope = make({T * num_q_heads, head_dim});
    s.k_rope = make({T * num_kv_heads, head_dim});
    s.attn_out = make({T, q_dim});
    s.q_seq = make({T, head_dim});
    s.k_seq = make({T, head_dim});
    s.v_seq = make({T, head_dim});
    s.scores = make({T, T});
    s.scaled_scores = make({T, T});
    s.masked_scores = make({T, T});
    s.probs = make({T, T});
    s.ctx_seq = make({T, head_dim});
    s.attn_proj = make({T, hidden_size});
    s.after_attn = make({T, hidden_size});
    s.mlp_in = make({T, hidden_size});
    s.gate = make({T, intermediate});
    s.up = make({T, intermediate});
    s.gated = make({T, intermediate});
    s.mlp_out = make({T, hidden_size});
}

void run_prefill_core(const Tensor& hidden,
                      const AttentionDecoderLayerWeights& weights,
                      const Tensor& cos_table,
                      const Tensor& sin_table,
                      const std::vector<int32_t>& row_to_t,
                      const PrefillAttentionShared& shared,
                      const AttentionDecoderLayerConfig& config,
                      AttentionLayerCache* cache,
                      int64_t cache_offset,
                      PrefillLayerScratch& scratch,
                      int64_t layer_index,
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
    if (shared.scale.shape() != std::vector<int64_t>{T, T} ||
        shared.causal_mask.shape() != std::vector<int64_t>{T, T} ||
        static_cast<int64_t>(shared.q_row_to_t.size()) != T * num_q_heads ||
        static_cast<int64_t>(shared.k_row_to_t.size()) != T * num_kv_heads) {
        throw std::runtime_error("prefill shared scale/mask/row map shape mismatch");
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

    ensure_prefill_scratch(scratch, T, hidden_size, q_dim, kv_dim,
                           num_q_heads, num_kv_heads, head_dim, intermediate);

    Tensor& normed = scratch.normed;
    {
        ProfileScope p(prefill_profile_name(layer_index, "input_norm"), stream);
        rms_norm(hidden, *weights.input_norm_weight, normed, config.rms_epsilon, stream);
    }

    Tensor& q_full = scratch.q_full;
    Tensor& k_full = scratch.k_full;
    Tensor& v_full = scratch.v_full;
    {
        ProfileScope p(prefill_profile_name(layer_index, "qkv"), stream);
        matmul_b_transposed(normed, *weights.q_proj_weight, q_full, stream);
        matmul_b_transposed(normed, *weights.k_proj_weight, k_full, stream);
        matmul_b_transposed(normed, *weights.v_proj_weight, v_full, stream);
    }

    Tensor& q_heads = scratch.q_heads;
    Tensor& k_heads = scratch.k_heads;
    {
        ProfileScope p(prefill_profile_name(layer_index, "head_pack"), stream);
        copy_heads_from_cols(q_full, num_q_heads, head_dim, q_heads, stream);
        copy_heads_from_cols(k_full, num_kv_heads, head_dim, k_heads, stream);
    }

    Tensor& q_rope = scratch.q_rope;
    Tensor& k_rope = scratch.k_rope;
    {
        ProfileScope p(prefill_profile_name(layer_index, "rope"), stream);
        apply_rope_prefill(q_heads, cos_table, sin_table, num_q_heads, config.rotary_dim, q_rope, stream);
        apply_rope_prefill(k_heads, cos_table, sin_table, num_kv_heads, config.rotary_dim, k_rope, stream);
    }

    if (cache != nullptr) {
        ProfileScope p(prefill_profile_name(layer_index, "cache_write"), stream);
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

    Tensor& attn_out = scratch.attn_out;

    {
        ProfileScope p(prefill_profile_name(layer_index, "attn_heads"), stream);
        if (T <= 256) {
            const float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
            prefill_attention_custom(q_rope, k_rope, v_full, T, num_q_heads, num_kv_heads,
                                     head_dim, attn_scale, attn_out, stream);
        } else {
            Tensor& q_seq = scratch.q_seq;
            Tensor& k_seq = scratch.k_seq;
            Tensor& v_seq = scratch.v_seq;
            Tensor& scores = scratch.scores;
            Tensor& scaled_scores = scratch.scaled_scores;
            Tensor& masked_scores = scratch.masked_scores;
            Tensor& probs = scratch.probs;
            Tensor& ctx_seq = scratch.ctx_seq;
            for (int64_t qh = 0; qh < num_q_heads; ++qh) {
                const int64_t kvh = qh / q_per_kv;
                copy_head_to_seq(q_rope, qh, num_q_heads, q_seq, stream);
                copy_head_to_seq(k_rope, kvh, num_kv_heads, k_seq, stream);
                copy_col_block(v_full, kvh * head_dim, v_seq, stream);
                matmul_b_transposed(q_seq, k_seq, scores, stream);
                mul(scores, shared.scale, scaled_scores, stream);
                add(scaled_scores, shared.causal_mask, masked_scores, stream);
                softmax_last_dim(masked_scores, probs, stream);
                matmul(probs, v_seq, ctx_seq, stream);
                copy_seq_to_head_block(ctx_seq, attn_out, qh * head_dim, stream);
            }
        }
    }

    Tensor& attn_proj = scratch.attn_proj;
    Tensor& after_attn = scratch.after_attn;
    {
        ProfileScope p(prefill_profile_name(layer_index, "o_proj"), stream);
        matmul_b_transposed(attn_out, *weights.o_proj_weight, attn_proj, stream);
        add(hidden, attn_proj, after_attn, stream);
    }

    Tensor& mlp_in = scratch.mlp_in;
    {
        ProfileScope p(prefill_profile_name(layer_index, "mlp_norm"), stream);
        rms_norm(after_attn, *weights.post_attention_norm_weight, mlp_in, config.rms_epsilon, stream);
    }

    Tensor& gate = scratch.gate;
    Tensor& up = scratch.up;
    Tensor& gated = scratch.gated;
    Tensor& mlp_out = scratch.mlp_out;
    {
        ProfileScope p(prefill_profile_name(layer_index, "mlp"), stream);
        matmul_b_transposed(mlp_in, *weights.gate_proj_weight, gate, stream);
        matmul_b_transposed(mlp_in, *weights.up_proj_weight, up, stream);
        silu_mul(gate, up, gated, stream);
        matmul_b_transposed(gated, *weights.down_proj_weight, mlp_out, stream);
        add(after_attn, mlp_out, out, stream);
    }
}

}  // namespace

void build_prefill_attention_shared(const std::vector<int32_t>& row_to_t,
                                    const AttentionDecoderLayerConfig& config,
                                    PrefillAttentionShared& shared) {
    const int64_t T = static_cast<int64_t>(row_to_t.size());
    if (T <= 0) {
        throw std::runtime_error("build_prefill_attention_shared requires non-empty row_to_t");
    }
    if (config.head_dim <= 0) {
        throw std::runtime_error("build_prefill_attention_shared invalid head_dim");
    }

    std::vector<uint16_t> scale_host(static_cast<size_t>(T * T),
                                     f32_to_f16_bits(1.0f / std::sqrt(static_cast<float>(config.head_dim))));
    shared.scale = Tensor({T, T}, DType::Float16);
    shared.scale.copy_from_host(scale_host.data(), scale_host.size() * sizeof(uint16_t));

    std::vector<uint16_t> mask_host(static_cast<size_t>(T * T));
    for (int64_t r = 0; r < T; ++r) {
        for (int64_t c = 0; c < T; ++c) {
            mask_host[static_cast<size_t>(r * T + c)] =
                f32_to_f16_bits(row_to_t[c] <= row_to_t[r] ? 0.0f : -65504.0f);
        }
    }
    shared.causal_mask = Tensor({T, T}, DType::Float16);
    shared.causal_mask.copy_from_host(mask_host.data(), mask_host.size() * sizeof(uint16_t));

    shared.q_row_to_t.resize(static_cast<size_t>(T * config.num_q_heads));
    shared.k_row_to_t.resize(static_cast<size_t>(T * config.num_kv_heads));
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t h = 0; h < config.num_q_heads; ++h) {
            shared.q_row_to_t[static_cast<size_t>(t * config.num_q_heads + h)] = row_to_t[static_cast<size_t>(t)];
        }
        for (int64_t h = 0; h < config.num_kv_heads; ++h) {
            shared.k_row_to_t[static_cast<size_t>(t * config.num_kv_heads + h)] = row_to_t[static_cast<size_t>(t)];
        }
    }
}

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
                                     aclrtStream stream) {
    run_prefill_core(hidden, weights, cos_table, sin_table, row_to_t, shared, config,
                     &cache, cache_offset, scratch, layer_index, out, stream);
}

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
    if (s.normed.shape() == std::vector<int64_t>{1, hidden_size} &&
        s.q_full.shape() == std::vector<int64_t>{1, q_dim} &&
        s.k_full.shape() == std::vector<int64_t>{1, kv_dim} &&
        s.v_full.shape() == std::vector<int64_t>{1, kv_dim} &&
        s.q_acc_i32.shape() == std::vector<int64_t>{1, q_dim} &&
        s.k_acc_i32.shape() == std::vector<int64_t>{1, kv_dim} &&
        s.v_acc_i32.shape() == std::vector<int64_t>{1, kv_dim} &&
        s.q_heads.shape() == std::vector<int64_t>{num_q_heads, head_dim} &&
        s.k_heads.shape() == std::vector<int64_t>{num_kv_heads, head_dim} &&
        s.o_acc_i32.shape() == std::vector<int64_t>{1, hidden_size} &&
        s.gate.shape() == std::vector<int64_t>{1, intermediate} &&
        s.gate_acc_i32.shape() == std::vector<int64_t>{1, intermediate} &&
        s.up_acc_i32.shape() == std::vector<int64_t>{1, intermediate} &&
        s.down_acc_i32.shape() == std::vector<int64_t>{1, hidden_size}) {
        return;
    }
    auto make = [](std::vector<int64_t> shape) {
        Tensor t(std::move(shape), DType::Float16);
        t.allocate();
        return t;
    };
    auto make_i8 = [](std::vector<int64_t> shape) {
        Tensor t(std::move(shape), DType::Int8);
        t.allocate();
        return t;
    };
    auto make_i32 = [](std::vector<int64_t> shape) {
        Tensor t(std::move(shape), DType::Int32);
        t.allocate();
        return t;
    };
    s.normed = make({1, hidden_size});
    s.q_full = make({1, q_dim});
    s.k_full = make({1, kv_dim});
    s.v_full = make({1, kv_dim});
    s.normed_i8 = make_i8({1, hidden_size});
    s.normed_scale = make({1});
    s.q_acc_i32 = make_i32({1, q_dim});
    s.k_acc_i32 = make_i32({1, kv_dim});
    s.v_acc_i32 = make_i32({1, kv_dim});
    s.q_heads = make({num_q_heads, head_dim});
    s.k_heads = make({num_kv_heads, head_dim});
    s.q_rope = make({num_q_heads, head_dim});
    s.attn_out = make({1, q_dim});
    s.attn_out_i8 = make_i8({1, q_dim});
    s.attn_out_scale = make({1});
    s.o_acc_i32 = make_i32({1, hidden_size});
    s.attn_proj = make({1, hidden_size});
    s.after_attn = make({1, hidden_size});
    s.mlp_in = make({1, hidden_size});
    s.mlp_i8 = make_i8({1, hidden_size});
    s.mlp_scale = make({1});
    s.gate_acc_i32 = make_i32({1, intermediate});
    s.up_acc_i32 = make_i32({1, intermediate});
    s.gate = make({1, intermediate});
    s.up = make({1, intermediate});
    s.gated = make({1, intermediate});
    s.gated_i8 = make_i8({1, intermediate});
    s.gated_scale = make({1});
    s.down_acc_i32 = make_i32({1, hidden_size});
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
    PrefillAttentionShared shared;
    build_prefill_attention_shared(row_to_t, config, shared);
    PrefillLayerScratch scratch;
    attention_decoder_layer_prefill(hidden, weights, cos_table, sin_table, row_to_t,
                                    shared, config, cache, 0, scratch, 0, out, stream);
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
    {
        ProfileScope p("decode.input_norm", stream);
        rms_norm_with_scratch(hidden, *weights.input_norm_weight, normed,
                              config.rms_epsilon, s.input_norm_scratch, stream);
    }

    Tensor& q_full = s.q_full;
    Tensor& k_full = s.k_full;
    Tensor& v_full = s.v_full;

    {
        ProfileScope p("decode.qkv", stream);
        if (w8a8_weight_ready(weights.q_proj_w8) || w8a8_weight_ready(weights.k_proj_w8) ||
            w8a8_weight_ready(weights.v_proj_w8)) {
            Tensor& normed_i8 = s.normed_i8;
            Tensor& normed_scale = s.normed_scale;
            w8a8_quantize(normed, normed_i8, normed_scale, stream);
            if (w8a8_weight_ready(weights.q_proj_w8)) {
                matmul_decode_w8a8_prequant(normed_i8, normed_scale, *weights.q_proj_w8, s.q_acc_i32, q_full, stream);
            } else {
                matmul_decode_dispatch(normed, weights.q_proj_weight, weights.q_proj_q, nullptr, q_full, stream);
            }
            if (w8a8_weight_ready(weights.k_proj_w8)) {
                matmul_decode_w8a8_prequant(normed_i8, normed_scale, *weights.k_proj_w8, s.k_acc_i32, k_full, stream);
            } else {
                matmul_decode_dispatch(normed, weights.k_proj_weight, weights.k_proj_q, nullptr, k_full, stream);
            }
            if (w8a8_weight_ready(weights.v_proj_w8)) {
                matmul_decode_w8a8_prequant(normed_i8, normed_scale, *weights.v_proj_w8, s.v_acc_i32, v_full, stream);
            } else {
                matmul_decode_dispatch(normed, weights.v_proj_weight, weights.v_proj_q, nullptr, v_full, stream);
            }
        } else {
            matmul_decode_dispatch(normed, weights.q_proj_weight, weights.q_proj_q, nullptr, q_full, stream);
            matmul_decode_dispatch(normed, weights.k_proj_weight, weights.k_proj_q, nullptr, k_full, stream);
            matmul_decode_dispatch(normed, weights.v_proj_weight, weights.v_proj_q, nullptr, v_full, stream);
        }
    }

    Tensor& q_heads = s.q_heads;
    Tensor& k_heads = s.k_heads;
    {
        ProfileScope p("decode.head_pack", stream);
        copy_heads_from_cols(q_full, num_q_heads, head_dim, q_heads, stream);
        copy_heads_from_cols(k_full, num_kv_heads, head_dim, k_heads, stream);
    }

    Tensor& q_rope = s.q_rope;
    {
        ProfileScope p("decode.rope_cache_write", stream);
        rope_cache_write(q_heads, k_heads, v_full, cache.k_cache, cache.v_cache,
                         cos_table, sin_table, pos, cache_len,
                         num_q_heads, num_kv_heads, head_dim, config.rotary_dim,
                         q_rope, stream);
    }

    Tensor& attn_out = s.attn_out;
    {
        ProfileScope p("decode.ifa", stream);
        const float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        incre_flash_attention(q_rope, cache.k_cache, cache.v_cache,
                              context, num_q_heads, num_kv_heads, head_dim,
                              attn_scale, attn_out, stream);
    }

    Tensor& attn_proj = s.attn_proj;
    Tensor& after_attn = s.after_attn;
    {
        ProfileScope p("decode.o_proj", stream);
        if (w8a8_weight_ready(weights.o_proj_w8)) {
            w8a8_quantize(attn_out, s.attn_out_i8, s.attn_out_scale, stream);
            matmul_decode_w8a8_prequant(s.attn_out_i8, s.attn_out_scale, *weights.o_proj_w8, s.o_acc_i32, attn_proj, stream);
        } else {
            matmul_decode_dispatch(attn_out, weights.o_proj_weight, weights.o_proj_q, nullptr, attn_proj, stream);
        }
        add(hidden, attn_proj, after_attn, stream);
    }

    Tensor& mlp_in = s.mlp_in;
    {
        ProfileScope p("decode.mlp_norm", stream);
        rms_norm_with_scratch(after_attn, *weights.post_attention_norm_weight, mlp_in,
                              config.rms_epsilon, s.mlp_norm_scratch, stream);
    }

    Tensor& gate = s.gate;
    Tensor& up = s.up;
    Tensor& gated = s.gated;
    Tensor& mlp_out = s.mlp_out;

    {
        ProfileScope p("decode.mlp", stream);
        if (w8a8_weight_ready(weights.gate_proj_w8) || w8a8_weight_ready(weights.up_proj_w8)) {
            Tensor& mlp_i8 = s.mlp_i8;
            Tensor& mlp_scale = s.mlp_scale;
            w8a8_quantize(mlp_in, mlp_i8, mlp_scale, stream);
            if (w8a8_weight_ready(weights.gate_proj_w8)) {
                matmul_decode_w8a8_prequant(mlp_i8, mlp_scale, *weights.gate_proj_w8, s.gate_acc_i32, gate, stream);
            } else {
                matmul_decode_dispatch(mlp_in, weights.gate_proj_weight, weights.gate_proj_q, nullptr, gate, stream);
            }
            if (w8a8_weight_ready(weights.up_proj_w8)) {
                matmul_decode_w8a8_prequant(mlp_i8, mlp_scale, *weights.up_proj_w8, s.up_acc_i32, up, stream);
            } else {
                matmul_decode_dispatch(mlp_in, weights.up_proj_weight, weights.up_proj_q, nullptr, up, stream);
            }
        } else {
            matmul_decode_dispatch(mlp_in, weights.gate_proj_weight, weights.gate_proj_q, nullptr, gate, stream);
            matmul_decode_dispatch(mlp_in, weights.up_proj_weight, weights.up_proj_q, nullptr, up, stream);
        }
        silu_mul(gate, up, gated, stream);
        if (w8a8_weight_ready(weights.down_proj_w8)) {
            w8a8_quantize(gated, s.gated_i8, s.gated_scale, stream);
            matmul_decode_w8a8_prequant(s.gated_i8, s.gated_scale, *weights.down_proj_w8, s.down_acc_i32, mlp_out, stream);
        } else {
            matmul_decode_dispatch(gated, weights.down_proj_weight, weights.down_proj_q, nullptr, mlp_out, stream);
        }
        add(after_attn, mlp_out, out, stream);
    }
}

}  // namespace minicpmv
