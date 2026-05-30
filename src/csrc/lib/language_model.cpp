#include "minicpmv/language_model.h"

#include "minicpmv/acl_context.h"
#include "minicpmv/ops.h"
#include "minicpmv/quantized_weight.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace minicpmv {

namespace {

const std::string& w8a8_decode_mode() {
    static const std::string mode = [] {
        const char* v = std::getenv("MINICPM_W8A8_DECODE");
        return v == nullptr ? std::string{} : std::string(v);
    }();
    return mode;
}

bool w8a8_decode_enabled() {
    const std::string& mode = w8a8_decode_mode();
    return !mode.empty() && mode != "0" && mode != "false";
}

bool profile_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("MINICPM_PROFILE");
        return v != nullptr && *v != '\0' && std::string(v) != "0" && std::string(v) != "false";
    }();
    return enabled;
}

class ProfileScope {
public:
    ProfileScope(const char* name, aclrtStream stream) : name_(name), stream_(stream), enabled_(profile_enabled()) {
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
    const char* name_;
    aclrtStream stream_;
    bool enabled_;
    Clock::time_point start_;
};

bool token_in_mode(const std::string& mode, const char* token) {
    std::istringstream iss(mode);
    std::string part;
    while (std::getline(iss, part, ',')) {
        if (part == token) return true;
    }
    return false;
}

bool w8a8_policy_allows(const char* token) {
    if (!w8a8_decode_enabled()) return false;
    const std::string& mode = w8a8_decode_mode();
    if (mode == "all") return true;
    const std::string t(token);
    if (mode == "1" || mode == "selective") {
        return t == "lm_head";
    }
    if (token_in_mode(mode, token)) return true;
    if (t == "q" && token_in_mode(mode, "full_q")) return true;
    if (t == "k" && token_in_mode(mode, "full_k")) return true;
    if (t == "v" && token_in_mode(mode, "full_v")) return true;
    if (t == "o" && token_in_mode(mode, "full_o")) return true;
    return false;
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

Tensor load_weight(WeightsIndex& index, const std::string& name) {
    return index.load_to_device_as(name, DType::Float16);
}

Tensor load_layer_weight(WeightsIndex& index, const LanguageModelConfig& cfg, int layer, const std::string& suffix) {
    return index.load_to_device_as(cfg.model_prefix + ".layers." + std::to_string(layer) + "." + suffix, DType::Float16);
}

void copy_row(const Tensor& src, int64_t src_row, Tensor& dst, int64_t dst_row, aclrtStream stream) {
    const size_t elem = dtype_size(src.dtype());
    const size_t row_bytes = static_cast<size_t>(src.shape()[1]) * elem;
    auto* s = static_cast<const uint8_t*>(src.data()) + static_cast<size_t>(src_row) * row_bytes;
    auto* d = static_cast<uint8_t*>(dst.data()) + static_cast<size_t>(dst_row) * row_bytes;
    check_acl(aclrtMemcpyAsync(d, row_bytes, s, row_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
              "lm copy_row");
    check_acl(aclrtSynchronizeStream(stream), "lm copy_row sync");
}

void copy_tensor(const Tensor& src, Tensor& dst, aclrtStream stream) {
    check_acl(aclrtMemcpyAsync(dst.data(), dst.size_bytes(), src.data(), src.size_bytes(),
                               ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
              "lm copy_tensor");
    check_acl(aclrtSynchronizeStream(stream), "lm copy_tensor sync");
}

Tensor load_matmul_weight_transposed(WeightsIndex& index, const LanguageModelConfig& cfg, int layer, const std::string& suffix) {
    Tensor src = load_layer_weight(index, cfg, layer, suffix);
    if (src.shape().size() != 2) {
        throw std::runtime_error("matmul weight " + suffix + " expected 2D");
    }
    const int64_t N = src.shape()[0];
    const int64_t K = src.shape()[1];
    if (N > 16384 || (N % 128) != 0) {
        return src;
    }
    std::vector<uint16_t> host_nk(static_cast<size_t>(N) * K);
    src.copy_to_host(host_nk.data(), host_nk.size() * sizeof(uint16_t));
    std::vector<uint16_t> host_kn(static_cast<size_t>(K) * N);
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t k = 0; k < K; ++k) {
            host_kn[static_cast<size_t>(k) * N + n] = host_nk[static_cast<size_t>(n) * K + k];
        }
    }
    Tensor dst({K, N}, DType::Float16);
    dst.copy_from_host(host_kn.data(), host_kn.size() * sizeof(uint16_t));
    return dst;
}

std::string layer_base(const LanguageModelConfig& cfg, int layer, const std::string& suffix) {
    return cfg.model_prefix + ".layers." + std::to_string(layer) + "." + suffix;
}

W4A16QuantizedWeight load_layer_w4a16_if_present(WeightsIndex& index, const LanguageModelConfig& cfg, int layer, const std::string& suffix) {
    const std::string base = layer_base(cfg, layer, suffix);
    if (!has_w4a16_quantized_weight(index, base)) {
        return {};
    }
    return load_w4a16_quantized_weight(index, base);
}

const W4A16QuantizedWeight* quant_ptr(const W4A16QuantizedWeight& w) {
    return w.w_int8.data() == nullptr ? nullptr : &w;
}

const W8A8QuantizedWeight* quant_ptr(const W8A8QuantizedWeight& w) {
    return w.w_int8.data() == nullptr ? nullptr : &w;
}

AttentionDecoderLayerConfig attention_config(const LanguageModelConfig& cfg) {
    return AttentionDecoderLayerConfig{cfg.num_q_heads, cfg.num_kv_heads,
                                       cfg.head_dim, cfg.rotary_dim, cfg.rms_epsilon};
}

AttentionDecoderLayerWeights attention_weights(const LanguageModelLayerWeights& lw) {
    return AttentionDecoderLayerWeights{
        &lw.input_norm_w, &lw.post_norm_w, &lw.q_w, &lw.k_w, &lw.v_w,
        &lw.o_w, &lw.gate_w, &lw.up_w, &lw.down_w,
        quant_ptr(lw.q_q), quant_ptr(lw.k_q), quant_ptr(lw.v_q), quant_ptr(lw.o_q),
        quant_ptr(lw.gate_q), quant_ptr(lw.up_q), quant_ptr(lw.down_q),
        quant_ptr(lw.q_w8), quant_ptr(lw.k_w8), quant_ptr(lw.v_w8), quant_ptr(lw.o_w8),
        quant_ptr(lw.gate_w8), quant_ptr(lw.up_w8), quant_ptr(lw.down_w8),
    };
}

float h16_to_f32(uint16_t h) {
    uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t out;
    if (exp == 0) {
        out = sign;
    } else if (exp == 31) {
        out = sign | 0x7f800000u | (mant << 13);
    } else {
        out = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

bool tensor_ready(const Tensor& t, const std::vector<int64_t>& shape, DType dtype) {
    return t.data() != nullptr && t.dtype() == dtype && t.shape() == shape;
}

void ensure_tensor(Tensor& t, std::vector<int64_t> shape, DType dtype) {
    if (tensor_ready(t, shape, dtype)) return;
    t = Tensor(std::move(shape), dtype);
    t.allocate();
}

void ensure_lm_head_scratch(LmHeadScratch& s,
                            int64_t hidden_size,
                            int64_t chunk_n,
                            int64_t num_chunks,
                            bool use_w8a8) {
    ensure_tensor(s.normed, {1, hidden_size}, DType::Float16);
    ensure_tensor(s.logits, {1, chunk_n}, DType::Float16);
    ensure_tensor(s.chunk_best_value, {1}, DType::Float16);
    ensure_tensor(s.chunk_best_index, {1}, DType::Int32);
    ensure_tensor(s.chunk_best_values, {num_chunks}, DType::Float16);
    ensure_tensor(s.chunk_best_indices, {num_chunks}, DType::Int32);
    if (use_w8a8) {
        ensure_tensor(s.logits_i32, {1, chunk_n}, DType::Int32);
        ensure_tensor(s.normed_i8, {1, hidden_size}, DType::Int8);
        ensure_tensor(s.normed_scale, {1}, DType::Float16);
    }
}

int64_t lm_head_greedy_with_scratch(const Tensor& last_hidden_1xH,
                                    const LanguageModelWeights& w,
                                    const LanguageModelConfig& cfg,
                                    LmHeadScratch& scratch,
                                    aclrtStream stream) {
    ProfileScope profile("lm_head_greedy", stream);
    if (last_hidden_1xH.shape() != std::vector<int64_t>{1, cfg.hidden_size}) {
        throw std::runtime_error("lm_head_greedy hidden must be [1, hidden_size]");
    }
    if (w.lm_head_chunks.empty()) {
        throw std::runtime_error("lm_head_greedy missing pre-built chunks");
    }

    const int64_t kChunkN = w.lm_head_chunks.front().weight_kn.shape()[1];
    const int64_t num_chunks = static_cast<int64_t>(w.lm_head_chunks.size());
    const bool use_w8a8 = w8a8_decode_enabled();
    ensure_lm_head_scratch(scratch, cfg.hidden_size, kChunkN, num_chunks, use_w8a8);

    Tensor& normed = scratch.normed;
    rms_norm_with_scratch(last_hidden_1xH, w.final_norm_w, normed,
                          cfg.rms_epsilon, scratch.norm_scratch, stream);

    Tensor& logits = scratch.logits;
    Tensor& chunk_best_value = scratch.chunk_best_value;
    Tensor& chunk_best_index = scratch.chunk_best_index;
    Tensor& chunk_best_values = scratch.chunk_best_values;
    Tensor& chunk_best_indices = scratch.chunk_best_indices;

    if (use_w8a8) {
        w8a8_quantize(normed, scratch.normed_i8, scratch.normed_scale, stream);
    }

    for (size_t chunk_idx = 0; chunk_idx < w.lm_head_chunks.size(); ++chunk_idx) {
        const auto& chunk = w.lm_head_chunks[chunk_idx];
        if (use_w8a8 && chunk.weight_w8.w_int8.data() != nullptr) {
            matmul_w8a8_i32(scratch.normed_i8, chunk.weight_w8.w_int8, scratch.logits_i32, stream);
            w8a8_dequant(scratch.logits_i32, scratch.normed_scale, chunk.weight_w8.w_scale, logits, stream);
        } else {
            matmul_b_transposed(normed, chunk.weight_kn, logits, stream);
        }
        const int64_t valid = std::min<int64_t>(kChunkN, cfg.vocab_size - chunk.start_vocab);
        logits_top1(logits, valid, chunk_best_value, chunk_best_index, stream);
        check_acl(aclrtMemcpyAsync(static_cast<uint8_t*>(chunk_best_values.data()) + chunk_idx * dtype_size(DType::Float16),
                                   dtype_size(DType::Float16),
                                   chunk_best_value.data(), dtype_size(DType::Float16),
                                   ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                  "lm_head collect chunk value");
        check_acl(aclrtMemcpyAsync(static_cast<uint8_t*>(chunk_best_indices.data()) + chunk_idx * dtype_size(DType::Int32),
                                   dtype_size(DType::Int32),
                                   chunk_best_index.data(), dtype_size(DType::Int32),
                                   ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                  "lm_head collect chunk index");
    }
    check_acl(aclrtSynchronizeStream(stream), "lm_head chunk top1 sync");

    std::vector<uint16_t> chunk_best_value_host(static_cast<size_t>(num_chunks));
    std::vector<int32_t> chunk_best_index_host(static_cast<size_t>(num_chunks));
    chunk_best_values.copy_to_host(chunk_best_value_host.data(), chunk_best_value_host.size() * sizeof(uint16_t));
    chunk_best_indices.copy_to_host(chunk_best_index_host.data(), chunk_best_index_host.size() * sizeof(int32_t));

    int64_t best_token = 0;
    float best_logit = -std::numeric_limits<float>::infinity();
    for (size_t chunk_idx = 0; chunk_idx < w.lm_head_chunks.size(); ++chunk_idx) {
        const float v = h16_to_f32(chunk_best_value_host[chunk_idx]);
        if (v > best_logit) {
            best_logit = v;
            best_token = w.lm_head_chunks[chunk_idx].start_vocab + chunk_best_index_host[chunk_idx];
        }
    }
    return best_token;
}

}  // namespace

LanguageModelConfig default_minicpm5_1b_lm_config() {
    return LanguageModelConfig{};
}

LanguageModelWeights load_language_model_weights(WeightsIndex& index,
                                                 const LanguageModelConfig& cfg) {
    LanguageModelWeights w;
    w.embed = load_weight(index, cfg.embed_weight_name);
    w.final_norm_w = load_weight(index, cfg.final_norm_weight_name);
    w.layers.resize(static_cast<size_t>(cfg.num_layers));

    for (int64_t layer = 0; layer < cfg.num_layers; ++layer) {
        auto& lw = w.layers[static_cast<size_t>(layer)];
        const int li = static_cast<int>(layer);
        lw.input_norm_w = load_layer_weight(index, cfg, li, "input_layernorm.weight");
        lw.post_norm_w = load_layer_weight(index, cfg, li, "post_attention_layernorm.weight");
        lw.gate_w = load_matmul_weight_transposed(index, cfg, li, "mlp.gate_proj.weight");
        lw.up_w = load_matmul_weight_transposed(index, cfg, li, "mlp.up_proj.weight");
        lw.down_w = load_matmul_weight_transposed(index, cfg, li, "mlp.down_proj.weight");
        lw.q_w = load_matmul_weight_transposed(index, cfg, li, "self_attn.q_proj.weight");
        lw.k_w = load_matmul_weight_transposed(index, cfg, li, "self_attn.k_proj.weight");
        lw.v_w = load_matmul_weight_transposed(index, cfg, li, "self_attn.v_proj.weight");
        lw.o_w = load_matmul_weight_transposed(index, cfg, li, "self_attn.o_proj.weight");

        lw.gate_q = load_layer_w4a16_if_present(index, cfg, li, "mlp.gate_proj");
        lw.up_q = load_layer_w4a16_if_present(index, cfg, li, "mlp.up_proj");
        lw.down_q = load_layer_w4a16_if_present(index, cfg, li, "mlp.down_proj");
        lw.q_q = load_layer_w4a16_if_present(index, cfg, li, "self_attn.q_proj");
        lw.k_q = load_layer_w4a16_if_present(index, cfg, li, "self_attn.k_proj");
        lw.v_q = load_layer_w4a16_if_present(index, cfg, li, "self_attn.v_proj");
        lw.o_q = load_layer_w4a16_if_present(index, cfg, li, "self_attn.o_proj");

        if (w8a8_policy_allows("mlp")) {
            lw.gate_w8 = quantize_dense_weight_w8a8(lw.gate_w);
            lw.up_w8 = quantize_dense_weight_w8a8(lw.up_w);
            lw.down_w8 = quantize_dense_weight_w8a8(lw.down_w);
        }
        if (w8a8_policy_allows("q")) {
            lw.q_w8 = quantize_dense_weight_w8a8(lw.q_w);
        }
        if (w8a8_policy_allows("k")) {
            lw.k_w8 = quantize_dense_weight_w8a8(lw.k_w);
        }
        if (w8a8_policy_allows("v")) {
            lw.v_w8 = quantize_dense_weight_w8a8(lw.v_w);
        }
        if (w8a8_policy_allows("o")) {
            lw.o_w8 = quantize_dense_weight_w8a8(lw.o_w);
        }
    }

    constexpr int64_t kChunkN = 16384;
    const int64_t H = cfg.hidden_size;
    const int64_t V = cfg.vocab_size;
    Tensor lm_head_loaded;
    const Tensor* lm_head_source = &w.embed;
    if (!cfg.lm_head_weight_name.empty()) {
        lm_head_loaded = load_weight(index, cfg.lm_head_weight_name);
        lm_head_source = &lm_head_loaded;
    }
    if (lm_head_source->shape().size() != 2 || lm_head_source->shape()[0] != V || lm_head_source->shape()[1] != H) {
        throw std::runtime_error("lm_head weight shape unexpected for lm_head chunking");
    }
    std::vector<uint16_t> embed_host(static_cast<size_t>(V) * H);
    lm_head_source->copy_to_host(embed_host.data(), embed_host.size() * sizeof(uint16_t));
    std::vector<uint16_t> chunk_host(static_cast<size_t>(H) * kChunkN);
    for (int64_t start = 0; start < V; start += kChunkN) {
        const int64_t valid = std::min<int64_t>(kChunkN, V - start);
        std::fill(chunk_host.begin(), chunk_host.end(), uint16_t{0});
        for (int64_t n = 0; n < valid; ++n) {
            const int64_t row = start + n;
            for (int64_t k = 0; k < H; ++k) {
                chunk_host[static_cast<size_t>(k) * kChunkN + n] =
                    embed_host[static_cast<size_t>(row) * H + k];
            }
        }
        LmHeadChunk chunk;
        chunk.start_vocab = start;
        chunk.weight_kn = Tensor({H, kChunkN}, DType::Float16);
        chunk.weight_kn.copy_from_host(chunk_host.data(), chunk_host.size() * sizeof(uint16_t));
        if (w8a8_policy_allows("lm_head")) {
            chunk.weight_w8 = quantize_dense_weight_w8a8(chunk.weight_kn);
        }
        w.lm_head_chunks.push_back(std::move(chunk));
    }

    return w;
}

void build_rope_tables(int64_t T,
                       const LanguageModelConfig& cfg,
                       Tensor& cos_table,
                       Tensor& sin_table) {
    const int64_t half = cfg.rotary_dim / 2;
    std::vector<uint16_t> cos_host(static_cast<size_t>(T * half));
    std::vector<uint16_t> sin_host(static_cast<size_t>(T * half));
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t i = 0; i < half; ++i) {
            float inv = std::pow(static_cast<float>(cfg.rope_theta),
                                 -2.0f * static_cast<float>(i) / static_cast<float>(cfg.rotary_dim));
            float theta = static_cast<float>(t) * inv;
            cos_host[static_cast<size_t>(t * half + i)] = f32_to_f16_bits(std::cos(theta));
            sin_host[static_cast<size_t>(t * half + i)] = f32_to_f16_bits(std::sin(theta));
        }
    }
    cos_table = Tensor({T, half}, DType::Float16);
    sin_table = Tensor({T, half}, DType::Float16);
    cos_table.copy_from_host(cos_host.data(), cos_host.size() * sizeof(uint16_t));
    sin_table.copy_from_host(sin_host.data(), sin_host.size() * sizeof(uint16_t));
}

Tensor prefill_from_embeddings(const Tensor& prompt_hidden,
                               const LanguageModelWeights& w,
                               const LanguageModelConfig& cfg,
                               const Tensor& cos_table,
                               const Tensor& sin_table,
                               DecodeState& state,
                               aclrtStream stream) {
    if (prompt_hidden.shape().size() != 2 || prompt_hidden.shape()[1] != cfg.hidden_size) {
        throw std::runtime_error("prefill_from_embeddings prompt_hidden must be [T, hidden_size]");
    }
    if (prompt_hidden.dtype() != DType::Float16) {
        throw std::runtime_error("prefill_from_embeddings prompt_hidden must be fp16");
    }
    const int64_t T = prompt_hidden.shape()[0];
    if (T <= 0) throw std::runtime_error("prefill_from_embeddings T must be > 0");
    if (state.seq_len != 0) {
        throw std::runtime_error("prefill_from_embeddings expects empty state");
    }
    if (T > state.max_seq_len) {
        throw std::runtime_error("prefill_from_embeddings T exceeds state.max_seq_len");
    }
    if (static_cast<int64_t>(state.layers.size()) != cfg.num_layers) {
        throw std::runtime_error("prefill_from_embeddings state layer count mismatch");
    }
    if (cos_table.shape()[0] < T) {
        throw std::runtime_error("prefill_from_embeddings cos_table too short");
    }

    Tensor hidden_a({T, cfg.hidden_size}, DType::Float16); hidden_a.allocate();
    Tensor hidden_b({T, cfg.hidden_size}, DType::Float16); hidden_b.allocate();
    copy_tensor(prompt_hidden, hidden_a, stream);

    Tensor* in = &hidden_a;
    Tensor* out = &hidden_b;

    std::vector<int32_t> row_to_t(static_cast<size_t>(T));
    for (int64_t t = 0; t < T; ++t) row_to_t[static_cast<size_t>(t)] = static_cast<int32_t>(t);

    const AttentionDecoderLayerConfig acfg = attention_config(cfg);
    PrefillAttentionShared prefill_shared;
    build_prefill_attention_shared(row_to_t, acfg, prefill_shared);
    PrefillLayerScratch prefill_scratch;
    for (int64_t layer = 0; layer < cfg.num_layers; ++layer) {
        const auto ww = attention_weights(w.layers[static_cast<size_t>(layer)]);
        std::string profile_name = "prefill.layer." + std::to_string(layer) + ".total";
        ProfileScope layer_profile(profile_name.c_str(), stream);
        attention_decoder_layer_prefill(*in, ww, cos_table, sin_table,
                                        row_to_t, prefill_shared, acfg,
                                        state.layers[static_cast<size_t>(layer)], 0,
                                        prefill_scratch, layer, *out, stream);
        std::swap(in, out);
    }
    state.seq_len = T;

    Tensor last_hidden({1, cfg.hidden_size}, DType::Float16); last_hidden.allocate();
    copy_row(*in, T - 1, last_hidden, 0, stream);
    return last_hidden;
}

int64_t lm_head_greedy(const Tensor& last_hidden_1xH,
                       const LanguageModelWeights& w,
                       const LanguageModelConfig& cfg,
                       aclrtStream stream) {
    LmHeadScratch scratch;
    return lm_head_greedy_with_scratch(last_hidden_1xH, w, cfg, scratch, stream);
}

int64_t decode_step_greedy(int32_t token_id,
                           const LanguageModelWeights& w,
                           const LanguageModelConfig& cfg,
                           const Tensor& cos_table,
                           const Tensor& sin_table,
                           DecodeState& state,
                           aclrtStream stream) {
    ProfileScope profile("decode_step_greedy", stream);
    if (state.seq_len >= state.max_seq_len) {
        throw std::runtime_error("decode_step_greedy state full");
    }
    if (static_cast<int64_t>(state.layers.size()) != cfg.num_layers) {
        throw std::runtime_error("decode_step_greedy state layer count mismatch");
    }

    if (state.hidden_a.shape() != std::vector<int64_t>{1, cfg.hidden_size}) {
        state.hidden_a = Tensor({1, cfg.hidden_size}, DType::Float16);
        state.hidden_b = Tensor({1, cfg.hidden_size}, DType::Float16);
        state.hidden_a.allocate();
        state.hidden_b.allocate();
    }
    Tensor& hidden_a = state.hidden_a;
    Tensor& hidden_b = state.hidden_b;
    {
        ProfileScope profile_embed("decode.embedding", stream);
        embedding_lookup(w.embed, {token_id}, hidden_a, stream);
    }

    Tensor* in = &hidden_a;
    Tensor* out = &hidden_b;

    const AttentionDecoderLayerConfig acfg = attention_config(cfg);
    for (int64_t layer = 0; layer < cfg.num_layers; ++layer) {
        const auto ww = attention_weights(w.layers[static_cast<size_t>(layer)]);
        const auto layer_start = std::chrono::steady_clock::now();
        attention_decoder_layer_step(*in, ww, cos_table, sin_table,
                                     static_cast<int32_t>(state.seq_len), state.seq_len,
                                     acfg, state.layers[static_cast<size_t>(layer)], *out, stream);
        std::swap(in, out);
        if (profile_enabled()) {
            check_acl(aclrtSynchronizeStream(stream), "profile decode layer sync");
            const double ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - layer_start).count();
            std::cerr << "# profile decode.layer." << layer << " ms=" << ms << '\n';
        }
    }
    ++state.seq_len;
    return lm_head_greedy_with_scratch(*in, w, cfg, state.lm_head_scratch, stream);
}

bool is_eos(int64_t token_id, const std::vector<int64_t>& eos_token_ids) {
    return std::find(eos_token_ids.begin(), eos_token_ids.end(), token_id) != eos_token_ids.end();
}

}  // namespace minicpmv
