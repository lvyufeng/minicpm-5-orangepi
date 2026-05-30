#include "minicpmv/ops.h"
#include "minicpmv/acl_context.h"

#include <cstdint>
#include <stdexcept>
#include <vector>

#include <aclnnop/aclnn_mm.h>
#include <aclnnop/aclnn_argmax.h>
#include <aclnnop/aclnn_add.h>
#include <aclnnop/aclnn_mul.h>
#include <aclnnop/aclnn_silu.h>
#include <aclnnop/aclnn_sigmoid.h>
#include <aclnnop/aclnn_softmax.h>
#include <aclnnop/aclnn_cast.h>
#include <aclnnop/aclnn_mean.h>
#include <aclnnop/aclnn_rsqrt.h>
#include <aclnnop/aclnn_sub.h>
#include "aclnn_logits_top1_custom.h"
#include "aclnn_matmul_w4a16_custom.h"
#include "aclnn_matmul_w8a8_i32_custom.h"
#include "aclnn_w8a8_quantize_custom.h"
#include "aclnn_w8a8_dequant_custom.h"
#include "aclnn_attention_step_custom.h"
#include "aclnn_rope_cache_write_custom.h"
#include "aclnn_rope_prefill_custom.h"
#include "aclnn_prefill_attention_custom.h"
#include "aclnn_silu_mul_custom.h"
#include "aclnn_rms_norm_custom.h"
#include "aclnn_matmul_vec_custom.h"
#include "aclnn_matmul_cube_custom.h"

namespace minicpmv {

void embedding_lookup(const Tensor& weight,
                      const std::vector<int32_t>& host_ids,
                      Tensor& out,
                      aclrtStream stream) {
    if (weight.shape().size() != 2) {
        throw std::runtime_error("embedding weight must be 2D [V, H]");
    }
    if (out.dtype() != weight.dtype()) {
        throw std::runtime_error("embedding out dtype must match weight dtype");
    }
    if (out.shape().size() != 2) {
        throw std::runtime_error("embedding out must be 2D [N, H]");
    }

    const int64_t vocab = weight.shape()[0];
    const int64_t hidden = weight.shape()[1];
    const int64_t n = out.shape()[0];

    if (out.shape()[1] != hidden) {
        throw std::runtime_error("embedding out hidden mismatch");
    }
    if (static_cast<int64_t>(host_ids.size()) != n) {
        throw std::runtime_error("embedding host_ids size must equal out rows");
    }

    const size_t row_bytes = static_cast<size_t>(hidden) * dtype_size(weight.dtype());
    auto* weight_base = static_cast<const uint8_t*>(weight.data());
    auto* out_base = static_cast<uint8_t*>(out.data());

    for (int64_t i = 0; i < n; ++i) {
        const int32_t id = host_ids[i];
        if (id < 0 || id >= vocab) {
            throw std::runtime_error("embedding id out of range");
        }
        const void* src = weight_base + static_cast<size_t>(id) * row_bytes;
        void* dst = out_base + static_cast<size_t>(i) * row_bytes;
        check_acl(aclrtMemcpyAsync(dst, row_bytes, src, row_bytes,
                                   ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                  "aclrtMemcpyAsync embedding row");
    }
}

namespace {

struct AclTensorHandle {
    aclTensor* tensor{nullptr};
    std::vector<int64_t> view_dims;
    std::vector<int64_t> strides;
    std::vector<int64_t> storage_dims;

    ~AclTensorHandle() { if (tensor) aclDestroyTensor(tensor); }
    AclTensorHandle() = default;
    AclTensorHandle(const AclTensorHandle&) = delete;
    AclTensorHandle& operator=(const AclTensorHandle&) = delete;
};

void make_acl_tensor(const Tensor& t, AclTensorHandle& h) {
    h.view_dims = t.shape();
    h.storage_dims = t.shape();
    h.strides.assign(t.shape().size(), 1);
    for (int i = static_cast<int>(t.shape().size()) - 2; i >= 0; --i) {
        h.strides[i] = h.strides[i + 1] * t.shape()[i + 1];
    }
    h.tensor = aclCreateTensor(
        h.view_dims.data(), h.view_dims.size(), to_acl_dtype(t.dtype()),
        h.strides.data(), 0, ACL_FORMAT_ND,
        h.storage_dims.data(), h.storage_dims.size(),
        t.data());
    if (h.tensor == nullptr) {
        throw std::runtime_error("aclCreateTensor returned null");
    }
}

}  // namespace

void matmul(const Tensor& a, const Tensor& b, Tensor& out, aclrtStream stream) {
    if (a.shape().size() != 2 || b.shape().size() != 2 || out.shape().size() != 2) {
        throw std::runtime_error("matmul tensors must be 2D");
    }
    if (a.shape()[1] != b.shape()[0]) {
        throw std::runtime_error("matmul K dim mismatch");
    }
    if (a.shape()[0] != out.shape()[0] || b.shape()[1] != out.shape()[1]) {
        throw std::runtime_error("matmul out shape mismatch");
    }

    AclTensorHandle ha, hb, ho;
    make_acl_tensor(a, ha);
    make_acl_tensor(b, hb);
    make_acl_tensor(out, ho);

    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    constexpr int8_t kCubeMathType = 1;  // ALLOW_FP32_DOWN_PRECISION
    auto ret = aclnnMmGetWorkspaceSize(ha.tensor, hb.tensor, ho.tensor,
                                       kCubeMathType, &ws_size, &executor);
    if (ret != 0) {
        throw std::runtime_error("aclnnMmGetWorkspaceSize failed: " + std::to_string(ret));
    }

    void* workspace = nullptr;
    if (ws_size > 0) {
        check_acl(aclrtMalloc(&workspace, ws_size, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc workspace");
    }

    ret = aclnnMm(workspace, ws_size, executor, stream);
    if (ret != 0) {
        if (workspace) aclrtFree(workspace);
        throw std::runtime_error("aclnnMm failed: " + std::to_string(ret));
    }

    auto sync_ret = aclrtSynchronizeStream(stream);
    if (workspace) aclrtFree(workspace);
    check_acl(sync_ret, "aclrtSynchronizeStream matmul");
}


void matmul_b_transposed(const Tensor& a, const Tensor& b, Tensor& out, aclrtStream stream) {
    if (a.shape().size() != 2 || b.shape().size() != 2 || out.shape().size() != 2) {
        throw std::runtime_error("matmul_b_transposed tensors must be 2D");
    }
    const int64_t M = a.shape()[0];
    const int64_t K = a.shape()[1];

    // B can be either [N, K] (legacy matmul_b_transposed convention) or [K, N]
    // (pre-transposed for cube fast path). When N != K only one matches.
    const bool bIsTransposed = (b.shape()[1] == K);  // storage [N, K]
    const bool bIsNatural    = (b.shape()[0] == K) && (b.shape()[1] != K);  // [K, N], unambiguous
    if (!bIsTransposed && !bIsNatural) {
        throw std::runtime_error("matmul_b_transposed K dim mismatch");
    }
    const int64_t N = bIsTransposed ? b.shape()[0] : b.shape()[1];
    if (a.shape()[0] != out.shape()[0] || N != out.shape()[1]) {
        throw std::runtime_error("matmul_b_transposed out shape mismatch");
    }

    // Cube fast path: B already pre-transposed to [K, N], M=1, N divisible by
    // 128 (= 8 cores * 16 align), N <= 16384. Cube beats aclnnMm 2-7x at these
    // shapes per bench_matmul_vec. Larger N (e.g., lm_head N=248094) falls
    // back to aclnnMm because the cube tiling currently produces wrong output
    // for N >= ~32k.
    if (bIsNatural && M == 1 && a.dtype() == DType::Float16 &&
        b.dtype() == DType::Float16 && out.dtype() == DType::Float16 &&
        N <= 16384 && (N % 128) == 0) {
        AclTensorHandle ha2, hb2, ho2;
        make_acl_tensor(a, ha2);
        make_acl_tensor(b, hb2);
        make_acl_tensor(out, ho2);
        uint64_t ws_size = 0;
        aclOpExecutor* executor = nullptr;
        auto ret = aclnnMatmulCubeCustomGetWorkspaceSize(ha2.tensor, hb2.tensor, ho2.tensor,
                                                          &ws_size, &executor);
        if (ret != 0) {
            throw std::runtime_error("aclnnMatmulCubeCustomGetWorkspaceSize failed: " + std::to_string(ret));
        }
        void* workspace = nullptr;
        if (ws_size > 0) {
            check_acl(aclrtMalloc(&workspace, ws_size, ACL_MEM_MALLOC_HUGE_FIRST), "matmul_cube ws malloc");
        }
        ret = aclnnMatmulCubeCustom(workspace, ws_size, executor, stream);
        auto sync_ret = aclrtSynchronizeStream(stream);
        if (workspace) aclrtFree(workspace);
        if (ret != 0) {
            throw std::runtime_error("aclnnMatmulCubeCustom failed: " + std::to_string(ret));
        }
        check_acl(sync_ret, "aclrtSynchronizeStream matmul_cube");
        return;
    }

    // CANN's precompiled MatMulV2_FP16 kernel binary doesn't cover the
    // (M >= 64, K > 4096) corner — kernel lookup returns "kernel pointer null"
    // (errno 361001). Empirically M=32 works at any K and M=1024 works at
    // K<=4096. For larger K we tile along the K dim, accumulate partials.
    constexpr int64_t kKTile = 4096;
    if (M >= 64 && K > kKTile && a.dtype() == DType::Float16 &&
        b.dtype() == DType::Float16 && out.dtype() == DType::Float16) {
        const int64_t num_chunks = (K + kKTile - 1) / kKTile;
        Tensor accum({M, N}, DType::Float16); accum.allocate();
        check_acl(aclrtMemsetAsync(accum.data(), accum.size_bytes(), 0,
                                   accum.size_bytes(), stream),
                  "matmul K-tile memset");

        const size_t elem = dtype_size(a.dtype());
        for (int64_t i = 0; i < num_chunks; ++i) {
            const int64_t k_start = i * kKTile;
            const int64_t k_chunk = std::min<int64_t>(kKTile, K - k_start);

            Tensor a_chunk(std::vector<int64_t>{M, k_chunk}, a.dtype()); a_chunk.allocate();
            // a[:, k_start : k_start+k_chunk]: copy row by row (M rows, each
            // k_chunk*elem bytes). aclrtMemcpy2dAsync turned out to corrupt
            // the stack on this CANN build; per-row async memcpy is fine.
            for (int64_t r = 0; r < M; ++r) {
                check_acl(aclrtMemcpyAsync(
                    static_cast<uint8_t*>(a_chunk.data()) + r * k_chunk * elem,
                    k_chunk * elem,
                    static_cast<const uint8_t*>(a.data()) + (r * K + k_start) * elem,
                    k_chunk * elem,
                    ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                    "matmul K-tile a row slice");
            }

            std::vector<int64_t> b_chunk_shape;
            if (bIsTransposed) b_chunk_shape = {N, k_chunk};
            else               b_chunk_shape = {k_chunk, N};
            Tensor b_chunk(b_chunk_shape, b.dtype()); b_chunk.allocate();

            if (bIsTransposed) {
                // b is [N, K]; want b[:, k_start : k_start+k_chunk] → [N, k_chunk]
                for (int64_t r = 0; r < N; ++r) {
                    check_acl(aclrtMemcpyAsync(
                        static_cast<uint8_t*>(b_chunk.data()) + r * k_chunk * elem,
                        k_chunk * elem,
                        static_cast<const uint8_t*>(b.data()) + (r * K + k_start) * elem,
                        k_chunk * elem,
                        ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                        "matmul K-tile b row slice (transposed)");
                }
            } else {
                // b is [K, N]; want b[k_start : k_start+k_chunk, :] — contiguous block.
                check_acl(aclrtMemcpyAsync(
                    b_chunk.data(), b_chunk.size_bytes(),
                    static_cast<const uint8_t*>(b.data()) + k_start * N * elem,
                    static_cast<size_t>(k_chunk * N) * elem,
                    ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                    "matmul K-tile b slice (natural)");
            }

            Tensor partial(std::vector<int64_t>{M, N}, DType::Float16); partial.allocate();
            matmul_b_transposed(a_chunk, b_chunk, partial, stream);
            add(accum, partial, accum, stream);
        }
        check_acl(aclrtMemcpyAsync(out.data(), out.size_bytes(),
                                   accum.data(), accum.size_bytes(),
                                   ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                  "matmul K-tile out copy");
        check_acl(aclrtSynchronizeStream(stream), "matmul K-tile sync");
        return;
    }

    AclTensorHandle ha, hb, ho;
    make_acl_tensor(a, ha);
    make_acl_tensor(out, ho);

    if (bIsNatural) {
        make_acl_tensor(b, hb);  // [K, N], no view dance — aclnnMm handles natural B
    } else {
        // Legacy: build a transposed view for B: storage [N,K], logical [K,N].
        hb.storage_dims = b.shape();
        hb.view_dims = {b.shape()[1], b.shape()[0]};
        hb.strides = {1, b.shape()[1]};
        hb.tensor = aclCreateTensor(
            hb.view_dims.data(), hb.view_dims.size(), to_acl_dtype(b.dtype()),
            hb.strides.data(), 0, ACL_FORMAT_ND,
            hb.storage_dims.data(), hb.storage_dims.size(),
            b.data());
        if (hb.tensor == nullptr) {
            throw std::runtime_error("aclCreateTensor returned null for transposed B view");
        }
    }

    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    constexpr int8_t kCubeMathType = 1;
    auto ret = aclnnMmGetWorkspaceSize(ha.tensor, hb.tensor, ho.tensor,
                                       kCubeMathType, &ws_size, &executor);
    if (ret != 0) {
        throw std::runtime_error("aclnnMmGetWorkspaceSize (B^T) failed: " + std::to_string(ret));
    }
    void* workspace = nullptr;
    if (ws_size > 0) {
        check_acl(aclrtMalloc(&workspace, ws_size, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc workspace");
    }
    ret = aclnnMm(workspace, ws_size, executor, stream);
    if (ret != 0) {
        if (workspace) aclrtFree(workspace);
        throw std::runtime_error("aclnnMm (B^T) failed: " + std::to_string(ret));
    }
    auto sync_ret = aclrtSynchronizeStream(stream);
    if (workspace) aclrtFree(workspace);
    check_acl(sync_ret, "aclrtSynchronizeStream matmul_b_transposed");
}

void argmax_last_dim(const Tensor& self, Tensor& out, aclrtStream stream) {
    if (self.shape().empty()) {
        throw std::runtime_error("argmax input must have rank >= 1");
    }
    const int64_t dim = static_cast<int64_t>(self.shape().size() - 1);

    AclTensorHandle hself, hout;
    make_acl_tensor(self, hself);
    make_acl_tensor(out, hout);

    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnArgMaxGetWorkspaceSize(hself.tensor, dim, false, hout.tensor, &ws_size, &executor);
    if (ret != 0) {
        throw std::runtime_error("aclnnArgMaxGetWorkspaceSize failed: " + std::to_string(ret));
    }

    void* workspace = nullptr;
    if (ws_size > 0) {
        check_acl(aclrtMalloc(&workspace, ws_size, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc workspace");
    }

    ret = aclnnArgMax(workspace, ws_size, executor, stream);
    if (ret != 0) {
        if (workspace) aclrtFree(workspace);
        throw std::runtime_error("aclnnArgMax failed: " + std::to_string(ret));
    }

    auto sync_ret = aclrtSynchronizeStream(stream);
    if (workspace) aclrtFree(workspace);
    check_acl(sync_ret, "aclrtSynchronizeStream argmax");
}

namespace {

void run_op(const char* name,
            uint64_t ws_size,
            aclOpExecutor* executor,
            aclrtStream stream,
            int (*launch)(void*, uint64_t, aclOpExecutor*, aclrtStream)) {
    void* workspace = nullptr;
    if (ws_size > 0) {
        check_acl(aclrtMalloc(&workspace, ws_size, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc workspace");
        check_acl(aclrtMemsetAsync(workspace, ws_size, 0, ws_size, stream), "aclrtMemsetAsync workspace");
    }
    auto ret = launch(workspace, ws_size, executor, stream);
    if (ret != 0) {
        if (workspace) aclrtFree(workspace);
        throw std::runtime_error(std::string(name) + " failed: " + std::to_string(ret));
    }
    if (workspace) {
        auto sync_ret = aclrtSynchronizeStream(stream);
        aclrtFree(workspace);
        check_acl(sync_ret, "aclrtSynchronizeStream");
    }
}

void check_same_shape(const Tensor& a, const Tensor& b, const char* op) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error(std::string(op) + " shape mismatch");
    }
}

}  // namespace

void incre_flash_attention(const Tensor& query,
                           const Tensor& k_cache,
                           const Tensor& v_cache,
                           int64_t context,
                           int64_t num_q_heads,
                           int64_t num_kv_heads,
                           int64_t head_dim,
                           float scale,
                           Tensor& out,
                           aclrtStream stream) {
    if (query.dtype() != DType::Float16 || k_cache.dtype() != DType::Float16 ||
        v_cache.dtype() != DType::Float16 || out.dtype() != DType::Float16) {
        throw std::runtime_error("incre_flash_attention requires fp16 tensors");
    }
    if (query.numel() != static_cast<size_t>(num_q_heads * head_dim)) {
        throw std::runtime_error("incre_flash_attention query numel mismatch");
    }
    if (out.numel() != static_cast<size_t>(num_q_heads * head_dim)) {
        throw std::runtime_error("incre_flash_attention out numel mismatch");
    }
    if (k_cache.shape().size() != 2 || k_cache.shape()[1] != num_kv_heads * head_dim) {
        throw std::runtime_error("incre_flash_attention k_cache shape mismatch");
    }
    if (v_cache.shape() != k_cache.shape()) {
        throw std::runtime_error("incre_flash_attention v_cache shape mismatch");
    }
    if (context <= 0 || context > k_cache.shape()[0]) {
        throw std::runtime_error("incre_flash_attention context out of range");
    }

    // Our custom op AttentionStepCustom: one block per q-head, computes
    // softmax(q · K_kvh^T * scale) · V_kvh and writes into out.
    AclTensorHandle hq, hk, hv, ho;
    make_acl_tensor(query, hq);
    make_acl_tensor(k_cache, hk);
    make_acl_tensor(v_cache, hv);
    make_acl_tensor(out, ho);

    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnAttentionStepCustomGetWorkspaceSize(
        hq.tensor, hk.tensor, hv.tensor,
        context, num_q_heads, num_kv_heads, static_cast<double>(scale),
        ho.tensor, &ws_size, &executor);
    if (ret != 0) {
        throw std::runtime_error("aclnnAttentionStepCustomGetWorkspaceSize failed: " + std::to_string(ret));
    }
    run_op("aclnnAttentionStepCustom", ws_size, executor, stream, aclnnAttentionStepCustom);
}

void silu_mul(const Tensor& gate, const Tensor& up, Tensor& out, aclrtStream stream) {
    if (gate.dtype() != DType::Float16 || up.dtype() != DType::Float16 || out.dtype() != DType::Float16) {
        throw std::runtime_error("silu_mul requires fp16 tensors");
    }
    if (gate.shape() != up.shape() || gate.shape() != out.shape()) {
        throw std::runtime_error("silu_mul shape mismatch");
    }
    AclTensorHandle hg, hu, ho;
    make_acl_tensor(gate, hg);
    make_acl_tensor(up, hu);
    make_acl_tensor(out, ho);

    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnSiluMulCustomGetWorkspaceSize(hg.tensor, hu.tensor, ho.tensor, &ws_size, &executor);
    if (ret != 0) {
        throw std::runtime_error("aclnnSiluMulCustomGetWorkspaceSize failed: " + std::to_string(ret));
    }
    run_op("aclnnSiluMulCustom", ws_size, executor, stream, aclnnSiluMulCustom);
}

void add(const Tensor& a, const Tensor& b, Tensor& out, aclrtStream stream) {
    check_same_shape(a, b, "add");
    check_same_shape(a, out, "add");
    AclTensorHandle ha, hb, ho;
    make_acl_tensor(a, ha);
    make_acl_tensor(b, hb);
    make_acl_tensor(out, ho);
    float alpha_value = 1.0f;
    aclScalar* alpha = aclCreateScalar(&alpha_value, ACL_FLOAT);
    if (alpha == nullptr) throw std::runtime_error("aclCreateScalar(alpha) failed");
    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(ha.tensor, hb.tensor, alpha, ho.tensor, &ws_size, &executor);
    if (ret != 0) {
        aclDestroyScalar(alpha);
        throw std::runtime_error("aclnnAddGetWorkspaceSize failed: " + std::to_string(ret));
    }
    try {
        run_op("aclnnAdd", ws_size, executor, stream, aclnnAdd);
    } catch (...) {
        aclDestroyScalar(alpha);
        throw;
    }
    aclDestroyScalar(alpha);
}

void mul(const Tensor& a, const Tensor& b, Tensor& out, aclrtStream stream) {
    check_same_shape(a, b, "mul");
    check_same_shape(a, out, "mul");
    AclTensorHandle ha, hb, ho;
    make_acl_tensor(a, ha);
    make_acl_tensor(b, hb);
    make_acl_tensor(out, ho);
    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnMulGetWorkspaceSize(ha.tensor, hb.tensor, ho.tensor, &ws_size, &executor);
    if (ret != 0) throw std::runtime_error("aclnnMulGetWorkspaceSize failed: " + std::to_string(ret));
    run_op("aclnnMul", ws_size, executor, stream, aclnnMul);
}

void silu(const Tensor& self, Tensor& out, aclrtStream stream) {
    check_same_shape(self, out, "silu");
    AclTensorHandle hs, ho;
    make_acl_tensor(self, hs);
    make_acl_tensor(out, ho);
    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnSiluGetWorkspaceSize(hs.tensor, ho.tensor, &ws_size, &executor);
    if (ret != 0) throw std::runtime_error("aclnnSiluGetWorkspaceSize failed: " + std::to_string(ret));
    run_op("aclnnSilu", ws_size, executor, stream, aclnnSilu);
}

void sigmoid(const Tensor& self, Tensor& out, aclrtStream stream) {
    check_same_shape(self, out, "sigmoid");
    AclTensorHandle hs, ho;
    make_acl_tensor(self, hs);
    make_acl_tensor(out, ho);
    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnSigmoidGetWorkspaceSize(hs.tensor, ho.tensor, &ws_size, &executor);
    if (ret != 0) throw std::runtime_error("aclnnSigmoidGetWorkspaceSize failed: " + std::to_string(ret));
    run_op("aclnnSigmoid", ws_size, executor, stream, aclnnSigmoid);
}

void softmax_last_dim(const Tensor& self, Tensor& out, aclrtStream stream) {
    check_same_shape(self, out, "softmax");
    if (self.shape().empty()) throw std::runtime_error("softmax input must have rank >= 1");
    const int64_t dim = static_cast<int64_t>(self.shape().size() - 1);
    AclTensorHandle hs, ho;
    make_acl_tensor(self, hs);
    make_acl_tensor(out, ho);
    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnSoftmaxGetWorkspaceSize(hs.tensor, dim, ho.tensor, &ws_size, &executor);
    if (ret != 0) throw std::runtime_error("aclnnSoftmaxGetWorkspaceSize failed: " + std::to_string(ret));
    run_op("aclnnSoftmax", ws_size, executor, stream, aclnnSoftmax);
}

namespace {

bool tensor_ready(const Tensor& t, const std::vector<int64_t>& shape, DType dtype) {
    return t.data() != nullptr && t.dtype() == dtype && t.shape() == shape;
}

void ensure_tensor(Tensor& t, const std::vector<int64_t>& shape, DType dtype) {
    if (tensor_ready(t, shape, dtype)) return;
    t = Tensor(shape, dtype);
    t.allocate();
}

void ensure_rms_norm_scratch(RmsNormScratch& s,
                             const std::vector<int64_t>& x_shape,
                             const std::vector<int64_t>& gamma_shape,
                             const std::vector<int64_t>& reduce_shape) {
    ensure_tensor(s.x_f32, x_shape, DType::Float32);
    ensure_tensor(s.gamma_f32, gamma_shape, DType::Float32);
    ensure_tensor(s.x_sq, x_shape, DType::Float32);
    ensure_tensor(s.mean_x_sq, reduce_shape, DType::Float32);
    ensure_tensor(s.rstd, reduce_shape, DType::Float32);
    ensure_tensor(s.scaled, x_shape, DType::Float32);
    ensure_tensor(s.normed_f32, x_shape, DType::Float32);
}

}  // namespace

void rms_norm_with_scratch(const Tensor& x, const Tensor& gamma, Tensor& out,
                           double epsilon, RmsNormScratch& scratch, aclrtStream stream) {
    if (x.shape() != out.shape()) {
        throw std::runtime_error("rms_norm x/out shape mismatch");
    }
    if (x.shape().empty()) {
        throw std::runtime_error("rms_norm input must have rank >= 1");
    }
    const int64_t hidden = x.shape().back();
    if (gamma.shape().size() != 1 || gamma.shape()[0] != hidden) {
        throw std::runtime_error("rms_norm gamma shape must be [H] matching last dim of x");
    }
    if (x.dtype() != gamma.dtype() || x.dtype() != out.dtype()) {
        throw std::runtime_error("rms_norm dtype mismatch (x/gamma/out must match)");
    }

    if (x.dtype() == DType::Float16 && hidden > 0 && hidden <= 2048) {
        int64_t rows = 1;
        for (size_t i = 0; i + 1 < x.shape().size(); ++i) {
            rows *= x.shape()[i];
        }
        if (rows > 0 && rows <= 1024) {
            AclTensorHandle hx, hg, ho;
            make_acl_tensor(x, hx);
            make_acl_tensor(gamma, hg);
            make_acl_tensor(out, ho);
            uint64_t ws_size = 0;
            aclOpExecutor* executor = nullptr;
            auto ret = aclnnRmsNormCustomGetWorkspaceSize(
                hx.tensor, hg.tensor, static_cast<float>(epsilon), ho.tensor,
                &ws_size, &executor);
            if (ret != 0) {
                throw std::runtime_error("aclnnRmsNormCustomGetWorkspaceSize failed: " + std::to_string(ret));
            }
            run_op("aclnnRmsNormCustom", ws_size, executor, stream, aclnnRmsNormCustom);
            return;
        }
    }

    // Build [..., 1] reduce shape
    std::vector<int64_t> reduce_shape = x.shape();
    reduce_shape.back() = 1;
    ensure_rms_norm_scratch(scratch, x.shape(), gamma.shape(), reduce_shape);

    Tensor& x_f32 = scratch.x_f32;
    Tensor& gamma_f32 = scratch.gamma_f32;
    Tensor& x_sq = scratch.x_sq;
    Tensor& mean_x_sq = scratch.mean_x_sq;
    Tensor& rstd = scratch.rstd;
    Tensor& scaled = scratch.scaled;
    Tensor& normed_f32 = scratch.normed_f32;

    cast(x, x_f32, stream);
    cast(gamma, gamma_f32, stream);

    // 1) x_sq = x * x
    {
        AclTensorHandle hx, hx2, hsq;
        make_acl_tensor(x_f32, hx);
        make_acl_tensor(x_f32, hx2);
        make_acl_tensor(x_sq, hsq);
        uint64_t ws_size = 0;
        aclOpExecutor* executor = nullptr;
        auto ret = aclnnMulGetWorkspaceSize(hx.tensor, hx2.tensor, hsq.tensor, &ws_size, &executor);
        if (ret != 0) throw std::runtime_error("rms_norm Mul(x,x) ws failed: " + std::to_string(ret));
        run_op("rms_norm Mul(x,x)", ws_size, executor, stream, aclnnMul);
    }

    // 2) mean(x_sq, dim=-1, keepDim=true)
    {
        AclTensorHandle hsq, hmean;
        make_acl_tensor(x_sq, hsq);
        make_acl_tensor(mean_x_sq, hmean);
        const int64_t last_dim = static_cast<int64_t>(x.shape().size() - 1);
        std::vector<int64_t> dim_data{last_dim};
        aclIntArray* dim = aclCreateIntArray(dim_data.data(), dim_data.size());
        if (dim == nullptr) throw std::runtime_error("rms_norm aclCreateIntArray failed");
        uint64_t ws_size = 0;
        aclOpExecutor* executor = nullptr;
        auto ret = aclnnMeanGetWorkspaceSize(hsq.tensor, dim, true,
                                             ACL_FLOAT,
                                             hmean.tensor, &ws_size, &executor);
        if (ret != 0) {
            aclDestroyIntArray(dim);
            throw std::runtime_error("rms_norm Mean ws failed: " + std::to_string(ret));
        }
        try {
            run_op("rms_norm Mean", ws_size, executor, stream, aclnnMean);
        } catch (...) {
            aclDestroyIntArray(dim);
            throw;
        }
        aclDestroyIntArray(dim);
    }

    // 3) mean += eps; rstd = rsqrt(mean)
    {
        AclTensorHandle hmean_in, hmean_out;
        make_acl_tensor(mean_x_sq, hmean_in);
        make_acl_tensor(rstd, hmean_out);

        float eps_f = static_cast<float>(epsilon);
        aclScalar* eps_scalar = aclCreateScalar(&eps_f, ACL_FLOAT);
        if (eps_scalar == nullptr) throw std::runtime_error("rms_norm aclCreateScalar(eps) failed");
        float alpha_f = 1.0f;
        aclScalar* alpha_scalar = aclCreateScalar(&alpha_f, ACL_FLOAT);
        if (alpha_scalar == nullptr) {
            aclDestroyScalar(eps_scalar);
            throw std::runtime_error("rms_norm aclCreateScalar(alpha) failed");
        }
        uint64_t ws_size = 0;
        aclOpExecutor* executor = nullptr;
        auto ret = aclnnAddsGetWorkspaceSize(hmean_in.tensor, eps_scalar, alpha_scalar,
                                             hmean_out.tensor, &ws_size, &executor);
        if (ret != 0) {
            aclDestroyScalar(eps_scalar);
            aclDestroyScalar(alpha_scalar);
            throw std::runtime_error("rms_norm Adds ws failed: " + std::to_string(ret));
        }
        try {
            run_op("rms_norm Adds", ws_size, executor, stream, aclnnAdds);
        } catch (...) {
            aclDestroyScalar(eps_scalar);
            aclDestroyScalar(alpha_scalar);
            throw;
        }
        aclDestroyScalar(eps_scalar);
        aclDestroyScalar(alpha_scalar);
    }
    {
        AclTensorHandle hin, hout;
        make_acl_tensor(rstd, hin);
        make_acl_tensor(rstd, hout);
        uint64_t ws_size = 0;
        aclOpExecutor* executor = nullptr;
        auto ret = aclnnRsqrtGetWorkspaceSize(hin.tensor, hout.tensor, &ws_size, &executor);
        if (ret != 0) throw std::runtime_error("rms_norm Rsqrt ws failed: " + std::to_string(ret));
        run_op("rms_norm Rsqrt", ws_size, executor, stream, aclnnRsqrt);
    }

    // 4) scaled = x * rstd (broadcast last dim)
    {
        AclTensorHandle hx, hrstd, hout;
        make_acl_tensor(x_f32, hx);
        make_acl_tensor(rstd, hrstd);
        make_acl_tensor(scaled, hout);
        uint64_t ws_size = 0;
        aclOpExecutor* executor = nullptr;
        auto ret = aclnnMulGetWorkspaceSize(hx.tensor, hrstd.tensor, hout.tensor, &ws_size, &executor);
        if (ret != 0) throw std::runtime_error("rms_norm Mul(x,rstd) ws failed: " + std::to_string(ret));
        run_op("rms_norm Mul(x,rstd)", ws_size, executor, stream, aclnnMul);
    }

    // 5) out = cast(scaled * gamma)
    {
        AclTensorHandle hscaled, hgamma, hout;
        make_acl_tensor(scaled, hscaled);
        make_acl_tensor(gamma_f32, hgamma);
        make_acl_tensor(normed_f32, hout);
        uint64_t ws_size = 0;
        aclOpExecutor* executor = nullptr;
        auto ret = aclnnMulGetWorkspaceSize(hscaled.tensor, hgamma.tensor, hout.tensor, &ws_size, &executor);
        if (ret != 0) throw std::runtime_error("rms_norm Mul(*,gamma) ws failed: " + std::to_string(ret));
        run_op("rms_norm Mul(*,gamma)", ws_size, executor, stream, aclnnMul);
    }
    cast(normed_f32, out, stream);
}

void rms_norm(const Tensor& x, const Tensor& gamma, Tensor& out,
              double epsilon, aclrtStream stream) {
    RmsNormScratch scratch;
    rms_norm_with_scratch(x, gamma, out, epsilon, scratch, stream);
}

void cast(const Tensor& self, Tensor& out, aclrtStream stream) {
    if (self.shape() != out.shape()) {
        throw std::runtime_error("cast shape mismatch");
    }
    AclTensorHandle hs, ho;
    make_acl_tensor(self, hs);
    make_acl_tensor(out, ho);
    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnCastGetWorkspaceSize(hs.tensor, to_acl_dtype(out.dtype()), ho.tensor,
                                         &ws_size, &executor);
    if (ret != 0) {
        throw std::runtime_error("aclnnCastGetWorkspaceSize failed: " + std::to_string(ret));
    }
    run_op("aclnnCast", ws_size, executor, stream, aclnnCast);
}

namespace {

void d2d_row_copy(const void* src, size_t src_row_stride_bytes,
                  void* dst, size_t dst_row_stride_bytes,
                  size_t row_bytes, int64_t rows, aclrtStream stream) {
    auto* s = static_cast<const uint8_t*>(src);
    auto* d = static_cast<uint8_t*>(dst);
    for (int64_t r = 0; r < rows; ++r) {
        check_acl(aclrtMemcpyAsync(d + r * dst_row_stride_bytes, row_bytes,
                                   s + r * src_row_stride_bytes, row_bytes,
                                   ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                  "aclrtMemcpyAsync row copy");
    }
}

void ensure_rope_scratch(RopeScratch& s, int64_t N, int64_t HalfRot) {
    const std::vector<int64_t> shape{N, HalfRot};
    if (s.x1.shape() == shape && s.x2.shape() == shape &&
        s.cos_e.shape() == shape && s.sin_e.shape() == shape &&
        s.a.shape() == shape && s.b.shape() == shape &&
        s.y1.shape() == shape && s.y2.shape() == shape) {
        return;
    }
    auto make = [](const std::vector<int64_t>& tensor_shape) {
        Tensor t(tensor_shape, DType::Float16);
        t.allocate();
        return t;
    };
    s.x1 = make(shape);
    s.x2 = make(shape);
    s.cos_e = make(shape);
    s.sin_e = make(shape);
    s.a = make(shape);
    s.b = make(shape);
    s.y1 = make(shape);
    s.y2 = make(shape);
}

}  // namespace

void apply_rope_partial_with_scratch(const Tensor& x,
                                     const Tensor& cos_table,
                                     const Tensor& sin_table,
                                     const std::vector<int32_t>& row_to_t,
                                     int64_t rot,
                                     RopeScratch& scratch,
                                     Tensor& out,
                                     aclrtStream stream) {
    if (x.dtype() != DType::Float16 || cos_table.dtype() != DType::Float16 ||
        sin_table.dtype() != DType::Float16 || out.dtype() != DType::Float16) {
        throw std::runtime_error("apply_rope_partial requires fp16 tensors");
    }
    if (x.shape().size() != 2 || out.shape().size() != 2) {
        throw std::runtime_error("apply_rope_partial expects x/out shape [N, D]");
    }
    if (x.shape() != out.shape()) {
        throw std::runtime_error("apply_rope_partial x/out shape mismatch");
    }
    if (cos_table.shape().size() != 2 || sin_table.shape().size() != 2 ||
        cos_table.shape() != sin_table.shape()) {
        throw std::runtime_error("apply_rope_partial cos/sin must have shape [T, rot/2]");
    }

    const int64_t N = x.shape()[0];
    const int64_t D = x.shape()[1];
    if (rot <= 0 || rot > D || rot % 2 != 0) {
        throw std::runtime_error("apply_rope_partial rot must be even and <= D");
    }
    const int64_t HalfRot = rot / 2;
    if (cos_table.shape()[1] != HalfRot) {
        throw std::runtime_error("apply_rope_partial cos/sin last dim must be rot/2");
    }
    if (static_cast<int64_t>(row_to_t.size()) != N) {
        throw std::runtime_error("apply_rope_partial row_to_t size must equal N");
    }
    const int64_t T = cos_table.shape()[0];
    for (auto t : row_to_t) {
        if (t < 0 || t >= T) {
            throw std::runtime_error("apply_rope_partial row_to_t entry out of range");
        }
    }

    ensure_rope_scratch(scratch, N, HalfRot);

    const size_t fp16_size = sizeof(uint16_t);
    const size_t row_bytes_x = static_cast<size_t>(D) * fp16_size;
    const size_t half_bytes = static_cast<size_t>(HalfRot) * fp16_size;
    const size_t tail_bytes = static_cast<size_t>(D - rot) * fp16_size;

    Tensor& x1 = scratch.x1;
    Tensor& x2 = scratch.x2;
    Tensor& cos_e = scratch.cos_e;
    Tensor& sin_e = scratch.sin_e;
    Tensor& a = scratch.a;
    Tensor& b = scratch.b;
    Tensor& y1 = scratch.y1;
    Tensor& y2 = scratch.y2;

    auto* x_base = static_cast<const uint8_t*>(x.data());
    auto* out_base = static_cast<uint8_t*>(out.data());
    auto* cos_base = static_cast<const uint8_t*>(cos_table.data());
    auto* sin_base = static_cast<const uint8_t*>(sin_table.data());

    // 1) gather slices: x1, x2, cos_e, sin_e
    for (int64_t n = 0; n < N; ++n) {
        const uint8_t* x_row = x_base + static_cast<size_t>(n) * row_bytes_x;
        check_acl(aclrtMemcpyAsync(static_cast<uint8_t*>(x1.data()) + n * half_bytes, half_bytes,
                                   x_row + 0 * half_bytes, half_bytes,
                                   ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                  "rope copy x1");
        check_acl(aclrtMemcpyAsync(static_cast<uint8_t*>(x2.data()) + n * half_bytes, half_bytes,
                                   x_row + 1 * half_bytes, half_bytes,
                                   ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                  "rope copy x2");
        const int64_t t = row_to_t[n];
        const uint8_t* cos_row = cos_base + static_cast<size_t>(t) * half_bytes;
        const uint8_t* sin_row = sin_base + static_cast<size_t>(t) * half_bytes;
        check_acl(aclrtMemcpyAsync(static_cast<uint8_t*>(cos_e.data()) + n * half_bytes, half_bytes,
                                   cos_row, half_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                  "rope copy cos");
        check_acl(aclrtMemcpyAsync(static_cast<uint8_t*>(sin_e.data()) + n * half_bytes, half_bytes,
                                   sin_row, half_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                  "rope copy sin");
    }

    // 2) a = x1 * cos, b = x2 * sin, y1 = a - b
    mul(x1, cos_e, a, stream);
    mul(x2, sin_e, b, stream);
    {
        AclTensorHandle ha, hb, hy1;
        make_acl_tensor(a, ha);
        make_acl_tensor(b, hb);
        make_acl_tensor(y1, hy1);
        float alpha_f = 1.0f;
        aclScalar* alpha = aclCreateScalar(&alpha_f, ACL_FLOAT);
        if (!alpha) throw std::runtime_error("rope sub alpha alloc failed");
        uint64_t ws_size = 0;
        aclOpExecutor* executor = nullptr;
        auto ret = aclnnSubGetWorkspaceSize(ha.tensor, hb.tensor, alpha, hy1.tensor,
                                            &ws_size, &executor);
        if (ret != 0) {
            aclDestroyScalar(alpha);
            throw std::runtime_error("rope Sub ws failed: " + std::to_string(ret));
        }
        try {
            run_op("rope Sub", ws_size, executor, stream, aclnnSub);
        } catch (...) { aclDestroyScalar(alpha); throw; }
        aclDestroyScalar(alpha);
    }

    // 3) a = x2 * cos, b = x1 * sin, y2 = a + b
    mul(x2, cos_e, a, stream);
    mul(x1, sin_e, b, stream);
    add(a, b, y2, stream);

    // 4) scatter y1, y2 to out[:, 0:HalfRot], out[:, HalfRot:rot]; tail copy
    for (int64_t n = 0; n < N; ++n) {
        uint8_t* o_row = out_base + static_cast<size_t>(n) * row_bytes_x;
        check_acl(aclrtMemcpyAsync(o_row + 0 * half_bytes, half_bytes,
                                   static_cast<uint8_t*>(y1.data()) + n * half_bytes, half_bytes,
                                   ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                  "rope scatter y1");
        check_acl(aclrtMemcpyAsync(o_row + 1 * half_bytes, half_bytes,
                                   static_cast<uint8_t*>(y2.data()) + n * half_bytes, half_bytes,
                                   ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                  "rope scatter y2");
        if (tail_bytes > 0) {
            const uint8_t* x_row = x_base + static_cast<size_t>(n) * row_bytes_x;
            check_acl(aclrtMemcpyAsync(o_row + static_cast<size_t>(rot) * fp16_size, tail_bytes,
                                       x_row + static_cast<size_t>(rot) * fp16_size, tail_bytes,
                                       ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                      "rope scatter tail");
        }
    }
}

void apply_rope_partial(const Tensor& x,
                        const Tensor& cos_table,
                        const Tensor& sin_table,
                        const std::vector<int32_t>& row_to_t,
                        int64_t rot,
                        Tensor& out,
                        aclrtStream stream) {
    RopeScratch scratch;
    apply_rope_partial_with_scratch(x, cos_table, sin_table, row_to_t, rot, scratch, out, stream);
}

void apply_rope_prefill(const Tensor& x,
                        const Tensor& cos_table,
                        const Tensor& sin_table,
                        int64_t heads,
                        int64_t rotary_dim,
                        Tensor& out,
                        aclrtStream stream) {
    if (x.dtype() != DType::Float16 || cos_table.dtype() != DType::Float16 ||
        sin_table.dtype() != DType::Float16 || out.dtype() != DType::Float16) {
        throw std::runtime_error("apply_rope_prefill requires fp16 tensors");
    }
    if (x.shape().size() != 2 || out.shape() != x.shape()) {
        throw std::runtime_error("apply_rope_prefill expects x/out [N, head_dim] same shape");
    }
    if (cos_table.shape().size() != 2 || sin_table.shape() != cos_table.shape()) {
        throw std::runtime_error("apply_rope_prefill cos/sin must be [T, rot/2]");
    }
    const int64_t N = x.shape()[0];
    const int64_t headDim = x.shape()[1];
    if (heads <= 0 || N % heads != 0) {
        throw std::runtime_error("apply_rope_prefill N must be a multiple of heads");
    }
    if (rotary_dim <= 0 || rotary_dim > headDim || rotary_dim % 2 != 0) {
        throw std::runtime_error("apply_rope_prefill rot must be even and <= head_dim");
    }
    if (cos_table.shape()[1] != rotary_dim / 2) {
        throw std::runtime_error("apply_rope_prefill cos/sin last dim must be rot/2");
    }

    AclTensorHandle hx, hcos, hsin, ho;
    make_acl_tensor(x, hx);
    make_acl_tensor(cos_table, hcos);
    make_acl_tensor(sin_table, hsin);
    make_acl_tensor(out, ho);

    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnRopePrefillCustomGetWorkspaceSize(hx.tensor, hcos.tensor, hsin.tensor,
                                                      heads, rotary_dim, ho.tensor,
                                                      &ws_size, &executor);
    if (ret != 0) {
        throw std::runtime_error("aclnnRopePrefillCustomGetWorkspaceSize failed: " + std::to_string(ret));
    }
    run_op("aclnnRopePrefillCustom", ws_size, executor, stream, aclnnRopePrefillCustom);
}

void prefill_attention_custom(const Tensor& q_rope,
                              const Tensor& k_rope,
                              const Tensor& v_full,
                              int64_t seq_len,
                              int64_t num_q_heads,
                              int64_t num_kv_heads,
                              int64_t head_dim,
                              float scale,
                              Tensor& out,
                              aclrtStream stream) {
    if (q_rope.dtype() != DType::Float16 || k_rope.dtype() != DType::Float16 ||
        v_full.dtype() != DType::Float16 || out.dtype() != DType::Float16) {
        throw std::runtime_error("prefill_attention_custom requires fp16 tensors");
    }
    if (seq_len <= 0 || num_q_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0 ||
        num_q_heads % num_kv_heads != 0) {
        throw std::runtime_error("prefill_attention_custom invalid dims");
    }
    if (q_rope.shape() != std::vector<int64_t>{seq_len * num_q_heads, head_dim}) {
        throw std::runtime_error("prefill_attention_custom q_rope shape mismatch");
    }
    if (k_rope.shape() != std::vector<int64_t>{seq_len * num_kv_heads, head_dim}) {
        throw std::runtime_error("prefill_attention_custom k_rope shape mismatch");
    }
    if (v_full.shape() != std::vector<int64_t>{seq_len, num_kv_heads * head_dim}) {
        throw std::runtime_error("prefill_attention_custom v_full shape mismatch");
    }
    if (out.shape() != std::vector<int64_t>{seq_len, num_q_heads * head_dim}) {
        throw std::runtime_error("prefill_attention_custom out shape mismatch");
    }

    AclTensorHandle hq, hk, hv, ho;
    make_acl_tensor(q_rope, hq);
    make_acl_tensor(k_rope, hk);
    make_acl_tensor(v_full, hv);
    make_acl_tensor(out, ho);

    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnPrefillAttentionCustomGetWorkspaceSize(
        hq.tensor, hk.tensor, hv.tensor,
        seq_len, num_q_heads, num_kv_heads, head_dim, static_cast<double>(scale),
        ho.tensor, &ws_size, &executor);
    if (ret != 0) {
        throw std::runtime_error("aclnnPrefillAttentionCustomGetWorkspaceSize failed: " + std::to_string(ret));
    }
    run_op("aclnnPrefillAttentionCustom", ws_size, executor, stream, aclnnPrefillAttentionCustom);
}

void logits_top1(const Tensor& logits,
                 int64_t valid,
                 Tensor& value,
                 Tensor& index,
                 aclrtStream stream) {
    if (logits.dtype() != DType::Float16 || value.dtype() != DType::Float16 || index.dtype() != DType::Int32) {
        throw std::runtime_error("logits_top1 dtype mismatch");
    }
    if (logits.shape().size() != 2 || logits.shape()[0] != 1 ||
        value.shape() != std::vector<int64_t>{1} || index.shape() != std::vector<int64_t>{1}) {
        throw std::runtime_error("logits_top1 shape mismatch");
    }
    if (valid <= 0 || valid > logits.shape()[1]) {
        throw std::runtime_error("logits_top1 valid out of range");
    }

    AclTensorHandle hl, hv, hi;
    make_acl_tensor(logits, hl);
    make_acl_tensor(value, hv);
    make_acl_tensor(index, hi);

    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnLogitsTop1CustomGetWorkspaceSize(hl.tensor, valid, hv.tensor, hi.tensor, &ws_size, &executor);
    if (ret != 0) {
        throw std::runtime_error("aclnnLogitsTop1CustomGetWorkspaceSize failed: " + std::to_string(ret));
    }
    run_op("aclnnLogitsTop1Custom", ws_size, executor, stream, aclnnLogitsTop1Custom);
}

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
                      aclrtStream stream) {
    if (q.dtype() != DType::Float16 || k.dtype() != DType::Float16 || v.dtype() != DType::Float16 ||
        k_cache.dtype() != DType::Float16 || v_cache.dtype() != DType::Float16 ||
        cos_table.dtype() != DType::Float16 || sin_table.dtype() != DType::Float16 ||
        q_rope.dtype() != DType::Float16) {
        throw std::runtime_error("rope_cache_write requires fp16 tensors");
    }
    if (q.shape() != std::vector<int64_t>{num_q_heads, head_dim} ||
        k.shape() != std::vector<int64_t>{num_kv_heads, head_dim} ||
        v.shape() != std::vector<int64_t>{1, num_kv_heads * head_dim} ||
        q_rope.shape() != q.shape() ||
        k_cache.shape().size() != 2 || k_cache.shape()[1] != num_kv_heads * head_dim ||
        v_cache.shape() != k_cache.shape()) {
        throw std::runtime_error("rope_cache_write shape mismatch");
    }
    if (cache_len < 0 || cache_len >= k_cache.shape()[0]) {
        throw std::runtime_error("rope_cache_write cache_len out of range");
    }
    if (pos < 0 || rotary_dim <= 0 || rotary_dim > head_dim || (rotary_dim % 2) != 0) {
        throw std::runtime_error("rope_cache_write invalid position or rotary dim");
    }
    if (cos_table.shape().size() != 2 || sin_table.shape() != cos_table.shape() ||
        cos_table.shape()[0] <= pos || cos_table.shape()[1] != rotary_dim / 2) {
        throw std::runtime_error("rope_cache_write table shape mismatch");
    }

    AclTensorHandle hq, hk, hv, hkc, hvc, hcos, hsin, hqr;
    make_acl_tensor(q, hq);
    make_acl_tensor(k, hk);
    make_acl_tensor(v, hv);
    make_acl_tensor(k_cache, hkc);
    make_acl_tensor(v_cache, hvc);
    make_acl_tensor(cos_table, hcos);
    make_acl_tensor(sin_table, hsin);
    make_acl_tensor(q_rope, hqr);

    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnRopeCacheWriteCustomGetWorkspaceSize(
        hq.tensor, hk.tensor, hv.tensor, hkc.tensor, hvc.tensor, hcos.tensor, hsin.tensor,
        pos, cache_len, num_q_heads, num_kv_heads, head_dim, rotary_dim,
        hqr.tensor, hkc.tensor, hvc.tensor, &ws_size, &executor);
    if (ret != 0) {
        throw std::runtime_error("aclnnRopeCacheWriteCustomGetWorkspaceSize failed: " + std::to_string(ret));
    }
    run_op("aclnnRopeCacheWriteCustom", ws_size, executor, stream, aclnnRopeCacheWriteCustom);
}


void matmul_w4a16(const Tensor& x,
                  const Tensor& w_int8,
                  const Tensor& scales,
                  Tensor& out,
                  aclrtStream stream) {
    // GPTQ W4A16 matmul (M=1 fast path). The weight has been pre-unpacked
    // and block-packed by output-column chunks to match the custom kernel's 8-core split.
    if (x.dtype() != DType::Float16) {
        throw std::runtime_error("matmul_w4a16 x must be fp16");
    }
    if (w_int8.dtype() != DType::Int8) {
        throw std::runtime_error("matmul_w4a16 w must be int8 (pre-unpacked from GPTQ)");
    }
    if (scales.dtype() != DType::Float16) {
        throw std::runtime_error("matmul_w4a16 scales must be fp16");
    }
    if (out.dtype() != DType::Float16) {
        throw std::runtime_error("matmul_w4a16 out must be fp16");
    }
    if (x.shape().size() != 2 || x.shape()[0] != 1) {
        throw std::runtime_error("matmul_w4a16 x must be [1, K]");
    }
    const int64_t K = x.shape()[1];
    constexpr int64_t G = 128;
    if (scales.shape().size() != 2) {
        throw std::runtime_error("matmul_w4a16 scales must be 2D");
    }
    const int64_t N = scales.shape()[1];
    if (w_int8.shape().size() != 2) {
        throw std::runtime_error("matmul_w4a16 w must be [packed_rows, tile_len]");
    }
    const int64_t tileLen = w_int8.shape()[1];
    if (tileLen <= 0 || tileLen > 448 || tileLen % 16 != 0) {
        throw std::runtime_error("matmul_w4a16 tile_len must be a multiple of 16 and <= 448");
    }
    if (K % G != 0 || N % 128 != 0) {
        throw std::runtime_error("matmul_w4a16 K and N must be multiples of 128");
    }
    const int64_t blockLen = (N + 7) / 8;
    const int64_t tilesPerBlock = (blockLen + tileLen - 1) / tileLen;
    const int64_t packedRows = (K / G) * 8 * tilesPerBlock * G;
    if (w_int8.shape()[0] != packedRows) {
        throw std::runtime_error("matmul_w4a16 w packed_rows mismatch");
    }
    if (scales.shape() != std::vector<int64_t>{K / G, N}) {
        throw std::runtime_error("matmul_w4a16 scales must be [K/128, N]");
    }
    if (out.shape() != std::vector<int64_t>{1, N}) {
        throw std::runtime_error("matmul_w4a16 out must be [1, N]");
    }

    AclTensorHandle hx, hw, hs, ho;
    make_acl_tensor(x, hx);
    make_acl_tensor(w_int8, hw);
    make_acl_tensor(scales, hs);
    make_acl_tensor(out, ho);

    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnMatmulW4a16CustomGetWorkspaceSize(hx.tensor, hw.tensor, hs.tensor,
                                                       ho.tensor, &ws_size, &executor);
    if (ret != 0) {
        throw std::runtime_error("aclnnMatmulW4a16CustomGetWorkspaceSize failed: " + std::to_string(ret));
    }
    run_op("aclnnMatmulW4a16Custom", ws_size, executor, stream, aclnnMatmulW4a16Custom);
}

void matmul_w8a8_i32(const Tensor& x,
                     const Tensor& w_int8,
                     Tensor& out,
                     aclrtStream stream) {
    if (x.dtype() != DType::Int8) {
        throw std::runtime_error("matmul_w8a8_i32 x must be int8");
    }
    if (w_int8.dtype() != DType::Int8) {
        throw std::runtime_error("matmul_w8a8_i32 w must be int8");
    }
    if (out.dtype() != DType::Int32) {
        throw std::runtime_error("matmul_w8a8_i32 out must be int32");
    }
    if (x.shape().size() != 2 || x.shape()[0] != 1) {
        throw std::runtime_error("matmul_w8a8_i32 x must be [1, K]");
    }
    if (w_int8.shape().size() != 2 || w_int8.shape()[0] != x.shape()[1]) {
        throw std::runtime_error("matmul_w8a8_i32 w must be [K, N]");
    }
    if (out.shape() != std::vector<int64_t>{1, w_int8.shape()[1]}) {
        throw std::runtime_error("matmul_w8a8_i32 out must be [1, N]");
    }

    AclTensorHandle hx, hw, ho;
    make_acl_tensor(x, hx);
    make_acl_tensor(w_int8, hw);
    make_acl_tensor(out, ho);

    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnMatmulW8a8I32CustomGetWorkspaceSize(hx.tensor, hw.tensor,
                                                         ho.tensor, &ws_size, &executor);
    if (ret != 0) {
        throw std::runtime_error("aclnnMatmulW8a8I32CustomGetWorkspaceSize failed: " + std::to_string(ret));
    }
    run_op("aclnnMatmulW8a8I32Custom", ws_size, executor, stream, aclnnMatmulW8a8I32Custom);
}

void w8a8_quantize(const Tensor& x,
                   Tensor& x_int8,
                   Tensor& x_scale,
                   aclrtStream stream) {
    if (x.dtype() != DType::Float16) {
        throw std::runtime_error("w8a8_quantize x must be fp16");
    }
    if (x_int8.dtype() != DType::Int8) {
        throw std::runtime_error("w8a8_quantize x_int8 must be int8");
    }
    if (x_scale.dtype() != DType::Float16) {
        throw std::runtime_error("w8a8_quantize x_scale must be fp16");
    }
    if (x.shape().size() != 2 || x.shape()[0] != 1) {
        throw std::runtime_error("w8a8_quantize x must be [1, K]");
    }
    if (x_int8.shape() != x.shape()) {
        throw std::runtime_error("w8a8_quantize x_int8 must match x shape");
    }
    if (x_scale.shape() != std::vector<int64_t>{1}) {
        throw std::runtime_error("w8a8_quantize x_scale must be [1]");
    }

    AclTensorHandle hx, hxq, hs;
    make_acl_tensor(x, hx);
    make_acl_tensor(x_int8, hxq);
    make_acl_tensor(x_scale, hs);

    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnW8a8QuantizeCustomGetWorkspaceSize(hx.tensor, hxq.tensor, hs.tensor,
                                                        &ws_size, &executor);
    if (ret != 0) {
        throw std::runtime_error("aclnnW8a8QuantizeCustomGetWorkspaceSize failed: " + std::to_string(ret));
    }
    run_op("aclnnW8a8QuantizeCustom", ws_size, executor, stream, aclnnW8a8QuantizeCustom);
}

void w8a8_dequant(const Tensor& acc,
                  const Tensor& x_scale,
                  const Tensor& w_scale,
                  Tensor& out,
                  aclrtStream stream) {
    if (acc.dtype() != DType::Int32 || x_scale.dtype() != DType::Float16 ||
        w_scale.dtype() != DType::Float16 || out.dtype() != DType::Float16) {
        throw std::runtime_error("w8a8_dequant requires int32 acc, fp16 scales, fp16 out");
    }
    if (acc.shape().size() != 2 || acc.shape()[0] != 1) {
        throw std::runtime_error("w8a8_dequant acc must be [1, N]");
    }
    const int64_t N = acc.shape()[1];
    if (x_scale.shape() != std::vector<int64_t>{1}) {
        throw std::runtime_error("w8a8_dequant x_scale must be [1]");
    }
    if (w_scale.shape() != std::vector<int64_t>{N}) {
        throw std::runtime_error("w8a8_dequant w_scale must be [N]");
    }
    if (out.shape() != std::vector<int64_t>{1, N}) {
        throw std::runtime_error("w8a8_dequant out must be [1, N]");
    }

    AclTensorHandle ha, hxs, hws, ho;
    make_acl_tensor(acc, ha);
    make_acl_tensor(x_scale, hxs);
    make_acl_tensor(w_scale, hws);
    make_acl_tensor(out, ho);

    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnW8a8DequantCustomGetWorkspaceSize(ha.tensor, hxs.tensor, hws.tensor,
                                                       ho.tensor, &ws_size, &executor);
    if (ret != 0) {
        throw std::runtime_error("aclnnW8a8DequantCustomGetWorkspaceSize failed: " + std::to_string(ret));
    }
    run_op("aclnnW8a8DequantCustom", ws_size, executor, stream, aclnnW8a8DequantCustom);
}

}  // namespace minicpmv
