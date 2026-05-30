#include "kernel_operator.h"

using namespace AscendC;

// Prefill causal self-attention. One kernel launch replaces the host-driven
// per-head loop (copy_head_to_seq / matmul / scale / mask / softmax / matmul / scatter).
//
// Inputs:
//   q_rope: fp16 [T*Hq, D] token-major flattened heads (row = tq*Hq + qh)
//   k_rope: fp16 [T*Hkv, D] token-major flattened heads (row = tk*Hkv + kvh)
//   v_full: fp16 [T, Hkv*D] token-major, heads packed in last dim
// Output:
//   out:    fp16 [T, Hq*D] row t = concatenated head contexts
//
// Causal: only attend to tk <= tq (loop bound + -inf padding for vectorized softmax).
// GQA: kvh = qh / qPerKv.
// Work partition: totalTasks = T*Hq, split across 8 blocks.

class KernelPrefillAttentionCustom {
public:
    __aicore__ inline KernelPrefillAttentionCustom() {}

    __aicore__ inline void Init(GM_ADDR qRope, GM_ADDR kRope, GM_ADDR vFull, GM_ADDR out,
                                uint32_t seqLen, uint32_t numQHeads, uint32_t numKvHeads,
                                uint32_t headDim, uint32_t qPerKv, float scale,
                                uint32_t totalTasks) {
        this->seqLen = seqLen;
        this->numQHeads = numQHeads;
        this->numKvHeads = numKvHeads;
        this->headDim = headDim;
        this->qPerKv = qPerKv;
        this->scale = scale;
        this->kvDim = numKvHeads * headDim;
        // Align scoreCapacity to 16 for vector ops (32-byte alignment for fp32)
        this->scoreCapacity = ((seqLen + 15) / 16) * 16;

        const uint32_t blocks = GetBlockNum();
        const uint32_t bIdx   = GetBlockIdx();
        const uint32_t chunk  = (totalTasks + blocks - 1) / blocks;
        this->taskStart = bIdx * chunk;
        uint32_t taskEnd = taskStart + chunk;
        if (taskEnd > totalTasks) taskEnd = totalTasks;
        this->taskLen = taskEnd - taskStart;

        qGm.SetGlobalBuffer((__gm__ half*)qRope, static_cast<uint64_t>(seqLen) * numQHeads * headDim);
        kGm.SetGlobalBuffer((__gm__ half*)kRope, static_cast<uint64_t>(seqLen) * numKvHeads * headDim);
        vGm.SetGlobalBuffer((__gm__ half*)vFull, static_cast<uint64_t>(seqLen) * kvDim);
        outGm.SetGlobalBuffer((__gm__ half*)out, static_cast<uint64_t>(seqLen) * numQHeads * headDim);

        pipe.InitBuffer(qFp16Buf,    headDim * sizeof(half));
        pipe.InitBuffer(qFp32Buf,    headDim * sizeof(float));
        pipe.InitBuffer(kvFp16Buf,   headDim * sizeof(half));
        pipe.InitBuffer(kvFp32Buf,   headDim * sizeof(float));
        pipe.InitBuffer(prodBuf,     headDim * sizeof(float));
        pipe.InitBuffer(accFp32Buf,  headDim * sizeof(float));
        pipe.InitBuffer(outFp16Buf,  headDim * sizeof(half));
        pipe.InitBuffer(scoresBuf,   scoreCapacity * sizeof(float));
        pipe.InitBuffer(reduceTmpBuf, scoreCapacity * sizeof(float));
        pipe.InitBuffer(reduceDstBuf, 32);
    }

    __aicore__ inline void Process() {
        if (taskLen == 0) return;

        LocalTensor<half>  qFp16   = qFp16Buf.Get<half>();
        LocalTensor<float> qFp32   = qFp32Buf.Get<float>();
        LocalTensor<half>  kvFp16  = kvFp16Buf.Get<half>();
        LocalTensor<float> kvFp32  = kvFp32Buf.Get<float>();
        LocalTensor<float> prod    = prodBuf.Get<float>();
        LocalTensor<float> acc     = accFp32Buf.Get<float>();
        LocalTensor<half>  outFp16 = outFp16Buf.Get<half>();
        LocalTensor<float> scores  = scoresBuf.Get<float>();
        LocalTensor<float> redTmp  = reduceTmpBuf.Get<float>();
        LocalTensor<float> redDst  = reduceDstBuf.Get<float>();

        for (uint32_t task = 0; task < taskLen; ++task) {
            const uint32_t linear = taskStart + task;
            const uint32_t tq = linear / numQHeads;
            const uint32_t qh = linear % numQHeads;
            const uint32_t kvh = qh / qPerKv;
            const uint32_t validLen = tq + 1;  // causal: attend to [0..tq]

            // Initialize scores to large negative so exp(-inf) ≈ 0 for padding
            Duplicate(scores, -65504.0f, scoreCapacity);
            PipeBarrier<PIPE_V>();

            // Load query row
            const uint64_t qOffset = static_cast<uint64_t>(tq * numQHeads + qh) * headDim;
            DataCopy(qFp16, qGm[qOffset], headDim);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Cast(qFp32, qFp16, RoundMode::CAST_NONE, headDim);
            PipeBarrier<PIPE_V>();

            // Pass 1: compute scores[0..tq]
            for (uint32_t tk = 0; tk < validLen; ++tk) {
                const uint64_t kOffset = static_cast<uint64_t>(tk * numKvHeads + kvh) * headDim;
                DataCopy(kvFp16, kGm[kOffset], headDim);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                Cast(kvFp32, kvFp16, RoundMode::CAST_NONE, headDim);
                PipeBarrier<PIPE_V>();
                Mul(prod, qFp32, kvFp32, headDim);
                PipeBarrier<PIPE_V>();
                ReduceSum<float>(redDst, prod, redTmp, headDim);
                SetFlag<HardEvent::V_S>(EVENT_ID0);
                WaitFlag<HardEvent::V_S>(EVENT_ID0);
                scores.SetValue(tk, redDst.GetValue(0) * scale);
                SetFlag<HardEvent::S_V>(EVENT_ID0);
                WaitFlag<HardEvent::S_V>(EVENT_ID0);
            }

            // Vectorized stable softmax over scoreCapacity elements
            // (padding slots are -65504 → exp ≈ 0, won't affect result)
            PipeBarrier<PIPE_V>();
            ReduceMax<float>(redDst, scores, redTmp, scoreCapacity);
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            const float maxScore = redDst.GetValue(0);
            SetFlag<HardEvent::S_V>(EVENT_ID0);
            WaitFlag<HardEvent::S_V>(EVENT_ID0);
            Adds(scores, scores, -maxScore, scoreCapacity);
            PipeBarrier<PIPE_V>();
            Exp(scores, scores, scoreCapacity);
            PipeBarrier<PIPE_V>();
            ReduceSum<float>(redDst, scores, redTmp, scoreCapacity);
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            const float invSum = 1.0f / redDst.GetValue(0);
            SetFlag<HardEvent::S_V>(EVENT_ID0);
            WaitFlag<HardEvent::S_V>(EVENT_ID0);
            Muls(scores, scores, invSum, scoreCapacity);
            PipeBarrier<PIPE_V>();

            // Pass 2: weighted V accumulation (only valid tokens contribute)
            Duplicate(acc, 0.0f, headDim);
            PipeBarrier<PIPE_V>();

            for (uint32_t tk = 0; tk < validLen; ++tk) {
                const uint64_t vOffset = static_cast<uint64_t>(tk) * kvDim + static_cast<uint64_t>(kvh) * headDim;
                DataCopy(kvFp16, vGm[vOffset], headDim);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                Cast(kvFp32, kvFp16, RoundMode::CAST_NONE, headDim);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_S>(EVENT_ID0);
                WaitFlag<HardEvent::V_S>(EVENT_ID0);
                const float prob = scores.GetValue(tk);
                SetFlag<HardEvent::S_V>(EVENT_ID0);
                WaitFlag<HardEvent::S_V>(EVENT_ID0);
                Axpy<float, float>(acc, kvFp32, prob, headDim);
                PipeBarrier<PIPE_V>();
            }

            // Write output
            Cast(outFp16, acc, RoundMode::CAST_RINT, headDim);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            const uint64_t outOffset = static_cast<uint64_t>(tq) * numQHeads * headDim
                                     + static_cast<uint64_t>(qh) * headDim;
            DataCopy(outGm[outOffset], outFp16, headDim);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
    }

private:
    uint32_t seqLen;
    uint32_t numQHeads;
    uint32_t numKvHeads;
    uint32_t headDim;
    uint32_t qPerKv;
    uint32_t kvDim;
    uint32_t scoreCapacity;
    float scale;
    uint32_t taskStart;
    uint32_t taskLen;
    GlobalTensor<half> qGm;
    GlobalTensor<half> kGm;
    GlobalTensor<half> vGm;
    GlobalTensor<half> outGm;
    TPipe pipe;
    TBuf<TPosition::VECCALC> qFp16Buf;
    TBuf<TPosition::VECCALC> qFp32Buf;
    TBuf<TPosition::VECCALC> kvFp16Buf;
    TBuf<TPosition::VECCALC> kvFp32Buf;
    TBuf<TPosition::VECCALC> prodBuf;
    TBuf<TPosition::VECCALC> accFp32Buf;
    TBuf<TPosition::VECCALC> outFp16Buf;
    TBuf<TPosition::VECCALC> scoresBuf;
    TBuf<TPosition::VECCALC> reduceTmpBuf;
    TBuf<TPosition::VECCALC> reduceDstBuf;
};

extern "C" __global__ __aicore__ void prefill_attention_custom(GM_ADDR q_rope, GM_ADDR k_rope,
                                                                GM_ADDR v_full, GM_ADDR out,
                                                                GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelPrefillAttentionCustom op;
    op.Init(q_rope, k_rope, v_full, out,
            tiling_data.seqLen, tiling_data.numQHeads, tiling_data.numKvHeads,
            tiling_data.headDim, tiling_data.qPerKv, tiling_data.scale,
            tiling_data.totalTasks);
    op.Process();
}
