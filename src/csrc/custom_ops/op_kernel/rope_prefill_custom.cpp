#include "kernel_operator.h"

using namespace AscendC;

// Partial RoPE for prefill. x is [N, headDim] laid out token-major as
// row = t*heads + h, so the token index for row n is t = n / heads. cos/sin are
// [T, rotaryDim/2]. For the first rotaryDim lanes of each row:
//   out[i]          = x[i]*cos[i]        - x[halfRot+i]*sin[i]
//   out[halfRot+i]  = x[halfRot+i]*cos[i] + x[i]*sin[i]
// lanes [rotaryDim, headDim) (the non-rotary tail) are copied through unchanged.
// Matches apply_rope_partial in src/csrc/lib/ops.cpp. One launch replaces the
// thousands of host-issued gather/scatter memcpys the host path used.

class KernelRopePrefillCustom {
public:
    __aicore__ inline KernelRopePrefillCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR cosTable, GM_ADDR sinTable, GM_ADDR out,
                                uint32_t N, uint32_t headDim, uint32_t rotaryDim,
                                uint32_t halfRot, uint32_t heads) {
        this->headDim = headDim;
        this->rotaryDim = rotaryDim;
        this->halfRot = halfRot;
        this->heads = heads;

        const uint32_t blocks = GetBlockNum();
        const uint32_t bIdx   = GetBlockIdx();
        const uint32_t base   = (N + blocks - 1) / blocks;
        uint32_t start = bIdx * base;
        uint32_t end   = start + base;
        if (end   > N) end   = N;
        if (start > N) start = N;
        this->rowStart = start;
        this->rowLen   = end - start;

        xGm.SetGlobalBuffer((__gm__ half*)x + static_cast<uint64_t>(start) * headDim,
                            static_cast<uint64_t>(rowLen) * headDim);
        outGm.SetGlobalBuffer((__gm__ half*)out + static_cast<uint64_t>(start) * headDim,
                              static_cast<uint64_t>(rowLen) * headDim);
        cosGm.SetGlobalBuffer((__gm__ half*)cosTable, halfRot);
        sinGm.SetGlobalBuffer((__gm__ half*)sinTable, halfRot);

        pipe.InitBuffer(xFp16Buf,   headDim * sizeof(half));
        pipe.InitBuffer(outFp16Buf, headDim * sizeof(half));
        pipe.InitBuffer(cosFp16Buf, halfRot * sizeof(half));
        pipe.InitBuffer(sinFp16Buf, halfRot * sizeof(half));
        pipe.InitBuffer(xFp32Buf,   headDim * sizeof(float));
        pipe.InitBuffer(cosFp32Buf, halfRot * sizeof(float));
        pipe.InitBuffer(sinFp32Buf, halfRot * sizeof(float));
        pipe.InitBuffer(y1Buf,      halfRot * sizeof(float));
        pipe.InitBuffer(y2Buf,      halfRot * sizeof(float));
        pipe.InitBuffer(tmpBuf,     halfRot * sizeof(float));
    }

    __aicore__ inline void Process() {
        if (rowLen == 0) return;

        LocalTensor<half>  xFp16  = xFp16Buf.Get<half>();
        LocalTensor<half>  outFp16 = outFp16Buf.Get<half>();
        LocalTensor<half>  cosFp16 = cosFp16Buf.Get<half>();
        LocalTensor<half>  sinFp16 = sinFp16Buf.Get<half>();
        LocalTensor<float> xFp32  = xFp32Buf.Get<float>();
        LocalTensor<float> cosFp32 = cosFp32Buf.Get<float>();
        LocalTensor<float> sinFp32 = sinFp32Buf.Get<float>();
        LocalTensor<float> y1     = y1Buf.Get<float>();
        LocalTensor<float> y2     = y2Buf.Get<float>();
        LocalTensor<float> tmp    = tmpBuf.Get<float>();

        const uint32_t tail = headDim - rotaryDim;  // non-rotary lanes, 0 for this model
        uint32_t lastT = 0xffffffffu;

        for (uint32_t i = 0; i < rowLen; ++i) {
            const uint32_t t = (rowStart + i) / heads;

            DataCopy(xFp16, xGm[static_cast<uint64_t>(i) * headDim], headDim);
            if (t != lastT) {
                DataCopy(cosFp16, cosGm[static_cast<uint64_t>(t) * halfRot], halfRot);
                DataCopy(sinFp16, sinGm[static_cast<uint64_t>(t) * halfRot], halfRot);
            }
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

            Cast(xFp32, xFp16, RoundMode::CAST_NONE, headDim);
            if (t != lastT) {
                Cast(cosFp32, cosFp16, RoundMode::CAST_NONE, halfRot);
                Cast(sinFp32, sinFp16, RoundMode::CAST_NONE, halfRot);
                lastT = t;
            }
            PipeBarrier<PIPE_V>();

            // x1 = xFp32[0:halfRot], x2 = xFp32[halfRot:rotaryDim]
            LocalTensor<float> x1 = xFp32;
            LocalTensor<float> x2 = xFp32[halfRot];

            // y1 = x1*cos - x2*sin
            Mul(y1, x1, cosFp32, halfRot);
            PipeBarrier<PIPE_V>();
            Mul(tmp, x2, sinFp32, halfRot);
            PipeBarrier<PIPE_V>();
            Sub(y1, y1, tmp, halfRot);
            PipeBarrier<PIPE_V>();

            // y2 = x2*cos + x1*sin
            Mul(y2, x2, cosFp32, halfRot);
            PipeBarrier<PIPE_V>();
            Mul(tmp, x1, sinFp32, halfRot);
            PipeBarrier<PIPE_V>();
            Add(y2, y2, tmp, halfRot);
            PipeBarrier<PIPE_V>();

            Cast(outFp16, y1, RoundMode::CAST_RINT, halfRot);
            Cast(outFp16[halfRot], y2, RoundMode::CAST_RINT, halfRot);
            if (tail > 0) {
                // Pass through non-rotary lanes unchanged.
                Cast(outFp16[rotaryDim], xFp32[rotaryDim], RoundMode::CAST_RINT, tail);
            }
            PipeBarrier<PIPE_V>();

            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopy(outGm[static_cast<uint64_t>(i) * headDim], outFp16, headDim);

            // Close the loop: out store (MTE3) must finish before the next iter's
            // Cast reuses these UB buffers (MTE3->V), and the next iter's input
            // DataCopy (MTE2) must not race this iter's V reads (V->MTE2).
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        }
    }

private:
    uint32_t headDim;
    uint32_t rotaryDim;
    uint32_t halfRot;
    uint32_t heads;
    uint32_t rowStart;
    uint32_t rowLen;
    GlobalTensor<half> xGm;
    GlobalTensor<half> outGm;
    GlobalTensor<half> cosGm;
    GlobalTensor<half> sinGm;
    TPipe pipe;
    TBuf<TPosition::VECCALC> xFp16Buf;
    TBuf<TPosition::VECCALC> outFp16Buf;
    TBuf<TPosition::VECCALC> cosFp16Buf;
    TBuf<TPosition::VECCALC> sinFp16Buf;
    TBuf<TPosition::VECCALC> xFp32Buf;
    TBuf<TPosition::VECCALC> cosFp32Buf;
    TBuf<TPosition::VECCALC> sinFp32Buf;
    TBuf<TPosition::VECCALC> y1Buf;
    TBuf<TPosition::VECCALC> y2Buf;
    TBuf<TPosition::VECCALC> tmpBuf;
};

extern "C" __global__ __aicore__ void rope_prefill_custom(GM_ADDR x, GM_ADDR cos_table,
                                                          GM_ADDR sin_table, GM_ADDR out,
                                                          GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelRopePrefillCustom op;
    op.Init(x, cos_table, sin_table, out,
            tiling_data.N, tiling_data.headDim, tiling_data.rotaryDim,
            tiling_data.halfRot, tiling_data.heads);
    op.Process();
}
