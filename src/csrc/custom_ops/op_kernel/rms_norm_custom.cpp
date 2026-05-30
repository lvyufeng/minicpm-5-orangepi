#include "kernel_operator.h"

using namespace AscendC;

// Fused RMSNorm for fp16 tensors. The last dimension is reduced per row:
// out[row, i] = x[row, i] * rsqrt(mean(x^2) + epsilon) * gamma[i].
// Each AICore block processes one or more complete rows. This decode-oriented
// kernel targets MiniCPM5 shapes where hidden <= 2048/1536 and rows is usually 1.

class KernelRmsNormCustom {
public:
    __aicore__ inline KernelRmsNormCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR out,
                                uint32_t rows, uint32_t hidden, float epsilon) {
        this->rows = rows;
        this->hidden = hidden;
        this->epsilon = epsilon;
        xGm.SetGlobalBuffer((__gm__ half*)x, static_cast<uint64_t>(rows) * hidden);
        gammaGm.SetGlobalBuffer((__gm__ half*)gamma, hidden);
        outGm.SetGlobalBuffer((__gm__ half*)out, static_cast<uint64_t>(rows) * hidden);

        pipe.InitBuffer(xFp16Buf, TILE_ELEMS * sizeof(half));
        pipe.InitBuffer(gammaFp16Buf, TILE_ELEMS * sizeof(half));
        pipe.InitBuffer(outFp16Buf, TILE_ELEMS * sizeof(half));
        pipe.InitBuffer(xFp32Buf, TILE_ELEMS * sizeof(float));
        pipe.InitBuffer(gammaFp32Buf, TILE_ELEMS * sizeof(float));
        pipe.InitBuffer(outFp32Buf, TILE_ELEMS * sizeof(float));
        pipe.InitBuffer(invRmsBuf, ALIGN_FLOAT * sizeof(float));
    }

    __aicore__ inline void Process() {
        if (hidden == 0 || rows == 0) return;
        const uint32_t blocks = GetBlockNum();
        const uint32_t bIdx = GetBlockIdx();
        for (uint32_t row = bIdx; row < rows; row += blocks) {
            ProcessRow(row);
        }
    }

private:
    __aicore__ inline uint32_t AlignFp16(uint32_t n) const {
        return ((n + ALIGN_FP16 - 1) / ALIGN_FP16) * ALIGN_FP16;
    }

    __aicore__ inline void ProcessRow(uint32_t row) {
        LocalTensor<half> xFp16 = xFp16Buf.Get<half>();
        LocalTensor<half> gammaFp16 = gammaFp16Buf.Get<half>();
        LocalTensor<half> outFp16 = outFp16Buf.Get<half>();
        LocalTensor<float> xFp32 = xFp32Buf.Get<float>();
        LocalTensor<float> gammaFp32 = gammaFp32Buf.Get<float>();
        LocalTensor<float> outFp32 = outFp32Buf.Get<float>();
        LocalTensor<float> invRmsTensor = invRmsBuf.Get<float>();

        float sumSq = 0.0f;
        const uint64_t rowBase = static_cast<uint64_t>(row) * hidden;
        for (uint32_t offset = 0; offset < hidden; offset += TILE_ELEMS) {
            uint32_t n = hidden - offset;
            if (n > TILE_ELEMS) n = TILE_ELEMS;
            const uint32_t nAligned = AlignFp16(n);
            DataCopy(xFp16, xGm[rowBase + offset], nAligned);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

            for (uint32_t i = 0; i < n; ++i) {
                const float v = static_cast<float>(xFp16.GetValue(i));
                sumSq += v * v;
            }
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        }

        invRmsTensor.SetValue(0, sumSq / static_cast<float>(static_cast<int32_t>(hidden)) + epsilon);
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Rsqrt(invRmsTensor, invRmsTensor, ALIGN_FLOAT);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        const float invRms = invRmsTensor.GetValue(0);
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        for (uint32_t offset = 0; offset < hidden; offset += TILE_ELEMS) {
            uint32_t n = hidden - offset;
            if (n > TILE_ELEMS) n = TILE_ELEMS;
            const uint32_t nAligned = AlignFp16(n);

            DataCopy(xFp16, xGm[rowBase + offset], nAligned);
            DataCopy(gammaFp16, gammaGm[offset], nAligned);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

            Cast(xFp32, xFp16, RoundMode::CAST_NONE, nAligned);
            Cast(gammaFp32, gammaFp16, RoundMode::CAST_NONE, nAligned);
            PipeBarrier<PIPE_V>();
            Muls(outFp32, xFp32, invRms, nAligned);
            PipeBarrier<PIPE_V>();
            Mul(outFp32, outFp32, gammaFp32, nAligned);
            PipeBarrier<PIPE_V>();
            Cast(outFp16, outFp32, RoundMode::CAST_RINT, nAligned);
            PipeBarrier<PIPE_V>();

            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopy(outGm[rowBase + offset], outFp16, nAligned);
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
    }

    static constexpr uint32_t TILE_ELEMS = 2048;
    static constexpr uint32_t ALIGN_FP16 = 16;
    static constexpr uint32_t ALIGN_FLOAT = 8;

    uint32_t rows{0};
    uint32_t hidden{0};
    float epsilon{0.0f};
    GlobalTensor<half> xGm;
    GlobalTensor<half> gammaGm;
    GlobalTensor<half> outGm;
    TPipe pipe;
    TBuf<TPosition::VECCALC> xFp16Buf;
    TBuf<TPosition::VECCALC> gammaFp16Buf;
    TBuf<TPosition::VECCALC> outFp16Buf;
    TBuf<TPosition::VECCALC> xFp32Buf;
    TBuf<TPosition::VECCALC> gammaFp32Buf;
    TBuf<TPosition::VECCALC> outFp32Buf;
    TBuf<TPosition::VECCALC> invRmsBuf;
};

extern "C" __global__ __aicore__ void rms_norm_custom(GM_ADDR x,
                                                        GM_ADDR gamma,
                                                        GM_ADDR out,
                                                        GM_ADDR workspace,
                                                        GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelRmsNormCustom op;
    op.Init(x, gamma, out, tiling_data.rows, tiling_data.hidden, tiling_data.epsilon);
    op.Process();
}
