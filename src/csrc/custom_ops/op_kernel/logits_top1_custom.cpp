#include "kernel_operator.h"

using namespace AscendC;

class KernelLogitsTop1Custom {
public:
    __aicore__ inline KernelLogitsTop1Custom() {}

    __aicore__ inline void Init(GM_ADDR logits, GM_ADDR value, GM_ADDR index, uint32_t valid) {
        this->valid = valid;
        logitsGm.SetGlobalBuffer((__gm__ half*)logits, valid);
        valueGm.SetGlobalBuffer((__gm__ half*)value, 1);
        indexGm.SetGlobalBuffer((__gm__ int32_t*)index, 1);
    }

    __aicore__ inline void Process() {
        float best = -3.4028234663852886e38f;
        int32_t bestIndex = 0;
        for (uint32_t i = 0; i < valid; ++i) {
            float v = static_cast<float>(logitsGm.GetValue(i));
            if (v > best) {
                best = v;
                bestIndex = static_cast<int32_t>(i);
            }
        }
        valueGm.SetValue(0, static_cast<half>(best));
        indexGm.SetValue(0, bestIndex);
    }

private:
    uint32_t valid;
    GlobalTensor<half> logitsGm;
    GlobalTensor<half> valueGm;
    GlobalTensor<int32_t> indexGm;
};

extern "C" __global__ __aicore__ void logits_top1_custom(GM_ADDR logits,
                                                          GM_ADDR value,
                                                          GM_ADDR index,
                                                          GM_ADDR workspace,
                                                          GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelLogitsTop1Custom op;
    op.Init(logits, value, index, tiling_data.valid);
    op.Process();
}
