#include "kernel_operator.h"

using namespace AscendC;

class KernelRopeCacheWriteCustom {
public:
    __aicore__ inline KernelRopeCacheWriteCustom() {}

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v,
                                GM_ADDR kCache, GM_ADDR vCache,
                                GM_ADDR cosTable, GM_ADDR sinTable,
                                GM_ADDR qRope, GM_ADDR kCacheOut, GM_ADDR vCacheOut,
                                uint32_t pos, uint32_t cacheLen,
                                uint32_t numQHeads, uint32_t numKvHeads,
                                uint32_t headDim, uint32_t rotaryDim) {
        this->pos = pos;
        this->cacheLen = cacheLen;
        this->numQHeads = numQHeads;
        this->numKvHeads = numKvHeads;
        this->headDim = headDim;
        this->kvDim = numKvHeads * headDim;
        this->rotaryDim = rotaryDim;
        this->halfRot = rotaryDim / 2;
        this->qHead = GetBlockIdx();
        this->isKvHead = qHead < numKvHeads;

        qGm.SetGlobalBuffer((__gm__ half*)q + qHead * headDim, headDim);
        if (isKvHead) {
            kGm.SetGlobalBuffer((__gm__ half*)k + qHead * headDim, headDim);
            vGm.SetGlobalBuffer((__gm__ half*)v + qHead * headDim, headDim);
            kCacheGm.SetGlobalBuffer((__gm__ half*)kCacheOut + static_cast<uint64_t>(cacheLen) * kvDim + qHead * headDim, headDim);
            vCacheGm.SetGlobalBuffer((__gm__ half*)vCacheOut + static_cast<uint64_t>(cacheLen) * kvDim + qHead * headDim, headDim);
        }
        (void)kCache;
        (void)vCache;
        cosGm.SetGlobalBuffer((__gm__ half*)cosTable + pos * halfRot, halfRot);
        sinGm.SetGlobalBuffer((__gm__ half*)sinTable + pos * halfRot, halfRot);
        qRopeGm.SetGlobalBuffer((__gm__ half*)qRope + qHead * headDim, headDim);
    }

    __aicore__ inline half rope_value(GlobalTensor<half>& x, uint32_t i) {
        if (i < rotaryDim) {
            uint32_t halfIdx = i;
            bool second = false;
            if (halfIdx >= halfRot) {
                halfIdx -= halfRot;
                second = true;
            }
            float x1 = static_cast<float>(x.GetValue(halfIdx));
            float x2 = static_cast<float>(x.GetValue(halfRot + halfIdx));
            float c = static_cast<float>(cosGm.GetValue(halfIdx));
            float s = static_cast<float>(sinGm.GetValue(halfIdx));
            float y = second ? (x2 * c + x1 * s) : (x1 * c - x2 * s);
            return static_cast<half>(y);
        }
        return x.GetValue(i);
    }

    __aicore__ inline void Process() {
        for (uint32_t i = 0; i < headDim; ++i) {
            qRopeGm.SetValue(i, rope_value(qGm, i));
        }
        if (isKvHead) {
            for (uint32_t i = 0; i < headDim; ++i) {
                kCacheGm.SetValue(i, rope_value(kGm, i));
                vCacheGm.SetValue(i, vGm.GetValue(i));
            }
        }
    }

private:
    uint32_t pos;
    uint32_t cacheLen;
    uint32_t numQHeads;
    uint32_t numKvHeads;
    uint32_t headDim;
    uint32_t kvDim;
    uint32_t rotaryDim;
    uint32_t halfRot;
    uint32_t qHead;
    bool isKvHead;
    GlobalTensor<half> qGm;
    GlobalTensor<half> kGm;
    GlobalTensor<half> vGm;
    GlobalTensor<half> cosGm;
    GlobalTensor<half> sinGm;
    GlobalTensor<half> qRopeGm;
    GlobalTensor<half> kCacheGm;
    GlobalTensor<half> vCacheGm;
};

extern "C" __global__ __aicore__ void rope_cache_write_custom(GM_ADDR q,
                                                               GM_ADDR k,
                                                               GM_ADDR v,
                                                               GM_ADDR k_cache,
                                                               GM_ADDR v_cache,
                                                               GM_ADDR cos_table,
                                                               GM_ADDR sin_table,
                                                               GM_ADDR q_rope,
                                                               GM_ADDR k_cache_out,
                                                               GM_ADDR v_cache_out,
                                                               GM_ADDR workspace,
                                                               GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelRopeCacheWriteCustom op;
    op.Init(q, k, v, k_cache, v_cache, cos_table, sin_table,
            q_rope, k_cache_out, v_cache_out,
            tiling_data.pos, tiling_data.cacheLen,
            tiling_data.numQHeads, tiling_data.numKvHeads,
            tiling_data.headDim, tiling_data.rotaryDim);
    op.Process();
}
