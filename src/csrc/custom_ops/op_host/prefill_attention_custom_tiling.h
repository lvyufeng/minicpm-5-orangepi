#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(PrefillAttentionCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, seqLen);
    TILING_DATA_FIELD_DEF(uint32_t, numQHeads);
    TILING_DATA_FIELD_DEF(uint32_t, numKvHeads);
    TILING_DATA_FIELD_DEF(uint32_t, headDim);
    TILING_DATA_FIELD_DEF(uint32_t, qPerKv);
    TILING_DATA_FIELD_DEF(float, scale);
    TILING_DATA_FIELD_DEF(uint32_t, totalTasks);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(PrefillAttentionCustom, PrefillAttentionCustomTilingData)
}  // namespace optiling
