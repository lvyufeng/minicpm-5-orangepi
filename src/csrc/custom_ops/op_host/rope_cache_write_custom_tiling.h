#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(RopeCacheWriteCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, pos);
    TILING_DATA_FIELD_DEF(uint32_t, cacheLen);
    TILING_DATA_FIELD_DEF(uint32_t, numQHeads);
    TILING_DATA_FIELD_DEF(uint32_t, numKvHeads);
    TILING_DATA_FIELD_DEF(uint32_t, headDim);
    TILING_DATA_FIELD_DEF(uint32_t, rotaryDim);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(RopeCacheWriteCustom, RopeCacheWriteCustomTilingData)
}  // namespace optiling
