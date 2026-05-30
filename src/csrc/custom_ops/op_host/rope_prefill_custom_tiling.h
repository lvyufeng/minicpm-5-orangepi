#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(RopePrefillCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, N);
    TILING_DATA_FIELD_DEF(uint32_t, headDim);
    TILING_DATA_FIELD_DEF(uint32_t, rotaryDim);
    TILING_DATA_FIELD_DEF(uint32_t, halfRot);
    TILING_DATA_FIELD_DEF(uint32_t, heads);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(RopePrefillCustom, RopePrefillCustomTilingData)
}  // namespace optiling
