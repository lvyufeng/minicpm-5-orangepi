#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(LogitsTop1CustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, valid);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LogitsTop1Custom, LogitsTop1CustomTilingData)
}  // namespace optiling
