#include "rope_cache_write_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    RopeCacheWriteCustomTilingData tiling;
    const auto* attrs = context->GetAttrs();
    tiling.set_pos(static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(0)));
    tiling.set_cacheLen(static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(1)));
    tiling.set_numQHeads(static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(2)));
    tiling.set_numKvHeads(static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(3)));
    tiling.set_headDim(static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(4)));
    tiling.set_rotaryDim(static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(5)));
    context->SetBlockDim(static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(2)));
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
    const gert::Shape* q_shape = context->GetInputShape(0);
    const gert::Shape* k_cache_shape = context->GetInputShape(3);
    const gert::Shape* v_cache_shape = context->GetInputShape(4);
    gert::Shape* q_rope_shape = context->GetOutputShape(0);
    gert::Shape* k_cache_out_shape = context->GetOutputShape(1);
    gert::Shape* v_cache_out_shape = context->GetOutputShape(2);
    *q_rope_shape = *q_shape;
    *k_cache_out_shape = *k_cache_shape;
    *v_cache_out_shape = *v_cache_shape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
    context->SetOutputDataType(0, ge::DT_FLOAT16);
    context->SetOutputDataType(1, ge::DT_FLOAT16);
    context->SetOutputDataType(2, ge::DT_FLOAT16);
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class RopeCacheWriteCustom : public OpDef {
public:
    explicit RopeCacheWriteCustom(const char* name) : OpDef(name) {
        this->Input("q").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("k").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("v").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("k_cache").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("v_cache").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("cos_table").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("sin_table").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("q_rope").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("k_cache_out").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("v_cache_out").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("pos").Int();
        this->Attr("cache_len").Int();
        this->Attr("num_q_heads").Int();
        this->Attr("num_kv_heads").Int();
        this->Attr("head_dim").Int();
        this->Attr("rotary_dim").Int();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");
    }
};

OP_ADD(RopeCacheWriteCustom);
}  // namespace ops
