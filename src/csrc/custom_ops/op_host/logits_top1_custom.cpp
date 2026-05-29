#include "logits_top1_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    LogitsTop1CustomTilingData tiling;
    const auto* attrs = context->GetAttrs();
    tiling.set_valid(static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(0)));
    context->SetBlockDim(1);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
    gert::Shape* value_shape = context->GetOutputShape(0);
    gert::Shape* index_shape = context->GetOutputShape(1);
    value_shape->SetDimNum(1);
    value_shape->SetDim(0, 1);
    index_shape->SetDimNum(1);
    index_shape->SetDim(0, 1);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
    context->SetOutputDataType(0, ge::DT_FLOAT16);
    context->SetOutputDataType(1, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class LogitsTop1Custom : public OpDef {
public:
    explicit LogitsTop1Custom(const char* name) : OpDef(name) {
        this->Input("logits").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("value").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("index").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("valid").Int();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");
    }
};

OP_ADD(LogitsTop1Custom);
}  // namespace ops
