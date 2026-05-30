#include "rope_prefill_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    RopePrefillCustomTilingData tiling;

    // x: [N, headDim]; cos/sin: [T, rotaryDim/2].
    const gert::StorageShape* xShape = context->GetInputShape(0);
    const auto& xs = xShape->GetStorageShape();
    int64_t N = 1;
    for (size_t i = 0; i + 1 < xs.GetDimNum(); ++i) {
        N *= xs.GetDim(i);
    }
    const int64_t headDim = xs.GetDim(xs.GetDimNum() - 1);

    const auto* attrs = context->GetAttrs();
    const int64_t heads = *attrs->GetAttrPointer<int64_t>(0);
    const int64_t rotaryDim = *attrs->GetAttrPointer<int64_t>(1);

    tiling.set_N(static_cast<uint32_t>(N));
    tiling.set_headDim(static_cast<uint32_t>(headDim));
    tiling.set_rotaryDim(static_cast<uint32_t>(rotaryDim));
    tiling.set_halfRot(static_cast<uint32_t>(rotaryDim / 2));
    tiling.set_heads(static_cast<uint32_t>(heads));

    context->SetBlockDim(8);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
    const gert::Shape* xShape = context->GetInputShape(0);
    gert::Shape* outShape = context->GetOutputShape(0);
    *outShape = *xShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class RopePrefillCustom : public OpDef {
public:
    explicit RopePrefillCustom(const char* name) : OpDef(name) {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("cos_table")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("sin_table")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("heads").Int();
        this->Attr("rotary_dim").Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");
    }
};

OP_ADD(RopePrefillCustom);
}  // namespace ops
