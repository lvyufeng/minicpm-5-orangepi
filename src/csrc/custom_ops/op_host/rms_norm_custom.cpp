#include "rms_norm_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    const gert::StorageShape* xShape = context->GetInputShape(0);
    const auto& xStorageShape = xShape->GetStorageShape();
    const size_t rank = xStorageShape.GetDimNum();
    uint32_t hidden = 1;
    uint32_t rows = 1;
    if (rank > 0) {
        hidden = static_cast<uint32_t>(xStorageShape.GetDim(rank - 1));
        for (size_t i = 0; i + 1 < rank; ++i) {
            rows *= static_cast<uint32_t>(xStorageShape.GetDim(i));
        }
    }

    const auto* attrs = context->GetAttrs();
    const float epsilon = *attrs->GetAttrPointer<float>(0);

    RmsNormCustomTilingData tiling;
    tiling.set_rows(rows);
    tiling.set_hidden(hidden);
    tiling.set_epsilon(epsilon);

    uint32_t blockDim = rows;
    if (blockDim < 1) blockDim = 1;
    if (blockDim > 8) blockDim = 8;
    context->SetBlockDim(blockDim);
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
class RmsNormCustom : public OpDef {
public:
    explicit RmsNormCustom(const char* name) : OpDef(name) {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("gamma")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("epsilon").Float();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");
    }
};

OP_ADD(RmsNormCustom);
}  // namespace ops
