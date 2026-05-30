#include "prefill_attention_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    PrefillAttentionCustomTilingData tiling;

    const auto* attrs = context->GetAttrs();
    const int64_t seqLen     = *attrs->GetAttrPointer<int64_t>(0);
    const int64_t numQHeads  = *attrs->GetAttrPointer<int64_t>(1);
    const int64_t numKvHeads = *attrs->GetAttrPointer<int64_t>(2);
    const int64_t headDim    = *attrs->GetAttrPointer<int64_t>(3);
    const float   scale      = *attrs->GetAttrPointer<float>(4);

    tiling.set_seqLen(static_cast<uint32_t>(seqLen));
    tiling.set_numQHeads(static_cast<uint32_t>(numQHeads));
    tiling.set_numKvHeads(static_cast<uint32_t>(numKvHeads));
    tiling.set_headDim(static_cast<uint32_t>(headDim));
    tiling.set_qPerKv(static_cast<uint32_t>(numQHeads / numKvHeads));
    tiling.set_scale(scale);
    tiling.set_totalTasks(static_cast<uint32_t>(seqLen * numQHeads));

    context->SetBlockDim(8);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
    // out = [T, num_q_heads*head_dim]. T = v_full rows; total elems = q_rope elems
    // (T*num_q_heads*head_dim), so cols = q_rope_total / T.
    const gert::Shape* qShape = context->GetInputShape(0);
    const gert::Shape* vShape = context->GetInputShape(2);
    gert::Shape* outShape = context->GetOutputShape(0);
    int64_t qTotal = 1;
    for (size_t i = 0; i < qShape->GetDimNum(); ++i) qTotal *= qShape->GetDim(i);
    const int64_t T = vShape->GetDim(0);
    outShape->SetDimNum(2);
    outShape->SetDim(0, T);
    outShape->SetDim(1, T > 0 ? qTotal / T : 0);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class PrefillAttentionCustom : public OpDef {
public:
    explicit PrefillAttentionCustom(const char* name) : OpDef(name) {
        this->Input("q_rope")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("k_rope")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("v_full")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("seq_len").Int();
        this->Attr("num_q_heads").Int();
        this->Attr("num_kv_heads").Int();
        this->Attr("head_dim").Int();
        this->Attr("scale").Float();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");
    }
};

OP_ADD(PrefillAttentionCustom);
}  // namespace ops
