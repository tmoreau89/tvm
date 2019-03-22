/*!
 *  Copyright (c) 2017 by Contributors
 * \file lowbit.cc
 * \brief Support operators for lowbit
 */
#include <tvm/tvm.h>
#include <tvm/expr.h>
#include <tvm/packed_func_ext.h>
#include <nnvm/op.h>
#include <nnvm/node.h>
#include <nnvm/layout.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/compiler/op_attr_types.h>
#include <nnvm/top/nn.h>
#include "./nn_common.h"
#include "../op_common.h"
#include "../elemwise_op_common.h"

namespace nnvm {
namespace top {

struct BitPackParam : public dmlc::Parameter<BitPackParam> {
  int lanes;

  DMLC_DECLARE_PARAMETER(BitPackParam) {
    DMLC_DECLARE_FIELD(lanes).set_lower_bound(1)
    .describe("Number of lanes packed in one element");
  }
};


// dense
DMLC_REGISTER_PARAMETER(BitPackParam);

inline bool BitPackInferShape(const nnvm::NodeAttrs& attrs,
                            std::vector<TShape>* in_shape,
                            std::vector<TShape>* out_shape) {
  const BitPackParam& param = nnvm::get<BitPackParam>(attrs.parsed);
  CHECK_EQ(out_shape->size(), 1U);
  if ((*in_shape)[DenseParam::kData].ndim() != 0) {
    TShape dshape = (*in_shape)[0];
    CHECK_EQ(dshape[dshape.ndim() - 1] % param.lanes, 0);
    dshape[dshape.ndim() - 1] /= param.lanes;
    NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, 0, dshape);
    return false;
  }
  return true;
}


NNVM_REGISTER_OP(bitpack)
.describe(R"code(Applies bit packing to innermost dimension.

)code" NNVM_ADD_FILELINE)
.add_argument("data", "nD Tensor", "Input data.")
.add_argument("weight", "2D Tensor", "Weight matrix.")
.add_argument("bias", "1D Tensor", "Bias parameter.")
.add_arguments(BitPackParam::__FIELDS__())
.set_attr_parser(ParamParser<BitPackParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<BitPackParam>)
.set_num_outputs(1)
.set_num_inputs(1)
.set_support_level(5)
.set_attr<FInferShape>("FInferShape", BitPackInferShape)
.set_attr<FInferType>("FInferType", ElemwiseType<-1, 1>);

}  // namespace top
}  // namespace nnvm
