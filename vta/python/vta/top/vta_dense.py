import logging
from collections import namedtuple

import tvm
import topi
from topi.util import get_const_int, get_const_tuple

from ..environment import get_env

Workload = namedtuple("DenseWorkload",
                      ('batch', 'in_dim', 'out_dim'))

Schedule = namedtuple("GroupConv2DSchedule", ('factor', ))


def find_schedules(layer, vt_only=False, best_only=False):
    return [Schedule(0, 0, 1, 0, 0, 0, 0, False)]


def packed_dense(data,
                 weight,
                 out_dtype="int32"):
    """ Packed conv2d function."""
    env = get_env()

    N, IN, B_BATCH, B_CI = get_const_tuple(data.shape)
    OUT, IN, B_OUT, B_IN = get_const_tuple(weight.shape)

    oshape = (N, OUT, B_BATCH, B_OUT)

    ko = tvm.reduce_axis((0, IN), name='ko')
    ki = tvm.reduce_axis((0, env.BLOCK_IN), name='ki')

    out = tvm.compute(
        oshape,
        lambda n, o, b_n, b_out: tvm.sum(data[n, ko, b_n, ki].astype(out_dtype) *
                                         weight[o, ko, b_out, ki].astype(out_dtype),
                                         axis=[ko, ki]),
        name="res", tag="packed_dense",
        attrs={'workload': (N, IN * B_CI, OUT * B_OUT)})
    return out


def schedule_packed_dense(outs):
    """ Schedule the packed conv2d.
    """
    assert len(outs) == 1
    output = outs[0]
    return tvm.create_schedule(output.op)

    def _traverse(op):
        if topi.tag.is_broadcast(op.tag):
            if not op.same_as(output.op):
                ewise_ops.append(op)
            for tensor in op.input_tensors:
                if isinstance(tensor.op, tvm.tensor.PlaceholderOp):
                    ewise_inputs.append((op, tensor))
                else:
                    _traverse(tensor.op)
        else:
            assert op.tag == "packed_group_conv2d"
            conv2d_res.append(op)

    _traverse(output.op)
    assert len(conv2d_res) == 1
    conv2d_stage = conv2d_res[0].output(0)

    data, kernel = conv2d_stage.op.input_tensors
    if isinstance(data.op, tvm.tensor.ComputeOp) and "pad" in data.op.tag:
        temp = data.op.input_tensors[0]
        pad_data = data
        data = temp
    else:
        pad_data = None
    wrkld = _get_workload(data, pad_data, kernel, output)
    plan = find_schedules(wrkld, vt_only=True, best_only=True)[0]
    logging.info("Trying to find plan for %s", wrkld)
    env = get_env()

    load_inp = load_wgt = load_out = store_out = env.dma_copy
    alu = env.alu
    gemm = env.gemm

    # schedule1
    oshape = topi.util.get_const_tuple(output.shape)
    s = tvm.create_schedule(output.op)

    # setup pad
    if pad_data is not None:
        cdata = pad_data
        s[pad_data].set_scope(env.inp_scope)
    else:
        cdata = s.cache_read(data, env.inp_scope, [conv2d_stage])
    ckernel = s.cache_read(kernel, env.wgt_scope, [conv2d_stage])
    s[conv2d_stage].set_scope(env.acc_scope)
    # cache read input
    cache_read_ewise = []

    for consumer, tensor in ewise_inputs:
        cache_read_ewise.append(
            s.cache_read(tensor, env.acc_scope, [consumer]))
    # set ewise scope
    for op in ewise_ops:
        s[op].set_scope(env.acc_scope)
        s[op].pragma(s[op].op.axis[0], alu)

    # tile
    oc_factor = (plan.oc_factor if plan.oc_factor else 1)
    h_factor = (plan.h_factor if plan.h_factor else 1)
    w_factor = (plan.w_factor if plan.w_factor else 1)

    x_bo, x_co, x_i, x_j, x_bi, x_ci = s[output].op.axis
    x_co0, x_co1 = s[output].split(x_co, factor=oc_factor)
    x_i0, x_i1 = s[output].split(x_i, factor=h_factor)
    x_j0, x_j1 = s[output].split(x_j, factor=w_factor)
    s[output].reorder(x_bo, x_i0, x_co0, x_j0, x_co1, x_i1, x_j1, x_bi, x_ci)
    store_pt = x_j0

    # set all compute scopes
    s[conv2d_stage].compute_at(s[output], store_pt)
    for op in ewise_ops:
        s[op].compute_at(s[output], store_pt)

    for tensor in cache_read_ewise:
        s[tensor].compute_at(s[output], store_pt)
        s[tensor].pragma(s[tensor].op.axis[0], load_out)

    # virtual threading along output channel axes
    if plan.oc_nthread > 1:
        _, v_t = s[output].split(x_co0, factor=plan.oc_nthread)
        s[output].reorder(v_t, x_bo)
        s[output].bind(v_t, tvm.thread_axis("cthread"))

    # virtual threading along spatial rows
    if plan.h_nthread > 1:
        _, v_t = s[output].split(x_i0, factor=plan.h_nthread)
        s[output].reorder(v_t, x_bo)
        s[output].bind(v_t, tvm.thread_axis("cthread"))

    x_bo, x_co, x_i, x_j, x_bi, x_ci = s[conv2d_stage].op.axis
    k_o, d_i, d_j, k_i = s[conv2d_stage].op.reduce_axis
    s[conv2d_stage].reorder(x_bo, k_o, x_j, d_j, d_i, x_co, x_i, x_bi, x_ci, k_i)

    if plan.ic_factor:
        k_o, _ = s[conv2d_stage].split(k_o, factor=plan.ic_factor)
        s[cdata].compute_at(s[conv2d_stage], k_o)
        s[ckernel].compute_at(s[conv2d_stage], k_o)

    # Use VTA instructions
    s[cdata].pragma(s[cdata].op.axis[0], load_inp)
    s[ckernel].pragma(s[ckernel].op.axis[0], load_wgt)
    s[conv2d_stage].tensorize(x_bi, gemm)
    s[output].pragma(x_co1, store_out)

    return s
