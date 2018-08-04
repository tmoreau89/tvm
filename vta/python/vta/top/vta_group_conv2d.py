import logging
from collections import namedtuple

import tvm
import topi


from topi.util import get_const_int, get_const_tuple
from tvm.contrib.util import get_lower_ir

from ..environment import get_env

Workload = namedtuple("GroupConv2DWorkload",
                      ('batch', 'height', 'width', 'in_filter', 'out_filter', 'groups',
                       'hkernel', 'wkernel', 'hpad', 'wpad', 'hstride', 'wstride'))

Schedule = namedtuple("GroupConv2DSchedule",
                      ('b_factor', 'oc_factor', 'ic_factor', 'h_factor', 'w_factor',
                       'oc_nthread', 'h_nthread', 'debug_sync'))


def find_schedules(layer, vt_only=False, best_only=False):
    return [Schedule(0, 0, 1, 0, 0, 0, 0, False)]


def _get_workload(data, pad_data, kernel, output):
    """ Get the workload structure.
    """
    o_shape = get_const_tuple(output.shape)
    d_shape = get_const_tuple(data.shape)
    k_shape = get_const_tuple(kernel.shape)
    o_b, o_c, o_h, o_w, ob_blk, o_blk = o_shape
    i_b, i_c, i_h, i_w, ib_blk, i_blk = d_shape
    k_o, k_i, k_h, k_w, ko_blk, ki_blk = k_shape
    # For now we need to assume that input channel blocking is the same
    # as the output channel blocking
    assert o_blk == i_blk
    assert ob_blk == ib_blk
    # Make sure that dimensions match
    assert o_b == i_b
    assert o_blk == ko_blk
    assert i_blk == ki_blk
    assert k_o == o_c
    groups = i_c // k_i
    assert i_c % groups == 0
    assert o_c % groups == 0

    # Scale the channel size
    i_c *= i_blk
    o_c *= o_blk
    if pad_data is not None:
        p_shape = topi.util.get_const_tuple(pad_data.shape)
        h_pad = (p_shape[2] - d_shape[2]) // 2
        w_pad = (p_shape[3] - d_shape[3]) // 2
    else:
        h_pad, w_pad = 0, 0
    h_str = (i_h + h_pad*2 - k_h) // (o_h - 1)
    w_str = (i_w + w_pad*2 - k_w) // (o_w - 1)
    return Workload(i_b, i_h, i_w, i_c, o_c, groups, k_h, k_w, h_pad, w_pad, h_str, w_str)


def packed_group_conv2d(data,
                        kernel,
                        padding,
                        strides,
                        group,
                        out_dtype="int32"):
    """ Packed conv2d function."""

    if padding[0]:
        pad_data = topi.nn.pad(data, [0, 0, padding[0], padding[1], 0, 0], name="pad_data")
    else:
        pad_data = data

    assert len(data.shape) == 6
    assert len(kernel.shape) == 6
    assert data.dtype == "int8", data.dtype
    assert kernel.dtype == "int8", kernel.dtype

    N, CI, IH, IW, B_BATCH, B_CI = get_const_tuple(data.shape)
    CO, CI_G, KH, KW, B_CO, B_CI = get_const_tuple(kernel.shape)
    PAD_H, PAD_W = padding
    STR_H, STR_W = strides

    OH = (IH + 2 * PAD_H - KH) // strides[0] + 1
    OW = (IW + 2 * PAD_W - KW) // strides[1] + 1

    assert group * CI_G == CI
    assert CO % group == 0

    oshape = (N, CO, OH, OW, B_BATCH, B_CO)

    kh = tvm.reduce_axis((0, KH), name='d_i')
    kw = tvm.reduce_axis((0, KW), name='d_j')
    ci_o = tvm.reduce_axis((0, CI_G), name='k_o')
    ci_i = tvm.reduce_axis((0, B_CI), name='k_ten')

    out = tvm.compute(
        oshape,
        lambda n, co, h, w, b_n, b_co: tvm.sum(
            pad_data[n, co // (CO // group) * CI_G + ci_o, h * STR_H + kh,
                     w * STR_W + kw, b_n, ci_i].astype(out_dtype) *
            kernel[co, ci_o, kh, kw, b_co, ci_i].astype(out_dtype),
            axis=[ci_o, kh, kw, ci_i]),
        name="res", tag="packed_group_conv2d")
    return out


def schedule_packed_group_conv2d(outs):
    """ Schedule the packed conv2d.
    """
    assert len(outs) == 1
    output = outs[0]
    ewise_inputs = []
    ewise_ops = []
    conv2d_res = []
    assert output.dtype == "int8"
    assert output.op.input_tensors[0].dtype == "int32"

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
