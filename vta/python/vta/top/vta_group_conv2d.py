"""Namespace for supporting group_conv2d of nnvm."""

import tvm
from tvm import autotvm
import topi
from topi.util import get_const_tuple

import numpy as np

from ..environment import get_env

@autotvm.register_topi_compute(topi.nn.group_conv2d_nchw, 'vta', 'direct')
def packed_group_conv2d(cfg,
                        data,
                        kernel,
                        strides,
                        padding,
                        dilation,
                        group,
                        out_dtype):
    """ Packed group conv2d nchw function."""
    assert dilation == (1, 1)

    print(padding)
    print(data.shape)

    if padding[0]:
        pad_data = topi.nn.pad(data, [0, 0, padding[0], padding[1], 0, 0], name="pad_data")
    else:
        pad_data = data
    assert len(data.shape) == 6
    assert len(kernel.shape) == 6
    assert data.dtype == "int8", data.dtype
    assert kernel.dtype == "int8", kernel.dtype
    assert out_dtype == "int32", out_dtype

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

    cfg.add_flop(2 * np.prod(topi.util.get_const_tuple(oshape)) *
                 KH * KW * CI * B_CI)
    return out


@autotvm.register_topi_schedule(topi.generic.schedule_group_conv2d_nchw, 'vta', 'direct')
def schedule_packed_group_conv2d(cfg, outs):
    """ Schedule the packed conv2d.
    """
    assert len(outs) == 1
    output = outs[0]
    const_ops = []
    ewise_inputs = []
    ewise_ops = []
    conv2d_res = []
    assert output.dtype == "int8"
    assert output.op.input_tensors[0].dtype == "int32"

    def _traverse(op):
        if topi.tag.is_broadcast(op.tag):
            if not op.same_as(output.op):
                if len(op.axis) == 0:
                    const_ops.append(op)
                else:
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
    s = tvm.create_schedule(output.op)

    ##### space definition begin #####
    b, co, h, w, bi, ci = s[conv2d_stage].op.axis
    ci, kh, kw, bci = s[conv2d_stage].op.reduce_axis
    cfg.define_split('tile_b', b, num_outputs=2)
    cfg.define_split('tile_h', h, num_outputs=2)
    cfg.define_split('tile_w', w, num_outputs=2)
    cfg.define_split('tile_ci', ci, num_outputs=2)
    cfg.define_split('tile_co', co, num_outputs=2)
    cfg.define_knob('oc_nthread', [1, 2])
    cfg.define_knob('h_nthread', [1, 2])
    ###### space definition end ######

    data, kernel = conv2d_stage.op.input_tensors
    if isinstance(data.op, tvm.tensor.ComputeOp) and "pad" in data.op.tag:
        temp = data.op.input_tensors[0]
        pad_data = data
        data = temp
    else:
        pad_data = None

    env = get_env()
    load_inp = load_wgt = load_acc = store_out = env.dma_copy
    alu = env.alu
    gemm = env.gemm

    # schedule
    oshape = topi.util.get_const_tuple(output.shape)

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

    for op in const_ops:
        s[op].compute_inline()

    # tile
    x_bo, x_co, x_i, x_j, x_bi, x_ci = s[output].op.axis
    x_co0, x_co1 = cfg['tile_co'].apply(s, output, x_co)
    x_i0, x_i1 = cfg['tile_h'].apply(s, output, x_i)
    x_j0, x_j1 = cfg['tile_w'].apply(s, output, x_j)
    s[output].reorder(x_bo, x_i0, x_co0, x_j0, x_co1, x_i1, x_j1, x_bi, x_ci)
    store_pt = x_j0

    # set all compute scopes
    s[conv2d_stage].compute_at(s[output], store_pt)
    for op in ewise_ops:
        s[op].compute_at(s[output], store_pt)

    for tensor in cache_read_ewise:
        s[tensor].compute_at(s[output], store_pt)
        s[tensor].pragma(s[tensor].op.axis[0], load_acc)

    # virtual threading along output channel axes
    if cfg['oc_nthread'].val > 1:
        _, v_t = s[output].split(x_co0, factor=cfg['oc_nthread'].val)
        s[output].reorder(v_t, x_bo)
        s[output].bind(v_t, tvm.thread_axis("cthread"))

    # virtual threading along spatial rows
    if cfg['h_nthread'].val > 1:
        _, v_t = s[output].split(x_i0, factor=cfg['h_nthread'].val)
        s[output].reorder(v_t, x_bo)
        s[output].bind(v_t, tvm.thread_axis("cthread"))

    x_bo, x_co, x_i, x_j, x_bi, x_ci = s[conv2d_stage].op.axis
    k_o, d_i, d_j, k_i = s[conv2d_stage].op.reduce_axis
    s[conv2d_stage].reorder(x_bo, k_o, x_j, d_j, d_i, x_co, x_i, x_bi, x_ci, k_i)

    k_o, _ = cfg['tile_ci'].apply(s, conv2d_stage, k_o)
    s[cdata].compute_at(s[conv2d_stage], k_o)
    s[ckernel].compute_at(s[conv2d_stage], k_o)

    # Use VTA instructions
    s[cdata].pragma(s[cdata].op.axis[0], load_inp)
    s[ckernel].pragma(s[ckernel].op.axis[0], load_wgt)
    s[conv2d_stage].tensorize(x_bi, gemm)
    s[output].pragma(x_co1, store_out)

    return s
