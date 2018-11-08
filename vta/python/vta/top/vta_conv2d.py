"""Namespace for supporting packed_conv2d + ewise variant of nnvm."""

import tvm
from tvm import autotvm
import topi

import numpy as np

from ..environment import get_env
from .op import is_packed_layout


@autotvm.register_topi_compute(topi.nn.conv2d, 'vta', 'direct')
def packed_conv2d(cfg, data, kernel, strides, padding, dilation, layout, out_dtype):
    """ Packed conv2d function."""
    if not is_packed_layout(layout):
        raise topi.InvalidShapeError()
    assert dilation == (1, 1)

    if padding[0]:
        pad_data = topi.nn.pad(data, [0, 0, padding[0], padding[1], 0, 0], name="pad_data")
    else:
        pad_data = data
    assert len(data.shape) == 6
    assert len(kernel.shape) == 6
    oheight = topi.util.get_const_int((pad_data.shape[2] - kernel.shape[2]) // strides[0] + 1)
    owidth = topi.util.get_const_int((pad_data.shape[3] - kernel.shape[3]) // strides[1] + 1)
    oshape = (data.shape[0], kernel.shape[0], oheight, owidth, data.shape[4], kernel.shape[4])

    ishape = topi.util.get_const_tuple(data.shape)
    kshape = topi.util.get_const_tuple(kernel.shape)
    d_i = tvm.reduce_axis((0, kshape[2]), name='d_i')
    d_j = tvm.reduce_axis((0, kshape[3]), name='d_j')
    k_o = tvm.reduce_axis((0, ishape[1]), name='k_o')
    k_i = tvm.reduce_axis((0, ishape[-1]), name='k_i')
    hstride, wstride = strides
    res = tvm.compute(
        oshape,
        lambda b_o, c_o, i, j, b_i, c_i: tvm.sum(
            pad_data[b_o, k_o, i*hstride+d_i, j*wstride+d_j, b_i, k_i].astype(out_dtype) *
            kernel[c_o, k_o, d_i, d_j, c_i, k_i].astype(out_dtype),
            axis=[k_o, d_i, d_j, k_i]),
        name="res", tag="packed_conv2d")

@tvm.register_func("nnvm.compiler.build_target", override=True)
def _build(funcs, target, target_host):
    tvm_t = tvm.target.create(target)
    if tvm_t.device_name == "vta":
        return tvm.build(funcs, target="ext_dev", target_host=target_host)
    if tvm_t.device_name == "rasp" or tvm_t.device_name == "vtacpu":
        return tvm.build(funcs, target=target_host)
    return tvm.build(funcs, target=target)


@tvm.register_func("nnvm.compiler.lower", override=True)
def _lower(sch, inputs, func_name, graph):
    import traceback
    # pylint: disable=broad-except
    try:
        f = tvm.lower(sch, inputs, name=func_name)
        if "quantized_conv2d" in func_name:
            logging.info(graph.ir(join_entry_attrs=["shape"]))
    except Exception:
        msg = traceback.format_exc()
        msg += "Error during compile graph\n"
        msg += "--------------------------\n"
        msg += graph.ir(join_entry_attrs=["shape"])
        raise RuntimeError(msg)
    return f if isinstance(
        f, (tvm.container.Array, tuple, list)) else [f]


@reg.register_compute("clip", level=15)
def compute_clip(attrs, inputs, _):
    """ Clip operator.
    """
    x = inputs[0]
    a_min = attrs.get_float("a_min")
    a_max = attrs.get_float("a_max")
    const_min = tvm.const(a_min, x.dtype)
    const_max = tvm.const(a_max, x.dtype)
    with tvm.tag_scope(topi.tag.ELEMWISE):
        x = tvm.compute(
            x.shape, lambda *i: tvm.min(x(*i), const_max), name="clipA")
        x = tvm.compute(
            x.shape, lambda *i: tvm.max(x(*i), const_min), name="clipB")
    return x

# override to force partition at copy
reg.register_pattern("copy", OpPattern.INJECTIVE, level=15)

def is_packed_layout(layout):
    """Check if layout is packed layout"""
    if layout == "NCHW":
        return False
    if "n" in layout and "c" in layout:
        return True
    return False

@reg.register_alter_op_layout("conv2d", level=15)
def alter_conv2d_layout(attrs, inputs, out):
    layout = attrs['layout']
    if is_packed_layout(layout):
        return None
    return _nn.alter_conv2d_layout(attrs, inputs, out)


@reg.register_compute("conv2d", level=15)
def compute_conv2d(attrs, inputs, out):
    """ 2D convolution algorithm.
    """
    padding = attrs.get_int_tuple("padding")
    strides = attrs.get_int_tuple("strides")
    dilation = attrs.get_int_tuple("dilation")
    groups = attrs.get_int("groups")
    layout = attrs["layout"]
    out_dtype = attrs['out_dtype']

    assert dilation == (1, 1), "not support dilate now"
    if is_packed_layout(layout):
        if groups == 1:
            assert groups == 1
            env = get_env()
            assert env.LOG_INP_WIDTH == 3, "only support 8bit inp for now"
            assert env.LOG_OUT_WIDTH == 3, "only support 8bit inp for now"
            inputs = list(inputs)
            w_pack_factor = 1 << (3 - env.LOG_WGT_WIDTH)
            assert inputs[1].dtype == "int8"

            # Apply bit packing if necessary
            if w_pack_factor != 1:
                kshape = list(topi.util.get_const_tuple(inputs[1].shape))
                kshape[-1] *= w_pack_factor
                inputs[1] = reinterpret(inputs[1], kshape, dtype=env.wgt_dtype)

            return packed_conv2d(inputs[0], inputs[1],
                                 padding, strides, out_dtype=out_dtype)
        else:
            return packed_group_conv2d(inputs[0], inputs[1],
                                       padding, strides, groups, out_dtype=out_dtype)
    return _nn.compute_conv2d(attrs, inputs, out)


@reg.register_schedule("conv2d", level=15)
def schedule_conv2d(attrs, outs, target):
    """ 2D convolution schedule.
    """
    layout = attrs["layout"]
    groups = attrs.get_int('groups')

    if is_packed_layout(layout):
        target = tvm.target.create(target)
        if target.device_name == "vta":
            if groups == 1:
                return schedule_packed_conv2d(outs)
            else:
                return schedule_packed_group_conv2d(outs)
        elif str(target).startswith("llvm"):
            return tvm.create_schedule([x.op for x in outs])
        raise RuntimeError("not support target %s" % target)
    return _nn.schedule_conv2d(attrs, outs, target)


@reg.register_compute("conv2d_transpose", level=15)
def compute_conv2d_transpose(attrs, inputs, out):
    """ 2D convolution algorithm.
    """
    padding = attrs.get_int_tuple("padding")
    strides = attrs.get_int_tuple("strides")
    dilation = attrs.get_int_tuple("dilation")
    layout = attrs["layout"]
    out_dtype = attrs['out_dtype']

    assert dilation == (1, 1), "not support dilate now"
    if is_packed_layout(layout):
        return packed_conv2d_transpose(inputs[0], inputs[1],
                                       padding, strides,
                                       out_dtype=out_dtype)
    return _nn.compute_conv2d_transpose(attrs, inputs, out)

    return res


@autotvm.register_topi_schedule(topi.generic.schedule_conv2d_nchw, 'vta', 'direct')
def schedule_packed_conv2d(cfg, outs,
                           skip_load_inp=False, skip_load_wgt=False, skip_load_acc=False,
                           skip_store_out=False, skip_alu=False, skip_gemm=False):
    assert len(outs) == 1
    output = outs[0]
    ewise_inputs = []
    ewise_ops = []
    conv2d_res = []
    assert "int" in output.op.input_tensors[0].dtype

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
            assert op.tag == "packed_conv2d"
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
    mock = env.mock
    load_inp = mock.dma_copy if skip_load_inp else env.dma_copy
    load_wgt = mock.dma_copy if skip_load_wgt else env.dma_copy
    load_acc = mock.dma_copy if skip_load_acc else env.dma_copy
    store_out = mock.dma_copy if skip_store_out else env.dma_copy
    alu = mock.alu if skip_alu else env.alu
    gemm = mock.gemm if skip_gemm else env.gemm

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

