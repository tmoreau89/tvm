"""Tuning a conv2d operator """
import tvm
import sys
import logging
from tvm import autotvm
from tvm.contrib.util import get_lower_ir
import topi

import vta
import vta.testing
from vta.top.testing import my_clip

env = vta.get_env()

def vta_build_func(measure_input, tmp_dir, **kwargs):
    import time
    import os
    from tvm.autotvm.measure.measure_methods import BuildResult
    from random import getrandbits
    from tvm.autotvm.util import get_const_tuple
    tic = time.time()
    try:
        filename = os.path.join(tmp_dir, "tmp_func_%0x.tar" % getrandbits(64))
        target, task, config = measure_input

        with target:
            s, args = task.instantiate(config)
            if not config.valid():
                raise InstantiationError(config.errors)

            func = vta.build(s, args, target='ext_dev', target_host=task.target_host)

        arg_info =  tuple((get_const_tuple(x.shape), x.dtype) for x in args)
        func.export_library(filename)
    except Exception as e:  # pylint: disable=broad-except
        return BuildResult(None, None, e, time.time() - tic)
    return BuildResult(filename, arg_info, None, time.time() - tic)


def schedule_packed_conv2d(cfg, outs,
                           skip_load_inp=False, skip_load_wgt=False, skip_load_acc=False,
                           skip_store_out=False, skip_alu=False, skip_gemm=False):
    """Schedule the packed conv2d.
    """
    assert len(outs) == 1
    output = outs[0]
    ewise_inputs = []
    ewise_ops = []
    conv2d_res = []
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

@autotvm.template
def conv2d(N, CI, H, W, CO, KH, KW, strides, padding, in_dtype, out_dtype):
    data_shape = (N//env.BATCH, CI//env.BLOCK_IN, H, W, env.BATCH, env.BLOCK_IN)
    kernel_shape = (CO//env.BLOCK_OUT, CI//env.BLOCK_IN, KH, KW, env.BLOCK_OUT, env.BLOCK_IN)
    bias_shape = (N//env.BATCH, CO//env.BLOCK_OUT, 1, 1, env.BATCH, env.BLOCK_OUT)

    OH = (H + 2 * padding[0] - KH) // strides[0] + 1
    OW = (W + 2 * padding[1] - KW) // strides[1] + 1

    data = tvm.placeholder(data_shape, name="data", dtype=env.inp_dtype)
    bias = tvm.placeholder(bias_shape, name="kernel", dtype=env.acc_dtype)

    w_pack_factor = 1 << (3 - env.LOG_WGT_WIDTH)
    kernel_shape_pack = kernel_shape[:-1] + (kernel_shape[-1] // w_pack_factor,)
    kernel_arg = tvm.placeholder(kernel_shape_pack, dtype="int8", name="kernel_arg")
    kernel = vta.reinterpret(kernel_arg, kernel_shape, dtype=env.wgt_dtype)

    res_conv = vta.top.packed_conv2d(data, kernel, padding=padding, strides=strides)
    res = topi.right_shift(res_conv, 8)
    res = topi.add(res, bias)
    res = my_clip(res, 0, 127)
    res = topi.cast(res, "int8")

    cfg = autotvm.get_config()
    s = schedule_packed_conv2d(cfg, [res])

    cfg.add_flop(2 * N * CI * OH * OW * CO * KH * KW)
    return s, [data, kernel_arg, bias, res] 

if __name__ == '__main__':
    N, CI, H, W, CO, KH, KW, strides, padding, in_dtype, out_dtype = \
        1, 64, 56, 56, 64, 3, 3, (1, 1), (1, 1), 'int8', 'int32'

    task = autotvm.task.create(conv2d, args=(N, CI, H, W, CO, KH, KW, strides, padding, in_dtype, out_dtype),
            target='ext_dev', target_host=env.target_host)
    print(task.config_space)

    # logging config (for printing tuning log to the screen)
    logging.getLogger('autotvm').setLevel(logging.DEBUG)
    logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

    measure_option = autotvm.measure_option(
            builder=autotvm.LocalBuilder(build_func=vta_build_func),
            runner=autotvm.RPCRunner(
                'ultra96', 'fleet', 9190))

    tuner = autotvm.tuner.RandomTuner(task)
    tuner.tune(n_trial=len(task.config_space),
               measure_option=measure_option,
               callbacks=[autotvm.callback.log_to_file('conv2d.log')])

    print(tuner.best_config)

