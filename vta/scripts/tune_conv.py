"""Tuning a single conv2d operator"""
import logging

import tvm
from tvm import autotvm
from tvm.contrib.util import get_lower_ir
import topi

import vta
import vta.testing

env = vta.get_env()

@tvm.tag_scope(tag=topi.tag.ELEMWISE)
def my_clip(x, a_min, a_max):
    """Unlike topi's current clip, put min and max into two stages."""
    const_min = tvm.const(a_min, x.dtype)
    const_max = tvm.const(a_max, x.dtype)
    x = tvm.compute(x.shape, lambda *i: tvm.min(x(*i), const_max), name="clipA")
    x = tvm.compute(x.shape, lambda *i: tvm.max(x(*i), const_min), name="clipB")
    return x

def conv2d(N, CI, H, W, CO, KH, KW, strides, padding, in_dtype, out_dtype):
    data_shape = (N//env.BATCH, CI//env.BLOCK_IN, H, W, env.BATCH, env.BLOCK_IN)
    kernel_shape = (CO//env.BLOCK_OUT, CI//env.BLOCK_IN, KH, KW, env.BLOCK_OUT, env.BLOCK_IN)
    bias_shape = (N//env.BATCH, CO//env.BLOCK_OUT, 1, 1, env.BATCH, env.BLOCK_OUT)

    OH = (H + 2 * padding[0] - KH) // strides[0] + 1
    OW = (W + 2 * padding[1] - KW) // strides[1] + 1

    data = tvm.placeholder(data_shape, name="data", dtype=env.inp_dtype)
    bias = tvm.placeholder(bias_shape, name="bias", dtype=env.acc_dtype)
    kernel = tvm.placeholder(kernel_shape, name="kernel", dtype=env.wgt_dtype)

    with tvm.target.vta():
        res = topi.nn.conv2d(data, kernel, padding=padding, strides=strides,
                             layout='NCHW%dn%dc' % (env.BATCH, env.BLOCK_IN), out_dtype='int32')
        res = topi.add(res, bias)
        res = topi.right_shift(res, 8)
        res = my_clip(res, 0, 127)
        res = topi.cast(res, "int8")

    if tvm.target.current_target().device_name == 'vta':
        s = topi.generic.schedule_conv2d_nchw([res])
    else:
        s = tvm.create_schedule([res.op])


    return s, [data, kernel, bias, res] 

if __name__ == '__main__':
    model = env.TARGET
    N, CI, H, W, CO, KH, KW, strides, padding, in_dtype, out_dtype = \
        1, 64, 56, 56, 64, 3, 3, (1, 1), (1, 1), 'int8', 'int32'

    task = autotvm.task.create(conv2d, args=(N, CI, H, W, CO, KH, KW, strides, padding, in_dtype, out_dtype),
            target=tvm.target.vta(model), target_host=env.target_host, template_key='direct')
    print(task.config_space)

    # logging config (for printing tuning log to the screen)
    logging.basicConfig()
    logging.getLogger('autotvm').setLevel(logging.DEBUG)

    measure_option = autotvm.measure_option(
            builder=autotvm.LocalBuilder(build_func=vta.vta_autotvm_build_func),
            runner=autotvm.RPCRunner(model, 'fleet', 9190, number=4, repeat=3, timeout=30,
                                     check_correctness=True))

    tuner = autotvm.tuner.RandomTuner(task)
    tuner.tune(n_trial=len(task.config_space),
               measure_option=measure_option,
               callbacks=[autotvm.callback.log_to_file('conv2d.log')])

    print(tuner.best_config)

