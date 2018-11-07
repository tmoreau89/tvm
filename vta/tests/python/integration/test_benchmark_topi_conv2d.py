"""Testing if we can generate code in topi style"""

import os
import json
from collections import namedtuple

import numpy as np

import tvm
from tvm import autotvm
from tvm.contrib import util
from tvm.contrib.pickle_memoize import memoize
import topi
import topi.testing
import vta
import vta.testing


Workload = namedtuple("Conv2DWorkload",
                      ['batch', 'height', 'width', 'in_filter', 'out_filter',
                       'hkernel', 'wkernel', 'hpad', 'wpad', 'hstride', 'wstride'])

# ResNet18 workloads
resnet_wkls = [
    # Workloads of resnet18 on imagenet
    # Workload(env.BATCH, 224, 224, env.BLOCK_IN, 64, 7, 7, 3, 3, 2, 2),
    ('resnet-18.C2',  Workload(1, 56, 56, 64,  64,  3, 3, 1, 1, 1, 1)),
    #('resnet-18.C3',  Workload(1, 56, 56, 64,  64,  1, 1, 0, 0, 1, 1)), # this is a workload from a wrong model
    ('resnet-18.C4',  Workload(1, 56, 56, 64,  128, 3, 3, 1, 1, 2, 2)),
    ('resnet-18.C5',  Workload(1, 56, 56, 64,  128, 1, 1, 0, 0, 2, 2)),
    ('resnet-18.C6',  Workload(1, 28, 28, 128, 128, 3, 3, 1, 1, 1, 1)),
    ('resnet-18.C7',  Workload(1, 28, 28, 128, 256, 3, 3, 1, 1, 2, 2)),
    ('resnet-18.C8',  Workload(1, 28, 28, 128, 256, 1, 1, 0, 0, 2, 2)),
    ('resnet-18.C9',  Workload(1, 14, 14, 256, 256, 3, 3, 1, 1, 1, 1)),
    ('resnet-18.C10', Workload(1, 14, 14, 256, 512, 3, 3, 1, 1, 2, 2)),
    ('resnet-18.C11', Workload(1, 14, 14, 256, 512, 1, 1, 0, 0, 2, 2)),
    ('resnet-18.C12', Workload(1, 7,  7,  512, 512, 3, 3, 1, 1, 1, 1)),
]

def run_cpu_conv2d(env, remote, wl, target):
    data_shape = (wl.batch, wl.in_filter, wl.height, wl.width)
    kernel_shape = (wl.out_filter, wl.in_filter, wl.hkernel, wl.wkernel)
    fout_height = (wl.height + 2 * wl.hpad - wl.hkernel) // wl.hstride + 1
    fout_width = (wl.width + 2 * wl.wpad - wl.wkernel) // wl.wstride + 1
    data = tvm.placeholder(data_shape, name="data", dtype=env.inp_dtype)
    kernel = tvm.placeholder(kernel_shape, name="kernel", dtype=env.wgt_dtype)

    with target:
        res_conv = topi.nn.conv2d(
            data, kernel, (wl.hstride, wl.wstride), (wl.hpad, wl.wpad), "NCHW", "int32")
        res = topi.right_shift(res_conv, 8)
        res = my_clip(res, 0, 127)
        res = topi.cast(res, "int8")
        s = topi.generic.schedule_conv2d_nchw([res])

    # To compute number of ops, use a x2 factor for FMA
    num_ops = 2 * wl.batch * fout_height * fout_width * wl.hkernel * wl.wkernel * wl.out_filter * wl.in_filter

    # get reference data
    a_shape = (wl.batch, wl.in_filter, wl.height, wl.width)
    w_shape = (wl.out_filter, wl.in_filter, wl.hkernel, wl.wkernel)
    stride = (wl.hstride, wl.wstride)
    data_dtype = data.dtype
    kernel_dtype = kernel.dtype
    acc_dtype = env.acc_dtype
    padding = wl.hpad

    @memoize("vta.tests.test_benchmark_topi.conv2d.cpu.verify_nhwc")
    def get_ref_data():
        a_np = (np.random.uniform(size=a_shape) * 4).astype(data_dtype)
        w_np = (np.random.uniform(size=w_shape) * 4).astype(kernel_dtype)
        a_np = np.abs(a_np)
        w_np = np.abs(w_np)
        b_np = topi.testing.conv2d_nchw_python(
            a_np.astype(acc_dtype), w_np.astype(acc_dtype), stride, padding).astype(acc_dtype)
        return a_np, w_np, b_np

    check_correctness = True
    print_ir = False

    # build
    mod = tvm.build(s, [data, kernel, res], target=target, target_host=env.target_host, name="conv2d")
    temp = util.tempdir()
    mod.save(temp.relpath("conv2d.o"))
    remote.upload(temp.relpath("conv2d.o"))
    f = remote.load_module("conv2d.o")

    # run
    ctx = remote.context(str(target))
    data_orig, kernel_orig, res_ref = get_ref_data()
    res_shape = topi.util.get_const_tuple(res.shape)
    res_np = np.zeros(res_shape).astype(res.dtype)

    data_arr = tvm.nd.array(data_orig, ctx)
    kernel_arr = tvm.nd.array(kernel_orig, ctx)
    res_arr = tvm.nd.array(res_np, ctx)
    time_f = f.time_evaluator("conv2d", ctx, number=4)
    cost = time_f(data_arr, kernel_arr, res_arr)
    res_unpack = res_arr.asnumpy()

    # verify
    if check_correctness:
        assert wl.hpad == wl.wpad
        stride = (wl.hstride, wl.wstride)
        padding = wl.hpad
        res_ref = res_ref >> 8
        res_ref = np.clip(res_ref, 0, 127)
        res_ref = res_ref.astype("int8")
        np.testing.assert_allclose(res_unpack, res_ref)

    if print_ir:
        print(tvm.lower(s, [data, kernel, res], simple_mode=True))
    gops = (num_ops / cost.mean) / float(10 ** 9)
    print("CPU TEST: Time cost = %g sec/op, %g GOPS\n" % (cost.mean, gops))


def test_cpu_conv2d():
    def _run(env, remote):
        target = env.target_vta_cpu

        with autotvm.tophub.context(target):  # load pre-tuned operator parameters for ARM CPU
            for name, wl in resnet_wkls:
                print(name, wl)
                run_cpu_conv2d(env, remote, wl, target)

    vta.testing.run(_run)


def _sign_extend(value, bits):
    sign_bit = 1 << (bits - 1)
    return (value & (sign_bit - 1)) - (value & sign_bit)
_vector_sign_extend = np.vectorize(_sign_extend)


def _pack(x, width):
    assert(len(x.shape)==6)
    assert(x.dtype=="int8")
    pack_factor = 8 // width
    mask = ((1 << width) - 1)

    s = x.shape
    s_reshape = s[:-1] + (s[-1] // pack_factor, pack_factor)
    s_pack = s[:-1] + (s[-1] // pack_factor,)
    x_reshape = x.reshape(s_reshape)
    x_packed = np.zeros(s_pack, dtype="int8")
    for i in range(0, pack_factor):
        x_packed |= (x_reshape[:,:,:,:,:,:,i] & mask) << (i * width)

    return x_packed

def _unpack(x, width):
    assert(len(x.shape)==6)
    assert(x.dtype=="int8")
    pack_factor = 8 // width
    mask = ((1 << width) - 1)

    s = x.shape
    x_unpack = np.zeros(s[:] + (pack_factor,), dtype=x.dtype)
    for i in range(0, pack_factor):
        x_unpack[:,:,:,:,:,:,i] = _vector_sign_extend(((x >> (i * width)) & mask), width)

    return x_unpack.reshape(s[:-1] + (s[-1] * pack_factor,))

@tvm.tag_scope(tag=topi.tag.ELEMWISE)
def my_clip(x, a_min, a_max):
    """Unlike topi's current clip, put min and max into two stages."""
    const_min = tvm.const(a_min, x.dtype)
    const_max = tvm.const(a_max, x.dtype)
    x = tvm.compute(x.shape, lambda *i: tvm.min(x(*i), const_max), name="clipA")
    x = tvm.compute(x.shape, lambda *i: tvm.max(x(*i), const_min), name="clipB")
    return x


def run_vta_conv2d(env, remote, wl, target, check_correctness=True, print_ir=False,
                   plan_str=None, samples=4, profileOnly=False,
                   skip_load_inp=False, skip_load_wgt=False, skip_load_acc=False,
                   skip_store_out=False, skip_alu=False, skip_gemm=False):

    data_shape = (wl.batch//env.BATCH, wl.in_filter//env.BLOCK_IN,
                  wl.height, wl.width, env.BATCH, env.BLOCK_IN)
    kernel_shape = (wl.out_filter//env.BLOCK_OUT, wl.in_filter//env.BLOCK_IN,
                    wl.hkernel, wl.wkernel, env.BLOCK_OUT, env.BLOCK_IN)
    bias_shape = (wl.batch//env.BATCH, wl.out_filter//env.BLOCK_OUT,
                  1, 1, env.BATCH, env.BLOCK_OUT)
    data = tvm.placeholder(data_shape, name="data", dtype=env.inp_dtype)
    bias = tvm.placeholder(bias_shape, name="kernel", dtype=env.acc_dtype)
    factor = tvm.placeholder(bias_shape, name="kernel", dtype=env.acc_dtype)

    # Handle quantized inputs (less than 8 bits)
    # x_pack_factor = 1 << (3 - env.LOG_INP_WIDTH)
    # data_shape_pack = data_shape[:-1] + (data_shape[-1]//x_pack_factor,)
    # data_arg = tvm.placeholder(
    #     data_shape_pack,
    #     dtype="int8", name="data_arg")
    # data = vta.reinterpret(data_arg, data_shape, dtype=env.inp_dtype)

    # Handle quantized kernels (less than 8 bits)
    w_pack_factor = 1 << (3 - env.LOG_WGT_WIDTH)
    kernel_shape_pack = kernel_shape[:-1] + (kernel_shape[-1]//w_pack_factor,)
    kernel_arg = tvm.placeholder(kernel_shape_pack, dtype="int8", name="kernel_arg")
    kernel = vta.reinterpret(kernel_arg, kernel_shape, dtype=env.wgt_dtype)

    with target:
        res_conv = topi.nn.conv2d(
            data, kernel, (wl.hstride, wl.wstride), (wl.hpad, wl.wpad),
            "NCHW%dn%dc" % (env.BATCH, env.BLOCK_IN), 'int32')
        res = topi.right_shift(res_conv, 8)
        res = topi.add(res, bias)
        res = topi.multiply(res, factor)
        res = my_clip(res, 0, (1 << env.OUT_WIDTH-1)-1)
        res = topi.cast(res, "int8")

        s = topi.generic.schedule_conv2d_nchw([res])
                                         #planStr=plan_str,
                                         #skip_load_inp=skip_load_inp,
                                         #skip_load_wgt=skip_load_wgt,
                                         #skip_load_acc=skip_load_acc,
                                         #skip_store_out=skip_store_out,
                                         #skip_alu=skip_alu,
                                         #skip_gemm=skip_gemm)
        if print_ir:
            print(vta.lower(s, [data, kernel_arg, bias, factor, res], simple_mode=True))

    # Handle quantized outputs (less than 8 bits)
    # o_pack_factor = 1 << (3 - env.LOG_OUT_WIDTH)
    res_shape = topi.util.get_const_tuple(res.shape)
    # res_shape_pack = res_shape[:-1] + (res_shape[-1]//o_pack_factor,)
    # res_arg = vta.reinterpret(res, res_shape_pack, dtype="int8")

    # generate referene data
    fout_height = (wl.height + 2 * wl.hpad - wl.hkernel) // wl.hstride + 1
    fout_width = (wl.width + 2 * wl.wpad - wl.wkernel) // wl.wstride + 1
    num_ops = 2 * wl.batch * fout_height * fout_width * wl.hkernel * wl.wkernel * wl.out_filter * wl.in_filter

    a_shape = (wl.batch, wl.in_filter, wl.height, wl.width)
    w_shape = (wl.out_filter, wl.in_filter, wl.hkernel, wl.wkernel)
    stride = (wl.hstride, wl.wstride)
    data_dtype, kernel_dtype, acc_dtype = data.dtype, kernel.dtype, env.acc_dtype
    padding = wl.hpad
    INP_WIDTH, WGT_WIDTH = env.INP_WIDTH, env.WGT_WIDTH

    @memoize("vta.tests.test_benchmark_topi.conv2d.verify_nhwc")
    def get_ref_data():
        # derive min max for input and weight types (max non inclusive)
        a_min, a_max = 0 - (1 << (INP_WIDTH - 1)), (1 << (INP_WIDTH - 1))
        w_min, w_max = 0 - (1 << (WGT_WIDTH - 1)), (1 << (WGT_WIDTH - 1))
        a_np = np.random.randint(a_min, a_max, size=a_shape).astype("int8")
        w_np = np.random.randint(w_min, w_max, size=w_shape).astype("int8")
        b_np = topi.testing.conv2d_nchw_python(
            a_np.astype(acc_dtype), w_np.astype(acc_dtype), stride, padding).astype(acc_dtype)
        return a_np, w_np, b_np

    mod = vta.build(s,
                    [data, kernel_arg, bias, factor, res],
                    target,
                    env.target_host, name="conv2d")

    # Data in original format
    data_orig, kernel_orig, res_ref = get_ref_data()
    bias_orig = np.random.uniform(size=(wl.batch, wl.out_filter,)) * (1 << (env.INP_WIDTH + env.WGT_WIDTH - 2))
    bias_orig = bias_orig.astype("int32")
    bias_orig = np.abs(bias_orig)
    factor_orig = (np.random.uniform(size=(wl.batch, wl.out_filter,)) * (1 << 7))
    factor_orig = factor_orig.astype("int32")
    factor_orig = np.abs(factor_orig)


    data_packed = data_orig.reshape(
        wl.batch//env.BATCH, env.BATCH,
        wl.in_filter//env.BLOCK_IN, env.BLOCK_IN,
        wl.height, wl.width).transpose((0, 2, 4, 5, 1, 3))
    kernel_packed = kernel_orig.reshape(
        wl.out_filter//env.BLOCK_OUT, env.BLOCK_OUT,
        wl.in_filter//env.BLOCK_IN, env.BLOCK_IN,
        wl.hkernel, wl.wkernel).transpose((0, 2, 4, 5, 1, 3))
    factor_packed = factor_orig.reshape(
        wl.batch // env.BATCH, wl.out_filter // env.BLOCK_OUT,
        1, 1, env.BATCH, env.BLOCK_OUT)
    bias_packed = bias_orig.reshape(
        wl.batch // env.BATCH, wl.out_filter // env.BLOCK_OUT,
        1, 1, env.BATCH, env.BLOCK_OUT)

    # Quantized packing
    data_qpacked = _pack(data_packed, env.INP_WIDTH)
    kernel_qpacked = _pack(kernel_packed, env.WGT_WIDTH)

    # Upload
    temp = util.tempdir()
    mod.save(temp.relpath("conv2d.o"))
    remote.upload(temp.relpath("conv2d.o"))
    f = remote.load_module("conv2d.o")

    ctx = remote.context(str(target))
    res_np = np.zeros(res_shape).astype(res.dtype)
    data_arr = tvm.nd.array(data_qpacked, ctx)
    kernel_arr = tvm.nd.array(kernel_qpacked, ctx)
    factor_arr = tvm.nd.array(factor_packed, ctx)
    bias_arr = tvm.nd.array(bias_packed, ctx)
    res_arr = tvm.nd.array(res_np, ctx)
    time_f = f.time_evaluator("conv2d", ctx, number=samples)

    # In sim mode, collect simulator runtime statistics
    stats = {}
    cost = None
    if env.TARGET == "sim":
        # Check if we're in local RPC mode (allows us to rebuild the
        # runtime on the fly when varying the VTA designs)
        local_rpc = int(os.environ.get("VTA_LOCAL_SIM_RPC", "0"))
        if local_rpc:
            remote.get_function("vta.simulator.profiler_clear")()
            if profileOnly:
                remote.get_function("vta.simulator.profiler_debug_mode")(1)
            cost = time_f(data_arr, kernel_arr, bias_arr, factor_arr, res_arr)
            stats = json.loads(remote.get_function("vta.simulator.profiler_status")())
        else:
            simulator.clear_stats()
            if profileOnly:
                simulator.debug_mode(1)
            cost = time_f(data_arr, kernel_arr, bias_arr, factor_arr, res_arr)
            stats = simulator.stats()
    else:
        cost = time_f(data_arr, kernel_arr, bias_arr, factor_arr, res_arr)

    # Check correctness
    correct = False
    if check_correctness:
        res_unpack = res_arr.asnumpy()
        res_unpack = _unpack(res_unpack.astype("int8"), env.OUT_WIDTH)
        res_unpack = res_unpack.transpose(
            (0, 4, 1, 5, 2, 3)).reshape(wl.batch, wl.out_filter, fout_height, fout_width)
        assert wl.hpad == wl.wpad
        stride = (wl.hstride, wl.wstride)
        padding = wl.hpad
        res_ref = res_ref >> 8
        res_ref += bias_orig.reshape(wl.out_filter, 1, 1)
        res_ref *= factor_orig.reshape(wl.out_filter, 1, 1)
        res_ref = np.clip(res_ref, 0, (1 << env.OUT_WIDTH-1)-1)
        res_ref = res_ref.astype("int8")
        correct = np.allclose(res_unpack, res_ref)

    gops = (num_ops / cost.mean) / float(10 ** 9)
    status = "PASSED" if correct else "FAILED"
    print("VTA TEST %s: Time cost = %g sec/op, %g GOPS" % (status, cost.mean, gops))

    return correct, cost, stats


def test_vta_conv2d():
    def _run(env, remote):
        target = env.target

        with autotvm.tophub.context(target):  # load pre-tuned operator parameters for ARM CPU
            for name, wl in resnet_wkls:
                print(name, wl)
                run_vta_conv2d(env, remote, wl, target)

    vta.testing.run(_run)


if __name__ == "__main__":
    # test_cpu_conv2d()
    test_vta_conv2d()

