"""Testing if we can generate code in topi style"""

import tvm
from tvm import autotvm
from tvm.contrib import util
from tvm.contrib.pickle_memoize import memoize
import topi
import topi.testing
import vta
import vta.testing
import numpy as np

Workload = vta.top.vta_group_conv2d.Workload


@tvm.tag_scope(tag=topi.tag.ELEMWISE)
def my_clip(x, a_min, a_max):
    """Unlike topi's current clip, put min and max into two stages."""
    const_min = tvm.const(a_min, x.dtype)
    const_max = tvm.const(a_max, x.dtype)
    x = tvm.compute(x.shape, lambda *i: tvm.min(x(*i), const_max), name="clipA")
    x = tvm.compute(x.shape, lambda *i: tvm.max(x(*i), const_min), name="clipB")
    return x


def test_vta_group_conv2d():
    def run_vta_group_conv2d(env, remote, name, wl, profile=True):
        assert wl.in_filter % wl.groups == 0
        assert wl.out_filter % wl.groups == 0
        assert wl.in_filter % (wl.groups * env.BLOCK_IN) == 0
        assert wl.batch % env.BATCH == 0
        assert wl.in_filter % env.BLOCK_IN == 0
        assert wl.out_filter % env.BLOCK_OUT == 0

        batch_size = wl.batch
        CI_G = wl.in_filter // wl.groups

        data_shape = (batch_size//env.BATCH, wl.in_filter//env.BLOCK_IN,
                      wl.height, wl.width, env.BATCH, env.BLOCK_IN)
        kernel_shape = (wl.out_filter//env.BLOCK_OUT, CI_G//env.BLOCK_IN,
                        wl.hkernel, wl.wkernel, env.BLOCK_OUT, env.BLOCK_IN)
        bias_shape = (batch_size//env.BATCH, wl.out_filter//env.BLOCK_OUT,
                      1, 1, env.BATCH, env.BLOCK_OUT)

        fout_height = (wl.height + 2 * wl.hpad - wl.hkernel) // wl.hstride + 1
        fout_width = (wl.width + 2 * wl.wpad - wl.wkernel) // wl.wstride + 1
        data = tvm.placeholder(data_shape, name="data", dtype=env.inp_dtype)
        kernel = tvm.placeholder(kernel_shape, name="kernel", dtype=env.wgt_dtype)
        bias = tvm.placeholder(bias_shape, name="bias", dtype=env.acc_dtype)

        res_conv = vta.top.packed_group_conv2d(
            data, kernel, (wl.hpad, wl.wpad), (wl.hstride, wl.wstride), wl.groups)
        res = topi.right_shift(res_conv, 8)
        res = topi.add(res, bias)
        res = my_clip(res, 0, 127)
        res = topi.cast(res, "int8")

        # To compute number of ops, use a x2 factor for FMA
        num_ops = 2 * batch_size * fout_height * fout_width * wl.hkernel * wl.wkernel * \
            wl.out_filter * wl.in_filter // wl.groups

        a_shape = (batch_size, wl.in_filter, wl.height, wl.width)
        w_shape = (wl.out_filter, CI_G, wl.hkernel, wl.wkernel)
        data_dtype = data.dtype
        kernel_dtype = kernel.dtype
        acc_dtype = env.acc_dtype
        stride = (wl.hstride, wl.wstride)
        padding = (wl.hpad, wl.wpad)
        groups = wl.groups

        @memoize("vta.tests.test_group_conv2d")
        def get_ref_data():
            a_np = (np.random.uniform(size=a_shape) * 4).astype(data_dtype)
            w_np = (np.random.uniform(size=w_shape) * 4).astype(kernel_dtype)
            a_np = np.abs(a_np)
            w_np = np.abs(w_np)
            b_np = topi.testing.group_conv2d_nchw_python(
                a_np.astype(acc_dtype), w_np.astype(acc_dtype), stride, padding, groups).astype(acc_dtype)
            return a_np, w_np, b_np

        def verify(s, check_correctness):
            mod = vta.build(s, [data, kernel, bias, res], "ext_dev",
                            env.target_host, name="group_conv2d")
            temp = util.tempdir()

            mod.save(temp.relpath("group_conv2d.o"))
            remote.upload(temp.relpath("group_conv2d.o"))
            f = remote.load_module("group_conv2d.o")
            # verify
            ctx = remote.ext_dev(0)
            # Data in original format
            data_orig, kernel_orig, res_ref = get_ref_data()
            bias_orig = (np.random.uniform(size=(wl.out_filter,)) * 4).astype("int32")
            bias_orig = np.abs(bias_orig)

            data_packed = data_orig.reshape(
                batch_size//env.BATCH, env.BATCH,
                wl.in_filter//env.BLOCK_IN, env.BLOCK_IN,
                wl.height, wl.width).transpose((0, 2, 4, 5, 1, 3))
            kernel_packed = kernel_orig.reshape(
                wl.out_filter//env.BLOCK_OUT, env.BLOCK_OUT,
                wl.in_filter//wl.groups//env.BLOCK_IN, env.BLOCK_IN,
                wl.hkernel, wl.wkernel).transpose((0, 2, 4, 5, 1, 3))
            bias_packed = bias_orig.reshape(
                1, wl.out_filter // env.BLOCK_OUT, 1, 1, env.BATCH, env.BLOCK_OUT)
            res_shape = topi.util.get_const_tuple(res.shape)

            res_np = np.zeros(res_shape).astype(res.dtype)
            data_arr = tvm.nd.array(data_packed, ctx)
            kernel_arr = tvm.nd.array(kernel_packed, ctx)
            bias_arr = tvm.nd.array(bias_packed, ctx)
            res_arr = tvm.nd.array(res_np, ctx)
            time_f = f.time_evaluator("group_conv2d", ctx, number=5)
            cost = time_f(data_arr, kernel_arr, bias_arr, res_arr)
            res_unpack = res_arr.asnumpy().transpose(
                (0, 4, 1, 5, 2, 3)).reshape(batch_size, wl.out_filter, fout_height, fout_width)
            if check_correctness:
                res_ref = res_ref >> 8
                res_ref += bias_orig.reshape(wl.out_filter, 1, 1)
                res_ref = np.clip(res_ref, 0, 127).astype("int8")
                np.testing.assert_allclose(res_unpack, res_ref)
            return cost

        def group_conv_normal(print_ir):
            print("----- Group conv2d End-to-End Test-------")
            with vta.build_config():
                s = vta.top.schedule_packed_group_conv2d([res])
                if print_ir:
                    print(vta.lower(s, [data, kernel, bias, res], simple_mode=True))
            cost = verify(s, True)
            gops = (num_ops / cost.mean) / float(10 ** 9)
            print("\tTime cost = %g sec/op, %g GOPS" % (cost.mean, gops))

        group_conv_normal(False)

    def _run(env, remote):
        tasks = [
            # mobilenet
            ('mobilenet.D1', Workload(1, 112, 112,  32,  32,  2, 3, 3, 1, 1, 1, 1)),
            ('mobilenet.D2', Workload(1, 112, 112,  64,  64,  4, 3, 3, 1, 1, 2, 2)),
            ('mobilenet.D3', Workload(1,  56,  56,  64,  64,  4, 3, 3, 1, 1, 1, 1)),
            ('mobilenet.D4', Workload(1,  56,  56, 128, 128,  8, 3, 3, 1, 1, 2, 2)),
            ('mobilenet.D5', Workload(1,  28,  28, 256, 256,  8, 3, 3, 1, 1, 1, 1)),
            ('mobilenet.D6', Workload(1,  28,  28, 256, 256, 16, 3, 3, 1, 1, 2, 2)),
            ('mobilenet.D7', Workload(1,  14,  14, 256, 256, 16, 3, 3, 1, 1, 1, 1)),
            ('mobilenet.D8', Workload(1,  14,  14, 256, 256, 16, 3, 3, 1, 1, 2, 2)),
            ('mobilenet.D9', Workload(1,  7,  7, 1024, 1024, 64, 3, 3, 1, 1, 1, 1)),
        ]

        for tsk in tasks:
            print(tsk)
            name, wkl = tsk
            run_vta_group_conv2d(env, remote, name, wkl)

    vta.testing.run(_run)

if __name__ == "__main__":
    test_vta_group_conv2d()
