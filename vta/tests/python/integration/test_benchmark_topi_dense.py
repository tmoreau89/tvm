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

Workload = vta.top.vta_dense.Workload


@tvm.tag_scope(tag=topi.tag.ELEMWISE)
def my_clip(x, a_min, a_max):
    """Unlike topi's current clip, put min and max into two stages."""
    const_min = tvm.const(a_min, x.dtype)
    const_max = tvm.const(a_max, x.dtype)
    x = tvm.compute(x.shape, lambda *i: tvm.min(x(*i), const_max), name="clipA")
    x = tvm.compute(x.shape, lambda *i: tvm.max(x(*i), const_min), name="clipB")
    return x


def test_vta_dense():
    def run_vta_dense(env, remote, name, wl, profile=True):
        data_shape = (wl.batch//env.BATCH, wl.in_dim//env.BLOCK_IN,
                      env.BATCH, env.BLOCK_IN)
        weight_shape = (wl.out_dim//env.BLOCK_OUT, wl.in_dim//env.BLOCK_IN,
                        env.BLOCK_OUT, env.BLOCK_IN)
        bias_shape = (wl.batch//env.BATCH, wl.out_dim//env.BLOCK_OUT,
                      env.BATCH, env.BLOCK_OUT)

        data = tvm.placeholder(data_shape, name="data", dtype=env.inp_dtype)
        weight = tvm.placeholder(weight_shape, name="kernel", dtype=env.wgt_dtype)
        bias = tvm.placeholder(bias_shape, name="bias", dtype=env.acc_dtype)
        data_dtype = data.dtype
        weight_dtype = weight.dtype

        res = vta.top.packed_dense(data, weight)
        res = topi.right_shift(res, 8)
        res = topi.add(res, bias)
        res = my_clip(res, 0, 127)
        res = topi.cast(res, "int8")

        # To compute number of ops, use a x2 factor for FMA
        num_ops = 2 * wl.batch * wl.in_dim * wl.out_dim
        a_shape = (wl.batch, wl.in_dim)
        w_shape = (wl.out_dim, wl.in_dim)
        acc_dtype = env.acc_dtype

        @memoize("vta.tests.test_dense")
        def get_ref_data():
            a_np = (np.random.uniform(size=a_shape) * 4).astype(data_dtype)
            w_np = (np.random.uniform(size=w_shape) * 4).astype(weight_dtype)
            a_np = np.abs(a_np)
            w_np = np.abs(w_np)
            b_np = np.dot(a_np.astype(acc_dtype), w_np.astype(acc_dtype).T).astype(acc_dtype)
            return a_np, w_np, b_np

        def verify(s, check_correctness):
            mod = vta.build(s, [data, weight, bias, res], "ext_dev",
                            env.target_host, name="dense")
            temp = util.tempdir()

            mod.save(temp.relpath("dense.o"))
            remote.upload(temp.relpath("dense.o"))
            f = remote.load_module("dense.o")
            # verify
            ctx = remote.ext_dev(0)
            # Data in original format
            data_orig, id_card_opriginal, res_ref = get_ref_data()
            bias_orig = (np.random.uniform(size=(wl.out_dim,)) * 4).astype("int32")
            bias_orig = np.ones_like(bias_orig)

            data_packed = data_orig.reshape(
                wl.batch//env.BATCH, env.BATCH,
                wl.in_dim//env.BLOCK_IN, env.BLOCK_IN).transpose((0, 2, 1, 3))
            weight_packed = id_card_opriginal.reshape(
                wl.out_dim//env.BLOCK_OUT, env.BLOCK_OUT,
                wl.in_dim//env.BLOCK_IN, env.BLOCK_IN).transpose((0, 2, 1, 3))
            bias_packed = bias_orig.reshape(
                1, wl.out_dim // env.BLOCK_OUT, 1, env.BLOCK_OUT)
            res_shape = topi.util.get_const_tuple(res.shape)

            res_np = np.zeros(res_shape).astype(res.dtype)
            data_arr = tvm.nd.array(data_packed, ctx)
            weight_arr = tvm.nd.array(weight_packed, ctx)
            bias_arr = tvm.nd.array(bias_packed, ctx)
            res_arr = tvm.nd.array(res_np, ctx)

            time_f = f.time_evaluator("dense", ctx, number=5)
            cost = time_f(data_arr, weight_arr, bias_arr, res_arr)
            res_unpack = res_arr.asnumpy().transpose(
                (0, 2, 1, 3)).reshape(wl.batch, wl.out_dim)
            if check_correctness:
                res_ref = res_ref >> 8
                res_ref += bias_orig.reshape(1, wl.out_dim)
                res_ref = np.clip(res_ref, 0, 127).astype("int8")
                np.testing.assert_allclose(res_unpack, res_ref)
            return cost

        def dense_normal(print_ir):
            print("----- dense End-to-End Test-------")
            with vta.build_config():
                s = vta.top.schedule_packed_dense([res])
                if print_ir:
                    print(vta.lower(s, [data, weight, bias, res], simple_mode=True))
            cost = verify(s, True)
            gops = (num_ops / cost.mean) / float(10 ** 9)
            print("\tTime cost = %g sec/op, %g GOPS" % (cost.mean, gops))

        dense_normal(False)

    def _run(env, remote):
        tasks = [
            ('dense.DEN1', Workload(1, 1024, 1024)),
            ('dense.DEN2', Workload(1, 512, 512)),
        ]

        for tsk in tasks:
            name, wkl = tsk
            run_vta_dense(env, remote, name, wkl)

    vta.testing.run(_run)

if __name__ == "__main__":
    test_vta_dense()
