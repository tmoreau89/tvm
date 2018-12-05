import numpy as np
import tvm
from tvm import relay
from tvm.relay import quantize as qtz

def test_simulated_quantize():
    data = relay.var("data", relay.ty.TensorType((3, 4, 5, 6), "float32"))
    scale = relay.var("scale")
    bit = relay.var("bit")
    clip_min = relay.var("clip_min")
    clip_max = relay.var("clip_max")
    out = qtz.simulated_quantize(data, scale, bit, clip_min, clip_max, sign=True, rounding='round', kind=0)
    out = relay.ir_pass.infer_type(out)
    assert out.checked_type == out.args[0].checked_type
    assert out.args[1].checked_type == relay.ty.TensorType(tuple(), "float32")
    assert out.args[2].checked_type == relay.ty.TensorType(tuple(), "int32")
    assert out.args[3].checked_type == relay.ty.TensorType(tuple(), "float32")
    assert out.args[4].checked_type == relay.ty.TensorType(tuple(), "float32")

def test_annotate_pass():
    n, c, h, w = 1, 3, 224, 224
    def residual_block(data, cnt):
        # conv
        weight = relay.var("conv_weight" + str(cnt))
        conv = relay.nn.conv2d(data, weight, kernel_size=(3, 3), padding=(1, 1), channels=c)
        scale = relay.var('bn_scale' + str(cnt), relay.TensorType((1, c, 1, 1)))
        bias = relay.var('bn_bias' + str(cnt), relay.TensorType((1, c, 1, 1)))
        bn = conv * scale + bias
        relu = relay.nn.relu(bn)
        return relu

    data = relay.var("data", relay.TensorType((n, c, h, w), "float32"))
    out = data
    for i in range(1):
        out = residual_block(out, i)

    out = relay.ir_pass.infer_type(out)
    out = relay.ir_pass.simplify_inference(out)

    def make_dataset(args, size=100):
        def create_arr(var):
            ttype = var.type_annotation
            np_arr = np.random.uniform(-1.0, 1.0, size=ttype.concrete_shape).astype(ttype.dtype)
            return tvm.ndarray.array(np_arr)

        params = {}
        for arg in args:
            if arg.name_hint == 'data':
                dataset = [{'data': create_arr(arg)} for _ in range(size)]
            else:
                params[arg.name_hint] = create_arr(arg)
        return dataset, params

    args = relay.ir_pass.free_vars(out)
    graph = relay.Function(args, out)
    dataset, params = make_dataset(args, 10)

    with qtz.qconfig(skip_k_conv=0, global_scale=4.0):
        print('before:')
        print(graph.astext(show_meta_data=False))

        qgraph = qtz.quantize(graph, params)
        print('after quantize:')
        print(qgraph.astext(show_meta_data=False))
        print('\n')


if __name__ == "__main__":
    test_simulated_quantize()
    test_annotate_pass()
