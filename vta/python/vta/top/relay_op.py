"""Namespace for supporting packed conv2d and element-wise variant of Relay."""
from __future__ import absolute_import as _abs

from collections import namedtuple

import logging

import tvm
from tvm import autotvm, relay
import topi

from tvm.relay.op import op as reg
from tvm.relay.op.op import OpPattern
from tvm.relay.op.nn import _nn

from ..environment import get_env
from ..ptr_alias import reinterpret

from .vta_group_conv2d import packed_group_conv2d, schedule_packed_group_conv2d
from .vta_conv2d_transpose import packed_conv2d_transpose, schedule_packed_conv2d_transpose

@reg.register_compute("clip", level=15)
def compute_clip(attrs, inputs, output_type, target):
    """ Clip operator. """
    x = inputs[0]
    a_min = attrs.a_min
    a_max = attrs.a_max
    const_min = tvm.const(a_min, x.dtype)
    const_max = tvm.const(a_max, x.dtype)
    with tvm.tag_scope(topi.tag.ELEMWISE):
        x = tvm.compute(
            x.shape, lambda *i: tvm.min(x(*i), const_max), name="clipA")
        x = tvm.compute(
            x.shape, lambda *i: tvm.max(x(*i), const_min), name="clipB")
    return [x]

# override to force partition at copy
reg.register_pattern("copy", OpPattern.OPAQUE, level=15)

def is_packed_layout(layout):
    """Check if layout is packed layout"""
    if layout == "NCHW":
        return False
    if "n" in layout and "c" in layout:
        return True
    return False

@reg.register_compute("nn.conv2d", level=15)
def compute_conv2d(attrs, inputs, output_type, target):
    """ 2D convolution algorithm.
    """
    padding = topi.util.get_const_tuple(attrs.padding)
    strides = topi.util.get_const_tuple(attrs.strides)
    dilation = tuple([int(d) for d in attrs.dilation])
    groups = attrs.groups
    layout = attrs.data_layout
    out_dtype = attrs.out_dtype

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

            return [topi.nn.conv2d(inputs[0], inputs[1], strides, padding, dilation, layout, out_dtype)]
        else:
            return [topi.nn.group_conv2d_nchw(inputs[0], inputs[1], strides, padding, dilation, groups, out_dtype)]

    with tvm.target.arm_cpu(tvm.target.current_target().model):
        return _nn.compute_conv2d(attrs, inputs, output_type, target)


@reg.register_schedule("nn.conv2d", level=15)
def schedule_conv2d(attrs, outputs, target):
    """ 2D convolution schedule.
    """
    layout = attrs.data_layout
    groups = attrs.groups

    if is_packed_layout(layout):
        target = tvm.target.create(target)
        if target.device_name == "vta":
            if groups == 1:
                return topi.generic.schedule_conv2d_nchw(outputs)
            else:
                return topi.generic.schedule_group_conv2d_nchw(outputs)
        elif str(target).startswith("llvm"):
            return tvm.create_schedule([x.op for x in outputs])
        else:
            raise RuntimeError("not support target %s" % target)

    with tvm.target.arm_cpu(tvm.target.current_target().model):
        return _nn.schedule_conv2d(attrs, outputs, tvm.target.current_target())

# @reg.register_alter_op_layout("conv2d", level=15)
# def alter_conv2d_layout(attrs, inputs, out):
#     layout = attrs['layout']
#     if is_packed_layout(layout):
#         return None

#     with tvm.target.arm_cpu(tvm.target.current_target().model):
#         return _nn.alter_conv2d_layout(attrs, inputs, out)


@reg.register_compute("nn.conv2d_transpose", level=15)
def compute_conv2d_transpose(attrs, inputs, output_type, target):
    """ 2D convolution algorithm.
    """
    padding = topi.util.get_const_tuple(attrs.padding)
    strides = topi.util.get_const_tuple(attrs.strides)
    dilation = tuple([int(d) for d in attrs.dilation])
    groups = attrs.groups
    layout = attrs.data_layout
    out_dtype = attrs.out_dtype

    assert dilation == (1, 1), "not support dilate now"
    if is_packed_layout(layout):
        return [packed_conv2d_transpose(inputs[0], inputs[1], padding, strides, out_dtype)]

    with tvm.target.arm_cpu(tvm.target.current_target().model):
        return _nn.compute_conv2d_transpose(attrs, inputs, output_type, target)


@reg.register_schedule("nn.conv2d_transpose", level=15)
def schedule_conv2d_transpose(attrs, outputs, target):
    """ 2D convolution schedule.
    """
    layout = attrs.data_layout

    if is_packed_layout(layout):
        target = tvm.target.create(target)
        if target.device_name == "vta":
            return schedule_packed_conv2d_transpose(outputs)
        elif str(target).startswith("llvm"):
            return tvm.create_schedule([x.op for x in outputs])
        else:
            raise RuntimeError("not support target %s" % target)

    with tvm.target.arm_cpu(tvm.target.current_target().model):
        return _nn.schedule_conv2d_transpose(attrs, outputs, tvm.target.current_target())
