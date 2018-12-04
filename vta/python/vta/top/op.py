"""Namespace for supporting packed_conv2d + ewise variant of nnvm."""
from __future__ import absolute_import as _abs

from collections import namedtuple

import logging

import tvm
from tvm import autotvm
import topi

from nnvm.top import registry as reg, OpPattern
from nnvm.top import nn as _nn

from ..environment import get_env
from ..ptr_alias import reinterpret

from .vta_group_conv2d import packed_group_conv2d, schedule_packed_group_conv2d
from .vta_conv2d_transpose import packed_conv2d_transpose, schedule_packed_conv2d_transpose

@reg.register_compute("clip", level=15)
def compute_clip(attrs, inputs, _):
    """ Clip operator. """
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

            return topi.nn.conv2d(inputs[0], inputs[1], strides, padding, dilation, layout, out_dtype)
        else:
            return topi.nn.group_conv2d_nchw(inputs[0], inputs[1], strides, padding, dilation, groups, out_dtype)

    with tvm.target.arm_cpu(tvm.target.current_target().model):
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
                return topi.generic.schedule_conv2d_nchw(outs)
            else:
                return topi.generic.schedule_group_conv2d_nchw(outs)
        elif str(target).startswith("llvm"):
            return tvm.create_schedule([x.op for x in outs])
        else:
            raise RuntimeError("not support target %s" % target)

    with tvm.target.arm_cpu(tvm.target.current_target().model):
        return _nn.schedule_conv2d(attrs, outs, tvm.target.current_target())


@reg.register_alter_op_layout("conv2d", level=15)
def alter_conv2d_layout(attrs, inputs, out):
    layout = attrs['layout']
    if is_packed_layout(layout):
        return None

    with tvm.target.arm_cpu(tvm.target.current_target().model):
        return _nn.alter_conv2d_layout(attrs, inputs, out)


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
        return topi.nn.conv2d_transpose_nchw(inputs[0], inputs[1],
                                             strides, padding,
                                             out_dtype)

    with tvm.target.arm_cpu(tvm.target.current_target().model):
        return _nn.compute_conv2d_transpose(attrs, inputs, out)


@reg.register_schedule("conv2d_transpose", level=15)
def schedule_conv2d_transpose(attrs, outs, target):
    """ 2D convolution schedule.
    """
    layout = attrs["layout"]

    if is_packed_layout(layout):
        target = tvm.target.create(target)
        if target.device_name == "vta":
            return topi.generic.schedule_conv2d_transpose_nchw(outs)
        elif str(target).startswith("llvm"):
            return tvm.create_schedule([x.op for x in outs])
        else:
            raise RuntimeError("not support target %s" % target)

    with tvm.target.arm_cpu(tvm.target.current_target().model):
        return _nn.schedule_conv2d_transpose(attrs, outs, tvm.target.current_target())

