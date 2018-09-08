"""Reuse conv2d schedule from ARM CPU"""

import tvm

from topi.nn import conv2d, conv2d_alter_layout
from topi import generic

@conv2d.register(["vta"])
def compute(*args, **kwargs):
    target = tvm.target.current_target()
    with tvm.target.arm_cpu(model=target.model):
        return conv2d(*args, **kwargs)

@generic.schedule_conv2d_nchw.register(["vta"])
def schedule(*args, **kwargs):
    target = tvm.target.current_target()
    with tvm.target.arm_cpu(model=target.model):
        return generic.schedule_conv2d_nchw(*args, **kwargs)

@conv2d_alter_layout.register(["vta"])
def alter(*args, **kwargs):
    target = tvm.target.current_target()
    with tvm.target.arm_cpu(model=target.model):
        return conv2d_alter_layout(*args, **kwargs)
