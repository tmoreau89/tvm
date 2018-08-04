"""Reuse conv2d schedule from ARM CPU"""

import tvm

from topi.nn import conv2d, conv2d_alter_layout
from topi import generic

_WORKLOADS = [
    # resnet 18
    Workload('float32', 'float32', 224, 224, 3, 64, 7, 7, 3, 3, 2, 2),
    Workload('int8', 'int32', 224, 224, 3, 64, 7, 7, 3, 3, 2, 2),
    Workload('int8', 'int32', 56, 56, 64, 64, 3, 3, 1, 1, 1, 1),
    Workload('int8', 'int32', 56, 56, 64, 64, 1, 1, 0, 0, 1, 1),
    Workload('int8', 'int32', 56, 56, 64, 128, 3, 3, 1, 1, 2, 2),
    Workload('int8', 'int32', 56, 56, 64, 128, 1, 1, 0, 0, 2, 2),
    Workload('int8', 'int32', 28, 28, 128, 128, 3, 3, 1, 1, 1, 1),
    Workload('int8', 'int32', 28, 28, 128, 256, 3, 3, 1, 1, 2, 2),
    Workload('int8', 'int32', 28, 28, 128, 256, 1, 1, 0, 0, 2, 2),
    Workload('int8', 'int32', 14, 14, 256, 256, 3, 3, 1, 1, 1, 1),
    Workload('int8', 'int32', 14, 14, 256, 512, 3, 3, 1, 1, 2, 2),
    Workload('int8', 'int32', 14, 14, 256, 512, 1, 1, 0, 0, 2, 2),
    Workload('int8', 'int32', 7, 7, 512, 512, 3, 3, 1, 1, 1, 1),

    # mobilenet float32
    Workload('float32', 'float32', 224, 224, 3, 32, 3, 3, 1, 1, 2, 2),
    Workload('float32', 'float32', 112, 112, 32, 64, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 56, 56, 64, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 56, 56, 128, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 28, 28, 128, 256, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 28, 28, 256, 256, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 256, 512, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 512, 512, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7,  512, 1024, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1024, 1024, 1, 1, 0, 0, 1, 1),

    # mobilenet int8
    Workload('float32', 'float32', 224, 224, 3, 32, 3, 3, 1, 1, 2, 2),
    Workload('int8', 'int32', 112, 112, 32, 64, 1, 1, 0, 0, 1, 1),
    Workload('int8', 'int32', 56, 56, 64, 128, 1, 1, 0, 0, 1, 1),
    Workload('int8', 'int32', 56, 56, 128, 128, 1, 1, 0, 0, 1, 1),
    Workload('int8', 'int32', 28, 28, 128, 256, 1, 1, 0, 0, 1, 1),
    Workload('int8', 'int32', 28, 28, 256, 256, 1, 1, 0, 0, 1, 1),
    Workload('int8', 'int32', 14, 14, 256, 512, 1, 1, 0, 0, 1, 1),
    Workload('int8', 'int32', 14, 14, 512, 512, 1, 1, 0, 0, 1, 1),
    Workload('int8', 'int32', 7, 7,  512, 1024, 1, 1, 0, 0, 1, 1),
    Workload('int8', 'int32', 7, 7, 1024, 1024, 1, 1, 0, 0, 1, 1),
]

_SCHEDULES = [
    # float32 imagenet
    SpatialPack(1, 8, 4, 1, 4, True),
    SpatialPack(1, 8, 4, 1, 4, True),
    SpatialPack(1, 7, 4, 2, 4, True),
    SpatialPack(1, 4, 8, 4, 1, True),
    SpatialPack(1, 4, 4, 1, 16, False),
    SpatialPack(1, 4, 8, 4, 8, False),
    SpatialPack(1, 7, 4, 3, 8, True),
    SpatialPack(1, 2, 8, 1, 8, True),
    SpatialPack(2, 1, 16, 1, 4, True),
    SpatialPack(1, 7, 4, 1, 1, True),
    Im2ColPack(7, 4, 1, 16, True),
    Im2ColPack(7, 4, 1, 8, False),
    Im2ColPack(7, 4, 1, 16, False),

    # float32 mobilenet
    SpatialPack(2, 2, 4, 28, 1, True),
    SpatialPack(1, 4, 8, 14, 1, False),
    SpatialPack(1, 2, 16, 8, 1, True),
    SpatialPack(1, 4, 8, 8, 8, True),
    SpatialPack(2, 2, 8, 1, 1, False),
    SpatialPack(1, 4, 8, 4, 8, False),
    SpatialPack(2, 2, 8, 1, 4, False),
    SpatialPack(2, 2, 8, 1, 8, False),
    Im2ColPack(7, 4, 1, 16, False),
    Im2ColPack(7, 4, 1, 4, True),

    # int8 mobilenet
    SpatialPack(2, 2, 4, 28, 1, True),
    SpatialPack(1, 4, 8, 14, 1, False),
    SpatialPack(1, 2, 16, 8, 1, True),
    SpatialPack(1, 4, 8, 8, 8, True),
    SpatialPack(2, 2, 8, 1, 1, False),
    SpatialPack(1, 4, 8, 4, 8, False),
    SpatialPack(2, 2, 8, 1, 4, False),
    SpatialPack(2, 2, 8, 1, 8, False),
    Im2ColPack(7, 4, 1, 16, False),
    Im2ColPack(7, 4, 1, 4, True),
]

@conv2d.register(["vtacpu", "vta"])
def compute(*args, **kwargs):
    with tvm.target.arm_cpu("vtacpu"):
        return conv2d(*args, **kwargs)

@generic.schedule_conv2d_nchw.register(["vtacpu", "vta"])
def schedule(*args, **kwargs):
    with tvm.target.arm_cpu("vtacpu"):
        return generic.schedule_conv2d_nchw(*args, **kwargs)

@conv2d_alter_layout.register(["vtacpu", "vta"])
def alter(*args, **kwargs):
    with tvm.target.arm_cpu("vtacpu"):
        return conv2d_alter_layout(*args, **kwargs)
