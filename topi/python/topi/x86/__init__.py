# pylint: disable=redefined-builtin, wildcard-import
"""x86 specific declaration and schedules."""
from __future__ import absolute_import as _abs

#from .conv2d import schedule_conv2d, schedule_conv2d_nhwc
# TODO(lmzheng): don't know why the schedule call of arm_cpu will jump
# to the compute call of x86 cpu. Some registerations are messed up.
# Disable x86 registers temporarily for vta.
from .binarize_pack import schedule_binarize_pack
from .binary_dense import schedule_binary_dense
from .nn import *
from .injective import *
from .pooling import schedule_pool, schedule_global_pool
from .bitserial_conv2d import schedule_bitserial_conv2d
from .depthwise_conv2d import schedule_depthwise_conv2d_NCHWc
