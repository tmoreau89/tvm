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
