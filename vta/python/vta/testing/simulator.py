"""Utilities to start simulator."""
import ctypes
import json
import tvm
from ..libinfo import find_libvta

def _load_lib():
    """Load local library, assuming they are simulator."""
    lib_path = find_libvta(optional=True)
    if not lib_path:
        return []
    try:
        return [ctypes.CDLL(lib_path[0], ctypes.RTLD_GLOBAL)]
    except OSError:
        return []


def enabled():
    """Check if simulator is enabled."""
    f = tvm.get_global_func("vta.simulator.profiler_clear", True)
    return f is not None


def clear_stats():
    """Clear profiler statistics"""
    f = tvm.get_global_func("vta.simulator.profiler_clear", True)
    if f:
        f()

# debug flag to skip execution.
DEBUG_SKIP_EXEC = 1

def debug_mode(flag):
    """Set debug mode

    Paramaters
    ----------
    flag : int
        The debug flag, 0 means clear all flags.
    """
    tvm.get_global_func("vta.simulator.profiler_debug_mode")(flag)


def stats():
    """Clear profiler statistics

    Returns
    -------
    stats : dict
        Current profiler statistics
    """
    x = tvm.get_global_func("vta.simulator.profiler_status")()
    return json.loads(x)


LIBS = _load_lib()
