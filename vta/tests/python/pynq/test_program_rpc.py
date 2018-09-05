import os
import tvm
from tvm import rpc
from vta import get_bitstream_path, download_bitstream, program_fpga, reconfig_runtime, get_env

env = get_env()

if env.TARGET == "pynq":
    host = os.environ.get("VTA_PYNQ_RPC_HOST", "pynq")
    port = int(os.environ.get("VTA_PYNQ_RPC_PORT", "9091"))
elif env.TARGET == "ultra96":
    host = os.environ.get("VTA_ULTRA96_RPC_HOST", "pynq")
    port = int(os.environ.get("VTA_ULTRA96_RPC_PORT", "9091"))
else:
    print("Programming in for target=\"{}\" is not supported".format(env.TARGET))
    exit()

def program_rpc_bitstream(path=None):
    """Program the FPGA on the RPC server

    Parameters
    ----------
    path : path to bitstream (optional)
    """
    assert tvm.module.enabled("rpc")
    remote = rpc.connect(host, port)
    program_fpga(remote, path)

def reconfig_rpc_runtime():
    """Reconfig the RPC server runtime
    """
    assert tvm.module.enabled("rpc")
    remote = rpc.connect(host, port)
    reconfig_runtime(remote)

program_rpc_bitstream()
reconfig_rpc_runtime()
