"""Programming utilities for Pynq.

Provides functions to program the Pynq FPGA.
"""
import os

def program_pynq(bitfile_name):
    """The method to download the bitstream onto PL.
    Note
    ----
    The class variables held by the singleton PL will also be updated.
    Returns
    -------
    None
    """
    BS_IS_PARTIAL = "/sys/devices/soc0/amba/f8007000.devcfg/" \
                    "is_partial_bitstream"
    BS_XDEVCFG = "/dev/xdevcfg"

    if not os.path.exists(BS_XDEVCFG):
        raise RuntimeError("Could not find programmable device")

    # Compose bitfile name, open bitfile
    with open(bitfile_name, 'rb') as f:
        buf = f.read()

    # Set is_partial_bitfile device attribute to the appropriate value
    with open(BS_IS_PARTIAL, 'w') as fd:
        fd.write('0')

    # Write bitfile to xdevcfg device
    with open(BS_XDEVCFG, 'wb') as f:
        f.write(buf)
