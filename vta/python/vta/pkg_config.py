# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""VTA Package configuration module

This module is dependency free and can be used to configure package.
"""
from __future__ import absolute_import as _abs

import json
import glob

class PkgConfig(object):
    """Simple package config tool for VTA.

    This is used to provide runtime specific configurations.

    Parameters
    ----------
    cfg : dict
        The config dictionary

    proj_root : str
        Path to the project root
    """
    cfg_keys = [
        "TARGET",
        "INP_DTYPE",
        "WGT_DTYPE",
        "ACC_DTYPE",
        "BATCH",
        "BLOCK",
        "UOP_BUFF_SIZE",
        "INP_BUFF_SIZE",
        "WGT_BUFF_SIZE",
        "ACC_BUFF_SIZE",
    ]

    # Datatype white list
    dtype_keys = [
        "int8",
        "int16",
        "int32"
    ]

    # Datatype width map
    dtype_width = {
        "int8": 8,
        "int16": 16,
        "int32": 32,
    }

    def __init__(self, cfg, proj_root):

        def _is_power_of_two(num):
            return num != 0 and ((num & (num - 1)) == 0)
        
        def _log2(num):
            return num.bit_length() - 1

        # Check parameters
        assert cfg["INP_DTYPE"] in self.dtype_keys
        assert cfg["WGT_DTYPE"] in self.dtype_keys
        assert cfg["ACC_DTYPE"] in self.dtype_keys
        assert _is_power_of_two(cfg["BATCH"])
        assert _is_power_of_two(cfg["BLOCK"])
        assert _is_power_of_two(cfg["UOP_BUFF_SIZE"])
        assert _is_power_of_two(cfg["INP_BUFF_SIZE"])
        assert _is_power_of_two(cfg["WGT_BUFF_SIZE"])
        assert _is_power_of_two(cfg["ACC_BUFF_SIZE"])

        # Derive parameters
        cfg["OUT_DTYPE"] = cfg["INP_DTYPE"]
        cfg["BLOCK_IN"] = cfg["BLOCK"]
        cfg["BLOCK_OUT"] = cfg["BLOCK"]
        cfg["INP_WIDTH"] = self.dtype_width[cfg["INP_DTYPE"]]
        cfg["WGT_WIDTH"] = self.dtype_width[cfg["WGT_DTYPE"]]
        cfg["ACC_WIDTH"] = self.dtype_width[cfg["ACC_DTYPE"]]
        cfg["OUT_WIDTH"] = self.dtype_width[cfg["OUT_DTYPE"]]
        cfg["OUT_BUFF_SIZE"] = (cfg["ACC_BUFF_SIZE"] *
                                cfg["OUT_WIDTH"] //
                                cfg["ACC_WIDTH"])

        # Derive log of parameters
        cfg["LOG_INP_WIDTH"] = _log2(cfg["INP_WIDTH"])
        cfg["LOG_WGT_WIDTH"] = _log2(cfg["WGT_WIDTH"])
        cfg["LOG_ACC_WIDTH"] = _log2(cfg["ACC_WIDTH"])
        cfg["LOG_OUT_WIDTH"] = _log2(cfg["OUT_WIDTH"])
        cfg["LOG_BATCH"] = _log2(cfg["BATCH"])
        cfg["LOG_BLOCK_IN"] = _log2(cfg["BLOCK_IN"])
        cfg["LOG_BLOCK_OUT"] = _log2(cfg["BLOCK_OUT"])
        cfg["LOG_UOP_BUFF_SIZE"] = _log2(cfg["UOP_BUFF_SIZE"])
        cfg["LOG_INP_BUFF_SIZE"] = _log2(cfg["INP_BUFF_SIZE"])
        cfg["LOG_WGT_BUFF_SIZE"] = _log2(cfg["WGT_BUFF_SIZE"])
        cfg["LOG_ACC_BUFF_SIZE"] = _log2(cfg["ACC_BUFF_SIZE"])
        cfg["LOG_OUT_BUFF_SIZE"] = _log2(cfg["OUT_BUFF_SIZE"])

        # Derive element bytes
        cfg["INP_ELEM_BITS"] = (cfg["BATCH"] *
                                cfg["BLOCK_IN"] *
                                cfg["INP_WIDTH"])
        cfg["WGT_ELEM_BITS"] = (cfg["BLOCK_OUT"] *
                                cfg["BLOCK_IN"] *
                                cfg["WGT_WIDTH"])
        cfg["ACC_ELEM_BITS"] = (cfg["BATCH"] *
                                cfg["BLOCK_OUT"] *
                                cfg["ACC_WIDTH"])
        cfg["OUT_ELEM_BITS"] = (cfg["BATCH"] *
                                cfg["BLOCK_OUT"] *
                                cfg["OUT_WIDTH"])
        cfg["INP_ELEM_BYTES"] = cfg["INP_ELEM_BITS"] // 8
        cfg["WGT_ELEM_BYTES"] = cfg["WGT_ELEM_BITS"] // 8
        cfg["ACC_ELEM_BYTES"] = cfg["ACC_ELEM_BITS"] // 8
        cfg["OUT_ELEM_BYTES"] = cfg["OUT_ELEM_BITS"] // 8

        # Update cfg now that we've extended it
        self.__dict__.update(cfg)

        # Include path
        self.include_path = [
            "-I%s/include" % proj_root,
            "-I%s/vta/include" % proj_root,
            "-I%s/3rdparty/dlpack/include" % proj_root,
            "-I%s/3rdparty/dmlc-core/include" % proj_root
        ]

        # List of source files that can be used to build standalone library.
        self.lib_source = []
        self.lib_source += glob.glob("%s/vta/src/*.cc" % proj_root)
        if self.TARGET in ["pynq", "ultra96"]:
            # add pynq drivers for any board that uses pynq driver stack (see pynq.io)
            self.lib_source += glob.glob("%s/vta/src/pynq/*.cc" % (proj_root))

        # Linker flags
        if self.TARGET in ["pynq", "ultra96"]:
            self.ldflags = [
                "-L/usr/lib",
                "-l:libcma.so"]
        else:
            self.ldflags = []

        # Derive bitstream config string.
        self.bitstream = "{}x{}_i{}w{}a{}_{}_{}_{}_{}".format(
            cfg["BATCH"],
            cfg["BLOCK"],
            cfg["INP_WIDTH"],
            cfg["WGT_WIDTH"],
            cfg["ACC_WIDTH"],
            cfg["LOG_UOP_BUFF_SIZE"],
            cfg["LOG_INP_BUFF_SIZE"],
            cfg["LOG_WGT_BUFF_SIZE"],
            cfg["LOG_ACC_BUFF_SIZE"])

        # Derive FPGA parameters from target
        #   - device:           part number
        #   - family:           fpga family
        #   - freq:             PLL frequency
        #   - per:              clock period to achieve in HLS
        #                       (how aggressively design is pipelined)
        #   - axi_bus_width:    axi bus width used for DMA transactions
        #                       (property of FPGA memory interface)
        #   - axi_cache_bits:   ARCACHE/AWCACHE signals for the AXI bus
        #                       (e.g. 1111 is write-back read and write allocate)
        #   - axi_prot_bits:    ARPROT/AWPROT signals for the AXI bus
        if self.TARGET == "ultra96":
            self.fpga_device = "xczu3eg-sbva484-1-e"
            self.fpga_family = "zynq-ultrascale+"
            self.fpga_freq = 333
            self.fpga_per = 2
            self.fpga_log_axi_bus_width = 7
            self.axi_prot_bits = '010'
            # IP register address map
            self.ip_reg_map_range = "0x1000"
            self.fetch_base_addr = "0xA0000000"
            self.load_base_addr = "0xA0001000"
            self.compute_base_addr = "0xA0002000"
            self.store_base_addr = "0xA0003000"
        else:
            # By default, we use the pynq parameters
            self.fpga_device = "xc7z020clg484-1"
            self.fpga_family = "zynq-7000"
            self.fpga_freq = 100
            self.fpga_per = 7
            self.fpga_log_axi_bus_width = 6
            self.axi_prot_bits = '000'
            # IP register address map
            self.ip_reg_map_range = "0x1000"
            self.fetch_base_addr = "0x43C00000"
            self.load_base_addr = "0x43C01000"
            self.compute_base_addr = "0x43C02000"
            self.store_base_addr = "0x43C03000"
        # Set coherence settings
        coherent = True
        if coherent:
            self.axi_cache_bits = '1111'
            self.coherent = True

        # Define IP memory mapped registers offsets.
        # In HLS 0x00-0x0C is reserved for block-level I/O protocol.
        # Make sure to leave 8B between register offsets to maintain
        # compatibility with 64bit systems.
        self.fetch_insn_count_offset = 0x10
        self.fetch_insn_addr_offset = self.fetch_insn_count_offset + 0x08
        self.load_inp_addr_offset = 0x10
        self.load_wgt_addr_offset = self.load_inp_addr_offset + 0x08
        self.compute_done_wr_offet = 0x10
        self.compute_done_rd_offet = self.compute_done_wr_offet + 0x08
        self.compute_uop_addr_offset = self.compute_done_rd_offet + 0x08
        self.compute_bias_addr_offset = self.compute_uop_addr_offset + 0x08
        self.store_out_addr_offset = 0x10

        # Derive SRAM parameters
        # The goal here is to determine how many memory banks are needed,
        # how deep and wide each bank needs to be. This is derived from
        # the size of each memory element (result of data width, and tensor shape),
        # and also how wide a memory can be as permitted by the FPGA tools.
        #
        # The mem axi ratio is a parameter used by HLS to resize memories
        # so memory read/write ports are the same size as the design axi bus width.
        #
        # Max bus width allowed (property of FPGA vendor toolchain)
        max_bus_width = 1024
        # Bus width of a memory interface
        mem_bus_width = 1 << self.fpga_log_axi_bus_width
        # Input memory
        self.inp_mem_banks = cfg["INP_ELEM_BITS"] // max_bus_width
        self.inp_mem_width = min(cfg["INP_ELEM_BITS"], max_bus_width)
        self.inp_mem_depth = cfg["INP_BUFF_SIZE"] // cfg["INP_ELEM_BYTES"]
        self.inp_mem_axi_ratio = self.inp_mem_width // mem_bus_width
        # Weight memory
        self.wgt_mem_banks = cfg["WGT_ELEM_BITS"] // max_bus_width
        self.wgt_mem_width = min(cfg["WGT_ELEM_BITS"], max_bus_width)
        self.wgt_mem_depth = cfg["WGT_BUFF_SIZE"] // cfg["WGT_ELEM_BYTES"]
        self.wgt_mem_axi_ratio = self.wgt_mem_width // mem_bus_width
        # Output memory
        self.out_mem_banks = cfg["OUT_ELEM_BITS"] // max_bus_width
        self.out_mem_width = min(cfg["OUT_ELEM_BITS"], max_bus_width)
        self.out_mem_depth = cfg["OUT_BUFF_SIZE"] // cfg["OUT_ELEM_BYTES"]
        self.out_mem_axi_ratio = self.out_mem_width // mem_bus_width

        # Macro defs
        self.macro_defs = []
        self.cfg_dict = {}
        for key in cfg:
            self.macro_defs.append("-DVTA_%s=%s" % (key, str(cfg[key])))
            self.cfg_dict[key] = cfg[key]
        self.macro_defs.append("-DVTA_LOG_BUS_WIDTH=%s" % (self.fpga_log_axi_bus_width))
        # Macros used by the VTA driver
        self.macro_defs.append("-DVTA_IP_REG_MAP_RANGE=%s" % (self.ip_reg_map_range))
        self.macro_defs.append("-DVTA_FETCH_ADDR=%s" % (self.fetch_base_addr))
        self.macro_defs.append("-DVTA_LOAD_ADDR=%s" % (self.load_base_addr))
        self.macro_defs.append("-DVTA_COMPUTE_ADDR=%s" % (self.compute_base_addr))
        self.macro_defs.append("-DVTA_STORE_ADDR=%s" % (self.store_base_addr))
        # IP register offsets
        self.macro_defs.append("-DVTA_FETCH_INSN_COUNT_OFFSET=%s" % \
                (self.fetch_insn_count_offset))
        self.macro_defs.append("-DVTA_FETCH_INSN_ADDR_OFFSET=%s" % \
                (self.fetch_insn_addr_offset))
        self.macro_defs.append("-DVTA_LOAD_INP_ADDR_OFFSET=%s" % \
                (self.load_inp_addr_offset))
        self.macro_defs.append("-DVTA_LOAD_WGT_ADDR_OFFSET=%s" % \
                (self.load_wgt_addr_offset))
        self.macro_defs.append("-DVTA_COMPUTE_DONE_WR_OFFSET=%s" % \
                (self.compute_done_wr_offet))
        self.macro_defs.append("-DVTA_COMPUTE_DONE_RD_OFFSET=%s" % \
                (self.compute_done_rd_offet))
        self.macro_defs.append("-DVTA_COMPUTE_UOP_ADDR_OFFSET=%s" % \
                (self.compute_uop_addr_offset))
        self.macro_defs.append("-DVTA_COMPUTE_BIAS_ADDR_OFFSET=%s" % \
                (self.compute_bias_addr_offset))
        self.macro_defs.append("-DVTA_STORE_OUT_ADDR_OFFSET=%s" % \
                (self.store_out_addr_offset))
        # Coherency
        if coherent:
            self.macro_defs.append("-DVTA_COHERENT_ACCESSES=true")
        else:
            self.macro_defs.append("-DVTA_COHERENT_ACCESSES=false")

    @property
    def cflags(self):
        return self.include_path + self.macro_defs

    @property
    def cfg_json(self):
        return json.dumps(self.cfg_dict, indent=2)

    def same_config(self, cfg):
        """Compare if cfg is same as current config.

        Parameters
        ----------
        cfg : the configuration
            The configuration

        Returns
        -------
        equal : bool
            Whether the configuration is the same.
        """
        for k, v in self.cfg_dict.items():
            if k not in cfg:
                return False
            if cfg[k] != v:
                return False
        return True
