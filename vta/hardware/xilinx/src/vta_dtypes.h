/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file vta.h
 * \brief Type definitions and prototype for VTA HLS design.
 */
#ifndef VTA_VTA_DTYPES_H_
#define VTA_VTA_DTYPES_H_

#include <ap_int.h>

/* \typedef bus_T memory bus datatype*/
typedef ap_uint<VTA_BUS_WIDTH> bus_T;

/* \typedef uop_T Micro-op datatype*/
typedef ap_uint<VTA_UOP_WIDTH> uop_T;

#if VTA_INP_TYPE_CLASS == VTA_DTYPE_CLASS_UINT
    /* \typedef inp_T Input datatype*/
    typedef ap_uint<VTA_INP_WIDTH> inp_T;
#elif VTA_INP_TYPE_CLASS == VTA_DTYPE_CLASS_INT
    /* \typedef inp_T Input datatype*/
    typedef ap_int<VTA_INP_WIDTH> inp_T;
#endif

#if VTA_WGT_TYPE_CLASS == VTA_DTYPE_CLASS_UINT
    /* \typedef wgt_T Weight datatype*/
    typedef ap_uint<VTA_WGT_WIDTH> wgt_T;
#elif VTA_WGT_TYPE_CLASS == VTA_DTYPE_CLASS_INT
    /* \typedef wgt_T Weight datatype*/
    typedef ap_int<VTA_WGT_WIDTH> wgt_T;
#endif

#if VTA_ACC_TYPE_CLASS == VTA_DTYPE_CLASS_UINT
    /* \typedef acc_T Accumulator datatype*/
    typedef ap_uint<VTA_ACC_WIDTH> acc_T;
#elif VTA_ACC_TYPE_CLASS == VTA_DTYPE_CLASS_INT
    /* \typedef acc_T Accumulator datatype*/
    typedef ap_int<VTA_ACC_WIDTH> acc_T;
#endif

#if VTA_OUT_TYPE_CLASS == VTA_DTYPE_CLASS_UINT
    /* \typedef out_T Output datatype*/
    typedef ap_uint<VTA_OUT_WIDTH> out_T;
#elif VTA_OUT_TYPE_CLASS == VTA_DTYPE_CLASS_INT
    /* \typedef out_T Output datatype*/
    typedef ap_int<VTA_OUT_WIDTH> out_T;
#endif

/* \typedef uop_idx_T Micro-op SRAM index datatype*/
typedef ap_uint<VTA_LOG_UOP_BUFF_DEPTH+1> uop_idx_T;

/* \typedef inp_idx_T Input SRAM index datatype*/
typedef ap_uint<VTA_LOG_INP_BUFF_DEPTH+1> inp_idx_T;

/* \typedef wgt_idx_T Weight SRAM index datatype*/
typedef ap_uint<VTA_LOG_WGT_BUFF_DEPTH+1> wgt_idx_T;

/* \typedef acc_idx_T Accumulator SRAM index datatype*/
typedef ap_uint<VTA_LOG_ACC_BUFF_DEPTH+1> acc_idx_T;

/* \typedef opcode_T Opcode datatype*/
typedef ap_uint<VTA_OPCODE_BIT_WIDTH> opcode_T;

/* \typedef insn_T Instruction datatype*/
typedef ap_uint<VTA_INS_WIDTH> insn_T;

/* \typedef loop_T Loop bound datatype*/
typedef ap_uint<VTA_LOOP_ITER_WIDTH> loop_T;

/* \typedef memop_id_T Memory operation ID datatype*/
typedef ap_uint<VTA_MEMOP_ID_BIT_WIDTH> memop_id_T;

/* \typedef memop_sram_T Memory operation SRAM index datatype*/
typedef ap_uint<VTA_MEMOP_SRAM_ADDR_BIT_WIDTH> memop_sram_T;

/* \typedef memop_dram_T Memory operation DRAM index datatype*/
typedef ap_uint<VTA_MEMOP_DRAM_ADDR_BIT_WIDTH> memop_dram_T;

/* \typedef memop_size_T Memory operation range datatype*/
typedef ap_uint<VTA_MEMOP_SIZE_BIT_WIDTH> memop_size_T;

/* \typedef memop_stride_T Memory operation stride datatype*/
typedef ap_uint<VTA_MEMOP_STRIDE_BIT_WIDTH> memop_stride_T;

/* \typedef memop_pad_T Memory operation pad width datatype*/
typedef ap_uint<VTA_MEMOP_PAD_BIT_WIDTH> memop_pad_T;

/* \typedef aluop_opcode_T ALU operation opcode datatype*/
typedef ap_uint<VTA_ALU_OPCODE_BIT_WIDTH> aluop_opcode_T;

/* \typedef aluop_imm_T ALU operation immediate datatype*/
typedef ap_int<VTA_ALUOP_IMM_BIT_WIDTH> aluop_imm_T;

/* \typedef aluop_shr_arg_T ALU operation shift right immediate datatype*/
typedef ap_int<VTA_SHR_ARG_BIT_WIDTH> aluop_shr_arg_T;

/* \typedef aluop_mul_arg_T ALU operation multiply datatype*/
typedef ap_int<VTA_MUL_ARG_BIT_WIDTH> aluop_mul_arg_T;

#endif  // VTA_VTA_DTYPES_H_
