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
#ifndef VTA_VTA_H_
#define VTA_VTA_H_

#include <ap_axi_sdata.h>
#include <ap_int.h>
#include <assert.h>
#include <hls_stream.h>

#include <vta/hw_spec.h>

#include "vta_dtypes.h"

/*!
* Define HLS stream depth 
*/
#define PRAGMA_SUB(x) _Pragma (#x)
#define PRAGMA_HLS(x) PRAGMA_SUB(x)
#define STREAM_IN_DEPTH 8

/*!
* \brief Fetch module.
*   Reads in \a insn_count instructions via DMA and pushes them to the
*   appropriate load, gemm or store queue.
* \param insns Instruction data base address in DRAM. AXI-4 master port.
* \param insn_count Total instruction count. AXI-lite memory mapped register.
* \param load_queue Load instruction queue. AXI-stream FIFO.
* \param gemm_queue GEMM instruction queue. AXI-stream FIFO.
* \param store_queue Store instruction queue. AXI-stream FIFO.
*/
void fetch(
  uint32_t insn_count,
  volatile insn_T *insns,
  hls::stream<insn_T> &load_queue,
  hls::stream<insn_T> &gemm_queue,
  hls::stream<insn_T> &store_queue);

/*!
* \brief Load module.
*   Reads in load instructions from the load queue, and performs appropriate
*   DMA load operation to the \a wgt_mem and \a inp_mem SRAM buffers from DRAM.
*   Updates dependence queues accordingly.
* \param inputs Input data base address in DRAM. AXI-4 master port.
* \param weights Weight data base address in DRAM. AXI-4 master port.
* \param load_queue Load instruction queue. AXI-stream FIFO.
* \param g2l_dep_queue Dependence queue from GEMM to load stage.
*   AXI-stream FIFO.
* \param l2g_dep_queue Dependence queue from load to GEMM stage.
*   AXI-stream FIFO.
* \param inp_mem Local input SRAM buffer. Write only single port BRAM.
* \param wgt_mem Local weight SRAM buffer. Write only single port BRAM.
*/
void load(
  volatile bus_T *inputs,
  volatile bus_T *weights,
  hls::stream<insn_T> &load_queue,
  hls::stream<bool> &g2l_dep_queue,
  hls::stream<bool> &l2g_dep_queue,
  bus_T inp_mem[VTA_INP_BUFF_DEPTH][INP_MAT_AXI_RATIO],
  bus_T wgt_mem[VTA_WGT_BUFF_DEPTH][WGT_MAT_AXI_RATIO]);

/*!
* \brief Compute module.
*   Reads in GEMM instructions from the gemm queue, and performs appropriate
*   GEMM/ALU instructions. Reads in data from the \a wgt_mem and \a inp_mem,
*   and writes computation results into the \a out_mem. Updates dependence
*   queues accordingly.
* \param done Signal that indicates that VLA is done.  AXI-lite memory mapped
*   register.
* \param uops Micro-op data base address in DRAM. AXI-4 master port.
* \param biases Bias data base address in DRAM. AXI-4 master port.
* \param gemm_queue GEMM instruction queue. AXI-stream FIFO.
* \param l2g_dep_queue Dependence queue from load to gemm stage.
*   AXI-stream FIFO.
* \param s2g_dep_queue Dependence queue from store to gemm stage.
*   AXI-stream FIFO.
* \param g2l_dep_queue Dependence queue from gemm to load stage.
*   AXI-stream FIFO.
* \param g2s_dep_queue Dependence queue from gemm to store stage.
*   AXI-stream FIFO.
* \param inp_mem Local input SRAM buffer. Read only single port BRAM.
* \param wgt_mem Local weight SRAM buffer. Read only single port BRAM.
* \param out_mem Local output SRAM buffer. Write only single port BRAM.
*/
void compute(
  volatile uint32_t &done,
  volatile uop_T *uops,
  volatile bus_T *biases,
  hls::stream<insn_T> &gemm_queue,
  hls::stream<bool> &l2g_dep_queue,
  hls::stream<bool> &s2g_dep_queue,
  hls::stream<bool> &g2l_dep_queue,
  hls::stream<bool> &g2s_dep_queue,
  bus_T inp_mem[VTA_INP_BUFF_DEPTH][INP_MAT_AXI_RATIO],
  bus_T wgt_mem[VTA_WGT_BUFF_DEPTH][WGT_MAT_AXI_RATIO],
  bus_T out_mem[VTA_ACC_BUFF_DEPTH][OUT_MAT_AXI_RATIO]);

/*!
* \brief Store module.
*   Reads in store instructions from the store queue, and performs appropriate
*   store instructions from the output buffer in SRAM to DRAM. Updates dependence
*   queues accordingly.
* \param outputs Output data base address in DRAM. AXI-4 master port.
* \param store_queue Store instruction queue. AXI-stream FIFO.
* \param g2s_dep_queue Dependence queue from gemm to store stage.
*   AXI-stream FIFO.
* \param s2g_dep_queue Dependence queue from store to gemm stage.
*   AXI-stream FIFO.
* \param out_mem Local output SRAM buffer. Read only single port BRAM.
*/
void store(
  volatile bus_T *outputs,
  hls::stream<insn_T> &store_queue,
  hls::stream<bool> &g2s_dep_queue,
  hls::stream<bool> &s2g_dep_queue,
  bus_T out_mem[VTA_ACC_BUFF_DEPTH][OUT_MAT_AXI_RATIO]);

/*!
* \brief VTA wrapper for simulation purpose only.
*   Orchestrates dataflow execution of the fetch, load, GEMM and store stages.
* \param insn_count Total instruction count. AXI-lite memory mapped register.
* \param insns Instruction data base address in DRAM. AXI-4 master port.
* \param uops Micro-op data base address in DRAM. AXI-4 master port.
* \param inputs Input data base address in DRAM. AXI-4 master port.
* \param weights Weight data base address in DRAM. AXI-4 master port.
* \param biases Bias data base address in DRAM. AXI-4 master port.
* \param outputs Output data base address in DRAM. AXI-4 master port.
*/
void vta(
  uint32_t insn_count,
  volatile insn_T *insns,
  volatile uop_T *uops,
  volatile bus_T *inputs,
  volatile bus_T *weights,
  volatile bus_T *biases,
  volatile bus_T *outputs);

#endif  // VTA_VTA_H_
