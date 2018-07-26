/*!
 *  Copyright (c) 2018 by Contributors
 * \file vta.cpp
 * \brief VTA HLS design.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vta.h"

void fetch(
  uint32_t insn_count,
  volatile insn_T *insns,
  hls::stream<insn_T> &load_queue,
  hls::stream<insn_T> &gemm_queue,
  hls::stream<insn_T> &store_queue) {
#pragma HLS INTERFACE s_axilite port = insn_count bundle = CONTROL_BUS
#pragma HLS INTERFACE m_axi port = insns offset = slave bundle = ins_port
#pragma HLS INTERFACE axis port = load_queue
#pragma HLS INTERFACE axis port = gemm_queue
#pragma HLS INTERFACE axis port = store_queue
#pragma HLS INTERFACE s_axilite port = return bundle = CONTROL_BUS

  INSN_DECODE: for (int pc = 0; pc < insn_count; pc++) {
#pragma HLS PIPELINE
    // Read instruction fields
    insn_T insn = insns[pc];
    // Do some partial decoding
    opcode_T opcode = insn.range(VTA_INSN_MEM_0_1, VTA_INSN_MEM_0_0);
    memop_id_T memory_type = insn.range(VTA_INSN_MEM_5_1, VTA_INSN_MEM_5_0);
    // Push to appropriate instruction queue
    if (opcode == VTA_OPCODE_STORE) {
      store_queue.write(insn);
    } else if (opcode == VTA_OPCODE_LOAD &&
          (memory_type == VTA_MEM_ID_INP || memory_type == VTA_MEM_ID_WGT)) {
      load_queue.write(insn);
    } else {
      gemm_queue.write(insn);
    }
  }
}

void reset_mem(
  memop_sram_T &sram_idx,
  memop_sram_T range,
  memop_id_T memory_type,
  axi_T inp_mem[VTA_INP_BUFF_DEPTH][INP_TENSOR_ELEMS],
  axi_T wgt_mem[VTA_WGT_BUFF_DEPTH][WGT_TENSOR_ELEMS]
  ) {

  if (memory_type == VTA_MEM_ID_INP) {
    for (int i = 0; i < range; i++) {
#pragma HLS PIPELINE
      for (int j = 0; j < INP_TENSOR_ELEMS; j++) {
        inp_mem[sram_idx][j] = 0;
      }
      sram_idx++;
    }
  } else {
    for (int i = 0; i < range; i++) {
#pragma HLS PIPELINE
      for (int j = 0; j < WGT_TENSOR_ELEMS; j++) {
        wgt_mem[sram_idx][j] = 0;
      }
      sram_idx++;
    }
  }

}

void load(
  volatile axi_T *inputs,
  volatile axi_T *weights,
  hls::stream<insn_T> &load_queue,
  hls::stream<bool> &g2l_dep_queue,
  hls::stream<bool> &l2g_dep_queue,
  axi_T inp_mem[VTA_INP_BUFF_DEPTH][INP_TENSOR_ELEMS],
  axi_T wgt_mem[VTA_WGT_BUFF_DEPTH][WGT_TENSOR_ELEMS]
  ) {
#pragma HLS INTERFACE m_axi port = weights offset = slave bundle = data_port
#pragma HLS INTERFACE m_axi port = inputs offset = slave bundle = data_port
#pragma HLS INTERFACE axis port = load_queue
#pragma HLS INTERFACE axis port = g2l_dep_queue
#pragma HLS INTERFACE axis port = l2g_dep_queue
#pragma HLS INTERFACE bram port = wgt_mem
#pragma HLS INTERFACE bram port = inp_mem
#pragma HLS INTERFACE s_axilite port = return bundle = CONTROL_BUS
#pragma HLS RESOURCE variable = inp_mem core = RAM_1P
#pragma HLS RESOURCE variable = wgt_mem core = RAM_1P

  // Pop load instruction
  insn_T insn = load_queue.read();

  // Decode instruction
  bool pop_prev_dependence = insn[VTA_INSN_MEM_1];
  bool pop_next_dependence = insn[VTA_INSN_MEM_2];
  bool push_prev_dependence = insn[VTA_INSN_MEM_3];
  bool push_next_dependence = insn[VTA_INSN_MEM_4];
  memop_id_T memory_type = insn.range(VTA_INSN_MEM_5_1, VTA_INSN_MEM_5_0);
  memop_sram_T sram_base = insn.range(VTA_INSN_MEM_6_1, VTA_INSN_MEM_6_0);
  memop_dram_T dram_base = insn.range(VTA_INSN_MEM_7_1, VTA_INSN_MEM_7_0);
  memop_size_T y_size = insn.range(VTA_INSN_MEM_8_1, VTA_INSN_MEM_8_0);
  memop_size_T x_size = insn.range(VTA_INSN_MEM_9_1, VTA_INSN_MEM_9_0);
  memop_stride_T x_stride = insn.range(VTA_INSN_MEM_A_1, VTA_INSN_MEM_A_0);
  memop_pad_T y_pad_0 = insn.range(VTA_INSN_MEM_B_1, VTA_INSN_MEM_B_0);
  memop_pad_T y_pad_1 = insn.range(VTA_INSN_MEM_C_1, VTA_INSN_MEM_C_0);
  memop_pad_T x_pad_0 = insn.range(VTA_INSN_MEM_D_1, VTA_INSN_MEM_D_0);
  memop_pad_T x_pad_1 = insn.range(VTA_INSN_MEM_E_1, VTA_INSN_MEM_E_0);

  // Pop dependence token if instructed
  if (pop_next_dependence) {
    g2l_dep_queue.read();
  }

  // Initialize indices
  memop_sram_T sram_idx = sram_base;
  memop_dram_T dram_idx = dram_base;

  // Pre-compute data and force to use no DSPs
  memop_sram_T x_line = x_pad_0 + x_size + x_pad_1;
  memop_sram_T y_offset_0 = x_line * y_pad_0;
#pragma HLS RESOURCE variable = y_offset_0 core = Mul_LUT
  memop_sram_T y_offset_1 = x_line * y_pad_1;
#pragma HLS RESOURCE variable = y_offset_1 core = Mul_LUT

  // Skip top padding
  sram_idx += y_offset_0;
  if (memory_type == VTA_MEM_ID_INP) {
    for (int y = 0; y < y_size; y++) {
#pragma HLS PIPELINE
      // Skip left padding
      sram_idx += x_pad_0;
      // Data transfer
      memcpy(&inp_mem[sram_idx][0],
             (const axi_T*) &inputs[dram_idx * INP_TENSOR_ELEMS],
             x_size * VTA_INP_ELEM_BYTES);
      sram_idx += x_size;
      dram_idx += x_stride;
      // Skip right Padding
      sram_idx += x_pad_1;
    }
  } else {
    for (int y = 0; y < y_size; y++) {
#pragma HLS PIPELINE
      // Skip left padding
      sram_idx += x_pad_0;
      // Data transfer
      memcpy(&wgt_mem[sram_idx][0],
             (const axi_T*) &weights[dram_idx * WGT_TENSOR_ELEMS],
             x_size * VTA_WGT_ELEM_BYTES);
      sram_idx += x_size;
      dram_idx += x_stride;
      // Skip right Padding
      sram_idx += x_pad_1;
    }
  }
  // Reset Index
  sram_idx = sram_base;

  // Top padding
  reset_mem(sram_idx, y_offset_0, memory_type, inp_mem, wgt_mem);
  for (int y = 0; y < y_size; y++) {
    // Left padding
    reset_mem(sram_idx, x_pad_0, memory_type, inp_mem, wgt_mem);
    // Skip line
    sram_idx += x_size;
    // Right Padding
    reset_mem(sram_idx, x_pad_1, memory_type, inp_mem, wgt_mem);
  }
  // Bottom padding
  reset_mem(sram_idx, y_offset_1, memory_type, inp_mem, wgt_mem);

  // Push dependence token if instructed
  if (push_next_dependence) {
    l2g_dep_queue.write(1);
  }
}

void compute(
  volatile uint32_t &done,
  volatile uop_T *uops,
  volatile axi_T *biases,
  hls::stream<insn_T> &gemm_queue,
  hls::stream<bool> &l2g_dep_queue,
  hls::stream<bool> &s2g_dep_queue,
  hls::stream<bool> &g2l_dep_queue,
  hls::stream<bool> &g2s_dep_queue,
  axi_T inp_mem[VTA_INP_BUFF_DEPTH][INP_TENSOR_ELEMS],
  axi_T wgt_mem[VTA_WGT_BUFF_DEPTH][WGT_TENSOR_ELEMS],
  axi_T out_mem[VTA_ACC_BUFF_DEPTH][OUT_TENSOR_ELEMS]
  ) {
#pragma HLS INTERFACE s_axilite port = done bundle = CONTROL_BUS
#pragma HLS INTERFACE m_axi port = uops offset = slave bundle = uop_port
#pragma HLS INTERFACE m_axi port = biases offset = slave bundle = data_port
#pragma HLS INTERFACE axis port = gemm_queue
#pragma HLS INTERFACE axis port = l2g_dep_queue
#pragma HLS INTERFACE axis port = s2g_dep_queue
#pragma HLS INTERFACE axis port = g2l_dep_queue
#pragma HLS INTERFACE axis port = g2s_dep_queue
#pragma HLS INTERFACE bram port = inp_mem
#pragma HLS INTERFACE bram port = wgt_mem
#pragma HLS INTERFACE bram port = out_mem
#pragma HLS INTERFACE s_axilite port = return bundle = CONTROL_BUS
#pragma HLS RESOURCE variable = inp_mem core = RAM_1P
#pragma HLS RESOURCE variable = wgt_mem core = RAM_1P
#pragma HLS RESOURCE variable = out_mem core = RAM_1P

  // Micro-op storage
  static uop_T uop_mem[VTA_UOP_BUFF_DEPTH];

  // Accumulator storage
  static axi_T acc_mem[VTA_ACC_BUFF_DEPTH][ACC_TENSOR_ELEMS];
// #pragma HLS ARRAY_PARTITION variable = acc_mem complete dim = 2
// This is necessary to obtain II=1
#pragma HLS DEPENDENCE variable = acc_mem inter false

  // Pop GEMM instruction
  insn_T insn = gemm_queue.read();

  // Decode
  opcode_T opcode = insn.range(VTA_INSN_MEM_0_1, VTA_INSN_MEM_0_0);
  bool pop_prev_dependence = insn[VTA_INSN_MEM_1];
  bool pop_next_dependence = insn[VTA_INSN_MEM_2];
  bool push_prev_dependence = insn[VTA_INSN_MEM_3];
  bool push_next_dependence = insn[VTA_INSN_MEM_4];

  // Pop dependence token if instructed
  if (pop_prev_dependence) {
    l2g_dep_queue.read();
  }
  if (pop_next_dependence) {
    s2g_dep_queue.read();
  }

  // Perform action based on opcode
  if (opcode == VTA_OPCODE_FINISH) {
    // Set done flag if we reach a FINISH instruction
    done = 1;
  } else if (opcode == VTA_OPCODE_LOAD || opcode == VTA_OPCODE_STORE) {
    // Set done value
    done = 0;

    // Decode instruction
    memop_id_T memory_type = insn.range(VTA_INSN_MEM_5_1, VTA_INSN_MEM_5_0);
    memop_sram_T sram_base = insn.range(VTA_INSN_MEM_6_1, VTA_INSN_MEM_6_0);
    memop_dram_T dram_base = insn.range(VTA_INSN_MEM_7_1, VTA_INSN_MEM_7_0);
    memop_size_T y_size = insn.range(VTA_INSN_MEM_8_1, VTA_INSN_MEM_8_0);
    memop_size_T x_size = insn.range(VTA_INSN_MEM_9_1, VTA_INSN_MEM_9_0);
    memop_stride_T x_stride = insn.range(VTA_INSN_MEM_A_1, VTA_INSN_MEM_A_0);
    memop_pad_T y_pad_0 = insn.range(VTA_INSN_MEM_B_1, VTA_INSN_MEM_B_0);
    memop_pad_T y_pad_1 = insn.range(VTA_INSN_MEM_C_1, VTA_INSN_MEM_C_0);
    memop_pad_T x_pad_0 = insn.range(VTA_INSN_MEM_D_1, VTA_INSN_MEM_D_0);
    memop_pad_T x_pad_1 = insn.range(VTA_INSN_MEM_E_1, VTA_INSN_MEM_E_0);

    // Initialize indices
    memop_sram_T sram_idx = sram_base;
    memop_dram_T dram_idx = dram_base;

    if (memory_type == VTA_MEM_ID_UOP) {
      // Perform data transfer
      memcpy(&uop_mem[sram_base],
             (const uop_T*) &uops[dram_base],
             x_size * sizeof(uop_T));
    } else {
      // Perform data transfer from DRAM
      for (int y = 0; y < y_size; y++) {
        // Perform data transfer
        memcpy(&acc_mem[sram_idx][0],
              (const axi_T*) &biases[dram_idx * ACC_TENSOR_ELEMS],
              x_size * VTA_ACC_ELEM_BYTES);
        sram_idx += x_size;
        dram_idx += x_stride;
      }
    }
  } else if (opcode == VTA_OPCODE_GEMM || opcode == VTA_OPCODE_ALU) {
    // Set done value
    done = 0;

    // Decode
    bool reset_out = insn[VTA_INSN_GEM_5];
    uop_idx_T uop_bgn = insn.range(VTA_INSN_GEM_6_1, VTA_INSN_GEM_6_0);
    uop_idx_T uop_end = insn.range(VTA_INSN_GEM_7_1, VTA_INSN_GEM_7_0);
    loop_T iter_out  = insn.range(VTA_INSN_GEM_8_1, VTA_INSN_GEM_8_0);
    loop_T iter_in  = insn.range(VTA_INSN_GEM_9_1, VTA_INSN_GEM_9_0);
    acc_idx_T dst_factor_out = insn.range(VTA_INSN_GEM_A_1, VTA_INSN_GEM_A_0);
    acc_idx_T dst_factor_in = insn.range(VTA_INSN_GEM_B_1, VTA_INSN_GEM_B_0);
    inp_idx_T src_factor_out = insn.range(VTA_INSN_GEM_C_1, VTA_INSN_GEM_C_0);
    inp_idx_T src_factor_in = insn.range(VTA_INSN_GEM_D_1, VTA_INSN_GEM_D_0);

    // GEMM-specific fields
    wgt_idx_T wgt_factor_out = insn.range(VTA_INSN_GEM_E_1, VTA_INSN_GEM_E_0);
    wgt_idx_T wgt_factor_in = insn.range(VTA_INSN_GEM_F_1, VTA_INSN_GEM_F_0);

    // ALU-specific field
    aluop_opcode_T alu_opcode = insn.range(VTA_INSN_ALU_E_1, VTA_INSN_ALU_E_0);
    bool use_imm = insn[VTA_INSN_ALU_F];
    aluop_imm_T imm = insn.range(VTA_INSN_ALU_G_1, VTA_INSN_ALU_G_0);
    acc_idx_T dst_offset_out = 0;
    inp_idx_T src_offset_out = 0;
    wgt_idx_T wgt_offset_out = 0;

    // Outer Loop
    EXE_OUT_LOOP: for (int it_out = 0; it_out < iter_out; it_out++) {
      acc_idx_T dst_offset_in = dst_offset_out;
      inp_idx_T src_offset_in = src_offset_out;
      wgt_idx_T wgt_offset_in = wgt_offset_out;

      // Inner Loop
      EXE_IN_LOOP: for (int it_in = 0; it_in < iter_in; it_in++) {
        // Perform appropriate computation based on opcode
        if (opcode == VTA_OPCODE_GEMM) {
          // Iterate over micro op
          READ_GEMM_UOP: for (int upc = uop_bgn; upc < uop_end; upc++) {

            // Read micro-op fields
            uop_T uop = uop_mem[upc];

            // Decode indices
            acc_idx_T dst_idx =
                uop.range(VTA_UOP_GEM_0_1, VTA_UOP_GEM_0_0) + dst_offset_in;
            inp_idx_T src_idx =
                uop.range(VTA_UOP_GEM_1_1, VTA_UOP_GEM_1_0) + src_offset_in;
            wgt_idx_T wgt_idx =
                uop.range(VTA_UOP_GEM_2_1, VTA_UOP_GEM_2_0) + wgt_offset_in;

            // Read in weight tensor
            wgt_T w_tensor[VTA_BLOCK_OUT][VTA_BLOCK_IN];
            for (int oc = 0; oc < VTA_BLOCK_OUT; oc++) {
              for (int p = 0; p < WGT_VEC_AXI_RATIO; p++) {
                axi_T packet = wgt_mem[wgt_idx][oc * WGT_VEC_AXI_RATIO + p];
                for (int w = 0; w < AXI_WGT_RATIO; w++) {
                  w_tensor[oc][p * AXI_WGT_RATIO + w] =
                      packet.range((w + 1) * VTA_WGT_WIDTH - 1, w * VTA_WGT_WIDTH);
                }
              }
            }

            // Read in input tensor
            inp_T i_tensor[VTA_BATCH][VTA_BLOCK_IN];
            for (int b = 0; b < VTA_BATCH; b++) {
              for (int p = 0; p < INP_VEC_AXI_RATIO; p++) {
                axi_T packet = inp_mem[src_idx][b * INP_VEC_AXI_RATIO + p];
                for (int w = 0; w < AXI_INP_RATIO; w++) {
                  i_tensor[b][p * AXI_INP_RATIO + w] =
                      packet.range((w + 1) * VTA_INP_WIDTH - 1, w * VTA_INP_WIDTH);
                }
              }
            }

            // Read in accum tensor
            acc_T a_tensor[VTA_BATCH][VTA_BLOCK_OUT];
            for (int b = 0; b < VTA_BATCH; b++) {
              for (int p = 0; p < ACC_VEC_AXI_RATIO; p++) {
                axi_T packet = acc_mem[dst_idx][b * ACC_VEC_AXI_RATIO + p];
                for (int w = 0; w < AXI_ACC_RATIO; w++) {
                  a_tensor[b][p * AXI_ACC_RATIO + w] =
                      packet.range((w + 1) * VTA_ACC_WIDTH - 1, w * VTA_ACC_WIDTH);
                }
              }
            }

            // Output tensor
            out_T o_tensor[VTA_BATCH][VTA_BLOCK_OUT];

            // Inner GEMM loop
            for (int b = 0; b < VTA_BATCH; b++) {
              for (int oc = 0; oc < VTA_BLOCK_OUT; oc++) {
                // Initialize the accumulator values
                acc_T accum = a_tensor[b][oc];
                // Dot product sum
                sum_T tmp = 0;
                // Inner matrix multiplication loop (input channel/feature)
                for (int ic = 0; ic < VTA_BLOCK_IN; ic++) {
                  wgt_T w_elem = w_tensor[oc][ic];
                  inp_T i_elem = i_tensor[b][ic];
                  mul_T prod = i_elem * w_elem;
#ifdef NO_DSP
#pragma HLS RESOURCE variable = prod core = Mul_LUT
#endif //  NO_DSP
                  tmp += (sum_T) prod;
                }
                // Update summation
                accum += (acc_T) tmp;
                // Write back result acc_mem
                a_tensor[b][oc] = reset_out ? (acc_T) 0 : accum;
                // And output vector
                o_tensor[b][oc] = (out_T) accum.range(VTA_OUT_WIDTH - 1, 0);
              }
            }

            // Write the results back into accumulator
            for (int b = 0; b < VTA_BATCH; b++) {
              for (int p = 0; p < ACC_VEC_AXI_RATIO; p++) {
                axi_T packet = 0;
                for (int w = 0; w < AXI_ACC_RATIO; w++) {
                  packet.range((w + 1) * VTA_ACC_WIDTH - 1, w * VTA_ACC_WIDTH) = a_tensor[b][p * AXI_ACC_RATIO + w];
                }
                acc_mem[dst_idx][b * ACC_VEC_AXI_RATIO + p] = packet;
              }
            }

            // Write the results back in the output buffer
            for (int b = 0; b < VTA_BATCH; b++) {
              for (int p = 0; p < OUT_VEC_AXI_RATIO; p++) {
                axi_T packet = 0;
                for (int w = 0; w < AXI_OUT_RATIO; w++) {
                  packet.range((w + 1) * VTA_OUT_WIDTH - 1, w * VTA_OUT_WIDTH) = o_tensor[b][p * AXI_OUT_RATIO + w];
                }
                out_mem[dst_idx][b * OUT_VEC_AXI_RATIO + p] = packet;
              }
            }
          }
        }
#ifndef NO_ALU
        else if (opcode == VTA_OPCODE_ALU) {
          // Iterate over micro op
          READ_ALU_UOP: for (int upc = uop_bgn; upc < uop_end; upc++) {
            // Read micro-op fields
            uop_T uop = uop_mem[upc];

            // Decode
            acc_idx_T dst_idx =
                uop.range(VTA_UOP_ALU_0_1, VTA_UOP_ALU_0_0) + dst_offset_in;
            acc_idx_T src_idx =
                uop.range(VTA_UOP_ALU_1_1, VTA_UOP_ALU_1_0) + src_offset_in;

            // Read in src tensor
            acc_T src_tensor[VTA_BATCH][VTA_BLOCK_OUT];
            for (int b = 0; b < VTA_BATCH; b++) {
              for (int p = 0; p < ACC_VEC_AXI_RATIO; p++) {
                axi_T packet = acc_mem[src_idx][b * ACC_VEC_AXI_RATIO + p];
                for (int w = 0; w < AXI_ACC_RATIO; w++) {
                  src_tensor[b][p * AXI_ACC_RATIO + w] =
                      packet.range((w + 1) * VTA_ACC_WIDTH - 1, w * VTA_ACC_WIDTH);
                }
              }
            }

            // Read in dst tensor
            acc_T dst_tensor[VTA_BATCH][VTA_BLOCK_OUT];
            for (int b = 0; b < VTA_BATCH; b++) {
              for (int p = 0; p < ACC_VEC_AXI_RATIO; p++) {
                axi_T packet = acc_mem[dst_idx][b * ACC_VEC_AXI_RATIO + p];
                for (int w = 0; w < AXI_ACC_RATIO; w++) {
                  dst_tensor[b][p * AXI_ACC_RATIO + w] =
                      packet.range((w + 1) * VTA_ACC_WIDTH - 1, w * VTA_ACC_WIDTH);
                }
              }
            }

            // Output tensor
            out_T o_tensor[VTA_BATCH][VTA_BLOCK_OUT];

            // Perform ALU op over matrix elements
            for (int i = 0; i < VTA_BATCH; i++) {
              for (int b = 0; b < VTA_BLOCK_OUT; b++) {
                // Read in operands
                acc_T src_0 = dst_tensor[i][b];
                acc_T src_1 = use_imm ? (acc_T) imm : src_tensor[i][b];
                // Compute Min/Max
                acc_T mix_val = src_0 < src_1 ?
                    (alu_opcode == VTA_ALU_OPCODE_MIN ? src_0 : src_1) :
                    (alu_opcode == VTA_ALU_OPCODE_MIN ? src_1 : src_0);
                dst_tensor[i][b] = mix_val;
                o_tensor[i][b] = (out_T) mix_val.range(VTA_OUT_WIDTH - 1, 0);
                // Compute Sum
                acc_T add_val =
                    src_0.range(VTA_ACC_WIDTH - 1, 0) + src_1.range(VTA_ACC_WIDTH - 1, 0);
                dst_tensor[i][b] = add_val;
                o_tensor[i][b] = (out_T) add_val.range(VTA_OUT_WIDTH - 1, 0);
                // Compute Shift Right
                acc_T shr_val = src_0 >> (aluop_sh_imm_T) src_1.range(VTA_LOG_ACC_WIDTH - 1, 0);
                dst_tensor[i][b] = shr_val;
                o_tensor[i][b] = (out_T) shr_val.range(VTA_OUT_WIDTH-1, 0);
              }
            }

            // Write the results back into accumulator
            for (int b = 0; b < VTA_BATCH; b++) {
              for (int p = 0; p < ACC_VEC_AXI_RATIO; p++) {
                axi_T packet = 0;
                for (int w = 0; w < AXI_ACC_RATIO; w++) {
                  packet.range((w + 1) * VTA_ACC_WIDTH - 1, w * VTA_ACC_WIDTH) = dst_tensor[b][p * AXI_ACC_RATIO + w];
                }
                acc_mem[dst_idx][b * ACC_VEC_AXI_RATIO + p] = packet;
              }
            }

            // Write the results back in the output buffer
            for (int b = 0; b < VTA_BATCH; b++) {
              for (int p = 0; p < OUT_VEC_AXI_RATIO; p++) {
                axi_T packet = 0;
                for (int w = 0; w < AXI_OUT_RATIO; w++) {
                  packet.range((w + 1) * VTA_OUT_WIDTH - 1, w * VTA_OUT_WIDTH) = o_tensor[b][p * AXI_OUT_RATIO + w];
                }
                out_mem[dst_idx][b * OUT_VEC_AXI_RATIO + p] = packet;
              }
            }
          }
        }
#endif  // NO_ALU

        // Update offsets
        dst_offset_in += dst_factor_in;
        src_offset_in += src_factor_in;
        wgt_offset_in += wgt_factor_in;
      }

      // Update offsets
      dst_offset_out += dst_factor_out;
      src_offset_out += src_factor_out;
      wgt_offset_out += wgt_factor_out;
    }
  }

  // Push dependence token if instructed
  if (push_prev_dependence) {
    g2l_dep_queue.write(1);
  }
  if (push_next_dependence) {
    g2s_dep_queue.write(1);
  }
}

void store(
  volatile axi_T *outputs,
  hls::stream<insn_T> &store_queue,
  hls::stream<bool> &g2s_dep_queue,
  hls::stream<bool> &s2g_dep_queue,
  axi_T out_mem[VTA_ACC_BUFF_DEPTH][OUT_TENSOR_ELEMS]
  ) {
#pragma HLS INTERFACE m_axi port = outputs offset = slave bundle = data_port
#pragma HLS INTERFACE axis port = store_queue
#pragma HLS INTERFACE axis port = g2s_dep_queue
#pragma HLS INTERFACE axis port = s2g_dep_queue
#pragma HLS INTERFACE bram port = out_mem
#pragma HLS INTERFACE s_axilite port = return bundle = CONTROL_BUS
#pragma HLS RESOURCE variable = out_mem core = RAM_1P

  // Load buffer
  insn_T insn = store_queue.read();

  // Decode
  bool pop_prev_dependence = insn[VTA_INSN_MEM_1];
  bool pop_next_dependence = insn[VTA_INSN_MEM_2];
  bool push_prev_dependence = insn[VTA_INSN_MEM_3];
  bool push_next_dependence = insn[VTA_INSN_MEM_4];
  memop_id_T memory_type = insn.range(VTA_INSN_MEM_5_1, VTA_INSN_MEM_5_0);
  memop_sram_T sram_base = insn.range(VTA_INSN_MEM_6_1, VTA_INSN_MEM_6_0);
  memop_dram_T dram_base = insn.range(VTA_INSN_MEM_7_1, VTA_INSN_MEM_7_0);
  memop_size_T y_size = insn.range(VTA_INSN_MEM_8_1, VTA_INSN_MEM_8_0);
  memop_size_T x_size = insn.range(VTA_INSN_MEM_9_1, VTA_INSN_MEM_9_0);
  memop_stride_T x_stride = insn.range(VTA_INSN_MEM_A_1, VTA_INSN_MEM_A_0);
  memop_pad_T y_pad_0 = insn.range(VTA_INSN_MEM_B_1, VTA_INSN_MEM_B_0);
  memop_pad_T y_pad_1 = insn.range(VTA_INSN_MEM_C_1, VTA_INSN_MEM_C_0);
  memop_pad_T x_pad_0 = insn.range(VTA_INSN_MEM_D_1, VTA_INSN_MEM_D_0);
  memop_pad_T x_pad_1 = insn.range(VTA_INSN_MEM_E_1, VTA_INSN_MEM_E_0);

  // Pop dependence token if instructed
  if (pop_prev_dependence) {
    g2s_dep_queue.read();
  }

  // Initialize indices
  memop_sram_T sram_idx = sram_base;
  memop_dram_T dram_idx = dram_base;

  // Copy along y dimension
  for (int y = 0; y < y_size; y++) {
#pragma HLS PIPELINE
    // Perform data transfer
    memcpy(
      const_cast<axi_T*>(&outputs[dram_idx * OUT_TENSOR_ELEMS]),
      (const axi_T*) &out_mem[sram_idx][0],
      x_size * VTA_OUT_ELEM_BYTES);
#pragma HLS RESOURCE variable = sram_idx core = Mul_LUT
    sram_idx += x_size;
    dram_idx += x_stride;
  }

  // Push dependence token if instructed
  if (push_prev_dependence) {
    s2g_dep_queue.write(1);
  }
}

void vta(
  uint32_t insn_count,
  volatile insn_T *insns,
  volatile uop_T *uops,
  volatile axi_T *inputs,
  volatile axi_T *weights,
  volatile axi_T *biases,
  volatile axi_T *outputs) {
#pragma HLS INTERFACE s_axilite port = insn_count bundle = CONTROL_BUS
#pragma HLS INTERFACE m_axi port = insns offset = slave bundle = ins_port
#pragma HLS INTERFACE m_axi port = uops offset = slave bundle = uop_port
#pragma HLS INTERFACE m_axi port = inputs offset = slave bundle = data_port
#pragma HLS INTERFACE m_axi port = weights offset = slave bundle = data_port
#pragma HLS INTERFACE m_axi port = biases offset = slave bundle = data_port
#pragma HLS INTERFACE m_axi port = outputs offset = slave bundle = data_port
#pragma HLS INTERFACE s_axilite port = return bundle = CONTROL_BUS

  // Instantiate temporary instruction queues (used for peeking)
  hls::stream<insn_T> tmp_load_queue;
  hls::stream<insn_T> tmp_gemm_queue;
  hls::stream<insn_T> tmp_store_queue;

  // Instatiate physical instruction queues
  hls::stream<insn_T> load_queue;
  hls::stream<insn_T> gemm_queue;
  hls::stream<insn_T> store_queue;

  // Dependence queues
  hls::stream<bool> l2g_dep_queue;
  hls::stream<bool> s2g_dep_queue;
  hls::stream<bool> g2l_dep_queue;
  hls::stream<bool> g2s_dep_queue;

  // Instantiate memories
  axi_T inp_mem[VTA_INP_BUFF_DEPTH][INP_TENSOR_ELEMS];
  axi_T wgt_mem[VTA_WGT_BUFF_DEPTH][WGT_TENSOR_ELEMS];
  axi_T out_mem[VTA_ACC_BUFF_DEPTH][OUT_TENSOR_ELEMS];

  // Push all instructions into the queues
  fetch(insn_count, insns, tmp_load_queue, tmp_gemm_queue, tmp_store_queue);

  // Global done indicator
  uint32_t done = 0;

  // Temporary instructions
  insn_T tmp_load;
  insn_T tmp_gemv;
  insn_T tmp_store;

  // Peeking status
  bool tmp_load_popped = false;
  bool tmp_gemm_popped = false;
  bool tmp_store_popped = false;
  int exit_counter = 0;

  // Main control loop
  while (true) {
    // First execute as many load instructions as possible
    while (!tmp_load_queue.empty() || tmp_load_popped == true) {
      // Pop the load instruction
      if (!tmp_load_popped) {
        tmp_load_queue.read(tmp_load);
        tmp_load_popped = true;
      }
      // Check dependences and invoke the load stage
      bool pop_next_dependence = tmp_load[VTA_INSN_MEM_2];
      if ((pop_next_dependence && !g2l_dep_queue.empty()) ||
          !pop_next_dependence) {
        // Push the instruction in the load queue
        load_queue.write(tmp_load);
        tmp_load_popped = false;
        load(inputs, weights, load_queue, g2l_dep_queue, l2g_dep_queue, inp_mem, wgt_mem);
      } else {
        // Execution of load stage pending on completion of other stages, so break here...
        break;
      }
    }
    // Next execute as many gemm instructions as possible
    while (!tmp_gemm_queue.empty() || tmp_gemm_popped == true) {
      // Pop the gemm instruction
      if (!tmp_gemm_popped) {
        tmp_gemm_queue.read(tmp_gemv);
        tmp_gemm_popped = true;
      }
      // Check dependences and invoke the load stage
      bool pop_prev_dependence = tmp_gemv[VTA_INSN_MEM_1];
      bool pop_next_dependence = tmp_gemv[VTA_INSN_MEM_2];
      if (
        (pop_prev_dependence && !l2g_dep_queue.empty() &&
         pop_next_dependence && !s2g_dep_queue.empty()) ||
        (!pop_prev_dependence && pop_next_dependence &&
         !s2g_dep_queue.empty()) ||
        (pop_prev_dependence && !l2g_dep_queue.empty() &&
        !pop_next_dependence) ||
        (!pop_prev_dependence && !pop_next_dependence)
      ) {
        // Push the instruction in the load queue
        gemm_queue.write(tmp_gemv);
        tmp_gemm_popped = false;
        compute(done, uops, biases, gemm_queue, l2g_dep_queue, s2g_dep_queue,
                g2l_dep_queue, g2s_dep_queue, inp_mem, wgt_mem, out_mem);
      } else {
        // Execution of load stage pending on completion of other stages,
        // so break here...
        break;
      }
    }
    // Finally execute as many store instructions as possible
    while (!tmp_store_queue.empty() || tmp_store_popped == true) {
      // Pop the load instruction
      if (!tmp_store_popped) {
        tmp_store_queue.read(tmp_store);
        tmp_store_popped = true;
      }
      // Check dependences and invoke the load stage
      bool pop_prev_dependence = tmp_store[VTA_INSN_MEM_1];
      if ((pop_prev_dependence && !g2s_dep_queue.empty()) ||
          !pop_prev_dependence) {
        // Push the instruction in the load queue
        store_queue.write(tmp_store);
        tmp_store_popped = false;
        store(outputs, store_queue, g2s_dep_queue, s2g_dep_queue, out_mem);
      } else {
        // Execution of load stage pending on completion of other stages, so break here...
        break;
      }
    }
    // Check if we get a signal that we are done
    if (done) {
      break;
    }
    exit_counter++;
    if (exit_counter > 1000) {
      if (tmp_load_popped) {
        if (g2l_dep_queue.empty()) {
          printf("waiting on g2l\n");
        }
      }
      if (tmp_gemm_popped) {
        if (l2g_dep_queue.empty() && tmp_gemv[VTA_INSN_MEM_1]) {
          printf("waiting on l2g\n");
        }
        if (s2g_dep_queue.empty() && tmp_gemv[VTA_INSN_MEM_2]) {
          printf("waiting on s2g\n");
        }
      }
      if (tmp_store_popped) {
        if (g2s_dep_queue.empty()) {
          printf("waiting on g2s\n");
        }
      }
      break;
    }
  }

  // Ensure that the tokens are empty
  bool tmp_tok;
  int l2g_count = 0;
  int s2g_count = 0;
  int g2l_count = 0;
  int g2s_count = 0;
  while (l2g_dep_queue.read_nb(tmp_tok)) {
    l2g_count++;
  }
  while (s2g_dep_queue.read_nb(tmp_tok)) {
    s2g_count++;
  }
  while (g2l_dep_queue.read_nb(tmp_tok)) {
    g2l_count++;
  }
  while (g2s_dep_queue.read_nb(tmp_tok)) {
    g2s_count++;
  }

  assert(l2g_count == 0 && g2s_count == 0 && g2l_count == 0 && g2s_count == 0);
}
