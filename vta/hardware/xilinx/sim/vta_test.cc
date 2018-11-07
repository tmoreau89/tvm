/*!
 *  Copyright (c) 2018 by Contributors
 * \file vta_test.cpp
 * \brief Simulation tests for the VTA design.
 */

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "../src/vta.h"
#include "../../../tests/hardware/common/test_lib.h"

int main(void) {
#if DEBUG == 1
    printParameters();
#endif

    int status = 0;

#ifdef ALU_EN
    // Run ALU test (vector-scalar operators)
    status |= alu_test(VTA_ALU_OPCODE_MIN, true, VTA_BLOCK_OUT, 128, true);
    status |= alu_test(VTA_ALU_OPCODE_MIN, true, VTA_BLOCK_OUT, 128, false);
    status |= alu_test(VTA_ALU_OPCODE_MAX, true, VTA_BLOCK_OUT, 128, true);
    status |= alu_test(VTA_ALU_OPCODE_MAX, true, VTA_BLOCK_OUT, 128, false);
    status |= alu_test(VTA_ALU_OPCODE_ADD, true, VTA_BLOCK_OUT, 128, true);
    status |= alu_test(VTA_ALU_OPCODE_ADD, true, VTA_BLOCK_OUT, 128, false);
    status |= alu_test(VTA_ALU_OPCODE_SHR, true, VTA_BLOCK_OUT, 128, true);
    status |= alu_test(VTA_ALU_OPCODE_SHR, true, VTA_BLOCK_OUT, 128, false);

    // Run ALU test (vector-vector operators)
    status |= alu_test(VTA_ALU_OPCODE_MIN, false, VTA_BLOCK_OUT, 128, true);
    status |= alu_test(VTA_ALU_OPCODE_MIN, false, VTA_BLOCK_OUT, 128, false);
    status |= alu_test(VTA_ALU_OPCODE_MAX, false, VTA_BLOCK_OUT, 128, true);
    status |= alu_test(VTA_ALU_OPCODE_MAX, false, VTA_BLOCK_OUT, 128, false);
    status |= alu_test(VTA_ALU_OPCODE_ADD, false, VTA_BLOCK_OUT, 128, true);
    status |= alu_test(VTA_ALU_OPCODE_ADD, false, VTA_BLOCK_OUT, 128, false);
    status |= alu_test(VTA_ALU_OPCODE_SHR, false, VTA_BLOCK_OUT, 128, true);
    status |= alu_test(VTA_ALU_OPCODE_SHR, false, VTA_BLOCK_OUT, 128, false);

#ifdef MUL_EN
    status |= alu_test(VTA_ALU_OPCODE_MUL, true, VTA_BLOCK_OUT, 128, true);
    status |= alu_test(VTA_ALU_OPCODE_MUL, true, VTA_BLOCK_OUT, 128, false);
    status |= alu_test(VTA_ALU_OPCODE_MUL, false, VTA_BLOCK_OUT, 128, true);
    status |= alu_test(VTA_ALU_OPCODE_MUL, false, VTA_BLOCK_OUT, 128, false);
#endif // MUL_EN

#endif // ALU_EN

    // Run blocked GEMM test
    // status |= blocked_gemm_test(256, 256, VTA_BLOCK_OUT*4, true, 2);
    status |= blocked_gemm_test(256, 256, VTA_BLOCK_OUT*4, false, 2);
    // status |= blocked_gemm_test(256, 256, VTA_BLOCK_OUT*4, true, 1);
    status |= blocked_gemm_test(256, 256, VTA_BLOCK_OUT*4, false, 1);

    // Simple GEMM unit test
    status |= gemm_test(4 * VTA_BATCH, 4 * VTA_BLOCK_OUT, 4 * VTA_BLOCK_IN, false);

    return status;
}
