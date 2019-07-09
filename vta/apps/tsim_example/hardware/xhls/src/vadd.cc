/*!
 *  Copyright (c) 2018 by Contributors
 * \file vadd.cc
 * \brief Vector Add HLS design.
 */

#include "./vadd.h"

void vadd(
  int len,
  hls::stream<int> &a,
  hls::stream<int> &b) {
#pragma HLS INTERFACE s_axilite port = len bundle = CONTROL_BUS
#pragma HLS INTERFACE axis port = a
#pragma HLS INTERFACE axis port = b

  for (int i = 0; i < len; i++) {
    b.write(a.read() + 1);
  }
}
