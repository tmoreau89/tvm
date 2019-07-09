/*!
 *  Copyright (c) 2018 by Contributors
 * \file vadd.cc
 * \brief Vector Add HLS design.
 */

#include "./vadd.h"

void vadd(
  hls::stream<int> &a,
  hls::stream<int> &b) {
#pragma HLS INTERFACE axis port = a
#pragma HLS INTERFACE axis port = b
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS pipeline II = 1
  b.write(a.read() + 1);
}
