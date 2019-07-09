/*!
 *  Copyright (c) 2018 by Contributors
 * \file vadd.h
 * \brief Vector Add HLS design.
 */

#ifndef XHLS_VADD_H_
#define XHLS_VADD_H_

#include <hls_stream.h>

void vadd(
  int len,
  hls::stream<int> &a,
  hls::stream<int> &b);

#endif  // XHLS_VADD_H_