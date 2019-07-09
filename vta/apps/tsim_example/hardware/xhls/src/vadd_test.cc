/*!
 *  Copyright (c) 2018 by Contributors
 * \file vadd_test.cpp
 * \brief Simulation tests for the vector add design.
 */

#include <stdio.h>
#include <stdlib.h>

#include "vadd.h"

#define VECTOR_LEN 1024

#define PRAGMA_SUB(x) _Pragma (#x)
#define PRAGMA_HLS(x) PRAGMA_SUB(x)
#define STREAM_IN_DEPTH (VECTOR_LEN)

unsigned globalSeed;


int main(void) {

  // Test outcome
  bool correct = true;

  // Input and output FIFOs
  hls::stream<int> a_q;
PRAGMA_HLS(HLS stream depth=VECTOR_LEN variable=a_q)
  hls::stream<int> b_q;
PRAGMA_HLS(HLS stream depth=VECTOR_LEN variable=b_q)

  // Input and output array initialization
  int *a = (int *) malloc(sizeof(int) * VECTOR_LEN);
  int *b = (int *) malloc(sizeof(int) * VECTOR_LEN);
  for (int i = 0; i < VECTOR_LEN; i++) {
    a[i] = rand_r(&globalSeed) % 1024 - 512;
    a_q.write(a[i]);
  }

  // Invoke the vector add module
  vadd(VECTOR_LEN, a_q, b_q);

  // Check the outputc
  for (int i = 0; i < VECTOR_LEN; i++) {
    b[i] = b_q.read();
    if (b[i] != a[i] + 1) {
      correct = false;
    }
  }

  // Free arrays
  free(a);
  free(b);

  if (correct) {
    printf("Test successful\n");
    return 0;
  } else {
    printf("Test unsuccessful\n");
    return -1;
  }
}
