#pragma once
#include "common.h"
namespace attn {
void mha_block_dense(const float* x, int T, int D,
                     const float* W_in, const float* b_in,
                     const float* W_out, const float* b_out,
                     int H, bool causal, float* y_out);
}
