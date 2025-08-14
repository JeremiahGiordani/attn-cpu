#pragma once
#include "common.h"
#include <cstring>
#include <vector>

namespace attn {

// Dh=16 specialization (fast path)
ATTN_ALWAYS_INLINE void scatter_qkv_headed_dh16(
    const float* row_q, const float* row_k, const float* row_v,
    int T, int D, int H, int t_idx,
    std::vector<float>& Qh, std::vector<float>& KhT, std::vector<float>& Vh) {
  const int Dh = 16;
  for (int h=0; h<H; ++h) {
    const float* qsrc = row_q + h*Dh;
    const float* ksrc = row_k + h*Dh;
    const float* vsrc = row_v + h*Dh;
    std::memcpy(Qh.data() + ((size_t)h*T + t_idx)*Dh, qsrc, Dh*sizeof(float));
    std::memcpy(Vh.data() + ((size_t)h*T + t_idx)*Dh, vsrc, Dh*sizeof(float));
#pragma unroll(16)
    for (int d0=0; d0<Dh; ++d0) {
      KhT.data()[((size_t)h*Dh + d0)*T + t_idx] = ksrc[d0];
    }
  }
}

// Generic Dh (handles any Dh, slower)
ATTN_ALWAYS_INLINE void scatter_qkv_headed_generic(
    const float* row_q, const float* row_k, const float* row_v,
    int T, int D, int H, int Dh, int t_idx,
    std::vector<float>& Qh, std::vector<float>& KhT, std::vector<float>& Vh) {
  for (int h=0; h<H; ++h) {
    const float* qsrc = row_q + h*Dh;
    const float* ksrc = row_k + h*Dh;
    const float* vsrc = row_v + h*Dh;
    std::memcpy(Qh.data() + ((size_t)h*T + t_idx)*Dh, qsrc, Dh*sizeof(float));
    std::memcpy(Vh.data() + ((size_t)h*T + t_idx)*Dh, vsrc, Dh*sizeof(float));
    for (int d0=0; d0<Dh; ++d0) {
      KhT.data()[((size_t)h*Dh + d0)*T + t_idx] = ksrc[d0];
    }
  }
}

} // namespace attn
