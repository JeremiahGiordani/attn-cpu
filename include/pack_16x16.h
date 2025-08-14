#pragma once
#include "common.h"
#include <vector>
#include <cstring>
#include <stdexcept>

namespace attn {
// Pack Wt[K,N] (row-major) as a sequence of 16x16 tiles (K,N multiples of 16)
ATTN_ALWAYS_INLINE void pack_wt_16x16(const float* Wt, int K, int N, std::vector<float>& packed) {
  if ((K % 16) || (N % 16)) throw std::runtime_error("pack_wt_16x16 expects K,N % 16 == 0");
  const int nb = N / 16, kb = K / 16;
  packed.resize((size_t)nb * kb * 256);
  size_t idx = 0;
  for (int nb_i=0; nb_i<nb; ++nb_i) {
    const int n0 = nb_i * 16;
    for (int kb_i=0; kb_i<kb; ++kb_i) {
      const int k0 = kb_i * 16;
      for (int kk=0; kk<16; ++kk) {
        const float* src = Wt + (size_t)(k0 + kk) * N + n0;
        std::memcpy(&packed[idx], src, 16*sizeof(float));
        idx += 16;
      }
    }
  }
}
} // namespace attn
