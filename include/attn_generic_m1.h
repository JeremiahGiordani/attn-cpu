#pragma once
#include "common.h"
#include "simd_math.h"

namespace attn::kernels {

ATTN_ALWAYS_INLINE void run_generic_m1_online(
  const std::vector<float>& Qh, const std::vector<float>& KhT, const std::vector<float>& Vh,
  int T, int H, int D, int Dh, bool causal, std::vector<float>& Ctx) {

  const float scale = 1.0f / std::sqrt((float)Dh);
  for (int t=0; t<T; ++t) {
    const int valid_len = causal ? (t+1) : T;
    float* ctx_t = Ctx.data() + (size_t)t * (size_t)(H*Dh);
    for (int h=0; h<H; ++h) {
      const float* q = Qh.data() + ((size_t)h*T + t) * Dh;
      float m = -std::numeric_limits<float>::infinity();
      float l = 0.f;
      std::vector<float> acc(Dh, 0.f);
      for (int j=0; j<valid_len; ++j) {
        // dot(q, K[j])
        const float* kj = &KhT[((size_t)h*Dh + 0)*T + j];
        float s=0.f;
        for (int d=0; d<Dh; ++d) s += q[d] * kj[(size_t)d*T];
        s *= scale;
        const float m_new = std::max(m, s);
        const float a = std::exp(m - m_new);
        const float w = std::exp(s - m_new);
        for (int d=0; d<Dh; ++d) acc[d] = acc[d]*a + w * Vh[((size_t)h*T + j)*Dh + d];
        l = l*a + w;
        m = m_new;
      }
      const float inv = 1.0f / l;
      for (int d=0; d<Dh; ++d) ctx_t[h*Dh + d] = acc[d] * inv;
    }
  }
}

} // namespace attn::kernels
