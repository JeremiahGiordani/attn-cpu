#pragma once
#include "common.h"
#include <immintrin.h>

namespace attn {

ATTN_ALWAYS_INLINE void gemm_row1_packed_16x16(const float* x,
                                               const float* packedWt,
                                               const float* bias,
                                               int K, int N, float* y) {
  const int nb = N / 16, kb = K / 16;
  for (int nb_i=0; nb_i<nb; ++nb_i) {
    __m512 acc = _mm512_loadu_ps(bias + nb_i*16);
    const size_t base_nb = (size_t)nb_i * kb * 256;
    for (int kb_i=0; kb_i<kb; ++kb_i) {
      const float* tile = packedWt + base_nb + (size_t)kb_i * 256;
      const float* xk   = x + kb_i*16;
#pragma unroll(16)
      for (int kk=0; kk<16; ++kk) {
        __m512 wv = _mm512_loadu_ps(tile + kk*16);
        acc = _mm512_fmadd_ps(wv, _mm512_set1_ps(xk[kk]), acc);
      }
    }
    _mm512_storeu_ps(y + nb_i*16, acc);
  }
}

ATTN_ALWAYS_INLINE void gemm_row4_packed_16x16(const float* x0,const float* x1,
                                               const float* x2,const float* x3,
                                               const float* packedWt,
                                               const float* bias,
                                               int K, int N,
                                               float* y0,float* y1,
                                               float* y2,float* y3) {
  const int nb = N / 16, kb = K / 16;
  for (int nb_i=0; nb_i<nb; ++nb_i) {
    __m512 acc0 = _mm512_loadu_ps(bias + nb_i*16);
    __m512 acc1 = acc0, acc2 = acc0, acc3 = acc0;
    const size_t base_nb = (size_t)nb_i * kb * 256;
    for (int kb_i=0; kb_i<kb; ++kb_i) {
      const float* tile = packedWt + base_nb + (size_t)kb_i * 256;
      const float* xk0  = x0 + kb_i*16;
      const float* xk1  = x1 + kb_i*16;
      const float* xk2  = x2 + kb_i*16;
      const float* xk3  = x3 + kb_i*16;
#pragma unroll(16)
      for (int kk=0; kk<16; ++kk) {
        __m512 wv = _mm512_loadu_ps(tile + kk*16);
        acc0 = _mm512_fmadd_ps(wv, _mm512_set1_ps(xk0[kk]), acc0);
        acc1 = _mm512_fmadd_ps(wv, _mm512_set1_ps(xk1[kk]), acc1);
        acc2 = _mm512_fmadd_ps(wv, _mm512_set1_ps(xk2[kk]), acc2);
        acc3 = _mm512_fmadd_ps(wv, _mm512_set1_ps(xk3[kk]), acc3);
      }
    }
    _mm512_storeu_ps(y0 + nb_i*16, acc0);
    _mm512_storeu_ps(y1 + nb_i*16, acc1);
    _mm512_storeu_ps(y2 + nb_i*16, acc2);
    _mm512_storeu_ps(y3 + nb_i*16, acc3);
  }
}

} // namespace attn
