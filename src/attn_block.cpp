#include "attn.h"
#include <algorithm>
#include <cmath>
#include <vector>
#include <cstring>
#if defined(ATTN_USE_AVX512)
  #include <immintrin.h>
#endif

namespace attn {

static inline const float* q_ptr(const TensorF32& Q, int h, int l) {
  return Q.data + h*Q.strideH + l*Q.strideT;
}
static inline const float* k_ptr(const TensorF32& K, int h, int t) {
  return K.data + h*K.strideH + t*K.strideT;
}
static inline const float* v_ptr(const TensorF32& V, int h, int t) {
  return V.data + h*V.strideH + t*V.strideT;
}
static inline float* o_ptr(const MutableTensor3F32& O, int h, int l) {
  return O.data + h*O.strideH + l*O.strideT;
}

#if defined(ATTN_USE_AVX512)
static inline float hsum512(__m512 v) {
#if defined(__AVX512F__) && defined(__INTEL_COMPILER)
  return _mm512_reduce_add_ps(v);
#else
  __m256 lo = _mm512_castps512_ps256(v);
  __m256 hi = _mm512_extractf32x8_ps(v, 1);
  __m256 sum256 = _mm256_add_ps(lo, hi);
  __m128 lo128 = _mm256_castps256_ps128(sum256);
  __m128 hi128 = _mm256_extractf128_ps(sum256, 1);
  __m128 sum128 = _mm_add_ps(lo128, hi128);
  sum128 = _mm_hadd_ps(sum128, sum128);
  sum128 = _mm_hadd_ps(sum128, sum128);
  return _mm_cvtss_f32(sum128);
#endif
}

static inline float dot_avx512(const float* __restrict a,
                               const float* __restrict b,
                               int Dh) {
  __m512 acc = _mm512_setzero_ps();
  int d = 0;
  for (; d + 16 <= Dh; d += 16) {
    __m512 av = _mm512_loadu_ps(a + d);
    __m512 bv = _mm512_loadu_ps(b + d);
    acc = _mm512_fmadd_ps(av, bv, acc);
  }
  if (d < Dh) {
    const int rem = Dh - d;
    const __mmask16 m = (1u << rem) - 1u;
    __m512 av = _mm512_maskz_loadu_ps(m, a + d);
    __m512 bv = _mm512_maskz_loadu_ps(m, b + d);
    acc = _mm512_fmadd_ps(av, bv, acc);
  }
  return hsum512(acc);
}

static inline void axpy_avx512(float* __restrict y,
                               const float* __restrict x,
                               float alpha,
                               int Dh) {
  __m512 a = _mm512_set1_ps(alpha);
  int d = 0;
  for (; d + 16 <= Dh; d += 16) {
    __m512 yv = _mm512_loadu_ps(y + d);
    __m512 xv = _mm512_loadu_ps(x + d);
    yv = _mm512_fmadd_ps(a, xv, yv);
    _mm512_storeu_ps(y + d, yv);
  }
  if (d < Dh) {
    const int rem = Dh - d;
    const __mmask16 m = (1u << rem) - 1u;
    __m512 yv = _mm512_maskz_loadu_ps(m, y + d);
    __m512 xv = _mm512_maskz_loadu_ps(m, x + d);
    yv = _mm512_fmadd_ps(a, xv, yv);
    _mm512_mask_storeu_ps(y + d, m, yv);
  }
}

static inline void scale_avx512(float* __restrict y,
                                float alpha,
                                int Dh) {
  __m512 a = _mm512_set1_ps(alpha);
  int d = 0;
  for (; d + 16 <= Dh; d += 16) {
    __m512 yv = _mm512_loadu_ps(y + d);
    yv = _mm512_mul_ps(yv, a);
    _mm512_storeu_ps(y + d, yv);
  }
  if (d < Dh) {
    const int rem = Dh - d;
    const __mmask16 m = (1u << rem) - 1u;
    __m512 yv = _mm512_maskz_loadu_ps(m, y + d);
    yv = _mm512_mul_ps(yv, a);
    _mm512_mask_storeu_ps(y + d, m, yv);
  }
}
#endif // ATTN_USE_AVX512

void attn_block_dense(const TensorF32& Q,
                      const TensorF32& K,
                      const TensorF32& V,
                      MutableTensor3F32& O,
                      bool causal) {
  if (!Q.data || !K.data || !V.data || !O.data)
    throw std::invalid_argument("Null data pointer");
  if (Q.H<=0 || Q.Dh<=0 || K.H!=Q.H || V.H!=Q.H)
    throw std::invalid_argument("H mismatch");
  if (K.Dh!=Q.Dh || V.Dh!=Q.Dh)
    throw std::invalid_argument("Dh mismatch");
  if (K.T<=0 || V.T!=K.T)
    throw std::invalid_argument("K/V T mismatch");
  if (O.H!=Q.H || O.Dh!=Q.Dh || O.T!=Q.T)
    throw std::invalid_argument("O must be [H,L,Dh] matching Q");
  if (causal && Q.T!=K.T)
    throw std::invalid_argument("causal=true requires L==T");

  const int H = Q.H;
  const int Dh= Q.Dh;
  const int L = Q.T;
  const int T = K.T;
  const float inv_sqrt_dh = 1.0f / std::sqrt((float)Dh);

  const bool contiguous_d =
      (Q.strideDh == 1 && K.strideDh == 1 && V.strideDh == 1 && O.strideDh == 1);

  // Parallel region with per-thread scratch buffer
  #if defined(ATTN_USE_OPENMP)
  #pragma omp parallel
  #endif
  {
    std::vector<float> logits_local(T);

    #if defined(ATTN_USE_OPENMP)
    #pragma omp for schedule(static)
    #endif
    for (int hl = 0; hl < H * L; ++hl) {
      const int h = hl / L;
      const int l = hl % L;

      const float* q = q_ptr(Q, h, l);
      float* out = o_ptr(O, h, l);

      // Pass 1: logits + max
      float m = -INFINITY;
      for (int t = 0; t < T; ++t) {
        if (causal && t > l) { logits_local[t] = -INFINITY; continue; }
        const float* krow = k_ptr(K, h, t);
        float dot = 0.f;
        if (contiguous_d) {
        #if defined(ATTN_USE_AVX512)
          __m512 acc = _mm512_setzero_ps();
          int d=0; for (; d+16<=Dh; d+=16) {
            __m512 av=_mm512_loadu_ps(q+d);
            __m512 bv=_mm512_loadu_ps(krow+d);
            acc=_mm512_fmadd_ps(av,bv,acc);
          }
          if (d<Dh){
            __mmask16 msk=(1u<<(Dh-d))-1u;
            __m512 av=_mm512_maskz_loadu_ps(msk,q+d);
            __m512 bv=_mm512_maskz_loadu_ps(msk,krow+d);
            acc=_mm512_fmadd_ps(av,bv,acc);
          }
          alignas(64) float tmp[16]; _mm512_store_ps(tmp,acc);
          #pragma unroll
          for(int i=0;i<16;++i) dot += tmp[i];
        #else
          for (int d=0; d<Dh; ++d) dot += q[d] * krow[d];
        #endif
        } else {
          for (int d=0; d<Dh; ++d) dot += q[d*Q.strideDh] * krow[d*K.strideDh];
        }

        float lgt = dot * inv_sqrt_dh;
        logits_local[t] = lgt;
        if (lgt > m) m = lgt;
      }

      // Pass 2: softmax normalize and mix V
      float denom = 0.f;
      // zero out
      if (contiguous_d) std::memset(out, 0, sizeof(float)*Dh);
      else for (int d=0; d<Dh; ++d) out[d*O.strideDh] = 0.f;

      for (int t = 0; t < T; ++t) {
        if (causal && t > l) continue;
        float w = std::exp(logits_local[t] - m);
        denom += w;
        const float* vrow = v_ptr(V, h, t);
        if (contiguous_d) {
        #if defined(ATTN_USE_AVX512)
          __m512 a = _mm512_set1_ps(w);
          int d=0; for (; d+16<=Dh; d+=16) {
            __m512 yv=_mm512_loadu_ps(out+d);
            __m512 xv=_mm512_loadu_ps(vrow+d);
            yv=_mm512_fmadd_ps(a,xv,yv);
            _mm512_storeu_ps(out+d,yv);
          }
          if (d<Dh){
            __mmask16 msk=(1u<<(Dh-d))-1u;
            __m512 yv=_mm512_maskz_loadu_ps(msk,out+d);
            __m512 xv=_mm512_maskz_loadu_ps(msk,vrow+d);
            yv=_mm512_fmadd_ps(a,xv,yv);
            _mm512_mask_storeu_ps(out+d,msk,yv);
          }
        #else
          for (int d=0; d<Dh; ++d) out[d] += w * vrow[d];
        #endif
        } else {
          for (int d=0; d<Dh; ++d) out[d*O.strideDh] += w * vrow[d*V.strideDh];
        }
      }

      const float inv_denom = 1.0f / denom;
      if (contiguous_d) {
      #if defined(ATTN_USE_AVX512)
        __m512 s=_mm512_set1_ps(inv_denom);
        int d=0; for (; d+16<=Dh; d+=16) {
          __m512 ov=_mm512_loadu_ps(out+d);
          ov=_mm512_mul_ps(ov,s);
          _mm512_storeu_ps(out+d,ov);
        }
        if (d<Dh){
          __mmask16 msk=(1u<<(Dh-d))-1u;
          __m512 ov=_mm512_maskz_loadu_ps(msk,out+d);
          ov=_mm512_mul_ps(ov,s);
          _mm512_mask_storeu_ps(out+d,msk,ov);
        }
      #else
        for (int d=0; d<Dh; ++d) out[d] *= inv_denom;
      #endif
      } else {
        for (int d=0; d<Dh; ++d) out[d*O.strideDh] *= inv_denom;
      }
    }
  } // omp parallel
}

} // namespace attn
