#include "attn.h"


// attn_mha.cpp
#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace attn {

//=============================
// Helpers: aligned scratch
//=============================
template <typename T, std::size_t Align>
struct AlignedAlloc {
  using value_type = T;
  AlignedAlloc() noexcept {}
  template <class U> AlignedAlloc(const AlignedAlloc<U, Align>&) noexcept {}
  T* allocate(std::size_t n) {
    void* p = nullptr;
#if defined(_MSC_VER)
    p = _aligned_malloc(n * sizeof(T), Align);
    if (!p) throw std::bad_alloc();
#else
    if (posix_memalign(&p, Align, n * sizeof(T))) throw std::bad_alloc();
#endif
    return reinterpret_cast<T*>(p);
  }
  void deallocate(T* p, std::size_t) noexcept {
#if defined(_MSC_VER)
    _aligned_free(p);
#else
    free(p);
#endif
  }
};
template <typename T, typename U, std::size_t A1, std::size_t A2>
inline bool operator==(const AlignedAlloc<T, A1>&, const AlignedAlloc<U, A2>&){ return A1==A2; }
template <typename T, typename U, std::size_t A1, std::size_t A2>
inline bool operator!=(const AlignedAlloc<T, A1>&, const AlignedAlloc<U, A2>&){ return !(A1==A2); }

template <typename T>
using avector = std::vector<T>;

//=============================
// AVX-512 math helpers
//=============================
#if defined(ATTN_USE_AVX512)
static inline float hsum_ps(__m512 v) {
  return _mm512_reduce_add_ps(v);
}
static inline float hmax_ps(__m512 v) {
  return _mm512_reduce_max_ps(v);
}

// Fast exp(x) for __m512 adapted from cephes/sse_mathfun style (good accuracy for softmax)
static inline __m512 exp512_ps(__m512 x) {
  const __m512 exp_hi = _mm512_set1_ps(88.3762626647949f);
  const __m512 exp_lo = _mm512_set1_ps(-88.3762626647949f);
  const __m512 log2ef = _mm512_set1_ps(1.44269504088896341f);  // 1/ln(2)
  const __m512 c1 = _mm512_set1_ps(0.693359375f);              // ln2_hi
  const __m512 c2 = _mm512_set1_ps(-2.12194440e-4f);           // ln2_lo

  const __m512 p0 = _mm512_set1_ps(1.9875691500e-4f);
  const __m512 p1 = _mm512_set1_ps(1.3981999507e-3f);
  const __m512 p2 = _mm512_set1_ps(8.3334519073e-3f);
  const __m512 p3 = _mm512_set1_ps(4.1665795894e-2f);
  const __m512 p4 = _mm512_set1_ps(1.6666665459e-1f);
  const __m512 p5 = _mm512_set1_ps(5.0000001201e-1f);

  // clamp
  x = _mm512_min_ps(x, exp_hi);
  x = _mm512_max_ps(x, exp_lo);

  // range reduction
  __m512 fx = _mm512_fmadd_ps(x, log2ef, _mm512_set1_ps(0.5f));
  fx = _mm512_floor_ps(fx);

  __m512 tmp = _mm512_fnmadd_ps(fx, c1, x);
  x = _mm512_fnmadd_ps(fx, c2, tmp);

  // polynomial
  __m512 z = _mm512_mul_ps(x, x);
  __m512 y = p0;
  y = _mm512_fmadd_ps(y, x, p1);
  y = _mm512_fmadd_ps(y, x, p2);
  y = _mm512_fmadd_ps(y, x, p3);
  y = _mm512_fmadd_ps(y, x, p4);
  y = _mm512_fmadd_ps(y, x, p5);
  y = _mm512_fmadd_ps(y, z, x);
  y = _mm512_add_ps(y, _mm512_set1_ps(1.0f));

  // build 2^fx
  __m512i emm0 = _mm512_cvttps_epi32(fx);
  emm0 = _mm512_add_epi32(emm0, _mm512_set1_epi32(127));
  emm0 = _mm512_slli_epi32(emm0, 23);
  __m512 pow2n = _mm512_castsi512_ps(emm0);

  return _mm512_mul_ps(y, pow2n);
}

static inline float dot_16(const float* a, const float* b) {
  __m512 va = _mm512_loadu_ps(a);
  __m512 vb = _mm512_loadu_ps(b);
  return hsum_ps(_mm512_mul_ps(va, vb));
}

#else
// Scalar fallbacks (compile everywhere)
static inline float hsum_ps(float v){ return v; }
static inline float hmax_ps(float v){ return v; }
static inline float dot_16(const float* a, const float* b) {
  float s=0.f; for (int i=0;i<16;i++) s += a[i]*b[i]; return s;
}
#endif

//=============================
// 4-row GEMV micro-kernel (row-major W: [N_out, K], x: [K])
//=============================
static inline void gemv4_rowmajor_f32(const float* W, const float* x, const float* b,
                                      int N_out, int K, float* y_out) {
#if defined(ATTN_USE_AVX512)
  int n=0;
  for (; n+3 < N_out; n += 4) {
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();

    for (int k=0; k<K; k+=16) {
      __m512 xv = _mm512_loadu_ps(&x[k]);
      __m512 w0 = _mm512_loadu_ps(&W[(n+0)*K + k]);
      __m512 w1 = _mm512_loadu_ps(&W[(n+1)*K + k]);
      __m512 w2 = _mm512_loadu_ps(&W[(n+2)*K + k]);
      __m512 w3 = _mm512_loadu_ps(&W[(n+3)*K + k]);
      acc0 = _mm512_fmadd_ps(w0, xv, acc0);
      acc1 = _mm512_fmadd_ps(w1, xv, acc1);
      acc2 = _mm512_fmadd_ps(w2, xv, acc2);
      acc3 = _mm512_fmadd_ps(w3, xv, acc3);
    }
    y_out[n+0] = hsum_ps(acc0) + b[n+0];
    y_out[n+1] = hsum_ps(acc1) + b[n+1];
    y_out[n+2] = hsum_ps(acc2) + b[n+2];
    y_out[n+3] = hsum_ps(acc3) + b[n+3];
  }
  for (; n < N_out; ++n) {
    __m512 acc = _mm512_setzero_ps();
    for (int k=0; k<K; k+=16) {
      __m512 xv = _mm512_loadu_ps(&x[k]);
      __m512 w  = _mm512_loadu_ps(&W[n*K + k]);
      acc = _mm512_fmadd_ps(w, xv, acc);
    }
    y_out[n] = hsum_ps(acc) + b[n];
  }
#else
  for (int n=0;n<N_out;++n) {
    float s=0.f;
    for (int k=0;k<K;++k) s += W[n*K + k]*x[k];
    y_out[n] = s + b[n];
  }
#endif
}

//=============================
// Main kernel
//=============================
void mha_block_dense(const float* x,   int T, int D,
                     const float* W_in, const float* b_in,   // [3D, D], [3D]
                     const float* W_out, const float* b_out, // [D, D],  [D]
                     int H,
                     bool causal,
                     float* y_out) {

  const int Dh = D / H;
  const float scale = 1.0f / std::sqrt((float)Dh);
  const float NEG_INF = -std::numeric_limits<float>::infinity();

  // thread-local scratch to avoid re-allocs across benchmark iterations
  struct Scratch {
    int T=-1, D=-1, H=-1;
    avector<float> Q;        // [T, D]
    avector<float> K;        // [T, D] (temporary, before transpose)
    avector<float> V;        // [T, D]
    avector<float> KT;       // [H, Dh, T] => index ((h*Dh + d)*T + t)
    avector<float> scores;   // [T] per row
    avector<float> ctx;      // [D] temp per row (after softmax*V concat heads)
  };
  static thread_local Scratch S;

  auto ensure = [&](int _T,int _D,int _H){
    if (S.T==_T && S.D==_D && S.H==_H) return;
    S.T=_T; S.D=_D; S.H=_H;
    S.Q .assign((size_t)_T*(size_t)_D, 0.f);
    S.K .assign((size_t)_T*(size_t)_D, 0.f);
    S.V .assign((size_t)_T*(size_t)_D, 0.f);
    S.KT.assign((size_t)_H*(size_t)Dh*(size_t)_T, 0.f);
    S.scores.assign((size_t)_T, 0.f);
    S.ctx.assign((size_t)_D, 0.f);
  };
  ensure(T,D,H);

  const float* Wq = W_in + 0*D*D;
  const float* Wk = W_in + 1*D*D;
  const float* Wv = W_in + 2*D*D;
  const float* bq = b_in + 0*D;
  const float* bk = b_in + 1*D;
  const float* bv = b_in + 2*D;

  // 1) Q,K,V = X * W^T + b  (GEMV per token, 4-row micro-kernel)
  for (int t = 0; t < T; ++t) {
    const float* xt = x + (size_t)t * D;
    float* Qt = S.Q.data() + (size_t)t * D;
    float* Kt = S.K.data() + (size_t)t * D;
    float* Vt = S.V.data() + (size_t)t * D;

    gemv4_rowmajor_f32(Wq, xt, bq, D, D, Qt);
    gemv4_rowmajor_f32(Wk, xt, bk, D, D, Kt);
    gemv4_rowmajor_f32(Wv, xt, bv, D, D, Vt);
  }

  // 2) Transpose K per head to [H, Dh, T] for block dot-products
  //    KT[(h*Dh + d)*T + t] = K[t, h*Dh + d]
  for (int h = 0; h < H; ++h) {
    for (int d = 0; d < Dh; ++d) {
      float* KT_hd = S.KT.data() + (size_t)(h*Dh + d) * T;
      for (int t = 0; t < T; ++t) {
        KT_hd[t] = S.K[(size_t)t*D + h*Dh + d];
      }
    }
  }

  // 3) For each t/head: compute softmax(QK^T/sqrt(Dh) [+ mask]) * V, write into ctx row
  const int blocks = (T + 15) / 16;
  avector<float> tmp_wblk(16); // reused small buffer

#if defined(ATTN_USE_AVX512)
  const __m512 vscl = _mm512_set1_ps(scale);
#endif

  for (int t = 0; t < T; ++t) {
    float* ctx_t = S.ctx.data(); // length D; we will fill per-head segments
    std::fill(ctx_t, ctx_t + D, 0.f);

    const int valid_len = causal ? (t + 1) : T;

    for (int h = 0; h < H; ++h) {
      const float* q_hd = S.Q.data() + (size_t)t*D + h*Dh;

      // Pass 1: compute scores and row_max
      float row_max = -std::numeric_limits<float>::infinity();

      for (int b = 0; b < blocks; ++b) {
        const int j0 = b * 16;
        const int take = std::min(16, valid_len - j0);
        if (take <= 0) break;

#if defined(ATTN_USE_AVX512)
        __m512 s = _mm512_setzero_ps();
        // s = sum_d q[d] * K_T[d, j0..j0+15]
        for (int d = 0; d < Dh; ++d) {
          __m512 qd = _mm512_set1_ps(q_hd[d]);
          const float* KT_row = S.KT.data() + (size_t)(h*Dh + d)*T + j0;
          __m512 kd = _mm512_maskz_loadu_ps((__mmask16)((1u<<take)-1), KT_row);
          s = _mm512_fmadd_ps(qd, kd, s);
        }
        s = _mm512_mul_ps(s, vscl);

        // store valid lanes to scores buffer and update row_max
        _mm512_mask_storeu_ps(S.scores.data() + j0, (__mmask16)((1u<<take)-1), s);
        float local_max = hmax_ps(_mm512_maskz_mov_ps((__mmask16)((1u<<take)-1), s));
        row_max = std::max(row_max, local_max);
#else
        // Scalar fallback
        for (int jj = 0; jj < take; ++jj) {
          float sc = 0.f;
          for (int d = 0; d < Dh; ++d) {
            sc += q_hd[d] * S.KT[(size_t)(h*Dh + d)*T + (j0 + jj)];
          }
          sc *= scale;
          S.scores[j0 + jj] = sc;
          row_max = std::max(row_max, sc);
        }
#endif
      }

      // Pass 2: compute weights (exp(scores - max)), denom, and accumulate context
      float denom = 0.f;

      // Prepare per-16-dim accumulators across Dh in 16-chunks
      const int segs = (Dh + 15) / 16;
      // We keep up to small fixed-size array of __m512; allocate on stack safely via vector
      avector<__m512> acc_vecs(segs);
      for (int s = 0; s < segs; ++s) {
#if defined(ATTN_USE_AVX512)
        acc_vecs[s] = _mm512_setzero_ps();
#else
        // not used in scalar path
#endif
      }

      for (int b = 0; b < blocks; ++b) {
        const int j0 = b * 16;
        const int take = std::min(16, valid_len - j0);
        if (take <= 0) break;

#if defined(ATTN_USE_AVX512)
        __m512 sv = _mm512_maskz_loadu_ps((__mmask16)((1u<<take)-1), S.scores.data() + j0);
        sv = _mm512_sub_ps(sv, _mm512_set1_ps(row_max));
        __m512 wv = exp512_ps(sv);
        // write weights to stack buffer for reuse across Dh segments
        _mm512_storeu_ps(tmp_wblk.data(), wv);
        denom += hsum_ps(_mm512_maskz_mov_ps((__mmask16)((1u<<take)-1), wv));
#else
        for (int jj=0; jj<take; ++jj) {
          tmp_wblk[jj] = std::exp(S.scores[j0+jj] - row_max);
          denom += tmp_wblk[jj];
        }
#endif

        // Accumulate acc_vecs[s] += sum_{l=0..take-1} w[l] * V[(j0+l), head, seg]
        for (int s = 0; s < segs; ++s) {
          const int d0 = s * 16;
          const int chunk = std::min(16, Dh - d0);

#if defined(ATTN_USE_AVX512)
          __m512 accs = acc_vecs[s];
          for (int l = 0; l < take; ++l) {
            __m512 w = _mm512_set1_ps(tmp_wblk[l]);
            const float* vrow = S.V.data() + (size_t)(j0 + l)*D + h*Dh + d0;
            __m512 vv = _mm512_maskz_loadu_ps((__mmask16)((chunk>=16)?0xFFFFu:((1u<<chunk)-1)), vrow);
            accs = _mm512_fmadd_ps(w, vv, accs);
          }
          acc_vecs[s] = accs;
#else
          // scalar (rarely used)
          for (int l = 0; l < take; ++l) {
            const float w = tmp_wblk[l];
            const float* vrow = S.V.data() + (size_t)(j0 + l)*D + h*Dh + d0;
            for (int dd=0; dd<chunk; ++dd) {
              reinterpret_cast<float*>(&acc_vecs[s])[dd] += w * vrow[dd];
            }
          }
#endif
        }
      } // blocks

      const float inv_denom = 1.f / denom;

      // write normalized context back to ctx row
      for (int s = 0; s < segs; ++s) {
        const int d0 = s * 16;
        const int chunk = std::min(16, Dh - d0);
#if defined(ATTN_USE_AVX512)
        __m512 outv = _mm512_mul_ps(acc_vecs[s], _mm512_set1_ps(inv_denom));
        _mm512_mask_storeu_ps(ctx_t + h*Dh + d0,
                              (__mmask16)((chunk>=16)?0xFFFFu:((1u<<chunk)-1)),
                              outv);
#else
        for (int dd=0; dd<chunk; ++dd) {
          ctx_t[h*Dh + d0 + dd] = (reinterpret_cast<float*>(&acc_vecs[s])[dd]) * inv_denom;
        }
#endif
      }
    } // heads

    // 4) Output projection: y[t] = ctx_t * W_out^T + b_out
    //    W_out: [D_out=D, K=D]
    gemv4_rowmajor_f32(W_out, ctx_t, b_out, D, D, y_out + (size_t)t*D);
  } // t
}

} // namespace attn
