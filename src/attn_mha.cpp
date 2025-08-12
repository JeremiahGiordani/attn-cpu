#include "attn.h"

#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace attn {

//=============================
// AVX-512 helpers
//=============================
#if defined(ATTN_USE_AVX512)
static inline float hsum_ps(__m512 v) { return _mm512_reduce_add_ps(v); }
static inline float hmax_ps(__m512 v) { return _mm512_reduce_max_ps(v); }

// Fast exp(x) for __m512 (sufficient for softmax)
static inline __m512 exp512_ps(__m512 x) {
  const __m512 exp_hi = _mm512_set1_ps(88.3762626647949f);
  const __m512 exp_lo = _mm512_set1_ps(-88.3762626647949f);
  const __m512 log2ef = _mm512_set1_ps(1.44269504088896341f);
  const __m512 c1 = _mm512_set1_ps(0.693359375f);
  const __m512 c2 = _mm512_set1_ps(-2.12194440e-4f);

  const __m512 p0 = _mm512_set1_ps(1.9875691500e-4f);
  const __m512 p1 = _mm512_set1_ps(1.3981999507e-3f);
  const __m512 p2 = _mm512_set1_ps(8.3334519073e-3f);
  const __m512 p3 = _mm512_set1_ps(4.1665795894e-2f);
  const __m512 p4 = _mm512_set1_ps(1.6666665459e-1f);
  const __m512 p5 = _mm512_set1_ps(5.0000001201e-1f);

  x = _mm512_min_ps(x, exp_hi);
  x = _mm512_max_ps(x, exp_lo);

  __m512 fx = _mm512_fmadd_ps(x, log2ef, _mm512_set1_ps(0.5f));
  fx = _mm512_floor_ps(fx);

  __m512 tmp = _mm512_fnmadd_ps(fx, c1, x);
  x = _mm512_fnmadd_ps(fx, c2, tmp);

  __m512 z = _mm512_mul_ps(x, x);
  __m512 y = p0;
  y = _mm512_fmadd_ps(y, x, p1);
  y = _mm512_fmadd_ps(y, x, p2);
  y = _mm512_fmadd_ps(y, x, p3);
  y = _mm512_fmadd_ps(y, x, p4);
  y = _mm512_fmadd_ps(y, x, p5);
  y = _mm512_fmadd_ps(y, z, x);
  y = _mm512_add_ps(y, _mm512_set1_ps(1.0f));

  __m512i emm0 = _mm512_cvttps_epi32(fx);
  emm0 = _mm512_add_epi32(emm0, _mm512_set1_epi32(127));
  emm0 = _mm512_slli_epi32(emm0, 23);
  __m512 pow2n = _mm512_castsi512_ps(emm0);
  return _mm512_mul_ps(y, pow2n);
}
#else
#error "This file expects AVX-512 enabled (ATTN_USE_AVX512)."
#endif

//=============================
// Weight packing for GEMM
// We compute Y[M,N] = X[M,K] * Wt[K,N] + bias[N].
// We pack Wt by blocks of N16=16 columns, K16=16 depth, laid out so that
// for each 16-depth slice we can broadcast X[m,k..k+15] scalars and FMA into 16 output lanes.
//=============================
static constexpr int N16 = 16;
static constexpr int K16 = 16;

// dst size: round_up(N,16) * round_up(K,16)
static inline void pack_wt_col16_k16(const float* Wt, int K, int N, std::vector<float>& packed) {
  const int Np = (N + N16 - 1) / N16 * N16;
  const int Kp = (K + K16 - 1) / K16 * K16;
  packed.assign((size_t)Np * (size_t)Kp, 0.0f);

  for (int n0 = 0; n0 < N; n0 += N16) {
    const int n_take = std::min(N16, N - n0);
    for (int k0 = 0; k0 < K; k0 += K16) {
      const int k_take = std::min(K16, K - k0);
      float* dst = packed.data() + (size_t)n0 * Kp + (size_t)k0 * Np;
      // dst layout for this tile: K16 "planes" of 16 columns (N-major inside)
      for (int kk = 0; kk < k_take; ++kk) {
        float* dst_plane = dst + (size_t)kk * Np;
        for (int nn = 0; nn < n_take; ++nn) {
          dst_plane[nn] = Wt[(k0 + kk) * N + (n0 + nn)];
        }
      }
    }
  }
}

static inline void pack_wt_16x16(const float* Wt, int K, int N, std::vector<float>& packed) {
  // Preconditions for our fast path
  if ((K % 16) != 0 || (N % 16) != 0) {
    throw std::runtime_error("pack_wt_16x16 requires K and N multiples of 16");
  }
  const int nb = N / 16;  // number of 16-col tiles
  const int kb = K / 16;  // number of 16-depth tiles
  packed.resize((size_t)nb * (size_t)kb * 256);

  size_t idx = 0;
  for (int nb_i = 0; nb_i < nb; ++nb_i) {
    const int n0 = nb_i * 16;
    for (int kb_i = 0; kb_i < kb; ++kb_i) {
      const int k0 = kb_i * 16;
      // Copy the 16x16 block: rows over kk, each row 16 contiguous cols
      for (int kk = 0; kk < 16; ++kk) {
        const float* src = Wt + (size_t)(k0 + kk) * N + n0;
        std::memcpy(&packed[idx], src, 16 * sizeof(float));
        idx += 16;
      }
    }
  }
}

// =============================
// GEMM microkernel for a single output row y[0..N)
// X row length K (multiple of 16), N multiple of 16.
// packedWt must be produced by pack_wt_16x16(Wt, K, N).
// =============================
static inline void gemm_row_packed_16x16(const float* x,
                                         const float* packedWt,
                                         const float* bias,
                                         int K, int N,
                                         float* y) {
  const int nb = N / 16;
  const int kb = K / 16;

  for (int nb_i = 0; nb_i < nb; ++nb_i) {
    // acc = bias[nb_i*16 .. nb_i*16+15]
    __m512 acc = _mm512_loadu_ps(bias + nb_i * 16);

    // Base offset for this N-tile (nb_i)
    const size_t base_nb = (size_t)nb_i * (size_t)kb * 256; // kb tiles, each 256 floats

    for (int kb_i = 0; kb_i < kb; ++kb_i) {
      const float* tile = packedWt + base_nb + (size_t)kb_i * 256;
      // Accumulate over 16 depth positions
      // acc += tile[kk*16 : kk*16+16) * x[kb_i*16 + kk]
      const float* xk = x + kb_i * 16;
#pragma unroll(16)
      for (int kk = 0; kk < 16; ++kk) {
        __m512 wv = _mm512_loadu_ps(tile + kk * 16);
        acc = _mm512_fmadd_ps(wv, _mm512_set1_ps(xk[kk]), acc);
      }
    }
    _mm512_storeu_ps(y + nb_i * 16, acc);
  }
}

// Microkernel: compute one row y[0..N) for a given x[0..K) against packed Wt.
// Bias is added at the end.
// N is handled in 16-col tiles; K in 16-depth tiles.
static inline void gemm_row_packed_wt(const float* x, const float* packedWt, const float* bias,
                                      int K, int N, float* y) {
  const int Np = (N + N16 - 1) / N16 * N16;
  const int Kp = (K + K16 - 1) / K16 * K16;

  int n0 = 0;
  for (; n0 + N16 <= N; n0 += N16) {
    __m512 acc = _mm512_setzero_ps(); // 16 outputs in lanes

    for (int k0 = 0; k0 < K; k0 += K16) {
      const float* xk = x + k0;
      const float* wblk = packedWt + (size_t)n0 * Kp + (size_t)k0 * Np;
      // accumulate over kk in [0..K16)
      // For each kk: acc += w[kk, 0..15] * x[k0+kk]
      // w[kk, 0..15] is contiguous in packed layout.
      for (int kk = 0; kk < K16; ++kk) {
        float xv = (k0 + kk < K) ? xk[kk] : 0.0f;
        __m512 wv = _mm512_loadu_ps(wblk + (size_t)kk * Np);
        acc = _mm512_fmadd_ps(wv, _mm512_set1_ps(xv), acc);
      }
    }
    // add bias and store
    __m512 bv = _mm512_loadu_ps(bias + n0);
    acc = _mm512_add_ps(acc, bv);
    _mm512_storeu_ps(y + n0, acc);
  }

  // ragged right
  if (n0 < N) {
    __m512 acc = _mm512_setzero_ps();
    const __mmask16 mask = (__mmask16)((1u << (N - n0)) - 1);
    for (int k0 = 0; k0 < K; k0 += K16) {
      const float* xk = x + k0;
      const float* wblk = packedWt + (size_t)n0 * ((K + K16 - 1) / K16 * K16) + (size_t)k0 * ((N + N16 - 1)/N16 * N16);
      for (int kk = 0; kk < K16; ++kk) {
        float xv = (k0 + kk < K) ? xk[kk] : 0.0f;
        __m512 wv = _mm512_maskz_loadu_ps(mask, wblk + (size_t)kk * ((N + N16 - 1)/N16 * N16));
        acc = _mm512_fmadd_ps(wv, _mm512_set1_ps(xv), acc);
      }
    }
    __m512 bv = _mm512_maskz_loadu_ps(mask, bias + n0);
    acc = _mm512_add_ps(acc, bv);
    _mm512_mask_storeu_ps(y + n0, mask, acc);
  }
}

//=============================
// Main kernel (single-thread)
//=============================
void mha_block_dense(const float* x,   int T, int D,
                     const float* W_in, const float* b_in,   // [3D, D], [3D]
                     const float* W_out, const float* b_out, // [D, D],  [D]
                     int H,
                     bool causal,
                     float* y_out)
{
  const int Dh = D / H;
  const float scale = 1.0f / std::sqrt((float)Dh);

  // ---- Scratch (persistent across calls would be even better) ----
  static thread_local int S_T=-1, S_D=-1, S_H=-1;
  static thread_local std::vector<float> Q;           // [T,D]
  static thread_local std::vector<float> K;           // [T,D]
  static thread_local std::vector<float> V;           // [T,D]
  static thread_local std::vector<float> Qh;          // [H,T,Dh]
  static thread_local std::vector<float> KhT;         // [H,Dh,T]  (transposed keys per head)
  static thread_local std::vector<float> Vh;          // [H,T,Dh]
  static thread_local std::vector<float> Ctx;         // [T,D]
  static thread_local std::vector<float> Win_packed;  // packed (Wt) for 3*D outputs
  static thread_local std::vector<float> Wout_packed; // packed (Wt) for D outputs
  static thread_local int packed_dims = -1;
  static thread_local std::vector<float> row3D_scratch;

  auto ensure = [&](int t, int d, int h) {
    if (S_T==t && S_D==d && S_H==h) return;
    S_T=t; S_D=d; S_H=h;
    Q.assign((size_t)T*(size_t)D, 0.f);
    K.assign((size_t)T*(size_t)D, 0.f);
    V.assign((size_t)T*(size_t)D, 0.f);
    Qh.assign((size_t)H*(size_t)T*(size_t)Dh, 0.f);
    KhT.assign((size_t)H*(size_t)Dh*(size_t)T, 0.f);
    Vh.assign((size_t)H*(size_t)T*(size_t)Dh, 0.f);
    Ctx.assign((size_t)T*(size_t)D, 0.f);
    packed_dims = -1; // force repack
    if ((int)row3D_scratch.size() != 3 * D) row3D_scratch.assign(3 * D, 0.0f);
  };
  ensure(T,D,H);

  // ---- 1) Fused input projection via GEMM: [T,D] x [D,3D] -> [T,3D] ----
  // W_in is [3D,D] row-major; we need Wt = [D,3D]
  // We'll pack Wt for our microkernel (N=3D).
  if (packed_dims != D*3) {
    // Build Wt temporarily (or pack from W_in directly). Simpler: build Wt then pack.
    std::vector<float> Wt((size_t)D * (size_t)(3 * D));
    for (int r = 0; r < D; ++r)
    for (int c = 0; c < 3 * D; ++c)
        Wt[(size_t)r * (3 * D) + c] = W_in[(size_t)c * D + r];
    pack_wt_16x16(Wt.data(), D, 3 * D, Win_packed);

    // Build WoutT = transpose(W_out: [D,D]) -> [D,D]
    std::vector<float> WoutT((size_t)D * (size_t)D);
    for (int r = 0; r < D; ++r)
    for (int c = 0; c < D; ++c)
        WoutT[(size_t)r * D + c] = W_out[(size_t)c * D + r];
    pack_wt_16x16(WoutT.data(), D, D, Wout_packed);
    packed_dims = D*3;
  }

  // Run GEMM row-by-row (single thread), then slice into Q/K/V and add bias.
  for (int t = 0; t < T; ++t) {
    const float* xt = x + (size_t)t*D;
    // temp buffer for [3D]
    float* row3D = row3D_scratch.data();
    gemm_row_packed_16x16(xt, Win_packed.data(), b_in, D, 3 * D, row3D);
    // slice into Q,K,V
    std::memcpy(Q.data() + (size_t)t*D,          row3D + 0*D, D*sizeof(float));
    std::memcpy(K.data() + (size_t)t*D,          row3D + 1*D, D*sizeof(float));
    std::memcpy(V.data() + (size_t)t*D,          row3D + 2*D, D*sizeof(float));
  }

  // ---- 2) Repack by head for tight inner loops ----
  // Qh[h,t,d] = Q[t, h*Dh + d]
  // KhT[h,d,t] = K[t, h*Dh + d]
  // Vh[h,t,d] = V[t, h*Dh + d]
  for (int h = 0; h < H; ++h) {
    for (int t = 0; t < T; ++t) {
      const float* qrow = Q.data() + (size_t)t*D + h*Dh;
      const float* krow = K.data() + (size_t)t*D + h*Dh;
      const float* vrow = V.data() + (size_t)t*D + h*Dh;
      float* qdst = Qh.data() + ((size_t)h*T + t)*Dh;
      float* vdst = Vh.data() + ((size_t)h*T + t)*Dh;
      std::memcpy(qdst, qrow, Dh*sizeof(float));
      std::memcpy(vdst, vrow, Dh*sizeof(float));
      for (int d0 = 0; d0 < Dh; ++d0) {
        KhT.data()[((size_t)h*Dh + d0)*T + t] = krow[d0];
      }
    }
  }

  // ---- 3) Attention per (t,h) with online softmax over key blocks ----
  // Block size in keys = 64 (4 * 16-lane vectors)
  static constexpr int BK = 64;
  const __m512 vscale = _mm512_set1_ps(scale);

  for (int t = 0; t < T; ++t) {
    float* ctx_t = Ctx.data() + (size_t)t*D; // output of attention before out-proj
    std::fill(ctx_t, ctx_t + D, 0.f);

    const int valid_len = causal ? (t + 1) : T;

    for (int h = 0; h < H; ++h) {
      const float* q = Qh.data() + ((size_t)h*T + t)*Dh;
      // Running stats
      float m = -std::numeric_limits<float>::infinity();
      float l = 0.0f;

      // Accumulator for context (Dh)
      const int segs = (Dh + 15) / 16;
      // Keep as registers per segment
      std::vector<__m512> acc(segs, _mm512_setzero_ps());

      for (int j0 = 0; j0 < valid_len; j0 += BK) {
        const int take = std::min(BK, valid_len - j0);
        // Compute logits for this block into four 16-lane vectors s0..s3
        __m512 s0 = _mm512_setzero_ps();
        __m512 s1 = _mm512_setzero_ps();
        __m512 s2 = _mm512_setzero_ps();
        __m512 s3 = _mm512_setzero_ps();

        for (int d0 = 0; d0 < Dh; ++d0) {
          const float qd = q[d0];
          const float* kt_base = KhT.data() + ((size_t)h*Dh + d0)*T + j0;
          __m512 kd0 = _mm512_maskz_loadu_ps((__mmask16)((take>=16)?0xFFFFu:((1u<<take)-1)), kt_base + 0);
          s0 = _mm512_fmadd_ps(kd0, _mm512_set1_ps(qd), s0);
          if (take > 16) {
            __m512 kd1 = _mm512_maskz_loadu_ps((__mmask16)((take>=32)?0xFFFFu:((1u<<(take-16))-1)), kt_base + 16);
            s1 = _mm512_fmadd_ps(kd1, _mm512_set1_ps(qd), s1);
          }
          if (take > 32) {
            __m512 kd2 = _mm512_maskz_loadu_ps((__mmask16)((take>=48)?0xFFFFu:((1u<<(take-32))-1)), kt_base + 32);
            s2 = _mm512_fmadd_ps(kd2, _mm512_set1_ps(qd), s2);
          }
          if (take > 48) {
            __m512 kd3 = _mm512_maskz_loadu_ps((__mmask16)((take>=64)?0xFFFFu:((1u<<(take-48))-1)), kt_base + 48);
            s3 = _mm512_fmadd_ps(kd3, _mm512_set1_ps(qd), s3);
          }
        }

        // scale
        s0 = _mm512_mul_ps(s0, vscale);
        if (take > 16) s1 = _mm512_mul_ps(s1, vscale);
        if (take > 32) s2 = _mm512_mul_ps(s2, vscale);
        if (take > 48) s3 = _mm512_mul_ps(s3, vscale);

        // Online softmax update for (m, l) and acc
        float block_max = hmax_ps(_mm512_maskz_mov_ps((__mmask16)((take>=16)?0xFFFFu:((1u<<take)-1)), s0));
        if (take > 16) block_max = std::max(block_max, hmax_ps(_mm512_maskz_mov_ps((__mmask16)((take>=32)?0xFFFFu:((1u<<(take-16))-1)), s1)));
        if (take > 32) block_max = std::max(block_max, hmax_ps(_mm512_maskz_mov_ps((__mmask16)((take>=48)?0xFFFFu:((1u<<(take-32))-1)), s2)));
        if (take > 48) block_max = std::max(block_max, hmax_ps(_mm512_maskz_mov_ps((__mmask16)((take>=64)?0xFFFFu:((1u<<(take-48))-1)), s3)));

        const float m_new = std::max(m, block_max);
        const float alpha = std::exp(m - m_new);   // scale for previous terms

        // weights in this block
        __m512 w0 = exp512_ps(_mm512_sub_ps(s0, _mm512_set1_ps(m_new)));
        __m512 w1 = (take > 16) ? exp512_ps(_mm512_sub_ps(s1, _mm512_set1_ps(m_new))) : _mm512_setzero_ps();
        __m512 w2 = (take > 32) ? exp512_ps(_mm512_sub_ps(s2, _mm512_set1_ps(m_new))) : _mm512_setzero_ps();
        __m512 w3 = (take > 48) ? exp512_ps(_mm512_sub_ps(s3, _mm512_set1_ps(m_new))) : _mm512_setzero_ps();

        float l_blk = hsum_ps(_mm512_maskz_mov_ps((__mmask16)((take>=16)?0xFFFFu:((1u<<take)-1)), w0));
        if (take > 16) l_blk += hsum_ps(_mm512_maskz_mov_ps((__mmask16)((take>=32)?0xFFFFu:((1u<<(take-16))-1)), w1));
        if (take > 32) l_blk += hsum_ps(_mm512_maskz_mov_ps((__mmask16)((take>=48)?0xFFFFu:((1u<<(take-32))-1)), w2));
        if (take > 48) l_blk += hsum_ps(_mm512_maskz_mov_ps((__mmask16)((take>=64)?0xFFFFu:((1u<<(take-48))-1)), w3));

        // Update acc: acc = acc * alpha + sum_j w_j * V[j]
        for (int s = 0; s < segs; ++s) {
          const int d0 = s * 16;
          const int chunk = std::min(16, Dh - d0);
          __m512 accs = acc[s];
          accs = _mm512_mul_ps(accs, _mm512_set1_ps(alpha));

          auto load_v = [&](int off, __m512 w, __mmask16 mask) {
            for (int jj = 0; jj < off; jj += 16) {
              // no-op; structured to keep pattern similar
            }
            // Iterate keys in this 16-lane group
            const int base = off;
            // Accumulate: accs += sum_l w[l] * V[j0 + base + l, d0:d0+chunk]
            // We can’t multiply vector w by vector v directly; we broadcast each lane of w.
            // But that would be 16 FMAs; good locality though.
            alignas(64) float wtmp[16];
            _mm512_store_ps(wtmp, w);

            for (int l16 = 0; l16 < 16 && base + l16 < take; ++l16) {
              const float wl = wtmp[l16];
              const float* vrow = Vh.data() + ((size_t)h*T + (j0 + base + l16))*Dh + d0;
              __m512 vv = _mm512_maskz_loadu_ps((__mmask16)((chunk>=16)?0xFFFFu:((1u<<chunk)-1)), vrow);
              accs = _mm512_fmadd_ps(vv, _mm512_set1_ps(wl), accs);
            }
          };

          // group 0..15
          {
            const __mmask16 m0 = (__mmask16)((take>=16)?0xFFFFu:((1u<<take)-1));
            load_v(0, w0, m0);
          }
          if (take > 16) {
            const __mmask16 m1 = (__mmask16)((take>=32)?0xFFFFu:((1u<<(take-16))-1));
            load_v(16, w1, m1);
          }
          if (take > 32) {
            const __mmask16 m2 = (__mmask16)((take>=48)?0xFFFFu:((1u<<(take-32))-1));
            load_v(32, w2, m2);
          }
          if (take > 48) {
            const __mmask16 m3 = (__mmask16)((take>=64)?0xFFFFu:((1u<<(take-48))-1));
            load_v(48, w3, m3);
          }

          acc[s] = accs;
        } // segs

        // update running m, l
        l = l * alpha + l_blk;
        m = m_new;
      } // key blocks

      // Normalize and write to ctx_t segment
      const float inv_l = 1.0f / l;
      for (int s = 0; s < segs; ++s) {
        const int d0 = s * 16;
        const int chunk = std::min(16, Dh - d0);
        __m512 outv = _mm512_mul_ps(acc[s], _mm512_set1_ps(inv_l));
        _mm512_mask_storeu_ps(ctx_t + h*Dh + d0,
                              (__mmask16)((chunk>=16)?0xFFFFu:((1u<<chunk)-1)),
                              outv);
      }
    } // heads
  } // t

  // ---- 4) Output projection via GEMM: [T,D] x [D,D] -> [T,D] ----
  for (int t = 0; t < T; ++t) {
    const float* ct = Ctx.data() + (size_t)t*D;
    float* yt = y_out + (size_t)t*D;
    gemm_row_packed_16x16(ct, Wout_packed.data(), b_out, D, D, yt);
  }
}

} // namespace attn