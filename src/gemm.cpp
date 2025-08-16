// sgemm_blocked.cpp — AVX-512 + OpenMP SGEMM (hi-impact updates)
// Changes vs. previous:
//  - Unmasked aligned B loads in core kernel (masked only in tails).
//  - Memory-form A broadcasts (_mm512_broadcastss_ps + _mm_load_ss).
//  - k-unroll = 8 with interleaved loads/FMAs (simple software pipeline).
//  - Keep NT stores only on the final k-panel; otherwise regular stores.
//  - Use pointer walking in the k loop to reduce AGU pressure.
//
// NOTE: Cross-kpanel C-scratch accumulation (avoid touching C between KC panels)
//       is not included here; it needs loop restructuring. This version keeps
//       your existing blocking and β_this logic.

#include <immintrin.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <omp.h>

namespace gemm {

// ===================== Tunables (adjust per CPU) =====================
static constexpr int MR = 8;     // rows per micro-kernel (broadcast scalar)
static constexpr int NR = 48;    // cols per micro-kernel (3× zmm lanes)
static constexpr int KC = 512;   // K blocking (fits L2 with A-pack)
static constexpr int MC = 256;   // M blocking (per-thread A-pack ~ MC*KC*4 bytes)
static constexpr int NC = 768;   // N blocking (B-pack ~ KC*NC*4 bytes)
static constexpr int K_UNROLL = 8; // increased unroll for better ILP
// Prefetching is toned down; rely mostly on HW prefetchers.
// You can experiment with light T1 for A if it helps your CPU.

// ===================== Aligned allocation helpers ====================
static inline float* aligned_alloc64(size_t n_floats) {
#if defined(_MSC_VER)
  return static_cast<float*>(_aligned_malloc(n_floats * sizeof(float), 64));
#else
  void* p = nullptr;
  if (posix_memalign(&p, 64, n_floats * sizeof(float)) != 0) return nullptr;
  return static_cast<float*>(p);
#endif
}
static inline void aligned_free64(float* p) {
#if defined(_MSC_VER)
  _aligned_free(p);
#else
  free(p);
#endif
}

// ===================== Packing routines ==============================
// A-pack (MR×kc) per micro-tile, k-major: Ap[k*MR + r] = A[r, k]
static inline void pack_A_mr_kc(const float* __restrict A, int ldA,
                                int mr_eff, int kc,
                                float* __restrict Ap) {
  for (int k = 0; k < kc; ++k) {
    const float* a_col = A + k;     // row-major: step by ldA down rows
    float* dst = Ap + (size_t)k * MR;
    int r = 0;
    for (; r < mr_eff; ++r) dst[r] = a_col[(size_t)r * ldA];
    for (; r < MR;     ++r) dst[r] = 0.0f;
  }
}

// Pack MC×kc block of A as consecutive MR-panels
static inline void pack_A_mc_kc(const float* __restrict A, int ldA,
                                int mc_eff, int kc,
                                float* __restrict Ap) {
  int tiles = (mc_eff + MR - 1) / MR;
  for (int t = 0; t < tiles; ++t) {
    int r0 = t * MR;
    int mr_eff = std::min(MR, mc_eff - r0);
    const float* A_src = A + (size_t)r0 * ldA;
    float* Ap_t = Ap + (size_t)t * kc * MR;
    pack_A_mr_kc(A_src, ldA, mr_eff, kc, Ap_t);
  }
}

// B-pack (kc×nr) per NR-tile: Bp[k*NR + j] = B[k, j]
static inline void pack_B_kc_nr(const float* __restrict B, int ldB,
                                int kc, int nr_eff,
                                float* __restrict Bp) {
  // Bp is 64B aligned; each 16-float slice is 64B and aligned
  for (int k = 0; k < kc; ++k) {
    const float* b_row = B + (size_t)k * ldB;
    float* dst = Bp + (size_t)k * NR;
    int j = 0;
    for (; j < nr_eff; ++j) dst[j] = b_row[j];
    for (; j < NR;     ++j) dst[j] = 0.0f; // pad tail
  }
}

// Pack kc×nc panel of B as NR tiles
static inline void pack_B_kc_nc(const float* __restrict B, int ldB,
                                int kc, int nc,
                                float* __restrict Bp) {
  int ntiles = (nc + NR - 1) / NR;
  for (int nt = 0; nt < ntiles; ++nt) {
    int j0 = nt * NR;
    int nr_eff = std::min(NR, nc - j0);
    const float* B_src = B + j0;             // row 0, col j0
    float* Bp_tile = Bp + (size_t)nt * kc * NR;
    pack_B_kc_nr(B_src, ldB, kc, nr_eff, Bp_tile);
  }
}

// ===================== 8×48 AVX-512 micro-kernel =====================
//  - Unmasked aligned loads for B (3 loads per k-step): _mm512_load_ps
//  - Memory-form broadcasts for A: _mm512_broadcastss_ps(_mm_load_ss(...))
//  - k-unroll = 8 with simple double-buffering of B vectors
//  - Stores honor beta==0/1 fast paths; NT only on final panel
static inline void microkernel_8x48_core_u8(
    const float* __restrict Ap,      // packed A: kc×MR (stride MR per k)
    const float* __restrict Bp,      // packed B: kc×NR (stride NR per k)
    float* __restrict C, int ldc,    // C tile top-left
    int kc,                          // depth for THIS k-panel
    float alpha, float beta,         // scaling
    bool final_kpanel)               // if true and beta==0 => NT stores
{
  __m512 acc0[MR], acc1[MR], acc2[MR];
#pragma unroll
  for (int r = 0; r < MR; ++r) {
    acc0[r] = _mm512_setzero_ps();
    acc1[r] = _mm512_setzero_ps();
    acc2[r] = _mm512_setzero_ps();
  }

  const float* a_ptr = Ap;
  const float* b_ptr = Bp;

  // Process by 8
  int k = 0;
  const int kend = kc & ~7;

  for (; k < kend; k += 8) {
    // Preload B vectors for 8 consecutive k’s (double-buffer-ish)
    const float* b0 = b_ptr + (size_t)0 * NR;
    const float* b1 = b_ptr + (size_t)1 * NR;
    const float* b2 = b_ptr + (size_t)2 * NR;
    const float* b3 = b_ptr + (size_t)3 * NR;
    const float* b4 = b_ptr + (size_t)4 * NR;
    const float* b5 = b_ptr + (size_t)5 * NR;
    const float* b6 = b_ptr + (size_t)6 * NR;
    const float* b7 = b_ptr + (size_t)7 * NR;

    __m512 b0_0 = _mm512_load_ps(b0 +  0);
    __m512 b0_1 = _mm512_load_ps(b0 + 16);
    __m512 b0_2 = _mm512_load_ps(b0 + 32);

    __m512 b1_0 = _mm512_load_ps(b1 +  0);
    __m512 b1_1 = _mm512_load_ps(b1 + 16);
    __m512 b1_2 = _mm512_load_ps(b1 + 32);

    __m512 b2_0 = _mm512_load_ps(b2 +  0);
    __m512 b2_1 = _mm512_load_ps(b2 + 16);
    __m512 b2_2 = _mm512_load_ps(b2 + 32);

    __m512 b3_0 = _mm512_load_ps(b3 +  0);
    __m512 b3_1 = _mm512_load_ps(b3 + 16);
    __m512 b3_2 = _mm512_load_ps(b3 + 32);

    __m512 b4_0 = _mm512_load_ps(b4 +  0);
    __m512 b4_1 = _mm512_load_ps(b4 + 16);
    __m512 b4_2 = _mm512_load_ps(b4 + 32);

    __m512 b5_0 = _mm512_load_ps(b5 +  0);
    __m512 b5_1 = _mm512_load_ps(b5 + 16);
    __m512 b5_2 = _mm512_load_ps(b5 + 32);

    __m512 b6_0 = _mm512_load_ps(b6 +  0);
    __m512 b6_1 = _mm512_load_ps(b6 + 16);
    __m512 b6_2 = _mm512_load_ps(b6 + 32);

    __m512 b7_0 = _mm512_load_ps(b7 +  0);
    __m512 b7_1 = _mm512_load_ps(b7 + 16);
    __m512 b7_2 = _mm512_load_ps(b7 + 32);

    // Issue 8*MR FMAs using memory-form broadcasts for A
#pragma unroll
    for (int r = 0; r < MR; ++r) {
      const float* a0 = a_ptr + (size_t)0 * MR + r;
      const float* a1 = a_ptr + (size_t)1 * MR + r;
      const float* a2 = a_ptr + (size_t)2 * MR + r;
      const float* a3 = a_ptr + (size_t)3 * MR + r;
      const float* a4 = a_ptr + (size_t)4 * MR + r;
      const float* a5 = a_ptr + (size_t)5 * MR + r;
      const float* a6 = a_ptr + (size_t)6 * MR + r;
      const float* a7 = a_ptr + (size_t)7 * MR + r;

      __m512 A0 = _mm512_broadcastss_ps(_mm_load_ss(a0));
      __m512 A1 = _mm512_broadcastss_ps(_mm_load_ss(a1));
      __m512 A2 = _mm512_broadcastss_ps(_mm_load_ss(a2));
      __m512 A3 = _mm512_broadcastss_ps(_mm_load_ss(a3));
      __m512 A4 = _mm512_broadcastss_ps(_mm_load_ss(a4));
      __m512 A5 = _mm512_broadcastss_ps(_mm_load_ss(a5));
      __m512 A6 = _mm512_broadcastss_ps(_mm_load_ss(a6));
      __m512 A7 = _mm512_broadcastss_ps(_mm_load_ss(a7));

      acc0[r] = _mm512_fmadd_ps(A0, b0_0, acc0[r]);
      acc1[r] = _mm512_fmadd_ps(A0, b0_1, acc1[r]);
      acc2[r] = _mm512_fmadd_ps(A0, b0_2, acc2[r]);

      acc0[r] = _mm512_fmadd_ps(A1, b1_0, acc0[r]);
      acc1[r] = _mm512_fmadd_ps(A1, b1_1, acc1[r]);
      acc2[r] = _mm512_fmadd_ps(A1, b1_2, acc2[r]);

      acc0[r] = _mm512_fmadd_ps(A2, b2_0, acc0[r]);
      acc1[r] = _mm512_fmadd_ps(A2, b2_1, acc1[r]);
      acc2[r] = _mm512_fmadd_ps(A2, b2_2, acc2[r]);

      acc0[r] = _mm512_fmadd_ps(A3, b3_0, acc0[r]);
      acc1[r] = _mm512_fmadd_ps(A3, b3_1, acc1[r]);
      acc2[r] = _mm512_fmadd_ps(A3, b3_2, acc2[r]);

      acc0[r] = _mm512_fmadd_ps(A4, b4_0, acc0[r]);
      acc1[r] = _mm512_fmadd_ps(A4, b4_1, acc1[r]);
      acc2[r] = _mm512_fmadd_ps(A4, b4_2, acc2[r]);

      acc0[r] = _mm512_fmadd_ps(A5, b5_0, acc0[r]);
      acc1[r] = _mm512_fmadd_ps(A5, b5_1, acc1[r]);
      acc2[r] = _mm512_fmadd_ps(A5, b5_2, acc2[r]);

      acc0[r] = _mm512_fmadd_ps(A6, b6_0, acc0[r]);
      acc1[r] = _mm512_fmadd_ps(A6, b6_1, acc1[r]);
      acc2[r] = _mm512_fmadd_ps(A6, b6_2, acc2[r]);

      acc0[r] = _mm512_fmadd_ps(A7, b7_0, acc0[r]);
      acc1[r] = _mm512_fmadd_ps(A7, b7_1, acc1[r]);
      acc2[r] = _mm512_fmadd_ps(A7, b7_2, acc2[r]);
    }

    a_ptr += (size_t)8 * MR;
    b_ptr += (size_t)8 * NR;
  }

  // Remainder k (0..7)
  for (; k < kc; ++k) {
    const float* b0 = b_ptr + (size_t)0 * NR;
    __m512 B0 = _mm512_load_ps(b0 +  0);
    __m512 B1 = _mm512_load_ps(b0 + 16);
    __m512 B2 = _mm512_load_ps(b0 + 32);
#pragma unroll
    for (int r = 0; r < MR; ++r) {
      const float* a0 = a_ptr + r;
      __m512 A0 = _mm512_broadcastss_ps(_mm_load_ss(a0));
      acc0[r] = _mm512_fmadd_ps(A0, B0, acc0[r]);
      acc1[r] = _mm512_fmadd_ps(A0, B1, acc1[r]);
      acc2[r] = _mm512_fmadd_ps(A0, B2, acc2[r]);
    }
    a_ptr += MR;
    b_ptr += NR;
  }

  // Optional alpha scaling
  if (alpha != 1.0f) {
    __m512 va = _mm512_set1_ps(alpha);
#pragma unroll
    for (int r = 0; r < MR; ++r) {
      acc0[r] = _mm512_mul_ps(acc0[r], va);
      acc1[r] = _mm512_mul_ps(acc1[r], va);
      acc2[r] = _mm512_mul_ps(acc2[r], va);
    }
  }

  // Stores (beta==0 fast path; beta==1 fast path; general beta)
  if (beta == 0.0f) {
#pragma unroll
    for (int r = 0; r < MR; ++r) {
      float* cptr = C + (size_t)r * ldc;
      if (final_kpanel) {
        _mm512_stream_ps(cptr +  0, acc0[r]);
        _mm512_stream_ps(cptr + 16, acc1[r]);
        _mm512_stream_ps(cptr + 32, acc2[r]);
      } else {
        _mm512_storeu_ps(cptr +  0, acc0[r]);
        _mm512_storeu_ps(cptr + 16, acc1[r]);
        _mm512_storeu_ps(cptr + 32, acc2[r]);
      }
    }
  } else if (beta == 1.0f) {
#pragma unroll
    for (int r = 0; r < MR; ++r) {
      float* cptr = C + (size_t)r * ldc;
      __m512 c0 = _mm512_loadu_ps(cptr +  0);
      __m512 c1 = _mm512_loadu_ps(cptr + 16);
      __m512 c2 = _mm512_loadu_ps(cptr + 32);
      c0 = _mm512_add_ps(c0, acc0[r]);
      c1 = _mm512_add_ps(c1, acc1[r]);
      c2 = _mm512_add_ps(c2, acc2[r]);
      _mm512_storeu_ps(cptr +  0, c0);
      _mm512_storeu_ps(cptr + 16, c1);
      _mm512_storeu_ps(cptr + 32, c2);
    }
  } else {
    __m512 vb = _mm512_set1_ps(beta);
#pragma unroll
    for (int r = 0; r < MR; ++r) {
      float* cptr = C + (size_t)r * ldc;
      __m512 c0 = _mm512_loadu_ps(cptr +  0);
      __m512 c1 = _mm512_loadu_ps(cptr + 16);
      __m512 c2 = _mm512_loadu_ps(cptr + 32);
      c0 = _mm512_fmadd_ps(c0, vb, acc0[r]);
      c1 = _mm512_fmadd_ps(c1, vb, acc1[r]);
      c2 = _mm512_fmadd_ps(c2, vb, acc2[r]);
      _mm512_storeu_ps(cptr +  0, c0);
      _mm512_storeu_ps(cptr + 16, c1);
      _mm512_storeu_ps(cptr + 32, c2);
    }
  }
}

// Tail-capable wrapper for edges (rare in your sizes). Keep simple & correct.
static inline void microkernel_8x48_tail(
    const float* __restrict Ap,
    const float* __restrict Bp,
    float* __restrict C, int ldc,
    int kc, int mr_eff, int nr_eff,
    float alpha, float beta,
    bool final_kpanel)
{
  // Fallback scalar (rare) — correctness first.
  for (int r = 0; r < mr_eff; ++r) {
    for (int j = 0; j < nr_eff; ++j) {
      float acc = 0.0f;
      for (int kk = 0; kk < kc; ++kk) {
        acc += Ap[(size_t)kk*MR + r] * Bp[(size_t)kk*NR + j];
      }
      float* cptr = C + (size_t)r * ldc + j;
      float out = (alpha==1.0f ? acc : alpha*acc) + beta * (*cptr);
      *cptr = out;
    }
  }
}

// ====================== Top-level SGEMM ==============================
void sgemm_blocked(const float* A, int M, int K,
                   const float* B, int N,
                   float* C,
                   float alpha, float beta)
{
  if (M <= 0 || N <= 0 || K <= 0) return;

  const int ldA = K;
  const int ldB = N;
  const int ldC = N;

  // alpha==0 -> C = beta*C
  if (alpha == 0.0f) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < M; ++i) {
      float* Crow = C + (size_t)i * ldC;
      if (beta == 0.0f) {
        std::memset(Crow, 0, sizeof(float) * (size_t)N);
      } else if (beta != 1.0f) {
        for (int j = 0; j < N; ++j) Crow[j] *= beta;
      }
    }
    return;
  }

  // Outer N blocking
  for (int n0 = 0; n0 < N; n0 += NC) {
    int nc = std::min(NC, N - n0);
    int ntiles = (nc + NR - 1) / NR;

    // K blocking
    for (int k0 = 0; k0 < K; k0 += KC) {
      int kc = std::min(KC, K - k0);

      // Pack B panel once (shared)
      const size_t Bp_elems = (size_t)kc * (size_t)ntiles * (size_t)NR;
      float* Bp = aligned_alloc64(Bp_elems);
      pack_B_kc_nc(B + (size_t)k0 * ldB + n0, ldB, kc, nc, Bp);

      const bool final_kpanel = (k0 + kc == K);
      const float beta_this = (k0 == 0) ? beta : 1.0f;

#pragma omp parallel
      {
        // Thread-private A pack buffer
        float* Ap = aligned_alloc64((size_t)MC * (size_t)kc);

#pragma omp for schedule(static)
        for (int m0 = 0; m0 < M; m0 += MC) {
          int mc = std::min(MC, M - m0);
          int a_tiles = (mc + MR - 1) / MR;

          // Pack A block once for all NR tiles
          pack_A_mc_kc(A + (size_t)m0 * ldA + k0, ldA, mc, kc, Ap);

          // Sweep NR tiles
          for (int nt = 0; nt < ntiles; ++nt) {
            int j0 = n0 + nt * NR;
            int nr_eff = std::min(NR, N - j0);
            const float* Bp_tile = Bp + (size_t)nt * kc * NR;

            // Iterate MR tiles
            for (int t = 0; t < a_tiles; ++t) {
              int r0 = t * MR;
              int mr_eff = std::min(MR, mc - r0);
              const float* Ap_tile = Ap + (size_t)t * kc * MR;
              float* C_tile = C + (size_t)(m0 + r0) * ldC + j0;

              if (mr_eff == MR && nr_eff == NR) {
                microkernel_8x48_core_u8(Ap_tile, Bp_tile, C_tile, ldC,
                                         kc, alpha, beta_this, final_kpanel);
              } else {
                microkernel_8x48_tail(Ap_tile, Bp_tile, C_tile, ldC,
                                      kc, mr_eff, nr_eff,
                                      alpha, beta_this, final_kpanel);
              }
            }
          }
        }

        aligned_free64(Ap);
      } // omp parallel

      aligned_free64(Bp);
    } // k0
  } // n0
}

} // namespace gemm
