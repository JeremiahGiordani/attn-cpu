// sgemm_blocked.cpp — AVX-512 SGEMM (global prepack + column-parallel “L2-locked B tiles”)
#include <immintrin.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <omp.h>

namespace gemm {

// ---------------- Tunables (picked for AVX-512 & 2048x1280x960) ---------------
static constexpr int MR = 8;      // rows per micro-kernel
static constexpr int NR = 48;     // cols per micro-kernel (3× zmm)
static constexpr int K_UNROLL = 4;
static constexpr int PREFETCH_DIST = 64;

// ---------------- Aligned alloc helpers ---------------------------------------
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
static inline bool aligned64(const void* p) {
  return ((reinterpret_cast<uintptr_t>(p) & 63u) == 0u);
}

// ---------------- Packing: global A (M×K) and B (K×N) --------------------------
// A packed per MR-row tile across full K; alpha fused.
static inline void pack_A_tile_mrK(const float* __restrict A, int ldA,
                                   int mr_eff, int K, float alpha,
                                   float* __restrict Ap) {
  for (int k = 0; k < K; ++k) {
    const float* a_col = A + k;
    float* dst = Ap + (size_t)k * MR;
    int r = 0;
    for (; r < mr_eff; ++r) dst[r] = a_col[(size_t)r * ldA] * alpha;
    for (; r < MR;     ++r) dst[r] = 0.0f;
  }
}

// B packed per NR-col tile across full K.
static inline void pack_B_tile_Knr(const float* __restrict B, int ldB,
                                   int K, int nr_eff,
                                   float* __restrict Bp) {
  // We assume Bp base is 64B aligned; each step is NR floats (NR*4B = 192B), still 64B aligned.
  for (int k = 0; k < K; ++k) {
    const float* b_row = B + (size_t)k * ldB;
    float* dst = Bp + (size_t)k * NR;
    int j = 0;
    for (; j < nr_eff; ++j) dst[j] = b_row[j];
    for (; j < NR;     ++j) dst[j] = 0.0f;
  }
}

// ---------------- 8×48 AVX-512 micro-kernel (beta==0 hot path) ----------------
// Full-tile kernel only (mr_eff==MR && nr_eff==NR). Edges handled elsewhere.
// A already has alpha fused; we compute over the full K in one sweep,
// then write C once (streaming if aligned).
static inline void microkernel_8x48_beta0(
    const float* __restrict Ap,      // packed A tile (K×MR, stride MR per k)
    const float* __restrict Bp,      // packed B tile (K×NR, stride NR per k)
    float* __restrict C, int ldc,    // C tile top-left
    int K,
    bool stream_if_aligned)
{
  __m512 acc0[MR], acc1[MR], acc2[MR];
#pragma unroll
  for (int r = 0; r < MR; ++r) {
    acc0[r] = _mm512_setzero_ps();
    acc1[r] = _mm512_setzero_ps();
    acc2[r] = _mm512_setzero_ps();
  }

  int k = 0;
  const int kend = K & ~(K_UNROLL - 1);

  for (; k < kend; k += K_UNROLL) {
    if (k + PREFETCH_DIST < K) {
      _mm_prefetch((const char*)(Ap + (size_t)(k + PREFETCH_DIST) * MR), _MM_HINT_T0);
      _mm_prefetch((const char*)(Bp + (size_t)(k + PREFETCH_DIST) * NR), _MM_HINT_T0);
    }
#pragma unroll
    for (int u = 0; u < K_UNROLL; ++u) {
      const float* a = Ap + (size_t)(k + u) * MR;
      const float* b = Bp + (size_t)(k + u) * NR;
      // Bp is 64B aligned; use aligned loads for better throughput.
      __m512 b0 = _mm512_load_ps(b +  0);
      __m512 b1 = _mm512_load_ps(b + 16);
      __m512 b2 = _mm512_load_ps(b + 32);
#pragma unroll
      for (int r = 0; r < MR; ++r) {
        __m512 ar = _mm512_set1_ps(a[r]);
        acc0[r] = _mm512_fmadd_ps(ar, b0, acc0[r]);
        acc1[r] = _mm512_fmadd_ps(ar, b1, acc1[r]);
        acc2[r] = _mm512_fmadd_ps(ar, b2, acc2[r]);
      }
    }
  }

  for (; k < K; ++k) {
    const float* a = Ap + (size_t)k * MR;
    const float* b = Bp + (size_t)k * NR;
    __m512 b0 = _mm512_load_ps(b +  0);
    __m512 b1 = _mm512_load_ps(b + 16);
    __m512 b2 = _mm512_load_ps(b + 32);
#pragma unroll
    for (int r = 0; r < MR; ++r) {
      __m512 ar = _mm512_set1_ps(a[r]);
      acc0[r] = _mm512_fmadd_ps(ar, b0, acc0[r]);
      acc1[r] = _mm512_fmadd_ps(ar, b1, acc1[r]);
      acc2[r] = _mm512_fmadd_ps(ar, b2, acc2[r]);
    }
  }

  // Final write: beta==0 → no RMW. Stream if 64B aligned.
#pragma unroll
  for (int r = 0; r < MR; ++r) {
    float* cptr = C + (size_t)r * ldc;
    if (stream_if_aligned && aligned64(cptr +  0)) _mm512_stream_ps(cptr +  0, acc0[r]);
    else                                           _mm512_storeu_ps (cptr +  0, acc0[r]);
    if (stream_if_aligned && aligned64(cptr + 16)) _mm512_stream_ps(cptr + 16, acc1[r]);
    else                                           _mm512_storeu_ps (cptr + 16, acc1[r]);
    if (stream_if_aligned && aligned64(cptr + 32)) _mm512_stream_ps(cptr + 32, acc2[r]);
    else                                           _mm512_storeu_ps (cptr + 32, acc2[r]);
  }
}

// Scalar edge handler (rare on your dims)
static inline void microkernel_tail_scalar(
    const float* __restrict Ap, const float* __restrict Bp,
    float* __restrict C, int ldc,
    int K, int mr_eff, int nr_eff, float beta)
{
  for (int r = 0; r < mr_eff; ++r) {
    for (int j = 0; j < nr_eff; ++j) {
      float acc = 0.f;
      for (int kk = 0; kk < K; ++kk) acc += Ap[(size_t)kk*MR + r] * Bp[(size_t)kk*NR + j];
      float* cptr = C + (size_t)r * ldc + j;
      if (beta == 0.0f) *cptr = acc;
      else if (beta == 1.0f) *cptr += acc;
      else *cptr = acc + beta * (*cptr);
    }
  }
}

// ------------------------------ Top-level -------------------------------------
void sgemm_blocked(const float* A, int M, int K,
                   const float* B, int N,
                   float* C,
                   float alpha, float beta)
{
  if (M <= 0 || N <= 0 || K <= 0) return;

  const int ldA = K, ldB = N, ldC = N;

  // alpha==0: C = beta*C
  if (alpha == 0.0f) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < M; ++i) {
      float* Crow = C + (size_t)i * ldC;
      if (beta == 0.0f) std::memset(Crow, 0, sizeof(float) * N);
      else if (beta != 1.0f) for (int j = 0; j < N; ++j) Crow[j] *= beta;
    }
    return;
  }

  // Tile counts
  const int MT = (M + MR - 1) / MR;
  const int NT = (N + NR - 1) / NR;

  // Global packs (64B aligned):
  float* A_pack = aligned_alloc64((size_t)MT * K * MR);
  float* B_pack = aligned_alloc64((size_t)NT * K * NR);

  // Pack A with alpha fused (parallel over row tiles)
#pragma omp parallel for schedule(static)
  for (int tm = 0; tm < MT; ++tm) {
    int r0 = tm * MR;
    int mr_eff = std::min(MR, M - r0);
    const float* A_src = A + (size_t)r0 * ldA;
    float* Ap = A_pack + (size_t)tm * K * MR;
    pack_A_tile_mrK(A_src, ldA, mr_eff, K, alpha, Ap);
  }

  // Pack B (parallel over column tiles). Ensure each tile base is 64B aligned in B_pack.
#pragma omp parallel for schedule(static)
  for (int tn = 0; tn < NT; ++tn) {
    int j0 = tn * NR;
    int nr_eff = std::min(NR, N - j0);
    const float* B_src = B + j0;
    float* Bp = B_pack + (size_t)tn * K * NR;
    pack_B_tile_Knr(B_src, ldB, K, nr_eff, Bp);
  }

  // ====== Column-parallel sweep: keep each B tile hot in the core’s L2 ======
  if (beta == 0.0f) {
    // Fast path: no RMW on C and streaming stores allowed.
#pragma omp parallel for schedule(static)
    for (int tn = 0; tn < NT; ++tn) {
      int j0 = tn * NR;
      const float* Bp = B_pack + (size_t)tn * K * NR;

      // Walk ALL A tiles with this B tile resident
      for (int tm = 0; tm < MT; ++tm) {
        int r0 = tm * MR;
        int mr_eff = std::min(MR, M - r0);
        int nr_eff = std::min(NR, N - j0);
        float* C_tile = C + (size_t)r0 * ldC + j0;
        const float* Ap = A_pack + (size_t)tm * K * MR;

        if (mr_eff == MR && nr_eff == NR) {
          // C is row-major; with N=960, ldC*4B == 3840B (64B multiple) so many rows are 64-aligned.
          bool stream_ok =
              aligned64(C_tile + 0) && aligned64(C_tile + 16) && aligned64(C_tile + 32);
          microkernel_8x48_beta0(Ap, Bp, C_tile, ldC, K, stream_ok);
        } else {
          microkernel_tail_scalar(Ap, Bp, C_tile, ldC, K, mr_eff, nr_eff, /*beta*/0.0f);
        }
      }
    }
  } else {
    // General path (rare in your benchmark) — keep correctness; a bit slower.
#pragma omp parallel for schedule(static)
    for (int tn = 0; tn < NT; ++tn) {
      int j0 = tn * NR;
      const float* Bp = B_pack + (size_t)tn * K * NR;

      for (int tm = 0; tm < MT; ++tm) {
        int r0 = tm * MR;
        int mr_eff = std::min(MR, M - r0);
        int nr_eff = std::min(NR, N - j0);
        float* C_tile = C + (size_t)r0 * ldC + j0;
        const float* Ap = A_pack + (size_t)tm * K * MR;

        if (mr_eff == MR && nr_eff == NR) {
          // Fold beta manually (load-add-store). Keep simple here.
          __m512 acc0[MR], acc1[MR], acc2[MR];
          for (int r = 0; r < MR; ++r) {
            acc0[r] = _mm512_setzero_ps();
            acc1[r] = _mm512_setzero_ps();
            acc2[r] = _mm512_setzero_ps();
          }
          int k = 0, kend = K & ~(K_UNROLL - 1);
          for (; k < kend; k += K_UNROLL) {
            if (k + PREFETCH_DIST < K) {
              _mm_prefetch((const char*)(Ap + (size_t)(k + PREFETCH_DIST) * MR), _MM_HINT_T0);
              _mm_prefetch((const char*)(Bp + (size_t)(k + PREFETCH_DIST) * NR), _MM_HINT_T0);
            }
            for (int u = 0; u < K_UNROLL; ++u) {
              const float* a = Ap + (size_t)(k + u) * MR;
              const float* b = Bp + (size_t)(k + u) * NR;
              __m512 b0 = _mm512_load_ps(b +  0);
              __m512 b1 = _mm512_load_ps(b + 16);
              __m512 b2 = _mm512_load_ps(b + 32);
              for (int r = 0; r < MR; ++r) {
                __m512 ar = _mm512_set1_ps(a[r]);
                acc0[r] = _mm512_fmadd_ps(ar, b0, acc0[r]);
                acc1[r] = _mm512_fmadd_ps(ar, b1, acc1[r]);
                acc2[r] = _mm512_fmadd_ps(ar, b2, acc2[r]);
              }
            }
          }
          for (; k < K; ++k) {
            const float* a = Ap + (size_t)k * MR;
            const float* b = Bp + (size_t)k * NR;
            __m512 b0 = _mm512_load_ps(b +  0);
            __m512 b1 = _mm512_load_ps(b + 16);
            __m512 b2 = _mm512_load_ps(b + 32);
            for (int r = 0; r < MR; ++r) {
              __m512 ar = _mm512_set1_ps(a[r]);
              acc0[r] = _mm512_fmadd_ps(ar, b0, acc0[r]);
              acc1[r] = _mm512_fmadd_ps(ar, b1, acc1[r]);
              acc2[r] = _mm512_fmadd_ps(ar, b2, acc2[r]);
            }
          }
          __m512 vb = _mm512_set1_ps(beta);
          for (int r = 0; r < MR; ++r) {
            float* cptr = C_tile + (size_t)r * ldC;
            __m512 c0 = _mm512_loadu_ps(cptr +  0);
            __m512 c1 = _mm512_loadu_ps(cptr + 16);
            __m512 c2 = _mm512_loadu_ps(cptr + 32);
            _mm512_storeu_ps(cptr +  0, _mm512_fmadd_ps(c0, vb, acc0[r]));
            _mm512_storeu_ps(cptr + 16, _mm512_fmadd_ps(c1, vb, acc1[r]));
            _mm512_storeu_ps(cptr + 32, _mm512_fmadd_ps(c2, vb, acc2[r]));
          }
        } else {
          microkernel_tail_scalar(Ap, Bp, C_tile, ldC, K, mr_eff, nr_eff, beta);
        }
      }
    }
  }

  aligned_free64(B_pack);
  aligned_free64(A_pack);
}

} // namespace gemm
