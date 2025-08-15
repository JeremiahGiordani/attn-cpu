// src/gemm.cpp
#include "gemm.h"

#include <algorithm>
#include <vector>
#include <cstring>
#include <immintrin.h>

#if defined(_OPENMP)
  #include <omp.h>
#endif

// Tunable prefetch distances (in k elements)
#ifndef GEMM_PREFETCH_DIST_K
#define GEMM_PREFETCH_DIST_K 64
#endif
#ifndef GEMM_PREFETCH_DIST_BJ
#define GEMM_PREFETCH_DIST_BJ 32
#endif

namespace {

// ---------------------------- Packing helpers -----------------------------

// Pack B panel: source B row-major [K x N] (ldB = N).
// Destination Bp is column-major inside the panel [Kp x Np]:
//   Bp[j*Kp + k] = B[(k0 + k), (n0 + j)]
inline void pack_B_panel_colmajor(const float* B, int ldB,
                                  int k0, int n0,
                                  int Kp, int Np,
                                  float* Bp)
{
    for (int j = 0; j < Np; ++j) {
        const float* Bj_src = B + (n0 + j);
        float*       Bj_dst = Bp + j * Kp;
        for (int k = 0; k < Kp; ++k) {
            Bj_dst[k] = Bj_src[(k0 + k) * ldB];
        }
    }
}

// Pack A micro-panel: source A row-major [M x K] (ldA = K).
// Destination Ap is column-major [Kp x mr]:
//   Ap[k*mr + r] = A[(m1 + r), (k0 + k)]   (zero-pad r>=mr_eff)
inline void pack_A_micropanel_colmajor(const float* A, int ldA,
                                       int m1, int k0,
                                       int mr, int mr_eff, int Kp,
                                       float* Ap)
{
    for (int k = 0; k < Kp; ++k) {
        const float* a_base = A + (m1 * ldA + (k0 + k));
        int r = 0;
        for (; r < mr_eff; ++r) {
            Ap[k*mr + r] = a_base[r * ldA];
        }
        for (; r < mr; ++r) {
            Ap[k*mr + r] = 0.0f;
        }
    }
}

// Scatter store 16 rows of a C column with stride ldC, optionally masked (mr_eff < 16).
inline void scatter_store_16_rows(const __m512 v, float* base, int ldC, int mr_eff)
{
    // idx[r] = r * ldC (floats). i32scatter uses a byte-scale; we pass scale=4.
    alignas(64) int idx_arr[16];
    const int lanes = 16;
    for (int r = 0; r < lanes; ++r) idx_arr[r] = r * ldC;
    const __m512i idx = _mm512_load_si512((const void*)idx_arr);

    if (mr_eff >= 16) {
        _mm512_i32scatter_ps(base, idx, v, /*scale=*/4);
    } else {
        const __mmask16 mask = (mr_eff <= 0) ? 0 : (__mmask16)((1u << mr_eff) - 1u);
        _mm512_mask_i32scatter_ps(base, mask, idx, v, /*scale=*/4);
    }
}

// ---------------------------- AVX-512 microkernel -----------------------------

// Rows = lanes (mr=16). Keep up to nr<=24 columns of C live in regs.
// K-unrolled by `ku` (typ. 4). B is scalar-broadcast per column; A is vector-load per k.
//
// Inputs:
//   Ap:      [Kp x 16], column-major: Ap[k*16 + r]
//   Bp_tile: [Kp x nr_eff], column-major per output column: Bp_tile[j*Kp + k]
//   Kp: depth of this K-slab
//   nr_eff: number of output columns in this micro-tile (<= 24)
//   ldC: leading dim of row-major C (N)
//   Cblk: top-left ptr of this C micro-tile (row-major)
//   alpha: scale factor for A*B
//   ku: K unroll (1,2,4,...)
//
// Notes:
//  - We rely on Ap being zero-padded when mr_eff < 16, so loads are unmasked.
//  - We do masked scatter on store if mr_eff < 16.
//  - Light prefetching for Ap and Bp_tile; no NT/streaming stores (by request).
inline void kernel_mr16_nrXX_ku_avx512(const float* Ap,
                                       const float* Bp_tile,
                                       int Kp, int nr_eff,
                                       int ldC, float* Cblk,
                                       float alpha, int ku,
                                       int mr_eff)
{
    // Limit nr_eff to 24 accumulators (fits nicely under 32 zmm with temps).
    const int NR_MAX = 24;
    if (nr_eff > NR_MAX) nr_eff = NR_MAX;

    __m512 Creg[NR_MAX];
    for (int j = 0; j < nr_eff; ++j) {
        Creg[j] = _mm512_setzero_ps();
    }

    int k = 0;

    // Unrolled K loop
    for (; k + ku <= Kp; k += ku) {
        __m512 Avec[8]; // support up to ku=8 (we'll only use [0..ku-1])
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            if (t < ku) {
                Avec[t] = _mm512_loadu_ps(&Ap[(k + t) * 16]); // 16 rows at depth k+t
                // Prefetch A ahead
                if (k + t + GEMM_PREFETCH_DIST_K < Kp) {
                    const float* pfA = &Ap[(k + t + GEMM_PREFETCH_DIST_K) * 16];
                    _mm_prefetch((const char*)(pfA), _MM_HINT_T0);
                }
            }
        }

        // Process each output column j
        for (int j = 0; j < nr_eff; ++j) {
            const float* Bj_col = Bp_tile + j * Kp;

            // Prefetch upcoming B scalars along k
            if (k + GEMM_PREFETCH_DIST_BJ < Kp) {
                _mm_prefetch((const char*)(Bj_col + k + GEMM_PREFETCH_DIST_BJ), _MM_HINT_T0);
            }

            // Accumulate ku FMAs for this column
            #pragma unroll
            for (int t = 0; t < 8; ++t) {
                if (t < ku) {
                    const float bj = (Bj_col[k + t]) * alpha;   // fold alpha here
                    const __m512 Bj_bcast = _mm512_set1_ps(bj);
                    Creg[j] = _mm512_fmadd_ps(Avec[t], Bj_bcast, Creg[j]);
                }
            }
        }
    }

    // Tail of K (ku-agnostic)
    for (; k < Kp; ++k) {
        const __m512 Avec = _mm512_loadu_ps(&Ap[k * 16]);
        for (int j = 0; j < nr_eff; ++j) {
            const float bj = (Bp_tile[j * Kp + k]) * alpha;
            const __m512 Bj_bcast = _mm512_set1_ps(bj);
            Creg[j] = _mm512_fmadd_ps(Avec, Bj_bcast, Creg[j]);
        }
    }

    // Store (+=) into C once (one column at a time, rows with stride ldC)
    for (int j = 0; j < nr_eff; ++j) {
        // Read current C column values, add Creg, write back. Since we avoided NT stores,
        // we simply accumulate in memory: C[r*ldC + j] += Creg[j][r].
        // Using scatter to avoid scalar inner loops.
        // Load current C column to temp, add, and scatter result.
        // To minimize loads, we do gather-add-scatter per column.

        // Prepare indices once (gather/scatter)
        alignas(64) int idx_arr[16];
        for (int r = 0; r < 16; ++r) idx_arr[r] = r * ldC;
        const __m512i idx = _mm512_load_si512((const void*)idx_arr);

        float* base = Cblk + j;

        // Gather current C
        __m512 Cold;
        if (mr_eff >= 16) {
            Cold = _mm512_i32gather_ps(idx, (const void*)base, /*scale=*/4);
        } else {
            const __mmask16 mask = (mr_eff <= 0) ? 0 : (__mmask16)((1u << mr_eff) - 1u);
            Cold = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), mask, idx, (const void*)base, 4);
        }

        // Add accumulators
        const __m512 Cnew = _mm512_add_ps(Cold, Creg[j]);

        // Scatter back
        scatter_store_16_rows(Cnew, base, ldC, mr_eff);
    }
}

// ---------------------------- Top level (blocked GEMM) -----------------------------

} // anonymous namespace

namespace gemm {

void sgemm_blocked(const float* A, int M, int K,
                   const float* B, int N,
                   float* C,
                   float alpha, float beta,
                   int Mb, int Nb, int Kb,
                   int mr, int nr, int ku)
{

    const char* env = std::getenv("OMP_NUM_THREADS");
    int nth       = env ? std::atoi(env) : omp_get_max_threads();
    // We specialize for mr=16, nr<=24. If caller passes something else, clamp/sanitize.
    if (mr != 16) mr = 16;
    if (nr > 24)  nr = 24;
    if (nr < 1)   nr = 1;
    if (ku < 1)   ku = 1;
    if (ku > 8)   ku = 8; // our kernel buffers up to 8 A-vectors

    const int ldA = K;
    const int ldB = N;
    const int ldC = N;

    // beta handling on C
    if (beta == 0.0f) {
        std::fill(C, C + static_cast<size_t>(M) * N, 0.0f);
    } else if (beta != 1.0f) {
        const size_t total = static_cast<size_t>(M) * N;
        #pragma omp parallel for if(total > (1<<15))
        for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(total); ++i) {
            C[i] *= beta;
        }
    }

    // Allocate once per call: max B panel (Kb x Nb) and A micro-panel (Kb x mr)
    std::vector<float> Bp; Bp.resize(static_cast<size_t>(Kb) * Nb);
    std::vector<float> Ap; Ap.resize(static_cast<size_t>(Kb) * mr);

    // OpenMP threads
    #if defined(_OPENMP)
    if (nth > 0) omp_set_num_threads(nth);
    #endif

    for (int n0 = 0; n0 < N; n0 += Nb) {
        const int Np = std::min(Nb, N - n0);

        for (int k0 = 0; k0 < K; k0 += Kb) {
            const int Kp = std::min(Kb, K - k0);

            // Pack B panel once for this (n0,k0)
            pack_B_panel_colmajor(B, ldB, k0, n0, Kp, Np, Bp.data());

            // Sweep M blocks (parallel loop) reusing same packed B panel
            #if defined(_OPENMP)
            #pragma omp parallel for schedule(static)
            #endif
            for (int m0 = 0; m0 < M; m0 += Mb) {
                const int Mp = std::min(Mb, M - m0);

                for (int m1 = m0; m1 < m0 + Mp; m1 += mr) {
                    const int mr_eff = std::min(mr, m0 + Mp - m1);

                    // Pack A micro-panel for rows m1.. and K-slab k0..k0+Kp
                    pack_A_micropanel_colmajor(A, ldA, m1, k0, mr, mr_eff, Kp, Ap.data());

                    for (int n1 = n0; n1 < n0 + Np; n1 += nr) {
                        const int nr_eff = std::min(nr, n0 + Np - n1);

                        float* Cblk = C + static_cast<size_t>(m1) * ldC + n1;
                        const float* Bp_tile = Bp.data() + (n1 - n0) * Kp;

                        kernel_mr16_nrXX_ku_avx512(
                            Ap.data(), Bp_tile,
                            Kp, nr_eff,
                            ldC, Cblk,
                            alpha, ku,
                            mr_eff
                        );
                    }
                }
            } // m0
        } // k0
    } // n0
}

} // namespace gemm
