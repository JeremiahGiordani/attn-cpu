// src/gemm.cpp
#include "gemm.h"

#include <algorithm>
#include <vector>
#include <cstring>
#include <immintrin.h>
#include <omp.h>
#include "kernels/microkernel_16x8_avx512.h"

// Prefetch distances (in "k" elements)
#ifndef GEMM_PREFETCH_DIST_K
#define GEMM_PREFETCH_DIST_K 64
#endif
#ifndef GEMM_PREFETCH_DIST_BJ
#define GEMM_PREFETCH_DIST_BJ 32
#endif

#ifndef GEMM_NR_MAX
#define GEMM_NR_MAX 12
#endif


namespace {

// ---------------------------- Packing helpers -----------------------------

// Pack B panel: source B row-major [K x N] (ldB = N).
// Destination Bp is column-major [Kp x Np]:
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

// ---------------------------- AVX-512 microkernel -----------------------------

inline __m512i make_row_index_bytes(int ldC) {
    alignas(64) int idx[16];
    for (int r = 0; r < 16; ++r) idx[r] = r * ldC * int(sizeof(float));
    return _mm512_load_si512(idx);
}

// AVX-512 microkernel: mr=16, 1<=nr_eff<=GEMM_NR_MAX, ku in [1..8]
// Accumulates C += alpha * Ap(16xKp) * Bp_tile(Kp x nr_eff).
inline void kernel_mr16_nrXX_ku_avx512(
    const float* __restrict Ap,       // [Kp x 16] packed col-major
    const float* __restrict Bp_tile,  // [Kp x nr_eff] packed col-major
    int Kp, int nr_eff,
    int ldC, float* __restrict Cblk,  // C starting at (m1, n1)
    float alpha, int ku,
    int mr_eff
){
    if (nr_eff > GEMM_NR_MAX) nr_eff = GEMM_NR_MAX;
    if (ku < 1) ku = 1; if (ku > 8) ku = 8;

    __m512 Creg[GEMM_NR_MAX];
    for (int j = 0; j < nr_eff; ++j) Creg[j] = _mm512_setzero_ps();

    const __m512 Alpha = _mm512_set1_ps(alpha);

    int k = 0;
    for (; k + ku <= Kp; k += ku) {
        __m512 Avec[8];
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            if (t >= ku) break;
            // load A(k + t, 16 rows) and scale by alpha once
            __m512 a = _mm512_loadu_ps(&Ap[(k + t) * 16]);
            Avec[t]  = _mm512_mul_ps(a, Alpha);

            if (k + t + GEMM_PREFETCH_DIST_K < Kp) {
                _mm_prefetch((const char*)(&Ap[(k + t + GEMM_PREFETCH_DIST_K) * 16]), _MM_HINT_T0);
            }
        }

        // FMAs across columns
        #pragma unroll
        for (int j = 0; j < nr_eff; ++j) {
            const float* __restrict Bj_col = Bp_tile + j * Kp;
            if (k + GEMM_PREFETCH_DIST_BJ < Kp) {
                _mm_prefetch((const char*)(Bj_col + k + GEMM_PREFETCH_DIST_BJ), _MM_HINT_T0);
            }
            #pragma unroll
            for (int t = 0; t < 8; ++t) {
                if (t >= ku) break;
                const __m512 bj = _mm512_set1_ps(Bj_col[k + t]);
                Creg[j] = _mm512_fmadd_ps(Avec[t], bj, Creg[j]);
            }
        }
    }

    // K tail
    for (; k < Kp; ++k) {
        __m512 Avec = _mm512_loadu_ps(&Ap[k * 16]);
        Avec = _mm512_mul_ps(Avec, Alpha);
        for (int j = 0; j < nr_eff; ++j) {
            const __m512 bj = _mm512_set1_ps(Bp_tile[j * Kp + k]);
            Creg[j] = _mm512_fmadd_ps(Avec, bj, Creg[j]);
        }
    }

    // ---- Write-back: dump columns to tiny temp, then contiguous row adds ----
    alignas(64) float Ct[GEMM_NR_MAX][16];
    for (int j = 0; j < nr_eff; ++j) {
        _mm512_storeu_ps(&Ct[j][0], Creg[j]);
    }

    if (mr_eff >= 16) {
        for (int r = 0; r < 16; ++r) {
            float* Crow = Cblk + r * ldC;  // contiguous over N
            #pragma unroll
            for (int j = 0; j < nr_eff; ++j) {
                Crow[j] += Ct[j][r];
            }
        }
    } else {
        for (int r = 0; r < mr_eff; ++r) {
            float* Crow = Cblk + r * ldC;
            #pragma unroll
            for (int j = 0; j < nr_eff; ++j) {
                Crow[j] += Ct[j][r];
            }
        }
    }
}


} // anonymous namespace

namespace gemm {

void sgemm_blocked(const float* __restrict A, int M, int K,
                   const float* __restrict B, int N,
                   float* __restrict C,
                   float alpha, float beta,
                   int Mb, int Nb, int Kb,
                   int mr, int nr, int ku)
{
    // Fixed kernel shape (we’ll do nr tail for <8)
    mr = GEMM_MR;            // 16
    nr = std::min(std::max(1, nr), GEMM_NR); // clamp to 8
    ku = std::min(std::max(1, ku), GEMM_KU); // clamp to 8

    const int ldA = K, ldB = N, ldC = N;

    // Initialize C (beta)
    if (beta == 0.0f) {
        std::fill(C, C + (size_t)M * N, 0.0f);
    } else if (beta != 1.0f) {
        const size_t total = (size_t)M * N;
        #pragma omp parallel for if (total > (1<<15))
        for (ptrdiff_t i = 0; i < (ptrdiff_t)total; ++i) C[i] *= beta;
    }

    // Tuned block sizes (good starting points on AVX-512):
    if (Mb <= 0) Mb = 192;                 // rows per macro-tile
    if (Nb <= 0) Nb = 256;                 // cols per B panel (multiple of 8)
    if (Kb <= 0) Kb = 384;                 // depth of K slab (fits L2 with A-pack)

    std::vector<float> Bp; Bp.resize((size_t)Kb * Nb);

    for (int n0 = 0; n0 < N; n0 += Nb) {
        const int Np = std::min(Nb, N - n0);

        for (int k0 = 0; k0 < K; k0 += Kb) {
            const int Kp = std::min(Kb, K - k0);

            // Pack B panel [Kp x Np]
            pack_B_panel_colmajor(B, ldB, k0, n0, Kp, Np, Bp.data());

            const bool first_panel = (k0 == 0) && (beta == 0.0f);
            const bool last_panel  = (k0 + Kp == K);

            #pragma omp parallel
            {
                float* Ap_thread = (float*)_mm_malloc(sizeof(float) * (size_t)Kp * mr, 64);

                #pragma omp for schedule(static)
                for (int m0 = 0; m0 < M; m0 += Mb) {
                    const int Mp = std::min(Mb, M - m0);

                    for (int m1 = m0; m1 < m0 + Mp; m1 += mr) {
                        const int mr_eff = std::min(mr, m0 + Mp - m1);

                        // Pack A micro-panel once per (m1, k0..k0+Kp)
                        pack_A_micropanel_colmajor(A, ldA, m1, k0, mr, mr_eff, Kp, Ap_thread);

                        // Walk columns in n1..n1+8 tiles
                        for (int n1 = n0; n1 < n0 + Np; n1 += GEMM_NR) {
                            const int nr_eff = std::min(GEMM_NR, n0 + Np - n1);

                            float* __restrict Cblk = C + (size_t)m1 * ldC + n1;
                            const float* __restrict Bp_tile = Bp.data() + (size_t)(n1 - n0) * Kp;

                            kernel_16x8_f32_avx512(
                                Ap_thread, Bp_tile,
                                Kp, nr_eff,
                                ldC, Cblk,
                                alpha,
                                first_panel,
                                last_panel,
                                mr_eff
                            );
                        }
                    }
                }

                _mm_free(Ap_thread);
            } // parallel
        } // k0
    } // n0
}
}
