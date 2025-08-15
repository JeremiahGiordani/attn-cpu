#include "gemm.h"
#include <algorithm>
#include <vector>
#include <cstring>

#if defined(_OPENMP)
  #include <omp.h>
#endif

// Optional light prefetch (portable-ish via builtin)
#ifndef GEMM_PREFETCH_DIST_K
#define GEMM_PREFETCH_DIST_K 64
#endif

namespace {

// Pack B panel: source B row-major [K x N] (ldB = N)
// Destination Bp is column-major *within the panel* [Kp x Np]:
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

// Pack A micro-panel: source A row-major [M x K] (ldA = K)
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

// Compute one micro-tile: C[m1..m1+mr_eff-1, n1..n1+nr_eff-1]
// Inputs:
//   Ap: [Kp x mr] col-major  (Ap[k*mr + r])
//   Bp_tile: points to start of B panel at column offset n1_off; col-major per output column
// Accumulates into Cblk row-major [*, ldC] with += (outer-product style).
inline void kernel_mrnr_ku_scalar(const float* Ap, const float* Bp_tile,
                                  int mr, int nr_eff, int Kp,
                                  int ldC, float* Cblk,
                                  float alpha, int ku)
{
    // Cacc[j][r] : conceptually registers; keep local for cache locality
    // Keep fixed upper bounds to allow compiler to unroll reasonably
    // (nr is typically <= 24)
    // Allocate on stack; if you worry about VLAs, use std::vector.
    std::vector<float> Cacc(nr_eff * mr, 0.0f);

    int k = 0;
    for (; k + ku <= Kp; k += ku) {
        // load ku A-vectors (each length mr)
        // Access pattern Ap[(k+t)*mr + r], r=0..mr-1
        for (int t = 0; t < ku; ++t) {
            const float* Avec = &Ap[(k + t) * mr];

            // Optionally prefetch B a bit ahead
            #if defined(__GNUC__) || defined(__clang__)
            __builtin_prefetch(Bp_tile + (k + t + GEMM_PREFETCH_DIST_K), 0, 1);
            #endif

            // For each output column j
            for (int j = 0; j < nr_eff; ++j) {
                const float b = *(Bp_tile + j * Kp + (k + t));
                float* ccol = &Cacc[j * mr];
                // Cacc[j,:] += alpha * Avec[:] * b
                // Unroll the short mr loop a bit (mr is typically 16)
                int r = 0;
                for (; r + 3 < mr; r += 4) {
                    ccol[r+0] += alpha * Avec[r+0] * b;
                    ccol[r+1] += alpha * Avec[r+1] * b;
                    ccol[r+2] += alpha * Avec[r+2] * b;
                    ccol[r+3] += alpha * Avec[r+3] * b;
                }
                for (; r < mr; ++r) {
                    ccol[r] += alpha * Avec[r] * b;
                }
            }
        }
    }
    // tail for K
    for (; k < Kp; ++k) {
        const float* Avec = &Ap[k * mr];
        for (int j = 0; j < nr_eff; ++j) {
            const float b = *(Bp_tile + j * Kp + k);
            float* ccol = &Cacc[j * mr];
            int r = 0;
            for (; r + 3 < mr; r += 4) {
                ccol[r+0] += alpha * Avec[r+0] * b;
                ccol[r+1] += alpha * Avec[r+1] * b;
                ccol[r+2] += alpha * Avec[r+2] * b;
                ccol[r+3] += alpha * Avec[r+3] * b;
            }
            for (; r < mr; ++r) {
                ccol[r] += alpha * Avec[r] * b;
            }
        }
    }

    // Store (+=) into C once
    for (int j = 0; j < nr_eff; ++j) {
        const float* ccol = &Cacc[j * mr];
        for (int r = 0; r < mr; ++r) {
            Cblk[r * ldC + j] += ccol[r];
        }
    }
}

} // anonymous namespace

namespace gemm {

void sgemm_blocked(const float* A, int M, int K,
                   const float* B, int N,
                   float* C,
                   float alpha, float beta,
                   int Mb, int Nb, int Kb,
                   int mr, int nr, int ku,
                   int num_threads)
{
    const int ldA = K;
    const int ldB = N;
    const int ldC = N;

    // Scale or zero C once at the beginning (beta)
    if (beta == 0.0f) {
        std::fill(C, C + static_cast<size_t>(M) * N, 0.0f);
    } else if (beta != 1.0f) {
        const size_t total = static_cast<size_t>(M) * N;
        for (size_t i = 0; i < total; ++i) C[i] *= beta;
    }

    // Allocate once per call: B panel and A micro-panel scratch
    // Max sizes; actual Kp/Np vary on tails.
    std::vector<float> Bp; Bp.resize(static_cast<size_t>(Kb) * Nb);
    std::vector<float> Ap; Ap.resize(static_cast<size_t>(Kb) * mr);

    // Threading over m0 for each (n0,k0). If num_threads<=0, use OMP default.
    #if defined(_OPENMP)
    if (num_threads > 0) omp_set_num_threads(num_threads);
    #endif

    for (int n0 = 0; n0 < N; n0 += Nb) {
        const int Np = std::min(Nb, N - n0);

        for (int k0 = 0; k0 < K; k0 += Kb) {
            const int Kp = std::min(Kb, K - k0);

            // Pack B panel once for this (n0,k0)
            pack_B_panel_colmajor(B, ldB, k0, n0, Kp, Np, Bp.data());

            // Sweep M blocks; parallelize this loop
            #if defined(_OPENMP)
            #pragma omp parallel for schedule(static)
            #endif
            for (int m0 = 0; m0 < M; m0 += Mb) {
                const int Mp = std::min(Mb, M - m0);

                // Walk micro-tiles in this (m0,n0) block
                for (int m1 = m0; m1 < m0 + Mp; m1 += mr) {
                    const int mr_eff = std::min(mr, m0 + Mp - m1);

                    // Pack A micro-panel for these rows vs Kp
                    pack_A_micropanel_colmajor(A, ldA,
                                               /*m1=*/m1, /*k0=*/k0,
                                               mr, mr_eff, Kp,
                                               Ap.data());

                    for (int n1 = n0; n1 < n0 + Np; n1 += nr) {
                        const int nr_eff = std::min(nr, n0 + Np - n1);

                        // Compute micro-tile and accumulate into C
                        float* Cblk = C + static_cast<size_t>(m1) * ldC + n1;
                        const float* Bp_tile = Bp.data() + (n1 - n0) * Kp;

                        kernel_mrnr_ku_scalar(Ap.data(), Bp_tile,
                                              mr, nr_eff, Kp,
                                              ldC, Cblk,
                                              alpha, ku);
                    }
                }
            } // m0
        } // k0
    } // n0
}

} // namespace gemm
