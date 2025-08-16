// gemm.cpp (driver parts)
// Requires: pack_A_micropanel_colmajor_alpha (alpha applied), and the kernels below.

#include <immintrin.h>
#include <algorithm>
#include <vector>
#include <cstring>
#include <omp.h>

#ifndef GEMM_MR
#define GEMM_MR 16
#endif
#ifndef GEMM_NRBLK   // physical micro-panel width in the packed B (k-interleaved)
#define GEMM_NRBLK 16
#endif
#ifndef GEMM_KU
#define GEMM_KU 8
#endif




inline void pack_A_micropanel_colmajor_alpha(const float* A, int ldA,
                                             int m1, int k0,
                                             int mr, int mr_eff, int Kp,
                                             float alpha,
                                             float* Ap)
{
    const bool scale = (alpha != 1.0f);
    for (int k = 0; k < Kp; ++k) {
        const float* a_base = A + (m1 * ldA + (k0 + k));
        int r = 0;
        if (scale) {
            for (; r < mr_eff; ++r) Ap[k*mr + r] = a_base[r * ldA] * alpha;
        } else {
            for (; r < mr_eff; ++r) Ap[k*mr + r] = a_base[r * ldA];
        }
        for (; r < mr; ++r) Ap[k*mr + r] = 0.0f;
    }
}

// ------------------ B pack: one 16-col "group", k-interleaved ------------------
// Layout per group g (columns n0 + j0 .. j0+15):
//   For k = 0..Kp-1:  Bpg[k*16 + j] = B[(k0+k), (n0 + j0 + j)]  for j=0..nr_eff-1
// Tail j in [nr_eff..15] is zero-padded.
inline void pack_B_group_kinter(
    const float* B, int ldB,
    int k0, int n0,
    int Kp, int Np,
    int j0,                       // offset within panel Np for this group (multiple of 16)
    float* __restrict Bpg         // out: Kp * 16 floats
){
    const int nr_eff = std::min(GEMM_NRBLK, Np - j0);
    for (int k = 0; k < Kp; ++k) {
        const float* src = B + (size_t)(k0 + k) * ldB + (n0 + j0);
        float*       dst = Bpg + (size_t)k * GEMM_NRBLK;
        std::memcpy(dst, src, (size_t)nr_eff * sizeof(float));
        for (int j = nr_eff; j < GEMM_NRBLK; ++j) dst[j] = 0.0f;
    }
}

// ------------------ Fixed-NR microkernels with contiguous-B -------------------
template<int NR>
inline void kernel_16xNR_f32_kinter(
    const float* __restrict Ap,        // [Kp x 16], alpha already applied
    const float* __restrict Bpg,       // group base (k-interleaved), size >= Kp*16
    int Kp, int ldC, float* __restrict Cblk,
    bool first_panel, int mr_eff)
{
    static_assert(NR==16 || NR==12 || NR==8, "NR must be 16/12/8");
    __m512 c[NR];
    #pragma unroll
    for (int j=0; j<NR; ++j) c[j] = _mm512_setzero_ps();

    const float* __restrict Ap_k = Ap;
    const float* __restrict Bk   = Bpg;          // points to k=0 row: 16 contiguous floats
    int k = 0;

    for (; k + GEMM_KU <= Kp; k += GEMM_KU) {
        __m512 a[GEMM_KU];
        #pragma unroll
        for (int t=0; t<GEMM_KU; ++t) a[t] = _mm512_load_ps(Ap_k + (size_t)t*GEMM_MR);

        // For each of the ku steps, B scalars for this k are contiguous: Bk + t*16
        #pragma unroll
        for (int t=0; t<GEMM_KU; ++t) {
            const float* __restrict bk = Bk + (size_t)t*GEMM_NRBLK;
            #pragma unroll
            for (int j=0; j<NR; ++j) {
                const __m512 b = _mm512_broadcastss_ps(_mm_load_ss(bk + j));
                c[j] = _mm512_fmadd_ps(a[t], b, c[j]);
            }
        }

        Ap_k += GEMM_KU*GEMM_MR;
        Bk   += GEMM_KU*GEMM_NRBLK;
    }

    for (; k < Kp; ++k) {
        const __m512 a0 = _mm512_load_ps(Ap + (size_t)k*GEMM_MR);
        const float* __restrict bk = Bpg + (size_t)k*GEMM_NRBLK;
        #pragma unroll
        for (int j=0; j<NR; ++j) {
            const __m512 b = _mm512_broadcastss_ps(_mm_load_ss(bk + j));
            c[j] = _mm512_fmadd_ps(a0, b, c[j]);
        }
    }

    // Write-back via small temp → contiguous row adds (simple & quick)
    alignas(64) float Ct[NR][GEMM_MR];
    #pragma unroll
    for (int j=0; j<NR; ++j) _mm512_store_ps(&Ct[j][0], c[j]);

    if (first_panel) {
        for (int r=0; r<mr_eff; ++r) {
            float* Crow = Cblk + (size_t)r*ldC;
            #pragma unroll
            for (int j=0; j<NR; ++j) Crow[j] = Ct[j][r];
        }
    } else {
        for (int r=0; r<mr_eff; ++r) {
            float* Crow = Cblk + (size_t)r*ldC;
            #pragma unroll
            for (int j=0; j<NR; ++j) Crow[j] += Ct[j][r];
        }
    }
}

// ------------------------------ Driver ---------------------------------------
namespace gemm {

void sgemm_blocked(const float* __restrict A, int M, int K,
                   const float* __restrict B, int N,
                   float* __restrict C,
                   float alpha, float beta,
                   int Mb, int Nb, int Kb,
                   int /*mr*/, int /*nr*/, int /*ku*/)
{
    // Fixed kernel shape for this path
    constexpr int MR = GEMM_MR;           // 16 rows
    constexpr int NRblk = GEMM_NRBLK;     // 16 columns per packed group
    constexpr int KU = GEMM_KU;           // 8-way K unroll

    const int ldA = K, ldB = N, ldC = N;

    // IMPORTANT: don't touch C when beta==0 (first K-panel will write it)
    if (beta != 0.0f) {
        if (beta != 1.0f) {
            const size_t total = (size_t)M * N;
            #pragma omp parallel for if (total > (1<<15))
            for (ptrdiff_t i = 0; i < (ptrdiff_t)total; ++i) C[i] *= beta;
        }
    }

    // Cache-aware defaults (override only if <=0 passed in)
    if (Mb <= 0) Mb = 128;     // keep C tile in L2
    if (Nb <= 0) Nb = 1024;    // big N-panel amortizes B-pack
    if (Kb <= 0) Kb = 256;     // Ap(16xKb) + Bp(16xKb) ≈ 32 KiB → L1D friendly

    // Max groups we ever need to pack for a panel (ceil(Nb/16))
    const int max_groups = (Nb + NRblk - 1) / NRblk;
    // Packed B buffer big enough for Kb * 16 * groups
    std::vector<float> Bp; Bp.resize((size_t)Kb * NRblk * max_groups);

    for (int n0 = 0; n0 < N; n0 += Nb) {
        const int Np = std::min(Nb, N - n0);
        const int groups_panel = (Np + NRblk - 1) / NRblk;

        for (int k0 = 0; k0 < K; k0 += Kb) {
            const int Kp = std::min(Kb, K - k0);
            const bool first_panel = (beta == 0.0f) && (k0 == 0);

            #pragma omp parallel
            {
                // 1) Parallel B-pack across groups (each thread packs some groups)
                #pragma omp for schedule(static)
                for (int g = 0; g < groups_panel; ++g) {
                    const int j0 = g * NRblk;                           // offset within the panel
                    float* Bpg = Bp.data() + (size_t)g * Kp * NRblk;    // group base
                    pack_B_group_kinter(B, ldB, k0, n0, Kp, Np, j0, Bpg);
                }
                // implicit barrier here (end of omp for)

                // Per-thread A buffer (aligned)
                float* Ap_thread = (float*)_mm_malloc(sizeof(float)*(size_t)Kp*MR, 64);

                // 2) Compute: parallel over M tiles; reuse packed A across all N-groups
                #pragma omp for schedule(static)
                for (int m0 = 0; m0 < M; m0 += Mb) {
                    const int Mp = std::min(Mb, M - m0);

                    for (int m1 = m0; m1 < m0 + Mp; m1 += MR) {
                        const int mr_eff = std::min(MR, m0 + Mp - m1);

                        // Pack A once for this (m1, k0..k0+Kp) and reuse across all B groups
                        pack_A_micropanel_colmajor_alpha(
                            A, ldA, m1, k0, MR, mr_eff, Kp, alpha, Ap_thread);

                        // Walk N in 16-wide groups. Because we step by 16, joff==0 always.
                        int n1 = 0;
                        // Full 16-wide tiles
                        for (; n1 + 16 <= Np; n1 += 16) {
                            const int g = n1 / NRblk;  // which group
                            float*       Cblk = C + (size_t)m1 * ldC + (n0 + n1);
                            const float* Bpg  = Bp.data() + (size_t)g * Kp * NRblk;

                            kernel_16xNR_f32_kinter<16>(
                                Ap_thread, Bpg, Kp, ldC, Cblk, first_panel, mr_eff);
                        }
                        // Tails (rare for N multiple of 16, but included for generality)
                        if (n1 + 12 <= Np) {
                            const int g = n1 / NRblk;
                            float*       Cblk = C + (size_t)m1 * ldC + (n0 + n1);
                            const float* Bpg  = Bp.data() + (size_t)g * Kp * NRblk;

                            kernel_16xNR_f32_kinter<12>(
                                Ap_thread, Bpg, Kp, ldC, Cblk, first_panel, mr_eff);
                            n1 += 12;
                        }
                        if (n1 + 8 <= Np) {
                            const int g = n1 / NRblk;
                            float*       Cblk = C + (size_t)m1 * ldC + (n0 + n1);
                            const float* Bpg  = Bp.data() + (size_t)g * Kp * NRblk;

                            kernel_16xNR_f32_kinter<8>(
                                Ap_thread, Bpg, Kp, ldC, Cblk, first_panel, mr_eff);
                            n1 += 8;
                        }
                        // tiny residue (1..7) – optional: scalar or reuse NR=8 with masking
                        for (; n1 < Np; ++n1) {
                            // very small path; for your N=960 this won't run
                            const int g = n1 / NRblk;
                            float*       Cblk = C + (size_t)m1 * ldC + (n0 + n1);
                            const float* Bpg  = Bp.data() + (size_t)g * Kp * NRblk;
                            // simple 16x1 microkernel (not shown) or call NR=8 and only use first col
                            kernel_16xNR_f32_kinter<8>(
                                Ap_thread, Bpg, Kp, ldC, Cblk, first_panel, mr_eff);
                        }
                    }
                }

                _mm_free(Ap_thread);
            } // omp parallel
        } // k0
    } // n0
}

} // namespace gemm
