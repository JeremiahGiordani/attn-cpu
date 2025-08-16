#pragma once
#include <immintrin.h>
#include <algorithm>
#include <cstdint>

#ifndef GEMM_MR
#define GEMM_MR 16
#endif
#ifndef GEMM_KU
#define GEMM_KU 8
#endif

// Compile-time 16×NR kernel; NR ∈ {16,12,8}
template<int NR>
inline void kernel_16xNR_f32_avx512(
    const float* __restrict Ap,       // [Kp x 16], alpha already applied
    const float* __restrict Bp_tile,  // [Kp x NR],  k-major, then columns 0..NR-1
    int Kp,
    int ldC, float* __restrict Cblk,  // C at (m1,n1)
    bool first_panel,                 // true if (k0==0 && beta==0)
    int mr_eff                        // 1..16
){
    static_assert(NR == 16 || NR == 12 || NR == 8, "NR must be 16/12/8");
    __m512 c[NR];
    #pragma unroll
    for (int j=0; j<NR; ++j) c[j] = _mm512_setzero_ps();

    int k = 0;
    const float* __restrict Ap_k = Ap;
    const float* __restrict Bk   = Bp_tile;

    for (; k + GEMM_KU <= Kp; k += GEMM_KU) {
        __m512 a[GEMM_KU];
        #pragma unroll
        for (int t=0; t<GEMM_KU; ++t)
            a[t] = _mm512_load_ps(Ap_k + (size_t)t*GEMM_MR);

        #pragma unroll
        for (int j=0; j<NR; ++j) {
            const float* __restrict bj = Bk + (size_t)j*Kp;
            #pragma unroll
            for (int t=0; t<GEMM_KU; ++t) {
                // explicitly broadcast from memory (generates vbroadcastss m32)
                const __m512 b = _mm512_broadcastss_ps(_mm_load_ss(bj + t));
                c[j] = _mm512_fmadd_ps(a[t], b, c[j]);
            }
        }
        Ap_k += (size_t)GEMM_KU*GEMM_MR;
        Bk   += GEMM_KU;
    }

    for (; k < Kp; ++k) {
        const __m512 a = _mm512_load_ps(Ap + (size_t)k*GEMM_MR);
        #pragma unroll
        for (int j=0; j<NR; ++j) {
            const __m512 b = _mm512_broadcastss_ps(_mm_load_ss(Bp_tile + (size_t)j*Kp + k));
            c[j] = _mm512_fmadd_ps(a, b, c[j]);
        }
    }

    // Write-back: dump columns to tiny temp, then contiguous row adds (fast & cache-friendly)
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
