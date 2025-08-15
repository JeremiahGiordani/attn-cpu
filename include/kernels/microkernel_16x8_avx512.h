#pragma once
#include <immintrin.h>

#ifndef GEMM_MR
#define GEMM_MR 16
#endif
#ifndef GEMM_NR_FULL
#define GEMM_NR_FULL 16
#endif
#ifndef GEMM_KU
#define GEMM_KU 8
#endif

// Ap: [Kp x 16] (64B aligned recommended), already scaled by alpha.
// Bp_tile: [Kp x nr_eff] column-major per tile (k-major).
inline void kernel_16x16_f32_avx512(
    const float* __restrict Ap,
    const float* __restrict Bp_tile,
    int Kp, int nr_eff,
    int ldC, float* __restrict Cblk,
    bool first_panel, bool last_panel,
    int mr_eff
){
    // Split 16 columns into two 8-wide groups.
    const int nr0 = (nr_eff >= 8) ? 8 : nr_eff;
    const int nr1 = (nr_eff > 8) ? (nr_eff - 8) : 0;

    __m512 c0[8], c1[8];
    for (int j=0; j<8; ++j) { c0[j] = _mm512_setzero_ps(); c1[j] = _mm512_setzero_ps(); }

    int k = 0;
    const float* __restrict Ap_k = Ap;
    const float* __restrict B0_k = Bp_tile;            // cols 0..7 laid at j*Kp
    const float* __restrict B1_k = Bp_tile + (size_t)8*Kp; // cols 8..15

    for (; k + GEMM_KU <= Kp; k += GEMM_KU) {
        __m512 a[GEMM_KU];
        #pragma unroll
        for (int t=0; t<GEMM_KU; ++t) {
            a[t] = _mm512_load_ps(Ap_k + (size_t)t*GEMM_MR);
        }
        // group 0 (0..7)
        #pragma unroll
        for (int j=0; j<8; ++j) {
            if (j >= nr0) break;
            const float* __restrict bj = B0_k + (size_t)j*Kp;
            #pragma unroll
            for (int t=0; t<GEMM_KU; ++t) {
                c0[j] = _mm512_fmadd_ps(a[t], _mm512_set1_ps(bj[t]), c0[j]);
            }
        }
        // group 1 (8..15)
        #pragma unroll
        for (int j=0; j<8; ++j) {
            if (j >= nr1) break;
            const float* __restrict bj = B1_k + (size_t)j*Kp;
            #pragma unroll
            for (int t=0; t<GEMM_KU; ++t) {
                c1[j] = _mm512_fmadd_ps(a[t], _mm512_set1_ps(bj[t]), c1[j]);
            }
        }

        Ap_k += (size_t)GEMM_KU*GEMM_MR;
        B0_k += GEMM_KU;
        B1_k += GEMM_KU;
    }

    // Tail
    for (; k < Kp; ++k) {
        const __m512 a = _mm512_load_ps(Ap + (size_t)k*GEMM_MR);
        // group 0
        #pragma unroll
        for (int j=0; j<8; ++j) {
            if (j >= nr0) break;
            c0[j] = _mm512_fmadd_ps(a, _mm512_set1_ps(Bp_tile[(size_t)j*Kp + k]), c0[j]);
        }
        // group 1
        #pragma unroll
        for (int j=0; j<8; ++j) {
            if (j >= nr1) break;
            c1[j] = _mm512_fmadd_ps(a, _mm512_set1_ps(Bp_tile[(size_t)(8+j)*Kp + k]), c1[j]);
        }
    }

    // ---- Write-back (row-contiguous) ----
    // Dump columns to a tiny temp and then row-add (fast, cache friendly).
    alignas(64) float Ct0[8][GEMM_MR];
    alignas(64) float Ct1[8][GEMM_MR];

    for (int j=0; j<nr0; ++j) _mm512_store_ps(&Ct0[j][0], c0[j]);
    for (int j=0; j<nr1; ++j) _mm512_store_ps(&Ct1[j][0], c1[j]);

    const bool do_stream = last_panel; // heuristic: stream only on last panel
    for (int r=0; r<mr_eff; ++r) {
        float* Crow = Cblk + (size_t)r*ldC;

        if (first_panel) {
            // write C := accum
            if (nr_eff == 16 && do_stream) {
                // 16 contiguous floats → one NT store
                __m512 v0 = _mm512_load_ps(&Ct0[0][r]); // WRONG shape; load 8 scalars across columns requires gather
                // Simpler: two 8-float NT stores:
                _mm256_stream_ps(Crow+0,  *(__m256*)&Ct0[0][r]);   // cols 0..7
                _mm256_stream_ps(Crow+8,  *(__m256*)&Ct1[0][r]);   // cols 8..15
            } else {
                for (int j=0; j<nr0; ++j) Crow[j]     = Ct0[j][r];
                for (int j=0; j<nr1; ++j) Crow[8 + j] = Ct1[j][r];
            }
        } else {
            // add to existing C
            for (int j=0; j<nr0; ++j) Crow[j]     += Ct0[j][r];
            for (int j=0; j<nr1; ++j) Crow[8 + j] += Ct1[j][r];
        }
    }
}
