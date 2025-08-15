// === microkernel_16x8_avx512.h ===
#pragma once
#include <immintrin.h>
#include <algorithm>

#ifndef GEMM_MR
#define GEMM_MR 16
#endif
#ifndef GEMM_NR
#define GEMM_NR 8
#endif
#ifndef GEMM_KU
#define GEMM_KU 8
#endif

// Ap: [Kp x 16] col-major (k-major, 16 contiguous)
// Bp_tile: [Kp x nr_eff] col-major (k-major)
// Cblk: C at (m1, n1)
// flags:
//   first_panel: true when (k0==0 && beta==0) → store, no read
//   last_panel : true when (k0+Kp==K)         → use stream store (NT) for final write
inline void kernel_16x8_f32_avx512(
    const float* __restrict Ap,
    const float* __restrict Bp_tile,
    int Kp, int nr_eff,
    int ldC, float* __restrict Cblk,
    float alpha,
    bool first_panel,
    bool last_panel,
    int mr_eff
){
    // We implement 16x8. If nr_eff<8 we mask lanes in the write-back.
    const __mmask8 colmask = (nr_eff >= 8) ? 0xFF : (__mmask8)((1u << nr_eff) - 1u);
    const __mmask16 rowmask = (mr_eff >= 16) ? 0xFFFF : (__mmask16)((1u << mr_eff) - 1u);

    // Eight accumulator vectors: each is 16 floats for one output column
    __m512 c[8];
    for (int j=0; j<8; ++j) c[j] = _mm512_setzero_ps();

    const __m512 Alpha = _mm512_set1_ps(alpha);

    int k = 0;
    // Unrolled main K loop (software pipeline: load A once, reuse across columns)
    for (; k + GEMM_KU <= Kp; k += GEMM_KU) {
        __m512 a[GEMM_KU];
        #pragma unroll
        for (int t=0; t<GEMM_KU; ++t) {
            // Load 16 rows from packed A and scale by alpha once
            a[t] = _mm512_mul_ps(_mm512_load_ps(&Ap[(k + t) * GEMM_MR]), Alpha);
        }

        // columns 0..7 (mask if nr_eff<8)
        #pragma unroll
        for (int j=0; j<8; ++j) {
            if (!(colmask & (1u<<j))) break;
            const float* __restrict bj = Bp_tile + j*Kp + k;

            // Unroll over the ku chunk
            #pragma unroll
            for (int t=0; t<GEMM_KU; ++t) {
                const __m512 b = _mm512_set1_ps(bj[t]);
                c[j] = _mm512_fmadd_ps(a[t], b, c[j]);
            }
        }
    }

    // K tail
    for (; k < Kp; ++k) {
        const __m512 a = _mm512_mul_ps(_mm512_load_ps(&Ap[k * GEMM_MR]), Alpha);
        #pragma unroll
        for (int j=0; j<8; ++j) {
            if (!(colmask & (1u<<j))) break;
            const __m512 b = _mm512_set1_ps(Bp_tile[j*Kp + k]);
            c[j] = _mm512_fmadd_ps(a, b, c[j]);
        }
    }

    // ---- Write-back: either direct store (first panel) or add-then-store ----
    // We write per row for contiguous stores across columns (good for caches & NT).
    for (int r=0; r<mr_eff; ++r) {
        float* Crow = Cblk + r*ldC;
        // Gather the 8 scalars [c0[r], c1[r], ...] into a small temp
        alignas(64) float tmp[8];
        #pragma unroll
        for (int j=0; j<8; ++j) {
            if (!(colmask & (1u<<j))) break;
            _mm_store_ss(&tmp[j], _mm_movehl_ps(_mm256_castps256_ps128(_mm512_extractf32x8_ps(c[j], 1)),
                                                _mm256_castps256_ps128(_mm512_extractf32x8_ps(c[j], 0))));
            // ↑ cheap way to get lane r? No — better to extract lane r with maskload:
        }
    }

    // Simpler and faster: store columns then add per row (your original idea), but
    // with “first/last panel” fast paths:
    alignas(64) float Ct[8][GEMM_MR];
    for (int j=0; j<8; ++j) {
        if (!(colmask & (1u<<j))) break;
        _mm512_mask_store_ps(&Ct[j][0], rowmask, c[j]);
    }

    if (first_panel) {
        // No read of C needed; we can stream the final store if also last_panel.
        if (last_panel) {
            for (int r=0; r<mr_eff; ++r) {
                float* Crow = Cblk + r*ldC;
                // stream 8 floats if we have full 8; else scalar
                if (nr_eff == 8) {
                    _mm256_stream_ps(Crow, _mm256_load_ps(&Ct[0][r])); // contiguous 8 values
                } else {
                    #pragma unroll
                    for (int j=0; j<nr_eff; ++j) Crow[j] = Ct[j][r];
                }
            }
        } else {
            for (int r=0; r<mr_eff; ++r) {
                float* Crow = Cblk + r*ldC;
                #pragma unroll
                for (int j=0; j<nr_eff; ++j) Crow[j] = Ct[j][r];
            }
        }
    } else {
        // Add to existing C
        for (int r=0; r<mr_eff; ++r) {
            float* Crow = Cblk + r*ldC;
            #pragma unroll
            for (int j=0; j<nr_eff; ++j) Crow[j] += Ct[j][r];
        }
    }
}
