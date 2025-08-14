#pragma once
#include "common.h"
#include "simd_math.h"
#include <cmath>
#include <vector>
// uses vectors Qh, KhT, Vh and writes to Ctx
namespace attn::kernels {

ATTN_ALWAYS_INLINE void run_dh16_m4_nocausal(const std::vector<float> &Qh,
                                             const std::vector<float> &KhT,
                                             const std::vector<float> &Vh,
                                             int T, int H, int D, int Dh,
                                             std::vector<float> &Ctx) {

    // === paste the "M=4 non-causal" attention from your current file ===
    // exact same logic you just fixed (with correct per-query weights),
    // parameterized to read/write Qh/KhT/Vh/Ctx.

    const int BK = 128; // key block
    const __m512 vscale = _mm512_set1_ps(1.0f / std::sqrt(16.0f));

    for (int t0 = 0; t0 < T; t0 += 4) {
        const int R = std::min(4, T - t0);
        // Pointers to 4 query rows (per head) and 4 ctx rows
        float *ctx0 = Ctx.data() + (size_t)(t0 + 0) * D;
        float *ctx1 = (R > 1) ? Ctx.data() + (size_t)(t0 + 1) * D : ctx0;
        float *ctx2 = (R > 2) ? Ctx.data() + (size_t)(t0 + 2) * D : ctx0;
        float *ctx3 = (R > 3) ? Ctx.data() + (size_t)(t0 + 3) * D : ctx0;

        for (int h = 0; h < H; ++h) {
            const float *q0 = Qh.data() + ((size_t)h * T + (t0 + 0)) * Dh;
            const float *q1 =
                (R > 1) ? Qh.data() + ((size_t)h * T + (t0 + 1)) * Dh : q0;
            const float *q2 =
                (R > 2) ? Qh.data() + ((size_t)h * T + (t0 + 2)) * Dh : q0;
            const float *q3 =
                (R > 3) ? Qh.data() + ((size_t)h * T + (t0 + 3)) * Dh : q0;

            // Running stats per query
            float m0 = -INFINITY, m1 = -INFINITY, m2 = -INFINITY,
                  m3 = -INFINITY;
            float l0 = 0.f, l1 = 0.f, l2 = 0.f, l3 = 0.f;
            __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps(), acc3 = _mm512_setzero_ps();

            for (int j0 = 0; j0 < T; j0 += BK) {
                const int take = std::min(BK, T - j0);

                // We do the block in two half-blocks of 64 to save
                // registers (each half has 4 groups of 16 keys).
                for (int half = 0; half < 2 && half * 64 < take; ++half) {
                    const int base = j0 + half * 64;
                    const int left = take - half * 64;
                    const int chunks =
                        std::min(4, (left + 15) / 16); // 1..4 groups of 16

                    // s[g][q]: 4 groups per half, 4 queries
                    __m512 s00 = _mm512_setzero_ps(), s01 = _mm512_setzero_ps(),
                           s02 = _mm512_setzero_ps(),
                           s03 = _mm512_setzero_ps(); // q0
                    __m512 s10 = _mm512_setzero_ps(), s11 = _mm512_setzero_ps(),
                           s12 = _mm512_setzero_ps(),
                           s13 = _mm512_setzero_ps(); // q1
                    __m512 s20 = _mm512_setzero_ps(), s21 = _mm512_setzero_ps(),
                           s22 = _mm512_setzero_ps(),
                           s23 = _mm512_setzero_ps(); // q2
                    __m512 s30 = _mm512_setzero_ps(), s31 = _mm512_setzero_ps(),
                           s32 = _mm512_setzero_ps(),
                           s33 = _mm512_setzero_ps(); // q3

// accumulate logits for this half-block
#pragma unroll(16)
                    for (int d0 = 0; d0 < 16; ++d0) {
                        const float *kt =
                            KhT.data() + ((size_t)h * Dh + d0) * T + base;
                        __m512 kd0 = (chunks >= 1) ? _mm512_loadu_ps(kt + 0)
                                                   : _mm512_setzero_ps();
                        __m512 kd1 = (chunks >= 2) ? _mm512_loadu_ps(kt + 16)
                                                   : _mm512_setzero_ps();
                        __m512 kd2 = (chunks >= 3) ? _mm512_loadu_ps(kt + 32)
                                                   : _mm512_setzero_ps();
                        __m512 kd3 = (chunks >= 4) ? _mm512_loadu_ps(kt + 48)
                                                   : _mm512_setzero_ps();

                        __m512 qv0 = _mm512_set1_ps(q0[d0]);
                        s00 = _mm512_fmadd_ps(kd0, qv0, s00);
                        if (chunks >= 2)
                            s01 = _mm512_fmadd_ps(kd1, qv0, s01);
                        if (chunks >= 3)
                            s02 = _mm512_fmadd_ps(kd2, qv0, s02);
                        if (chunks >= 4)
                            s03 = _mm512_fmadd_ps(kd3, qv0, s03);

                        if (R > 1) {
                            __m512 qv1 = _mm512_set1_ps(q1[d0]);
                            s10 = _mm512_fmadd_ps(kd0, qv1, s10);
                            if (chunks >= 2)
                                s11 = _mm512_fmadd_ps(kd1, qv1, s11);
                            if (chunks >= 3)
                                s12 = _mm512_fmadd_ps(kd2, qv1, s12);
                            if (chunks >= 4)
                                s13 = _mm512_fmadd_ps(kd3, qv1, s13);
                        }
                        if (R > 2) {
                            __m512 qv2 = _mm512_set1_ps(q2[d0]);
                            s20 = _mm512_fmadd_ps(kd0, qv2, s20);
                            if (chunks >= 2)
                                s21 = _mm512_fmadd_ps(kd1, qv2, s21);
                            if (chunks >= 3)
                                s22 = _mm512_fmadd_ps(kd2, qv2, s22);
                            if (chunks >= 4)
                                s23 = _mm512_fmadd_ps(kd3, qv2, s23);
                        }
                        if (R > 3) {
                            __m512 qv3 = _mm512_set1_ps(q3[d0]);
                            s30 = _mm512_fmadd_ps(kd0, qv3, s30);
                            if (chunks >= 2)
                                s31 = _mm512_fmadd_ps(kd1, qv3, s31);
                            if (chunks >= 3)
                                s32 = _mm512_fmadd_ps(kd2, qv3, s32);
                            if (chunks >= 4)
                                s33 = _mm512_fmadd_ps(kd3, qv3, s33);
                        }
                    }

                    // scale
                    s00 = _mm512_mul_ps(s00, vscale);
                    if (chunks >= 2)
                        s01 = _mm512_mul_ps(s01, vscale);
                    if (chunks >= 3)
                        s02 = _mm512_mul_ps(s02, vscale);
                    if (chunks >= 4)
                        s03 = _mm512_mul_ps(s03, vscale);

                    if (R > 1) {
                        s10 = _mm512_mul_ps(s10, vscale);
                        if (chunks >= 2)
                            s11 = _mm512_mul_ps(s11, vscale);
                        if (chunks >= 3)
                            s12 = _mm512_mul_ps(s12, vscale);
                        if (chunks >= 4)
                            s13 = _mm512_mul_ps(s13, vscale);
                    }
                    if (R > 2) {
                        s20 = _mm512_mul_ps(s20, vscale);
                        if (chunks >= 2)
                            s21 = _mm512_mul_ps(s21, vscale);
                        if (chunks >= 3)
                            s22 = _mm512_mul_ps(s22, vscale);
                        if (chunks >= 4)
                            s23 = _mm512_mul_ps(s23, vscale);
                    }
                    if (R > 3) {
                        s30 = _mm512_mul_ps(s30, vscale);
                        if (chunks >= 2)
                            s31 = _mm512_mul_ps(s31, vscale);
                        if (chunks >= 3)
                            s32 = _mm512_mul_ps(s32, vscale);
                        if (chunks >= 4)
                            s33 = _mm512_mul_ps(s33, vscale);
                    }

                    // block max per query
                    auto blkmax4 = [&](const __m512 a0, const __m512 a1,
                                       const __m512 a2, const __m512 a3,
                                       int ch) -> float {
                        float bm = -INFINITY;
                        bm = std::max(bm, hmax_ps(a0));
                        if (ch >= 2)
                            bm = std::max(bm, hmax_ps(a1));
                        if (ch >= 3)
                            bm = std::max(bm, hmax_ps(a2));
                        if (ch >= 4)
                            bm = std::max(bm, hmax_ps(a3));
                        return bm;
                    };
                    const float bm0 = blkmax4(s00, s01, s02, s03, chunks);
                    const float bm1 = (R > 1)
                                          ? blkmax4(s10, s11, s12, s13, chunks)
                                          : -INFINITY;
                    const float bm2 = (R > 2)
                                          ? blkmax4(s20, s21, s22, s23, chunks)
                                          : -INFINITY;
                    const float bm3 = (R > 3)
                                          ? blkmax4(s30, s31, s32, s33, chunks)
                                          : -INFINITY;

                    const float m0_new = std::max(m0, bm0);
                    const float m1_new = (R > 1) ? std::max(m1, bm1) : m0_new;
                    const float m2_new = (R > 2) ? std::max(m2, bm2) : m0_new;
                    const float m3_new = (R > 3) ? std::max(m3, bm3) : m0_new;

                    const float a0 = std::exp(m0 - m0_new);
                    const float a1 = (R > 1) ? std::exp(m1 - m1_new) : a0;
                    const float a2 = (R > 2) ? std::exp(m2 - m2_new) : a0;
                    const float a3 = (R > 3) ? std::exp(m3 - m3_new) : a0;

                    acc0 = _mm512_mul_ps(acc0, _mm512_set1_ps(a0));
                    if (R > 1)
                        acc1 = _mm512_mul_ps(acc1, _mm512_set1_ps(a1));
                    if (R > 2)
                        acc2 = _mm512_mul_ps(acc2, _mm512_set1_ps(a2));
                    if (R > 3)
                        acc3 = _mm512_mul_ps(acc3, _mm512_set1_ps(a3));

                    // weights (vs per-query m_new)
                    __m512 w00 =
                        exp512_ps(_mm512_sub_ps(s00, _mm512_set1_ps(m0_new)));
                    __m512 w01 = (chunks >= 2)
                                     ? exp512_ps(_mm512_sub_ps(
                                           s01, _mm512_set1_ps(m0_new)))
                                     : _mm512_setzero_ps();
                    __m512 w02 = (chunks >= 3)
                                     ? exp512_ps(_mm512_sub_ps(
                                           s02, _mm512_set1_ps(m0_new)))
                                     : _mm512_setzero_ps();
                    __m512 w03 = (chunks >= 4)
                                     ? exp512_ps(_mm512_sub_ps(
                                           s03, _mm512_set1_ps(m0_new)))
                                     : _mm512_setzero_ps();

                    __m512 w10 = _mm512_setzero_ps(), w11 = _mm512_setzero_ps(),
                           w12 = _mm512_setzero_ps(), w13 = _mm512_setzero_ps();
                    __m512 w20 = _mm512_setzero_ps(), w21 = _mm512_setzero_ps(),
                           w22 = _mm512_setzero_ps(), w23 = _mm512_setzero_ps();
                    __m512 w30 = _mm512_setzero_ps(), w31 = _mm512_setzero_ps(),
                           w32 = _mm512_setzero_ps(), w33 = _mm512_setzero_ps();
                    if (R > 1) {
                        w10 = exp512_ps(
                            _mm512_sub_ps(s10, _mm512_set1_ps(m1_new)));
                        if (chunks >= 2)
                            w11 = exp512_ps(
                                _mm512_sub_ps(s11, _mm512_set1_ps(m1_new)));
                        if (chunks >= 3)
                            w12 = exp512_ps(
                                _mm512_sub_ps(s12, _mm512_set1_ps(m1_new)));
                        if (chunks >= 4)
                            w13 = exp512_ps(
                                _mm512_sub_ps(s13, _mm512_set1_ps(m1_new)));
                    }
                    if (R > 2) {
                        w20 = exp512_ps(
                            _mm512_sub_ps(s20, _mm512_set1_ps(m2_new)));
                        if (chunks >= 2)
                            w21 = exp512_ps(
                                _mm512_sub_ps(s21, _mm512_set1_ps(m2_new)));
                        if (chunks >= 3)
                            w22 = exp512_ps(
                                _mm512_sub_ps(s22, _mm512_set1_ps(m2_new)));
                        if (chunks >= 4)
                            w23 = exp512_ps(
                                _mm512_sub_ps(s23, _mm512_set1_ps(m2_new)));
                    }
                    if (R > 3) {
                        w30 = exp512_ps(
                            _mm512_sub_ps(s30, _mm512_set1_ps(m3_new)));
                        if (chunks >= 2)
                            w31 = exp512_ps(
                                _mm512_sub_ps(s31, _mm512_set1_ps(m3_new)));
                        if (chunks >= 3)
                            w32 = exp512_ps(
                                _mm512_sub_ps(s32, _mm512_set1_ps(m3_new)));
                        if (chunks >= 4)
                            w33 = exp512_ps(
                                _mm512_sub_ps(s33, _mm512_set1_ps(m3_new)));
                    }

                    float l_blk0 = hsum_ps(w00);
                    if (chunks >= 2)
                        l_blk0 += hsum_ps(w01);
                    if (chunks >= 3)
                        l_blk0 += hsum_ps(w02);
                    if (chunks >= 4)
                        l_blk0 += hsum_ps(w03);

                    float l_blk1 = 0.f, l_blk2 = 0.f, l_blk3 = 0.f;
                    if (R > 1) {
                        l_blk1 = hsum_ps(w10);
                        if (chunks >= 2)
                            l_blk1 += hsum_ps(w11);
                        if (chunks >= 3)
                            l_blk1 += hsum_ps(w12);
                        if (chunks >= 4)
                            l_blk1 += hsum_ps(w13);
                    }
                    if (R > 2) {
                        l_blk2 = hsum_ps(w20);
                        if (chunks >= 2)
                            l_blk2 += hsum_ps(w21);
                        if (chunks >= 3)
                            l_blk2 += hsum_ps(w22);
                        if (chunks >= 4)
                            l_blk2 += hsum_ps(w23);
                    }
                    if (R > 3) {
                        l_blk3 = hsum_ps(w30);
                        if (chunks >= 2)
                            l_blk3 += hsum_ps(w31);
                        if (chunks >= 3)
                            l_blk3 += hsum_ps(w32);
                        if (chunks >= 4)
                            l_blk3 += hsum_ps(w33);
                    }

                    // ---- FIX: store weights for ALL queries, not just
                    // q0/q1
                    alignas(64) float wq[4][4][16]; // [query][group][lane]
                    _mm512_store_ps(wq[0][0], w00);
                    if (chunks >= 2)
                        _mm512_store_ps(wq[0][1], w01);
                    if (chunks >= 3)
                        _mm512_store_ps(wq[0][2], w02);
                    if (chunks >= 4)
                        _mm512_store_ps(wq[0][3], w03);

                    if (R > 1) {
                        _mm512_store_ps(wq[1][0], w10);
                        if (chunks >= 2)
                            _mm512_store_ps(wq[1][1], w11);
                        if (chunks >= 3)
                            _mm512_store_ps(wq[1][2], w12);
                        if (chunks >= 4)
                            _mm512_store_ps(wq[1][3], w13);
                    }
                    if (R > 2) {
                        _mm512_store_ps(wq[2][0], w20);
                        if (chunks >= 2)
                            _mm512_store_ps(wq[2][1], w21);
                        if (chunks >= 3)
                            _mm512_store_ps(wq[2][2], w22);
                        if (chunks >= 4)
                            _mm512_store_ps(wq[2][3], w23);
                    }
                    if (R > 3) {
                        _mm512_store_ps(wq[3][0], w30);
                        if (chunks >= 2)
                            _mm512_store_ps(wq[3][1], w31);
                        if (chunks >= 3)
                            _mm512_store_ps(wq[3][2], w32);
                        if (chunks >= 4)
                            _mm512_store_ps(wq[3][3], w33);
                    }

                    // accumulate V for this half-block (load V once, FMA
                    // into 4 accs)
                    // ---- FIX: pass the correct weight arrays per query
                    auto fma_keys = [&](int group_idx, int count) {
                        const int off = group_idx * 16;
                        for (int l = 0; l < count; ++l) {
                            const float *vrow =
                                Vh.data() + ((size_t)h * T + (base + off + l)) *
                                                Dh; // Dh=16
                            __m512 vv = _mm512_loadu_ps(vrow);
                            acc0 = _mm512_fmadd_ps(
                                vv, _mm512_set1_ps(wq[0][group_idx][l]), acc0);
                            if (R > 1)
                                acc1 = _mm512_fmadd_ps(
                                    vv, _mm512_set1_ps(wq[1][group_idx][l]),
                                    acc1);
                            if (R > 2)
                                acc2 = _mm512_fmadd_ps(
                                    vv, _mm512_set1_ps(wq[2][group_idx][l]),
                                    acc2);
                            if (R > 3)
                                acc3 = _mm512_fmadd_ps(
                                    vv, _mm512_set1_ps(wq[3][group_idx][l]),
                                    acc3);
                        }
                    };

                    const int c0 = std::min(16, left - 0);
                    const int c1 = std::max(0, std::min(16, left - 16));
                    const int c2 = std::max(0, std::min(16, left - 32));
                    const int c3 = std::max(0, std::min(16, left - 48));
                    if (c0)
                        fma_keys(0, c0);
                    if (c1)
                        fma_keys(1, c1);
                    if (c2)
                        fma_keys(2, c2);
                    if (c3)
                        fma_keys(3, c3);

                    // update running stats
                    l0 = l0 * a0 + l_blk0;
                    m0 = m0_new;
                    if (R > 1) {
                        l1 = l1 * a1 + l_blk1;
                        m1 = m1_new;
                    }
                    if (R > 2) {
                        l2 = l2 * a2 + l_blk2;
                        m2 = m2_new;
                    }
                    if (R > 3) {
                        l3 = l3 * a3 + l_blk3;
                        m3 = m3_new;
                    }
                } // half-block
            } // blocks

            // Normalize and store contexts for these queries
            __m512 inv0 = _mm512_set1_ps(1.0f / l0);
            _mm512_storeu_ps(ctx0 + h * Dh, _mm512_mul_ps(acc0, inv0));
            if (R > 1) {
                __m512 inv1 = _mm512_set1_ps(1.0f / l1);
                _mm512_storeu_ps(ctx1 + h * Dh, _mm512_mul_ps(acc1, inv1));
            }
            if (R > 2) {
                __m512 inv2 = _mm512_set1_ps(1.0f / l2);
                _mm512_storeu_ps(ctx2 + h * Dh, _mm512_mul_ps(acc2, inv2));
            }
            if (R > 3) {
                __m512 inv3 = _mm512_set1_ps(1.0f / l3);
                _mm512_storeu_ps(ctx3 + h * Dh, _mm512_mul_ps(acc3, inv3));
            }
        } // heads
    } // t0
}

} // namespace attn::kernels
