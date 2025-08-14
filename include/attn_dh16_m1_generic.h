#pragma once
#include "common.h"
#include "simd_math.h"
#include <vector>

namespace attn::kernels {

ATTN_ALWAYS_INLINE void run_dh16_m1_online(
  const std::vector<float>& Qh, const std::vector<float>& KhT, const std::vector<float>& Vh,
  int T, int H, int D, int Dh, bool causal, std::vector<float>& Ctx) {

    const int BK = 128;
    const __m512 vscale = _mm512_set1_ps(1.0f / std::sqrt(16.0f));

    for (int t = 0; t < T; ++t) {
            float *ctx_t =
                Ctx.data() +
                (size_t)t *
                    D; // we will overwrite all D entries; no need to clear
            const int valid_len = causal ? (t + 1) : T;

            for (int h = 0; h < H; ++h) {
                const float *q = Qh.data() + ((size_t)h * T + t) * Dh; // 16
                __m512 qv = _mm512_loadu_ps(q);

                float m = -std::numeric_limits<float>::infinity();
                float l = 0.0f;
                __m512 acc = _mm512_setzero_ps(); // Dh=16 accumulator

                for (int j0 = 0; j0 < valid_len; j0 += BK) {
                    const int take = std::min(BK, valid_len - j0);

                    // 1) Accumulate logits for the whole block into s[0..7]
                    // (each __m512 = 16 keys)
                    __m512 s[8];
                    s[0] = s[1] = s[2] = s[3] = s[4] = s[5] = s[6] = s[7] =
                        _mm512_setzero_ps();

                    // masks for ragged tail
                    const int c0 = std::min(16, take - 0);
                    const __mmask16 m0 =
                        (__mmask16)((c0 > 0) ? ((1u << c0) - 1) : 0);
                    const int c1 = std::min(16, take - 16);
                    const __mmask16 m1 =
                        (__mmask16)((c1 > 0) ? ((1u << c1) - 1) : 0);
                    const int c2 = std::min(16, take - 32);
                    const __mmask16 m2 =
                        (__mmask16)((c2 > 0) ? ((1u << c2) - 1) : 0);
                    const int c3 = std::min(16, take - 48);
                    const __mmask16 m3 =
                        (__mmask16)((c3 > 0) ? ((1u << c3) - 1) : 0);
                    const int c4 = std::min(16, take - 64);
                    const __mmask16 m4 =
                        (__mmask16)((c4 > 0) ? ((1u << c4) - 1) : 0);
                    const int c5 = std::min(16, take - 80);
                    const __mmask16 m5 =
                        (__mmask16)((c5 > 0) ? ((1u << c5) - 1) : 0);
                    const int c6 = std::min(16, take - 96);
                    const __mmask16 m6 =
                        (__mmask16)((c6 > 0) ? ((1u << c6) - 1) : 0);
                    const int c7 = std::min(16, take - 112);
                    const __mmask16 m7 =
                        (__mmask16)((c7 > 0) ? ((1u << c7) - 1) : 0);

#pragma unroll(16)
                    for (int d0 = 0; d0 < 16; ++d0) {
                        const float qd = q[d0];
                        const float *kt =
                            KhT.data() + ((size_t)h * Dh + d0) * T + j0;
                        if (m0) {
                            __m512 kd = _mm512_maskz_loadu_ps(m0, kt + 0);
                            s[0] =
                                _mm512_fmadd_ps(kd, _mm512_set1_ps(qd), s[0]);
                        }
                        if (m1) {
                            __m512 kd = _mm512_maskz_loadu_ps(m1, kt + 16);
                            s[1] =
                                _mm512_fmadd_ps(kd, _mm512_set1_ps(qd), s[1]);
                        }
                        if (m2) {
                            __m512 kd = _mm512_maskz_loadu_ps(m2, kt + 32);
                            s[2] =
                                _mm512_fmadd_ps(kd, _mm512_set1_ps(qd), s[2]);
                        }
                        if (m3) {
                            __m512 kd = _mm512_maskz_loadu_ps(m3, kt + 48);
                            s[3] =
                                _mm512_fmadd_ps(kd, _mm512_set1_ps(qd), s[3]);
                        }
                        if (m4) {
                            __m512 kd = _mm512_maskz_loadu_ps(m4, kt + 64);
                            s[4] =
                                _mm512_fmadd_ps(kd, _mm512_set1_ps(qd), s[4]);
                        }
                        if (m5) {
                            __m512 kd = _mm512_maskz_loadu_ps(m5, kt + 80);
                            s[5] =
                                _mm512_fmadd_ps(kd, _mm512_set1_ps(qd), s[5]);
                        }
                        if (m6) {
                            __m512 kd = _mm512_maskz_loadu_ps(m6, kt + 96);
                            s[6] =
                                _mm512_fmadd_ps(kd, _mm512_set1_ps(qd), s[6]);
                        }
                        if (m7) {
                            __m512 kd = _mm512_maskz_loadu_ps(m7, kt + 112);
                            s[7] =
                                _mm512_fmadd_ps(kd, _mm512_set1_ps(qd), s[7]);
                        }
                    }

                    // scale all present groups
                    if (m0)
                        s[0] = _mm512_mul_ps(s[0], vscale);
                    if (m1)
                        s[1] = _mm512_mul_ps(s[1], vscale);
                    if (m2)
                        s[2] = _mm512_mul_ps(s[2], vscale);
                    if (m3)
                        s[3] = _mm512_mul_ps(s[3], vscale);
                    if (m4)
                        s[4] = _mm512_mul_ps(s[4], vscale);
                    if (m5)
                        s[5] = _mm512_mul_ps(s[5], vscale);
                    if (m6)
                        s[6] = _mm512_mul_ps(s[6], vscale);
                    if (m7)
                        s[7] = _mm512_mul_ps(s[7], vscale);

                    // 2) One block max across all present lanes
                    float block_max = -std::numeric_limits<float>::infinity();
                    if (m0)
                        block_max = std::max(
                            block_max, hmax_ps(_mm512_maskz_mov_ps(m0, s[0])));
                    if (m1)
                        block_max = std::max(
                            block_max, hmax_ps(_mm512_maskz_mov_ps(m1, s[1])));
                    if (m2)
                        block_max = std::max(
                            block_max, hmax_ps(_mm512_maskz_mov_ps(m2, s[2])));
                    if (m3)
                        block_max = std::max(
                            block_max, hmax_ps(_mm512_maskz_mov_ps(m3, s[3])));
                    if (m4)
                        block_max = std::max(
                            block_max, hmax_ps(_mm512_maskz_mov_ps(m4, s[4])));
                    if (m5)
                        block_max = std::max(
                            block_max, hmax_ps(_mm512_maskz_mov_ps(m5, s[5])));
                    if (m6)
                        block_max = std::max(
                            block_max, hmax_ps(_mm512_maskz_mov_ps(m6, s[6])));
                    if (m7)
                        block_max = std::max(
                            block_max, hmax_ps(_mm512_maskz_mov_ps(m7, s[7])));

                    const float m_new = std::max(m, block_max);
                    const float alpha = std::exp(m - m_new);

                    // 3) Weights vs m_new for all lanes; accumulate denom and
                    // context
                    __m512 w[8];
                    float l_blk = 0.0f;
                    if (m0) {
                        w[0] = exp512_ps(
                            _mm512_sub_ps(s[0], _mm512_set1_ps(m_new)));
                        l_blk += hsum_ps(_mm512_maskz_mov_ps(m0, w[0]));
                    }
                    if (m1) {
                        w[1] = exp512_ps(
                            _mm512_sub_ps(s[1], _mm512_set1_ps(m_new)));
                        l_blk += hsum_ps(_mm512_maskz_mov_ps(m1, w[1]));
                    }
                    if (m2) {
                        w[2] = exp512_ps(
                            _mm512_sub_ps(s[2], _mm512_set1_ps(m_new)));
                        l_blk += hsum_ps(_mm512_maskz_mov_ps(m2, w[2]));
                    }
                    if (m3) {
                        w[3] = exp512_ps(
                            _mm512_sub_ps(s[3], _mm512_set1_ps(m_new)));
                        l_blk += hsum_ps(_mm512_maskz_mov_ps(m3, w[3]));
                    }
                    if (m4) {
                        w[4] = exp512_ps(
                            _mm512_sub_ps(s[4], _mm512_set1_ps(m_new)));
                        l_blk += hsum_ps(_mm512_maskz_mov_ps(m4, w[4]));
                    }
                    if (m5) {
                        w[5] = exp512_ps(
                            _mm512_sub_ps(s[5], _mm512_set1_ps(m_new)));
                        l_blk += hsum_ps(_mm512_maskz_mov_ps(m5, w[5]));
                    }
                    if (m6) {
                        w[6] = exp512_ps(
                            _mm512_sub_ps(s[6], _mm512_set1_ps(m_new)));
                        l_blk += hsum_ps(_mm512_maskz_mov_ps(m6, w[6]));
                    }
                    if (m7) {
                        w[7] = exp512_ps(
                            _mm512_sub_ps(s[7], _mm512_set1_ps(m_new)));
                        l_blk += hsum_ps(_mm512_maskz_mov_ps(m7, w[7]));
                    }

                    // scale previous accumulators
                    acc = _mm512_mul_ps(acc, _mm512_set1_ps(alpha));

                    // accumulate V with the new weights
                    alignas(64) float wb[8][16];
                    if (m0)
                        _mm512_store_ps(wb[0], w[0]);
                    if (m1)
                        _mm512_store_ps(wb[1], w[1]);
                    if (m2)
                        _mm512_store_ps(wb[2], w[2]);
                    if (m3)
                        _mm512_store_ps(wb[3], w[3]);
                    if (m4)
                        _mm512_store_ps(wb[4], w[4]);
                    if (m5)
                        _mm512_store_ps(wb[5], w[5]);
                    if (m6)
                        _mm512_store_ps(wb[6], w[6]);
                    if (m7)
                        _mm512_store_ps(wb[7], w[7]);

                    auto fma_group = [&](int base, const float *wlane,
                                         int cnt) {
                        for (int l = 0; l < cnt; ++l) {
                            const float *vrow =
                                Vh.data() +
                                ((size_t)h * T + (j0 + base + l)) * Dh; // Dh=16
                            __m512 vv = _mm512_loadu_ps(vrow);
                            acc = _mm512_fmadd_ps(vv, _mm512_set1_ps(wlane[l]),
                                                  acc);
                        }
                    };
                    if (c0)
                        fma_group(0, wb[0], c0);
                    if (c1)
                        fma_group(16, wb[1], c1);
                    if (c2)
                        fma_group(32, wb[2], c2);
                    if (c3)
                        fma_group(48, wb[3], c3);
                    if (c4)
                        fma_group(64, wb[4], c4);
                    if (c5)
                        fma_group(80, wb[5], c5);
                    if (c6)
                        fma_group(96, wb[6], c6);
                    if (c7)
                        fma_group(112, wb[7], c7);

                    // update running stats
                    l = l * alpha + l_blk;
                    m = m_new;
                }

                const __m512 inv_l = _mm512_set1_ps(1.0f / l);
                __m512 outv = _mm512_mul_ps(acc, inv_l);
                _mm512_storeu_ps(ctx_t + h * Dh, outv); // Dh==16
            } // heads
        } // t
  // === paste the per-(t,h) Dh=16 path you had before M=4 batching ===
  // (the corrected full-block max version)
}

} // namespace attn::kernels




