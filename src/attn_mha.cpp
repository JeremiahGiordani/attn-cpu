#include "attn.h"
#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace attn {

#if !defined(ATTN_USE_AVX512)
#error "Compile with ATTN_USE_AVX512 and AVX-512 flags."
#endif

// --------- AVX-512 helpers ----------
static inline float hsum_ps(__m512 v) {
    return _mm512_reduce_add_ps(v);
}
static inline float hmax_ps(__m512 v) {
    return _mm512_reduce_max_ps(v);
}

// Fast exp for softmax
static inline __m512 exp512_ps(__m512 x) {
    const __m512 exp_hi = _mm512_set1_ps(88.3762626647949f);
    const __m512 exp_lo = _mm512_set1_ps(-88.3762626647949f);
    const __m512 log2ef = _mm512_set1_ps(1.44269504088896341f);
    const __m512 c1 = _mm512_set1_ps(0.693359375f);
    const __m512 c2 = _mm512_set1_ps(-2.12194440e-4f);

    const __m512 p0 = _mm512_set1_ps(1.9875691500e-4f);
    const __m512 p1 = _mm512_set1_ps(1.3981999507e-3f);
    const __m512 p2 = _mm512_set1_ps(8.3334519073e-3f);
    const __m512 p3 = _mm512_set1_ps(4.1665795894e-2f);
    const __m512 p4 = _mm512_set1_ps(1.6666665459e-1f);
    const __m512 p5 = _mm512_set1_ps(5.0000001201e-1f);

    x = _mm512_min_ps(x, exp_hi);
    x = _mm512_max_ps(x, exp_lo);

    __m512 fx = _mm512_fmadd_ps(x, log2ef, _mm512_set1_ps(0.5f));
    fx = _mm512_floor_ps(fx);

    __m512 tmp = _mm512_fnmadd_ps(fx, c1, x);
    x = _mm512_fnmadd_ps(fx, c2, tmp);

    __m512 z = _mm512_mul_ps(x, x);
    __m512 y = p0;
    y = _mm512_fmadd_ps(y, x, p1);
    y = _mm512_fmadd_ps(y, x, p2);
    y = _mm512_fmadd_ps(y, x, p3);
    y = _mm512_fmadd_ps(y, x, p4);
    y = _mm512_fmadd_ps(y, x, p5);
    y = _mm512_fmadd_ps(y, z, x);
    y = _mm512_add_ps(y, _mm512_set1_ps(1.0f));

    __m512i emm0 = _mm512_cvttps_epi32(fx);
    emm0 = _mm512_add_epi32(emm0, _mm512_set1_epi32(127));
    emm0 = _mm512_slli_epi32(emm0, 23);
    __m512 pow2n = _mm512_castsi512_ps(emm0);
    return _mm512_mul_ps(y, pow2n);
}

// --------- 16×16 packing (K,N multiples of 16) ----------
static inline void pack_wt_16x16(const float *Wt, int K, int N,
                                 std::vector<float> &packed) {
    if ((K % 16) || (N % 16))
        throw std::runtime_error("pack_wt_16x16: K,N must be multiples of 16");
    const int nb = N / 16, kb = K / 16;
    packed.resize((size_t)nb * kb * 256);
    size_t idx = 0;
    for (int nb_i = 0; nb_i < nb; ++nb_i) {
        const int n0 = nb_i * 16;
        for (int kb_i = 0; kb_i < kb; ++kb_i) {
            const int k0 = kb_i * 16;
            for (int kk = 0; kk < 16; ++kk) {
                const float *src = Wt + (size_t)(k0 + kk) * N + n0;
                std::memcpy(&packed[idx], src, 16 * sizeof(float));
                idx += 16;
            }
        }
    }
}

// --------- GEMM microkernels (N multiple of 16, K multiple of 16) ----------

// 1 row × 16 cols × K16 (fallback / tails of M4)
static inline void gemm_row1_packed_16x16(const float *x, const float *packedWt,
                                          const float *bias, int K, int N,
                                          float *y) {
    const int nb = N / 16, kb = K / 16;
    for (int nb_i = 0; nb_i < nb; ++nb_i) {
        __m512 acc = _mm512_loadu_ps(bias + nb_i * 16);
        const size_t base_nb = (size_t)nb_i * kb * 256;
        for (int kb_i = 0; kb_i < kb; ++kb_i) {
            const float *tile = packedWt + base_nb + (size_t)kb_i * 256;
            const float *xk = x + kb_i * 16;
#pragma unroll(16)
            for (int kk = 0; kk < 16; ++kk) {
                __m512 wv = _mm512_loadu_ps(tile + kk * 16);
                acc = _mm512_fmadd_ps(wv, _mm512_set1_ps(xk[kk]), acc);
            }
        }
        _mm512_storeu_ps(y + nb_i * 16, acc);
    }
}

// 4 rows × 16 cols × K16 (preferred)
static inline void gemm_row4_packed_16x16(const float *x0, const float *x1,
                                          const float *x2, const float *x3,
                                          const float *packedWt,
                                          const float *bias, int K, int N,
                                          float *y0, float *y1, float *y2,
                                          float *y3) {
    const int nb = N / 16, kb = K / 16;
    for (int nb_i = 0; nb_i < nb; ++nb_i) {
        __m512 acc0 = _mm512_loadu_ps(bias + nb_i * 16);
        __m512 acc1 = acc0;
        __m512 acc2 = acc0;
        __m512 acc3 = acc0;

        const size_t base_nb = (size_t)nb_i * kb * 256;
        for (int kb_i = 0; kb_i < kb; ++kb_i) {
            const float *tile = packedWt + base_nb + (size_t)kb_i * 256;
            const float *xk0 = x0 + kb_i * 16;
            const float *xk1 = x1 + kb_i * 16;
            const float *xk2 = x2 + kb_i * 16;
            const float *xk3 = x3 + kb_i * 16;
#pragma unroll(16)
            for (int kk = 0; kk < 16; ++kk) {
                __m512 wv = _mm512_loadu_ps(tile + kk * 16);
                acc0 = _mm512_fmadd_ps(wv, _mm512_set1_ps(xk0[kk]), acc0);
                acc1 = _mm512_fmadd_ps(wv, _mm512_set1_ps(xk1[kk]), acc1);
                acc2 = _mm512_fmadd_ps(wv, _mm512_set1_ps(xk2[kk]), acc2);
                acc3 = _mm512_fmadd_ps(wv, _mm512_set1_ps(xk3[kk]), acc3);
            }
        }
        _mm512_storeu_ps(y0 + nb_i * 16, acc0);
        _mm512_storeu_ps(y1 + nb_i * 16, acc1);
        _mm512_storeu_ps(y2 + nb_i * 16, acc2);
        _mm512_storeu_ps(y3 + nb_i * 16, acc3);
    }
}

// --------- Main kernel ----------
void mha_block_dense(const float *x, int T, int D, const float *W_in,
                     const float *b_in,                      // [3D, D], [3D]
                     const float *W_out, const float *b_out, // [D, D],  [D]
                     int H, bool causal, float *y_out) {
    if ((D % 16) || ((3 * D) % 16))
        throw std::runtime_error("D and 3D must be multiples of 16");
    if ((D % H) != 0)
        throw std::runtime_error("D must be divisible by H");

    const int Dh = D / H;
    if (Dh != 16)
        throw std::runtime_error("This optimized path expects Dh=16 (D/H==16)");
    const __m512 vscale = _mm512_set1_ps(1.0f / std::sqrt(16.0f));
    const int BK = 128; // key block size

    // Thread-local scratch
    static thread_local int S_T = -1, S_D = -1, S_H = -1;
    static thread_local std::vector<float> Qh;          // [H,T,Dh]
    static thread_local std::vector<float> KhT;         // [H,Dh,T]
    static thread_local std::vector<float> Vh;          // [H,T,Dh]
    static thread_local std::vector<float> Ctx;         // [T,D]
    static thread_local std::vector<float> Win_packed;  // packed W_in^T: [D,3D]
    static thread_local std::vector<float> Wout_packed; // packed W_out^T: [D,D]
    static thread_local std::vector<float> row3D_4;     // [4, 3D]
    static thread_local int packed_dims = -1;

    auto ensure = [&](int t, int d, int h) {
        if (S_T == t && S_D == d && S_H == h)
            return;
        S_T = t;
        S_D = d;
        S_H = h;
        Qh.assign((size_t)H * (size_t)T * (size_t)Dh, 0.0f);
        KhT.assign((size_t)H * (size_t)Dh * (size_t)T, 0.0f);
        Vh.assign((size_t)H * (size_t)T * (size_t)Dh, 0.0f);
        Ctx.assign((size_t)T * (size_t)D, 0.0f);
        row3D_4.assign((size_t)4 * (size_t)(3 * D), 0.0f);
        packed_dims = -1;
    };
    ensure(T, D, H);

    // --- Pack W_in^T and W_out^T (once per shape) ---
    if (packed_dims != D * 3) {
        // W_in: [3D,D] -> Wt_in: [D,3D]
        std::vector<float> Wt_in((size_t)D * (size_t)(3 * D));
        for (int r = 0; r < D; ++r)
            for (int c = 0; c < 3 * D; ++c)
                Wt_in[(size_t)r * (3 * D) + c] = W_in[(size_t)c * D + r];
        pack_wt_16x16(Wt_in.data(), D, 3 * D, Win_packed);

        // W_out: [D,D] -> Wt_out: [D,D]
        std::vector<float> Wt_out((size_t)D * (size_t)D);
        for (int r = 0; r < D; ++r)
            for (int c = 0; c < D; ++c)
                Wt_out[(size_t)r * D + c] = W_out[(size_t)c * D + r];
        pack_wt_16x16(Wt_out.data(), D, D, Wout_packed);

        packed_dims = D * 3;
    }

    // --- 1) Q/K/V via GEMM (M=4 microkernel) + direct scatter to head layouts
    // ---
    for (int t0 = 0; t0 < T; t0 += 4) {
        const int R = std::min(4, T - t0);
        float *y0 = row3D_4.data() + 0 * (3 * D);
        float *y1 = row3D_4.data() + 1 * (3 * D);
        float *y2 = row3D_4.data() + 2 * (3 * D);
        float *y3 = row3D_4.data() + 3 * (3 * D);

        const float *x0 = x + (size_t)(t0 + 0) * D;
        const float *x1 = (R > 1) ? x + (size_t)(t0 + 1) * D : x0;
        const float *x2 = (R > 2) ? x + (size_t)(t0 + 2) * D : x0;
        const float *x3 = (R > 3) ? x + (size_t)(t0 + 3) * D : x0;

        if (R == 4) {
            gemm_row4_packed_16x16(x0, x1, x2, x3, Win_packed.data(), b_in, D,
                                   3 * D, y0, y1, y2, y3);
        } else {
            // tails
            gemm_row1_packed_16x16(x0, Win_packed.data(), b_in, D, 3 * D, y0);
            if (R > 1)
                gemm_row1_packed_16x16(x1, Win_packed.data(), b_in, D, 3 * D,
                                       y1);
            if (R > 2)
                gemm_row1_packed_16x16(x2, Win_packed.data(), b_in, D, 3 * D,
                                       y2);
            if (R > 3)
                gemm_row1_packed_16x16(x3, Win_packed.data(), b_in, D, 3 * D,
                                       y3);
        }

        // Scatter each produced row directly into Qh/KhT/Vh (Dh=16, H=D/16)
        auto scatter_row = [&](int t_idx, const float *row) {
            const float *qrow = row + 0 * D;
            const float *krow = row + 1 * D;
            const float *vrow = row + 2 * D;
            for (int h = 0; h < H; ++h) {
                // Qh[h,t,:]
                std::memcpy(Qh.data() + ((size_t)h * T + t_idx) * Dh,
                            qrow + h * Dh, Dh * sizeof(float));
                // Vh[h,t,:]
                std::memcpy(Vh.data() + ((size_t)h * T + t_idx) * Dh,
                            vrow + h * Dh, Dh * sizeof(float));
                // KhT[h,:,t]
                const float *ks = krow + h * Dh;
                float *kdst = KhT.data() + ((size_t)h * Dh + 0) * T + t_idx;
#pragma unroll(16)
                for (int d0 = 0; d0 < Dh; ++d0) {
                    kdst[d0 * (size_t)T] = ks[d0];
                }
            }
        };

        scatter_row(t0 + 0, y0);
        if (R > 1)
            scatter_row(t0 + 1, y1);
        if (R > 2)
            scatter_row(t0 + 2, y2);
        if (R > 3)
            scatter_row(t0 + 3, y3);
    }

    // --- 2) Attention (Dh=16 specialized), online softmax, per (t,h) ---
    if (!causal && Dh == 16) {
        // Non-causal fast path: process 4 queries at a time per head, reuse K/V
        // tiles.
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
                        __m512 s00 = _mm512_setzero_ps(),
                               s01 = _mm512_setzero_ps(),
                               s02 = _mm512_setzero_ps(),
                               s03 = _mm512_setzero_ps(); // q0
                        __m512 s10 = _mm512_setzero_ps(),
                               s11 = _mm512_setzero_ps(),
                               s12 = _mm512_setzero_ps(),
                               s13 = _mm512_setzero_ps(); // q1
                        __m512 s20 = _mm512_setzero_ps(),
                               s21 = _mm512_setzero_ps(),
                               s22 = _mm512_setzero_ps(),
                               s23 = _mm512_setzero_ps(); // q2
                        __m512 s30 = _mm512_setzero_ps(),
                               s31 = _mm512_setzero_ps(),
                               s32 = _mm512_setzero_ps(),
                               s33 = _mm512_setzero_ps(); // q3

// accumulate logits for this half-block
#pragma unroll(16)
                        for (int d0 = 0; d0 < 16; ++d0) {
                            const float *kt =
                                KhT.data() + ((size_t)h * Dh + d0) * T + base;
                            __m512 kd0 = (chunks >= 1) ? _mm512_loadu_ps(kt + 0)
                                                       : _mm512_setzero_ps();
                            __m512 kd1 = (chunks >= 2)
                                             ? _mm512_loadu_ps(kt + 16)
                                             : _mm512_setzero_ps();
                            __m512 kd2 = (chunks >= 3)
                                             ? _mm512_loadu_ps(kt + 32)
                                             : _mm512_setzero_ps();
                            __m512 kd3 = (chunks >= 4)
                                             ? _mm512_loadu_ps(kt + 48)
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
                        const float bm1 =
                            (R > 1) ? blkmax4(s10, s11, s12, s13, chunks)
                                    : -INFINITY;
                        const float bm2 =
                            (R > 2) ? blkmax4(s20, s21, s22, s23, chunks)
                                    : -INFINITY;
                        const float bm3 =
                            (R > 3) ? blkmax4(s30, s31, s32, s33, chunks)
                                    : -INFINITY;

                        const float m0_new = std::max(m0, bm0);
                        const float m1_new =
                            (R > 1) ? std::max(m1, bm1) : m0_new;
                        const float m2_new =
                            (R > 2) ? std::max(m2, bm2) : m0_new;
                        const float m3_new =
                            (R > 3) ? std::max(m3, bm3) : m0_new;

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
                        __m512 w00 = exp512_ps(
                            _mm512_sub_ps(s00, _mm512_set1_ps(m0_new)));
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

                        __m512 w10 = _mm512_setzero_ps(),
                               w11 = _mm512_setzero_ps(),
                               w12 = _mm512_setzero_ps(),
                               w13 = _mm512_setzero_ps();
                        __m512 w20 = _mm512_setzero_ps(),
                               w21 = _mm512_setzero_ps(),
                               w22 = _mm512_setzero_ps(),
                               w23 = _mm512_setzero_ps();
                        __m512 w30 = _mm512_setzero_ps(),
                               w31 = _mm512_setzero_ps(),
                               w32 = _mm512_setzero_ps(),
                               w33 = _mm512_setzero_ps();
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
                                    Vh.data() +
                                    ((size_t)h * T + (base + off + l)) *
                                        Dh; // Dh=16
                                __m512 vv = _mm512_loadu_ps(vrow);
                                acc0 = _mm512_fmadd_ps(
                                    vv, _mm512_set1_ps(wq[0][group_idx][l]),
                                    acc0);
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
    } else {
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
    }

    // --- 3) Output projection via GEMM (M=4 microkernel) ---
    for (int t0 = 0; t0 < T; t0 += 4) {
        const int R = std::min(4, T - t0);
        const float *c0 = Ctx.data() + (size_t)(t0 + 0) * D;
        const float *c1 = (R > 1) ? Ctx.data() + (size_t)(t0 + 1) * D : c0;
        const float *c2 = (R > 2) ? Ctx.data() + (size_t)(t0 + 2) * D : c0;
        const float *c3 = (R > 3) ? Ctx.data() + (size_t)(t0 + 3) * D : c0;

        float *y0 = y_out + (size_t)(t0 + 0) * D;
        float *y1 = y_out + (size_t)(t0 + 1) * D;
        float *y2 = y_out + (size_t)(t0 + 2) * D;
        float *y3 = y_out + (size_t)(t0 + 3) * D;

        if (R == 4) {
            gemm_row4_packed_16x16(c0, c1, c2, c3, Wout_packed.data(), b_out, D,
                                   D, y0, y1, y2, y3);
        } else {
            gemm_row1_packed_16x16(c0, Wout_packed.data(), b_out, D, D, y0);
            if (R > 1)
                gemm_row1_packed_16x16(c1, Wout_packed.data(), b_out, D, D, y1);
            if (R > 2)
                gemm_row1_packed_16x16(c2, Wout_packed.data(), b_out, D, D, y2);
            if (R > 3)
                gemm_row1_packed_16x16(c3, Wout_packed.data(), b_out, D, D, y3);
        }
    }
}

} // namespace attn