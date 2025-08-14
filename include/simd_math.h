#pragma once
#include "common.h"
#include <immintrin.h>
namespace attn {
ATTN_ALWAYS_INLINE float hsum_ps(__m512 v) { return _mm512_reduce_add_ps(v); }
ATTN_ALWAYS_INLINE float hmax_ps(__m512 v) { return _mm512_reduce_max_ps(v); }
ATTN_ALWAYS_INLINE __m512 exp512_ps(__m512 x) {
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
} // namespace attn
