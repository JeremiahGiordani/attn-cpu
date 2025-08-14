#pragma once
#include "common.h"
#include "simd_math.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <immintrin.h>
#include <vector>

namespace attn::kernels {

// Fast path: NON-CAUSAL, Dh = k*16, M=4 queries at a time.
// Qh:  [H,T,Dh], KhT:[H,Dh,T], Vh:[H,T,Dh], Ctx:[T,D]
// Works best when Dh is a multiple of 16 (e.g., 16/32/48/...), up to 16*ATTN_MAX_SEG.
#ifndef ATTN_MAX_SEG
#define ATTN_MAX_SEG 16  // supports Dh up to 256 by default
#endif

ATTN_ALWAYS_INLINE void run_dhk16_m4_nocausal(
    const std::vector<float>& Qh,
    const std::vector<float>& KhT,
    const std::vector<float>& Vh,
    int T, int H, int D, int Dh,
    std::vector<float>& Ctx)
{
  (void)D; // currently unused here; kept for signature consistency

  const int S = Dh / 16;
  if (ATTN_UNLIKELY(Dh % 16 != 0 || S > ATTN_MAX_SEG)) {
    return; // dispatcher shouldn't call us in this case
  }

  const __m512 vscale_all = _mm512_set1_ps(1.0f / std::sqrt((float)Dh));
  const int BK = 128;

  const __m512 NEG_INF = _mm512_set1_ps(-std::numeric_limits<float>::infinity());

  for (int t0 = 0; t0 < T; t0 += 4) {
    const int R = std::min(4, T - t0);

    float* ctx0 = Ctx.data() + (size_t)(t0+0) * D;
    float* ctx1 = (R>1) ? Ctx.data() + (size_t)(t0+1) * D : ctx0;
    float* ctx2 = (R>2) ? Ctx.data() + (size_t)(t0+2) * D : ctx0;
    float* ctx3 = (R>3) ? Ctx.data() + (size_t)(t0+3) * D : ctx0;

    for (int h = 0; h < H; ++h) {
      const float* q0 = Qh.data() + ((size_t)h*T + (t0+0))*Dh;
      const float* q1 = (R>1) ? Qh.data() + ((size_t)h*T + (t0+1))*Dh : q0;
      const float* q2 = (R>2) ? Qh.data() + ((size_t)h*T + (t0+2))*Dh : q0;
      const float* q3 = (R>3) ? Qh.data() + ((size_t)h*T + (t0+3))*Dh : q0;

      // Running stats per query (rename to avoid mask name collisions)
      float m0r=-INFINITY, m1r=-INFINITY, m2r=-INFINITY, m3r=-INFINITY;
      float l0=0.f,        l1=0.f,        l2=0.f,        l3=0.f;

      // Stack accumulators per 16-wide segment
      __m512 acc0[ATTN_MAX_SEG], acc1[ATTN_MAX_SEG], acc2[ATTN_MAX_SEG], acc3[ATTN_MAX_SEG];
      for (int seg=0; seg<S; ++seg) {
        acc0[seg] = _mm512_setzero_ps();
        acc1[seg] = _mm512_setzero_ps();
        acc2[seg] = _mm512_setzero_ps();
        acc3[seg] = _mm512_setzero_ps();
      }

      for (int j0 = 0; j0 < T; j0 += BK) {
        const int take = std::min(BK, T - j0);

        // Two 64-key halves
        for (int half = 0; half < 2 && half*64 < take; ++half) {
          const int base   = j0 + half*64;
          const int left   = take - half*64;
          const int groups = std::min(4, (left + 15) / 16);

          // Tail counts + masks (rename masks to avoid shadowing)
          const int c0 = std::min(16, left -  0);
          const int c1 = std::max(0, std::min(16, left - 16));
          const int c2 = std::max(0, std::min(16, left - 32));
          const int c3 = std::max(0, std::min(16, left - 48));
          const __mmask16 mask0 = (__mmask16)((c0>0)?((1u<<c0)-1):0);
          const __mmask16 mask1 = (__mmask16)((c1>0)?((1u<<c1)-1):0);
          const __mmask16 mask2 = (__mmask16)((c2>0)?((1u<<c2)-1):0);
          const __mmask16 mask3 = (__mmask16)((c3>0)?((1u<<c3)-1):0);

          // s[g][q] logits
          __m512 s00=_mm512_setzero_ps(), s01=_mm512_setzero_ps(),
                 s02=_mm512_setzero_ps(), s03=_mm512_setzero_ps();
          __m512 s10=_mm512_setzero_ps(), s11=_mm512_setzero_ps(),
                 s12=_mm512_setzero_ps(), s13=_mm512_setzero_ps();
          __m512 s20=_mm512_setzero_ps(), s21=_mm512_setzero_ps(),
                 s22=_mm512_setzero_ps(), s23=_mm512_setzero_ps();
          __m512 s30=_mm512_setzero_ps(), s31=_mm512_setzero_ps(),
                 s32=_mm512_setzero_ps(), s33=_mm512_setzero_ps();

          // Accumulate logits across Dh in 16-wide segments
          for (int seg = 0; seg < S; ++seg) {
            const int dbase = seg * 16;
#pragma unroll(16)
            for (int d0 = 0; d0 < 16; ++d0) {
              const int d = dbase + d0;
              const float* kt = KhT.data() + ((size_t)h*Dh + d)*T + base;

              __m512 kd0 = mask0 ? _mm512_maskz_loadu_ps(mask0, kt +  0) : _mm512_setzero_ps();
              __m512 kd1 = mask1 ? _mm512_maskz_loadu_ps(mask1, kt + 16) : _mm512_setzero_ps();
              __m512 kd2 = mask2 ? _mm512_maskz_loadu_ps(mask2, kt + 32) : _mm512_setzero_ps();
              __m512 kd3 = mask3 ? _mm512_maskz_loadu_ps(mask3, kt + 48) : _mm512_setzero_ps();

              __m512 qv0 = _mm512_set1_ps(q0[d]);
              s00 = _mm512_fmadd_ps(kd0, qv0, s00);
              if (groups>=2) s01 = _mm512_fmadd_ps(kd1, qv0, s01);
              if (groups>=3) s02 = _mm512_fmadd_ps(kd2, qv0, s02);
              if (groups>=4) s03 = _mm512_fmadd_ps(kd3, qv0, s03);

              if (R>1) {
                __m512 qv1 = _mm512_set1_ps(q1[d]);
                s10 = _mm512_fmadd_ps(kd0, qv1, s10);
                if (groups>=2) s11 = _mm512_fmadd_ps(kd1, qv1, s11);
                if (groups>=3) s12 = _mm512_fmadd_ps(kd2, qv1, s12);
                if (groups>=4) s13 = _mm512_fmadd_ps(kd3, qv1, s13);
              }
              if (R>2) {
                __m512 qv2 = _mm512_set1_ps(q2[d]);
                s20 = _mm512_fmadd_ps(kd0, qv2, s20);
                if (groups>=2) s21 = _mm512_fmadd_ps(kd1, qv2, s21);
                if (groups>=3) s22 = _mm512_fmadd_ps(kd2, qv2, s22);
                if (groups>=4) s23 = _mm512_fmadd_ps(kd3, qv2, s23);
              }
              if (R>3) {
                __m512 qv3 = _mm512_set1_ps(q3[d]);
                s30 = _mm512_fmadd_ps(kd0, qv3, s30);
                if (groups>=2) s31 = _mm512_fmadd_ps(kd1, qv3, s31);
                if (groups>=3) s32 = _mm512_fmadd_ps(kd2, qv3, s32);
                if (groups>=4) s33 = _mm512_fmadd_ps(kd3, qv3, s33);
              }
            }
          }

          // scale by 1/sqrt(Dh)
          s00=_mm512_mul_ps(s00, vscale_all); if (groups>=2) s01=_mm512_mul_ps(s01, vscale_all);
          if (groups>=3) s02=_mm512_mul_ps(s02, vscale_all); if (groups>=4) s03=_mm512_mul_ps(s03, vscale_all);
          if (R>1){ s10=_mm512_mul_ps(s10, vscale_all); if (groups>=2) s11=_mm512_mul_ps(s11, vscale_all);
                    if (groups>=3) s12=_mm512_mul_ps(s12, vscale_all); if (groups>=4) s13=_mm512_mul_ps(s13, vscale_all); }
          if (R>2){ s20=_mm512_mul_ps(s20, vscale_all); if (groups>=2) s21=_mm512_mul_ps(s21, vscale_all);
                    if (groups>=3) s22=_mm512_mul_ps(s22, vscale_all); if (groups>=4) s23=_mm512_mul_ps(s23, vscale_all); }
          if (R>3){ s30=_mm512_mul_ps(s30, vscale_all); if (groups>=2) s31=_mm512_mul_ps(s31, vscale_all);
                    if (groups>=3) s32=_mm512_mul_ps(s32, vscale_all); if (groups>=4) s33=_mm512_mul_ps(s33, vscale_all); }

          // masked hmax
          auto hmax_masked = [&](const __m512 s, __mmask16 k){ return attn::hmax_ps(_mm512_mask_mov_ps(NEG_INF, k, s)); };
          auto blkmax4 = [&](const __m512 a0,const __m512 a1,const __m512 a2,const __m512 a3)->float{
            float bm = hmax_masked(a0, mask0);
            if (groups>=2) bm = std::max(bm, hmax_masked(a1, mask1));
            if (groups>=3) bm = std::max(bm, hmax_masked(a2, mask2));
            if (groups>=4) bm = std::max(bm, hmax_masked(a3, mask3));
            return bm;
          };

          const float bm0 = blkmax4(s00,s01,s02,s03);
          const float bm1 = (R>1) ? blkmax4(s10,s11,s12,s13) : -INFINITY;
          const float bm2 = (R>2) ? blkmax4(s20,s21,s22,s23) : -INFINITY;
          const float bm3 = (R>3) ? blkmax4(s30,s31,s32,s33) : -INFINITY;

          const float m0_new = std::max(m0r, bm0);
          const float m1_new = (R>1) ? std::max(m1r, bm1) : m0_new;
          const float m2_new = (R>2) ? std::max(m2r, bm2) : m0_new;
          const float m3_new = (R>3) ? std::max(m3r, bm3) : m0_new;

          const float a0 = std::exp(m0r - m0_new);
          const float a1 = (R>1) ? std::exp(m1r - m1_new) : a0;
          const float a2 = (R>2) ? std::exp(m2r - m2_new) : a0;
          const float a3 = (R>3) ? std::exp(m3r - m3_new) : a0;

          for (int seg=0; seg<S; ++seg) {
            acc0[seg] = _mm512_mul_ps(acc0[seg], _mm512_set1_ps(a0));
            if (R>1) acc1[seg] = _mm512_mul_ps(acc1[seg], _mm512_set1_ps(a1));
            if (R>2) acc2[seg] = _mm512_mul_ps(acc2[seg], _mm512_set1_ps(a2));
            if (R>3) acc3[seg] = _mm512_mul_ps(acc3[seg], _mm512_set1_ps(a3));
          }

          // weights
          __m512 w00 = attn::exp512_ps(_mm512_sub_ps(s00, _mm512_set1_ps(m0_new)));
          __m512 w01 = (groups>=2) ? attn::exp512_ps(_mm512_sub_ps(s01, _mm512_set1_ps(m0_new))) : _mm512_setzero_ps();
          __m512 w02 = (groups>=3) ? attn::exp512_ps(_mm512_sub_ps(s02, _mm512_set1_ps(m0_new))) : _mm512_setzero_ps();
          __m512 w03 = (groups>=4) ? attn::exp512_ps(_mm512_sub_ps(s03, _mm512_set1_ps(m0_new))) : _mm512_setzero_ps();

          __m512 w10=_mm512_setzero_ps(), w11=_mm512_setzero_ps(), w12=_mm512_setzero_ps(), w13=_mm512_setzero_ps();
          __m512 w20=_mm512_setzero_ps(), w21=_mm512_setzero_ps(), w22=_mm512_setzero_ps(), w23=_mm512_setzero_ps();
          __m512 w30=_mm512_setzero_ps(), w31=_mm512_setzero_ps(), w32=_mm512_setzero_ps(), w33=_mm512_setzero_ps();
          if (R>1){ w10=attn::exp512_ps(_mm512_sub_ps(s10,_mm512_set1_ps(m1_new)));
                    if (groups>=2) w11=attn::exp512_ps(_mm512_sub_ps(s11,_mm512_set1_ps(m1_new)));
                    if (groups>=3) w12=attn::exp512_ps(_mm512_sub_ps(s12,_mm512_set1_ps(m1_new)));
                    if (groups>=4) w13=attn::exp512_ps(_mm512_sub_ps(s13,_mm512_set1_ps(m1_new))); }
          if (R>2){ w20=attn::exp512_ps(_mm512_sub_ps(s20,_mm512_set1_ps(m2_new)));
                    if (groups>=2) w21=attn::exp512_ps(_mm512_sub_ps(s21,_mm512_set1_ps(m2_new)));
                    if (groups>=3) w22=attn::exp512_ps(_mm512_sub_ps(s22,_mm512_set1_ps(m2_new)));
                    if (groups>=4) w23=attn::exp512_ps(_mm512_sub_ps(s23,_mm512_set1_ps(m2_new))); }
          if (R>3){ w30=attn::exp512_ps(_mm512_sub_ps(s30,_mm512_set1_ps(m3_new)));
                    if (groups>=2) w31=attn::exp512_ps(_mm512_sub_ps(s31,_mm512_set1_ps(m3_new)));
                    if (groups>=3) w32=attn::exp512_ps(_mm512_sub_ps(s32,_mm512_set1_ps(m3_new)));
                    if (groups>=4) w33=attn::exp512_ps(_mm512_sub_ps(s33,_mm512_set1_ps(m3_new))); }

          // masked denominators
          float l_blk0 = attn::hsum_ps(_mm512_maskz_mov_ps(mask0, w00));
          if (groups>=2) l_blk0 += attn::hsum_ps(_mm512_maskz_mov_ps(mask1, w01));
          if (groups>=3) l_blk0 += attn::hsum_ps(_mm512_maskz_mov_ps(mask2, w02));
          if (groups>=4) l_blk0 += attn::hsum_ps(_mm512_maskz_mov_ps(mask3, w03));

          float l_blk1 = 0.f, l_blk2 = 0.f, l_blk3 = 0.f;
          if (R>1){ l_blk1  = attn::hsum_ps(_mm512_maskz_mov_ps(mask0, w10));
                    if (groups>=2) l_blk1 += attn::hsum_ps(_mm512_maskz_mov_ps(mask1, w11));
                    if (groups>=3) l_blk1 += attn::hsum_ps(_mm512_maskz_mov_ps(mask2, w12));
                    if (groups>=4) l_blk1 += attn::hsum_ps(_mm512_maskz_mov_ps(mask3, w13)); }
          if (R>2){ l_blk2  = attn::hsum_ps(_mm512_maskz_mov_ps(mask0, w20));
                    if (groups>=2) l_blk2 += attn::hsum_ps(_mm512_maskz_mov_ps(mask1, w21));
                    if (groups>=3) l_blk2 += attn::hsum_ps(_mm512_maskz_mov_ps(mask2, w22));
                    if (groups>=4) l_blk2 += attn::hsum_ps(_mm512_maskz_mov_ps(mask3, w23)); }
          if (R>3){ l_blk3  = attn::hsum_ps(_mm512_maskz_mov_ps(mask0, w30));
                    if (groups>=2) l_blk3 += attn::hsum_ps(_mm512_maskz_mov_ps(mask1, w31));
                    if (groups>=3) l_blk3 += attn::hsum_ps(_mm512_maskz_mov_ps(mask2, w32));
                    if (groups>=4) l_blk3 += attn::hsum_ps(_mm512_maskz_mov_ps(mask3, w33)); }

          // store weights for reuse with V (one load per V row → 4 FMAs)
          alignas(64) float wq[4][4][16];
          _mm512_store_ps(wq[0][0], w00);
          if (groups>=2) _mm512_store_ps(wq[0][1], w01);
          if (groups>=3) _mm512_store_ps(wq[0][2], w02);
          if (groups>=4) _mm512_store_ps(wq[0][3], w03);
          if (R>1){ _mm512_store_ps(wq[1][0], w10);
            if (groups>=2) _mm512_store_ps(wq[1][1], w11);
            if (groups>=3) _mm512_store_ps(wq[1][2], w12);
            if (groups>=4) _mm512_store_ps(wq[1][3], w13); }
          if (R>2){ _mm512_store_ps(wq[2][0], w20);
            if (groups>=2) _mm512_store_ps(wq[2][1], w21);
            if (groups>=3) _mm512_store_ps(wq[2][2], w22);
            if (groups>=4) _mm512_store_ps(wq[2][3], w23); }
          if (R>3){ _mm512_store_ps(wq[3][0], w30);
            if (groups>=2) _mm512_store_ps(wq[3][1], w31);
            if (groups>=3) _mm512_store_ps(wq[3][2], w32);
            if (groups>=4) _mm512_store_ps(wq[3][3], w33); }

          auto fma_group = [&](int gidx, int cnt){
            const int off = gidx * 16;
            for (int l = 0; l < cnt; ++l) {
              const float* vrow = Vh.data() + ((size_t)h*T + (base + off + l))*Dh;
              for (int seg=0; seg<S; ++seg) {
                __m512 vv = _mm512_loadu_ps(vrow + seg*16);
                acc0[seg] = _mm512_fmadd_ps(vv, _mm512_set1_ps(wq[0][gidx][l]), acc0[seg]);
                if (R>1) acc1[seg] = _mm512_fmadd_ps(vv, _mm512_set1_ps(wq[1][gidx][l]), acc1[seg]);
                if (R>2) acc2[seg] = _mm512_fmadd_ps(vv, _mm512_set1_ps(wq[2][gidx][l]), acc2[seg]);
                if (R>3) acc3[seg] = _mm512_fmadd_ps(vv, _mm512_set1_ps(wq[3][gidx][l]), acc3[seg]);
              }
            }
          };

          if (c0) fma_group(0, c0);
          if (c1) fma_group(1, c1);
          if (c2) fma_group(2, c2);
          if (c3) fma_group(3, c3);

          // update running stats (using the non-colliding names)
          l0 = l0 * a0 + l_blk0; m0r = m0_new;
          if (R>1){ l1 = l1 * a1 + l_blk1; m1r = m1_new; }
          if (R>2){ l2 = l2 * a2 + l_blk2; m2r = m2_new; }
          if (R>3){ l3 = l3 * a3 + l_blk3; m3r = m3_new; }
        } // half
      } // blocks

      // Normalize & store contexts
      const float inv0 = 1.0f / l0;
      for (int seg=0; seg<S; ++seg) {
        _mm512_storeu_ps(ctx0 + h*Dh + seg*16, _mm512_mul_ps(acc0[seg], _mm512_set1_ps(inv0)));
      }
      if (R>1){
        const float inv1 = 1.0f / l1;
        for (int seg=0; seg<S; ++seg) {
          _mm512_storeu_ps(ctx1 + h*Dh + seg*16, _mm512_mul_ps(acc1[seg], _mm512_set1_ps(inv1)));
        }
      }
      if (R>2){
        const float inv2 = 1.0f / l2;
        for (int seg=0; seg<S; ++seg) {
          _mm512_storeu_ps(ctx2 + h*Dh + seg*16, _mm512_mul_ps(acc2[seg], _mm512_set1_ps(inv2)));
        }
      }
      if (R>3){
        const float inv3 = 1.0f / l3;
        for (int seg=0; seg<S; ++seg) {
          _mm512_storeu_ps(ctx3 + h*Dh + seg*16, _mm512_mul_ps(acc3[seg], _mm512_set1_ps(inv3)));
        }
      }
    } // h
  } // t0
}

} // namespace kernels
