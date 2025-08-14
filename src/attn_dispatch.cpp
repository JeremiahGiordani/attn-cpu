#include "attn_api.h"
#include "common.h"
#include "simd_math.h"
#include "pack_16x16.h"
#include "gemm_16x16.h"
#include "layout.h"
#include "attn_dh16_m4_nocausal.h"
#include "attn_dh16_m1_generic.h"
#include "attn_generic_m1.h"
#include <vector>
#include <cstring>
#include <stdexcept>

namespace attn {

void mha_block_dense(const float* x, int T, int D,
                     const float* W_in, const float* b_in,
                     const float* W_out, const float* b_out,
                     int H, bool causal, float* y_out) {

  if ((D % H) != 0) throw std::runtime_error("D must be divisible by H");
  const int Dh = D / H;

  // thread-local scratch (as before)
  static thread_local int S_T=-1, S_D=-1, S_H=-1;
  static thread_local std::vector<float> Qh, KhT, Vh, Ctx;
  static thread_local std::vector<float> Win_packed, Wout_packed;
  static thread_local std::vector<float> row3D_4;
  static thread_local int packed_dims=-1;

  auto ensure = [&](int t,int d,int h){
    if (S_T==t && S_D==d && S_H==h) return;
    S_T=t; S_D=d; S_H=h;
    Qh.assign((size_t)H*(size_t)T*(size_t)Dh, 0.0f);
    KhT.assign((size_t)H*(size_t)Dh*(size_t)T, 0.0f);
    Vh.assign((size_t)H*(size_t)T*(size_t)Dh, 0.0f);
    Ctx.assign((size_t)T*(size_t)D, 0.0f);
    row3D_4.assign((size_t)4*(size_t)(3*D), 0.0f);
    packed_dims = -1;
  };
  ensure(T,D,H);

  // Pack W_in^T and W_out^T once per shape (same as before)
  if (ATTN_LIKELY(D % 16 == 0 && (3*D) % 16 == 0)) {
    if (packed_dims != D*3) {
      std::vector<float> Wt_in((size_t)D*(size_t)(3*D));
      for (int r=0;r<D;++r)
        for (int c=0;c<3*D;++c)
          Wt_in[(size_t)r*(3*D)+c] = W_in[(size_t)c*D + r];
      pack_wt_16x16(Wt_in.data(), D, 3*D, Win_packed);

      std::vector<float> Wt_out((size_t)D*(size_t)D);
      for (int r=0;r<D;++r)
        for (int c=0;c<D;++c)
          Wt_out[(size_t)r*D + c] = W_out[(size_t)c*D + r];
      pack_wt_16x16(Wt_out.data(), D, D, Wout_packed);

      packed_dims = D*3;
    }

    // QKV GEMM (row4 preferred) + scatter
    for (int t0=0; t0<T; t0+=4) {
      const int R = std::min(4, T - t0);
      float* y0 = row3D_4.data() + 0*(3*D);
      float* y1 = row3D_4.data() + 1*(3*D);
      float* y2 = row3D_4.data() + 2*(3*D);
      float* y3 = row3D_4.data() + 3*(3*D);

      const float* x0 = x + (size_t)(t0+0)*D;
      const float* x1 = (R>1) ? x + (size_t)(t0+1)*D : x0;
      const float* x2 = (R>2) ? x + (size_t)(t0+2)*D : x0;
      const float* x3 = (R>3) ? x + (size_t)(t0+3)*D : x0;

      if (R==4) gemm_row4_packed_16x16(x0,x1,x2,x3, Win_packed.data(), b_in, D, 3*D, y0,y1,y2,y3);
      else {
        gemm_row1_packed_16x16(x0, Win_packed.data(), b_in, D, 3*D, y0);
        if (R>1) gemm_row1_packed_16x16(x1, Win_packed.data(), b_in, D, 3*D, y1);
        if (R>2) gemm_row1_packed_16x16(x2, Win_packed.data(), b_in, D, 3*D, y2);
        if (R>3) gemm_row1_packed_16x16(x3, Win_packed.data(), b_in, D, 3*D, y3);
      }

      // scatter
      if (Dh == 16) {
        scatter_qkv_headed_dh16(y0 + 0*D, y0 + 1*D, y0 + 2*D, T, D, H, t0+0, Qh, KhT, Vh);
        if (R>1) scatter_qkv_headed_dh16(y1 + 0*D, y1 + 1*D, y1 + 2*D, T, D, H, t0+1, Qh, KhT, Vh);
        if (R>2) scatter_qkv_headed_dh16(y2 + 0*D, y2 + 1*D, y2 + 2*D, T, D, H, t0+2, Qh, KhT, Vh);
        if (R>3) scatter_qkv_headed_dh16(y3 + 0*D, y3 + 1*D, y3 + 2*D, T, D, H, t0+3, Qh, KhT, Vh);
      } else {
        scatter_qkv_headed_generic(y0 + 0*D, y0 + 1*D, y0 + 2*D, T, D, H, Dh, t0+0, Qh, KhT, Vh);
        if (R>1) scatter_qkv_headed_generic(y1 + 0*D, y1 + 1*D, y1 + 2*D, T, D, H, Dh, t0+1, Qh, KhT, Vh);
        if (R>2) scatter_qkv_headed_generic(y2 + 0*D, y2 + 1*D, y2 + 2*D, T, D, H, Dh, t0+2, Qh, KhT, Vh);
        if (R>3) scatter_qkv_headed_generic(y3 + 0*D, y3 + 1*D, y3 + 2*D, T, D, H, Dh, t0+3, Qh, KhT, Vh);
      }
    }

    // Attention
    if (!causal && Dh == 16) {
      kernels::run_dh16_m4_nocausal(Qh, KhT, Vh, T, H, D, Dh, Ctx);
    } else if (Dh == 16) {
      kernels::run_dh16_m1_online(Qh, KhT, Vh, T, H, D, Dh, causal, Ctx);
    } else {
      kernels::run_generic_m1_online(Qh, KhT, Vh, T, H, D, Dh, causal, Ctx);
    }

    // Output GEMM
    for (int t0=0; t0<T; t0+=4) {
      const int R = std::min(4, T - t0);
      const float* c0 = Ctx.data() + (size_t)(t0+0)*D;
      const float* c1 = (R>1) ? Ctx.data() + (size_t)(t0+1)*D : c0;
      const float* c2 = (R>2) ? Ctx.data() + (size_t)(t0+2)*D : c0;
      const float* c3 = (R>3) ? Ctx.data() + (size_t)(t0+3)*D : c0;
      float* y0 = y_out + (size_t)(t0+0)*D;
      float* y1 = y_out + (size_t)(t0+1)*D;
      float* y2 = y_out + (size_t)(t0+2)*D;
      float* y3 = y_out + (size_t)(t0+3)*D;
      if (R==4) gemm_row4_packed_16x16(c0,c1,c2,c3, Wout_packed.data(), b_out, D, D, y0,y1,y2,y3);
      else {
        gemm_row1_packed_16x16(c0, Wout_packed.data(), b_out, D, D, y0);
        if (R>1) gemm_row1_packed_16x16(c1, Wout_packed.data(), b_out, D, D, y1);
        if (R>2) gemm_row1_packed_16x16(c2, Wout_packed.data(), b_out, D, D, y2);
        if (R>3) gemm_row1_packed_16x16(c3, Wout_packed.data(), b_out, D, D, y3);
      }
    }
    return;
  }

  // Fallback for D not multiple of 16: use generic packing or row-GEMV path (not hot)
  throw std::runtime_error("D not multiple of 16 fast path not implemented yet.");
}

} // namespace attn
