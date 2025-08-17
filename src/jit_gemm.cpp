// jit_gemm.cpp — AVX-512 SGEMM with Xbyak JIT micro-kernel
// Notes:
//  - Fast path (default): K split into L1 panels (KC_L1), k-unroll=2, NT stores.
//  - Optional STRICT_NUMERICS: process full K in one pass (no K panels) to make
//    accumulation order closer to a simple loop (OFF by default).
//
// Build: g++ -O3 -march=native -fopenmp -Ithird_party/xbyak src/jit_gemm.cpp -c
// For stricter accumulation order (slightly different perf): -DSTRICT_NUMERICS=1

#include <immintrin.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <omp.h>
#if defined(__linux__)
  #include <sys/mman.h>
#endif
#include "xbyak/xbyak.h"

namespace gemm {

static constexpr int MR    = 8;
static constexpr int NR    = 48;
#ifndef STRICT_NUMERICS
static constexpr int KC_L1 = 768;   // fast default: MR*KC_L1*4 ≈ 24KB
#else
static constexpr int KC_L1 = 1 << 30; // effectively "no panel split"
#endif

static inline float* aligned_alloc64(size_t n_floats) {
#if defined(_MSC_VER)
  return static_cast<float*>(_aligned_malloc(n_floats * sizeof(float), 64));
#else
  void* p = nullptr;
  if (posix_memalign(&p, 64, n_floats * sizeof(float)) != 0) return nullptr;
  return static_cast<float*>(p);
#endif
}
static inline void aligned_free64(float* p) {
#if defined(_MSC_VER)
  _aligned_free(p);
#else
  free(p);
#endif
}
static inline bool aligned64(const void* p) {
  return ((reinterpret_cast<uintptr_t>(p) & 63u) == 0u);
}
static inline void advise_hugepages(void* p, size_t n_bytes) {
#if defined(__linux__)
  madvise(p, n_bytes, MADV_WILLNEED);
  madvise(p, n_bytes, MADV_HUGEPAGE);
#endif
}

// ---------- packing ----------
static inline void pack_A_tile_mrK(const float* __restrict A, int ldA,
                                   int mr_eff, int K, float alpha,
                                   float* __restrict Ap) {
  for (int k = 0; k < K; ++k) {
    const float* a_col = A + k;
    float* dst = Ap + (size_t)k * MR;
    int r = 0;
    for (; r < mr_eff; ++r) dst[r] = a_col[(size_t)r * ldA] * alpha;
    for (; r < MR;     ++r) dst[r] = 0.0f;
  }
}
static inline void pack_B_tile_Knr(const float* __restrict B, int ldB,
                                   int K, int nr_eff,
                                   float* __restrict Bp) {
  for (int k = 0; k < K; ++k) {
    const float* b_row = B + (size_t)k * ldB;
    float* dst = Bp + (size_t)k * NR;
    int j = 0;
    for (; j < nr_eff; ++j) dst[j] = b_row[j];
    for (; j < NR;     ++j) dst[j] = 0.0f;
  }
}

// ---------- JIT 8×48, k-unroll=2, beta==0, aligned-B, NT stores ----------
struct Jit8x48_Beta0_Align : public Xbyak::CodeGenerator {
  using Fn = void(*)(const float*, const float*, float*, int, int, int);
  Jit8x48_Beta0_Align() {
    using namespace Xbyak;

    const Reg64 Ap = rdi, Bp = rsi, C = rdx, ldc = rcx, K = r8, stream = r9;
    const Reg64 kbase = r10, a_ptr = r11, b_ptr = r12, c_row = r13, ldc_bytes = r14;
    const Reg64 tmp = r15, kcnt = rax, Kc = rbx;

    // prologue (save callee-saved we use)
    push(rbp); mov(rbp, rsp);
    push(rbx); push(r12); push(r13); push(r14); push(r15);

    // zero accs
    for (int r = 0; r < MR; ++r) vxorps(Zmm(0 + r),  Zmm(0 + r),  Zmm(0 + r));
    for (int r = 0; r < MR; ++r) vxorps(Zmm(8 + r),  Zmm(8 + r),  Zmm(8 + r));
    for (int r = 0; r < MR; ++r) vxorps(Zmm(16 + r), Zmm(16 + r), Zmm(16 + r));

    mov(ldc_bytes, ldc);
    imul(ldc_bytes, ldc_bytes, 4);

    xor_(kbase, kbase);

    Label L_outer, L_outer_done, L_panel_loop, L_panel_tail, L_panel_done, L_no_clamp;
    L(L_outer);
      cmp(kbase, K);
      jge(L_outer_done, T_NEAR);

      mov(kcnt, K); sub(kcnt, kbase);            // remaining
      mov(Kc, (uint64_t)KC_L1);                  // Kc = min(KC_L1, remaining)
      cmp(kcnt, KC_L1);
      jge(L_no_clamp, T_NEAR);
      mov(Kc, kcnt);
      L(L_no_clamp);

      mov(tmp, kbase); imul(tmp, tmp, MR*4);     // a_ptr = Ap + kbase*MR*4
      lea(a_ptr, ptr[Ap + tmp]);
      mov(tmp, kbase); imul(tmp, tmp, NR*4);     // b_ptr = Bp + kbase*NR*4
      lea(b_ptr, ptr[Bp + tmp]);

      mov(kcnt, Kc);
      L(L_panel_loop);
        cmp(kcnt, 2);
        jl(L_panel_tail, T_NEAR);

        vmovaps(Zmm(24), ptr[b_ptr +  0*4]);     // B(k)
        vmovaps(Zmm(25), ptr[b_ptr + 16*4]);
        vmovaps(Zmm(26), ptr[b_ptr + 32*4]);
        vmovaps(Zmm(27), ptr[b_ptr + NR*4 +  0*4]); // B(k+1)
        vmovaps(Zmm(28), ptr[b_ptr + NR*4 + 16*4]);
        vmovaps(Zmm(29), ptr[b_ptr + NR*4 + 32*4]);

        for (int r = 0; r < MR; ++r) {
          vbroadcastss(Zmm(30), ptr[a_ptr + r*4]);
          vfmadd231ps(Zmm(0 + r),  Zmm(24), Zmm(30));
          vfmadd231ps(Zmm(8 + r),  Zmm(25), Zmm(30));
          vfmadd231ps(Zmm(16 + r), Zmm(26), Zmm(30));

          vbroadcastss(Zmm(31), ptr[a_ptr + MR*4 + r*4]);
          vfmadd231ps(Zmm(0 + r),  Zmm(27), Zmm(31));
          vfmadd231ps(Zmm(8 + r),  Zmm(28), Zmm(31));
          vfmadd231ps(Zmm(16 + r), Zmm(29), Zmm(31));
        }

        add(a_ptr, MR*2*4);
        add(b_ptr, NR*2*4);
        sub(kcnt, 2);
        jmp(L_panel_loop, T_NEAR);

      L(L_panel_tail);
        cmp(kcnt, 0);
        je(L_panel_done, T_NEAR);
        vmovaps(Zmm(24), ptr[b_ptr +  0*4]);
        vmovaps(Zmm(25), ptr[b_ptr + 16*4]);
        vmovaps(Zmm(26), ptr[b_ptr + 32*4]);
        for (int r = 0; r < MR; ++r) {
          vbroadcastss(Zmm(30), ptr[a_ptr + r*4]);
          vfmadd231ps(Zmm(0 + r),  Zmm(24), Zmm(30));
          vfmadd231ps(Zmm(8 + r),  Zmm(25), Zmm(30));
          vfmadd231ps(Zmm(16 + r), Zmm(26), Zmm(30));
        }
      L(L_panel_done);

      add(kbase, Kc);
      jmp(L_outer, T_NEAR);

    L(L_outer_done);

    // one branch: NT vs storeu
    Label L_storeu_all, L_done_all;
    test(stream, stream);
    jz(L_storeu_all, T_NEAR);

    for (int r = 0; r < MR; ++r) {
      mov(c_row, C);
      if (r != 0) { mov(tmp, ldc_bytes); imul(tmp, tmp, r); add(c_row, tmp); }
      vmovntps(ptr[c_row +  0*4], Zmm(0  + r));
      vmovntps(ptr[c_row + 16*4], Zmm(8  + r));
      vmovntps(ptr[c_row + 32*4], Zmm(16 + r));
    }
    jmp(L_done_all, T_NEAR);

    L(L_storeu_all);
    for (int r = 0; r < MR; ++r) {
      mov(c_row, C);
      if (r != 0) { mov(tmp, ldc_bytes); imul(tmp, tmp, r); add(c_row, tmp); }
      vmovups(ptr[c_row +  0*4], Zmm(0  + r));
      vmovups(ptr[c_row + 16*4], Zmm(8  + r));
      vmovups(ptr[c_row + 32*4], Zmm(16 + r));
    }

    L(L_done_all);

    // epilogue
    pop(r15); pop(r14); pop(r13); pop(r12); pop(rbx);
    pop(rbp);
    ret();
  }
  Fn get() const { return getCode<Fn>(); }
};

static Jit8x48_Beta0_Align::Fn jit_kernel_8x48_beta0_aligned() {
  static Jit8x48_Beta0_Align gen;
  return gen.get();
}

// ---------- portable tail / beta!=0 ----------
static inline void microkernel_tail_scalar(
    const float* __restrict Ap, const float* __restrict Bp,
    float* __restrict C, int ldc,
    int K, int mr_eff, int nr_eff, float beta)
{
  for (int r = 0; r < mr_eff; ++r) {
    for (int j = 0; j < nr_eff; ++j) {
      float acc = 0.f;
      for (int kk = 0; kk < K; ++kk)
        acc += Ap[(size_t)kk*MR + r] * Bp[(size_t)kk*NR + j];
      float* cptr = C + (size_t)r * ldc + j;
      if (beta == 0.0f) *cptr = acc;
      else if (beta == 1.0f) *cptr += acc;
      else *cptr = acc + beta * (*cptr);
    }
  }
}

// ============================== Top-level ==============================
void gemm_blocked_jit(const float* A, int M, int K,
                      const float* B, int N,
                      float* C,
                      float alpha, float beta)
{
  if (M <= 0 || N <= 0 || K <= 0) return;

  const int ldA = K, ldB = N, ldC = N;

  if (alpha == 0.0f) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < M; ++i) {
      float* Crow = C + (size_t)i * ldC;
      if (beta == 0.0f) std::memset(Crow, 0, sizeof(float) * (size_t)N);
      else if (beta != 1.0f) for (int j = 0; j < N; ++j) Crow[j] *= beta;
    }
    return;
  }

  const int MT = (M + MR - 1) / MR;
  const int NT = (N + NR - 1) / NR;

  const size_t A_pack_elems = (size_t)MT * (size_t)K * (size_t)MR;
  const size_t B_pack_elems = (size_t)NT * (size_t)K * (size_t)NR;
  float* A_pack = aligned_alloc64(A_pack_elems);
  float* B_pack = aligned_alloc64(B_pack_elems);
#if defined(__linux__)
  advise_hugepages(A_pack, A_pack_elems * sizeof(float));
  advise_hugepages(B_pack, B_pack_elems * sizeof(float));
#endif

  // pack A (alpha fused)
#pragma omp parallel for schedule(static)
  for (int tm = 0; tm < MT; ++tm) {
    int r0 = tm * MR;
    int mr_eff = std::min(MR, M - r0);
    const float* A_src = A + (size_t)r0 * ldA;
    float* Ap = A_pack + (size_t)tm * K * MR;
    pack_A_tile_mrK(A_src, ldA, mr_eff, K, alpha, Ap);
  }

  // pack B
#pragma omp parallel for schedule(static)
  for (int tn = 0; tn < NT; ++tn) {
    int j0 = tn * NR;
    int nr_eff = std::min(NR, N - j0);
    const float* B_src = B + j0;
    float* Bp = B_pack + (size_t)tn * K * NR;
    pack_B_tile_Knr(B_src, ldB, K, nr_eff, Bp);
  }

  if (beta == 0.0f) {
    auto ker = jit_kernel_8x48_beta0_aligned();

#pragma omp parallel
    {
      // column-parallel assignment
#pragma omp for schedule(static)
      for (int tn = 0; tn < NT; ++tn) {
        int j0 = tn * NR;
        const float* Bp = B_pack + (size_t)tn * K * NR;

        for (int tm = 0; tm < MT; ++tm) {
          int r0 = tm * MR;
          int mr_eff = std::min(MR, M - r0);
          int nr_eff = std::min(NR, N - j0);
          float* C_tile = C + (size_t)r0 * ldC + j0;
          const float* Ap = A_pack + (size_t)tm * K * MR;

          if (mr_eff == MR && nr_eff == NR) {
            int stream_flag = (aligned64(C_tile +  0) &&
                               aligned64(C_tile + 16) &&
                               aligned64(C_tile + 32)) ? 1 : 0;
            ker(Ap, Bp, C_tile, ldC, K, stream_flag);
          } else {
            microkernel_tail_scalar(Ap, Bp, C_tile, ldC, K, mr_eff, nr_eff, 0.0f);
          }
        }
      }
      _mm_sfence(); // drain NT stores once per thread
    }
  } else {
#pragma omp parallel for schedule(static)
    for (int tn = 0; tn < NT; ++tn) {
      int j0 = tn * NR;
      const float* Bp = B_pack + (size_t)tn * K * NR;

      for (int tm = 0; tm < MT; ++tm) {
        int r0 = tm * MR;
        int mr_eff = std::min(MR, M - r0);
        int nr_eff = std::min(NR, N - j0);
        float* C_tile = C + (size_t)r0 * ldC + j0;
        const float* Ap = A_pack + (size_t)tm * K * MR;

        if (mr_eff == MR && nr_eff == NR) {
          __m512 acc0[MR], acc1[MR], acc2[MR];
          for (int r = 0; r < MR; ++r) {
            acc0[r] = _mm512_setzero_ps();
            acc1[r] = _mm512_setzero_ps();
            acc2[r] = _mm512_setzero_ps();
          }
          for (int kbase = 0; kbase < K; kbase += KC_L1) {
            int Kc = std::min(KC_L1, K - kbase);
            const float* a_ptr = Ap + (size_t)kbase * MR;
            const float* b_ptr = Bp + (size_t)kbase * NR;
            int k = 0, kend = Kc & ~1;
            for (; k < kend; k += 2) {
              const float* bk = b_ptr + (size_t)k * NR;
              __m512 bk0 = _mm512_load_ps(bk +  0);
              __m512 bk1 = _mm512_load_ps(bk + 16);
              __m512 bk2 = _mm512_load_ps(bk + 32);
              const float* b1 = bk + NR;
              __m512 b10 = _mm512_load_ps(b1 +  0);
              __m512 b11 = _mm512_load_ps(b1 + 16);
              __m512 b12 = _mm512_load_ps(b1 + 32);
              for (int r = 0; r < MR; ++r) {
                __m512 A0 = _mm512_broadcastss_ps(_mm_load_ss(a_ptr + r));
                acc0[r] = _mm512_fmadd_ps(A0, bk0, acc0[r]);
                acc1[r] = _mm512_fmadd_ps(A0, bk1, acc1[r]);
                acc2[r] = _mm512_fmadd_ps(A0, bk2, acc2[r]);
                __m512 A1 = _mm512_broadcastss_ps(_mm_load_ss(a_ptr + MR + r));
                acc0[r] = _mm512_fmadd_ps(A1, b10, acc0[r]);
                acc1[r] = _mm512_fmadd_ps(A1, b11, acc1[r]);
                acc2[r] = _mm512_fmadd_ps(A1, b12, acc2[r]);
              }
              a_ptr += (size_t)2 * MR;
            }
            for (; k < Kc; ++k) {
              const float* bk = b_ptr + (size_t)k * NR;
              __m512 B0 = _mm512_load_ps(bk +  0);
              __m512 B1 = _mm512_load_ps(bk + 16);
              __m512 B2 = _mm512_load_ps(bk + 32);
              for (int r = 0; r < MR; ++r) {
                __m512 A0 = _mm512_broadcastss_ps(_mm_load_ss(a_ptr + r));
                acc0[r] = _mm512_fmadd_ps(A0, B0, acc0[r]);
                acc1[r] = _mm512_fmadd_ps(A0, B1, acc1[r]);
                acc2[r] = _mm512_fmadd_ps(A0, B2, acc2[r]);
              }
              a_ptr += MR;
            }
          }
          __m512 vb = _mm512_set1_ps(beta);
          for (int r = 0; r < MR; ++r) {
            float* cptr = C_tile + (size_t)r * ldC;
            __m512 c0 = _mm512_loadu_ps(cptr +  0);
            __m512 c1 = _mm512_loadu_ps(cptr + 16);
            __m512 c2 = _mm512_loadu_ps(cptr + 32);
            _mm512_storeu_ps(cptr +  0, _mm512_fmadd_ps(c0, vb, acc0[r]));
            _mm512_storeu_ps(cptr + 16, _mm512_fmadd_ps(c1, vb, acc1[r]));
            _mm512_storeu_ps(cptr + 32, _mm512_fmadd_ps(c2, vb, acc2[r]));
          }
        } else {
          microkernel_tail_scalar(Ap, Bp, C_tile, ldC, K, mr_eff, nr_eff, beta);
        }
      }
    }
  }

  aligned_free64(B_pack);
  aligned_free64(A_pack);
}

} // namespace gemm
