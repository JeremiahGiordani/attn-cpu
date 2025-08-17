// jit_gemm.cpp — AVX-512 SGEMM with Xbyak JIT micro-kernels
// Baseline: prefetching + asymmetric one-shot autotune of (NR, UNROLL, KC_L1), cached.
// NR ∈ {48, 32, 16} with UNROLL filtered by register feasibility.
//
// Build: g++ -O3 -march=native -fopenmp -Ithird_party/xbyak -c src/jit_gemm.cpp
// Optional: -DSTRICT_NUMERICS=1 to process full K in one pass (no panel split).

#include <immintrin.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <mutex>
#include <array>
#include <atomic>
#include <limits>
#include <cstdio>
#include <omp.h>
#include <iostream>
#if defined(__linux__)
  #include <sys/mman.h>
#endif
#include "xbyak/xbyak.h"

namespace gemm {

// ===================== Tunables (prefetch-only) ======================
static constexpr int PF_DIST_ITER_U2 = 2;  // JIT k-loop (k-unroll>=2) lookahead
static constexpr int PF_DIST_ITER_U1 = 4;  // JIT tail (k-unroll=1) lookahead
static constexpr int PF_KDIST_BETA   = 16; // β≠0 path prefetch distance
static constexpr bool PF_PACKERS     = true;

// ============================ Blocking ===============================
static constexpr int MR   = 8;           // register row block
static constexpr int NR_48 = 48;
static constexpr int NR_32 = 32;
static constexpr int NR_16 = 16;

// ============================ Utilities ==============================
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

// ============================ Packing ================================
static inline void pack_A_tile_mrK(const float* __restrict A, int ldA,
                                   int mr_eff, int K, float alpha,
                                   float* __restrict Ap) {
  for (int k = 0; k < K; ++k) {
    const float* a_col = A + k;
    float* dst = Ap + (size_t)k * MR;

    if constexpr (PF_PACKERS) {
      int k_pf = k + 8;
      if (k_pf < K) _mm_prefetch(reinterpret_cast<const char*>(A + k_pf), _MM_HINT_T0);
    }

    int r = 0;
    for (; r < mr_eff; ++r) dst[r] = a_col[(size_t)r * ldA] * alpha;
    for (; r < MR;     ++r) dst[r] = 0.0f;
  }
}

template<int NR>
static inline void pack_B_tile_Knr_NR(const float* __restrict B, int ldB,
                                      int K, int nr_eff,
                                      float* __restrict Bp) {
  for (int k = 0; k < K; ++k) {
    const float* b_row = B + (size_t)k * ldB;
    float* dst = Bp + (size_t)k * NR;

    if constexpr (PF_PACKERS) {
      int k_pf = k + 8;
      if (k_pf < K) {
        _mm_prefetch(reinterpret_cast<const char*>(B + (size_t)k_pf * ldB), _MM_HINT_T0);
      }
    }

    int j = 0;
    for (; j < nr_eff; ++j) dst[j] = b_row[j];
    for (; j < NR;     ++j) dst[j] = 0.0f;
  }
}

// ===================== Tails / β!=0 portable path ====================
static inline void microkernel_tail_scalar(
    const float* __restrict Ap, const float* __restrict Bp,
    float* __restrict C, int ldc,
    int K, int mr_eff, int nr_eff, int NR_stride, float beta)
{
  for (int r = 0; r < mr_eff; ++r) {
    for (int j = 0; j < nr_eff; ++j) {
      float acc = 0.f;
      for (int kk = 0; kk < K; ++kk)
        acc += Ap[(size_t)kk*MR + r] * Bp[(size_t)kk*NR_stride + j];
      float* cptr = C + (size_t)r * ldc + j;
      if (beta == 0.0f) *cptr = acc;
      else if (beta == 1.0f) *cptr += acc;
      else *cptr = acc + beta * (*cptr);
    }
  }
}

template<int NR, int KC_L1>
static inline void beta_not_zero_fulltile_intrinsics(
    const float* __restrict Ap, const float* __restrict Bp,
    float* __restrict C_tile, int ldC, int K, float beta)
{
  constexpr int NB = NR / 16;
  __m512 acc[MR][NB];
  for (int r = 0; r < MR; ++r)
    for (int b = 0; b < NB; ++b)
      acc[r][b] = _mm512_setzero_ps();

  for (int kbase = 0; kbase < K; kbase += KC_L1) {
    int Kc = std::min(KC_L1, K - kbase);
    const float* a_ptr = Ap + (size_t)kbase * MR;
    const float* b_ptr = Bp + (size_t)kbase * NR;

    int k = 0, kend = Kc & ~1;
    for (; k < kend; k += 2) {
      int kpf = k + PF_KDIST_BETA;
      if (kpf < Kc) {
        _mm_prefetch(reinterpret_cast<const char*>(b_ptr + (size_t)kpf * NR), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(a_ptr + (size_t)kpf * MR), _MM_HINT_T0);
      }

      const float* bk0 = b_ptr + (size_t)k * NR;
      const float* bk1 = bk0 + NR;

      __m512 Bk0[NB], Bk1[NB];
      for (int b = 0; b < NB; ++b) {
        Bk0[b] = _mm512_load_ps(bk0 + b*16);
        Bk1[b] = _mm512_load_ps(bk1 + b*16);
      }

      for (int r = 0; r < MR; ++r) {
        __m512 A0 = _mm512_broadcastss_ps(_mm_load_ss(a_ptr + r));
        __m512 A1 = _mm512_broadcastss_ps(_mm_load_ss(a_ptr + MR + r));
        for (int b = 0; b < NB; ++b) {
          acc[r][b] = _mm512_fmadd_ps(A0, Bk0[b], acc[r][b]);
          acc[r][b] = _mm512_fmadd_ps(A1, Bk1[b], acc[r][b]);
        }
      }
      a_ptr += (size_t)2 * MR;
    }

    for (; k < Kc; ++k) {
      int kpf = k + PF_KDIST_BETA;
      if (kpf < Kc) {
        _mm_prefetch(reinterpret_cast<const char*>(b_ptr + (size_t)kpf * NR), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(a_ptr + (size_t)kpf * MR), _MM_HINT_T0);
      }
      const float* bk = b_ptr + (size_t)k * NR;
      __m512 Bk[NB];
      for (int b = 0; b < NB; ++b) Bk[b] = _mm512_load_ps(bk + b*16);
      for (int r = 0; r < MR; ++r) {
        __m512 A0 = _mm512_broadcastss_ps(_mm_load_ss(a_ptr + r));
        for (int b = 0; b < NB; ++b)
          acc[r][b] = _mm512_fmadd_ps(A0, Bk[b], acc[r][b]);
      }
      a_ptr += MR;
    }
  }

  // Warm C rows before RMW
  for (int r = 0; r < MR; ++r) {
    float* cptr = C_tile + (size_t)r * ldC;
    for (int b = 0; b < (NR/16); ++b)
      _mm_prefetch(reinterpret_cast<const char*>(cptr + b*16), _MM_HINT_T0);
  }

  __m512 vb = _mm512_set1_ps(beta);
  for (int r = 0; r < MR; ++r) {
    float* cptr = C_tile + (size_t)r * ldC;
    for (int b = 0; b < (NR/16); ++b) {
      __m512 c = _mm512_loadu_ps(cptr + b*16);
      _mm512_storeu_ps(cptr + b*16, _mm512_fmadd_ps(c, vb, acc[r][b]));
    }
  }
}

// ================== JIT micro-kernels (beta==0, aligned B) ===========
// Generic generator for NR ∈ {16,32,48}, UNROLL ∈ {1..8}, KC_L1 baked in.
template<int NR, int UNROLL, int KC_L1>
struct Jit8xNR_Beta0 final : public Xbyak::CodeGenerator {
  using Fn = void(*)(const float*, const float*, float*, int, int, int);
  Jit8xNR_Beta0() {
    using namespace Xbyak;

    constexpr int NB = NR / 16;
    // Register budget: accumulators + B regs; keep zmm30/31 free for broadcast
    static_assert(NB >= 1 && NB <= 3, "NR must be 16, 32, or 48");
    static_assert(UNROLL >= 1 && UNROLL <= 8, "UNROLL out of supported range");
    static_assert(MR*NB + NB*UNROLL <= 30, "Register pressure too high (keep zmm30/31 free)");

    const Reg64 Ap = rdi, Bp = rsi, C = rdx, ldc = rcx, K = r8, stream = r9;
    const Reg64 kbase = r10, a_ptr = r11, b_ptr = r12, c_row = r13, ldc_bytes = r14;
    const Reg64 tmp = r15, kcnt = rax, Kc = rbx;

    // Prefetch byte distances (compile-time)
    static constexpr int APF_U2 = MR * 2 * 4 * PF_DIST_ITER_U2;
    static constexpr int BPF_U2 = NR * 2 * 4 * PF_DIST_ITER_U2;
    static constexpr int APF_U1 = MR * 1 * 4 * PF_DIST_ITER_U1;
    static constexpr int BPF_U1 = NR * 1 * 4 * PF_DIST_ITER_U1;

    // prologue
    push(rbp); mov(rbp, rsp);
    push(rbx); push(r12); push(r13); push(r14); push(r15);

    // zero accs: zmm0..zmm(MR*NB-1)
    for (int id = 0; id < MR*NB; ++id)
      vxorps(Zmm(id), Zmm(id), Zmm(id));

    mov(ldc_bytes, ldc);
    imul(ldc_bytes, ldc_bytes, 4);
    xor_(kbase, kbase);

    Label L_outer, L_outer_done, L_panel_loop, L_panel_tail, L_panel_done, L_no_clamp;
    L(L_outer);
      cmp(kbase, K);
      jge(L_outer_done, T_NEAR);

#ifndef STRICT_NUMERICS
      mov(kcnt, K); sub(kcnt, kbase);            // remaining
      mov(Kc, (uint64_t)KC_L1);                  // Kc = min(KC_L1, remaining)
      cmp(kcnt, KC_L1);
      jge(L_no_clamp, T_NEAR);
      mov(Kc, kcnt);
      L(L_no_clamp);
#else
      mov(Kc, K); sub(Kc, kbase);
#endif

      mov(tmp, kbase); imul(tmp, tmp, MR*4);     // a_ptr = Ap + kbase*MR*4
      lea(a_ptr, ptr[Ap + tmp]);
      mov(tmp, kbase); imul(tmp, tmp, NR*4);     // b_ptr = Bp + kbase*NR*4
      lea(b_ptr, ptr[Bp + tmp]);

      mov(kcnt, Kc);
      L(L_panel_loop);
        cmp(kcnt, UNROLL);
        jl(L_panel_tail, T_NEAR);

        // Prefetch next bodies for UNROLL>=2; for UNROLL=1 we use PF_U1 below
        if (UNROLL >= 2) {
          if (BPF_U2 > 0) {
            prefetcht0(ptr[b_ptr + BPF_U2]);
            if ((NR/16) >= 1) prefetcht0(ptr[b_ptr + BPF_U2 + NR*4]);
          }
          if (APF_U2 > 0) {
            prefetcht0(ptr[a_ptr + APF_U2]);
          }
        }

        // Load B rows for UNROLL steps
        const int baseB = MR*(NR/16); // B regs start after accumulators
        for (int t = 0; t < UNROLL; ++t) {
          for (int b = 0; b < (NR/16); ++b) {
            vmovaps(Zmm(baseB + t*(NR/16) + b), ptr[b_ptr + (t*NR + b*16)*4]);
          }
        }

        // FMAs
        for (int r = 0; r < MR; ++r) {
          for (int t = 0; t < UNROLL; ++t) {
            vbroadcastss(Zmm(30), ptr[a_ptr + (t*MR + r)*4]);
            for (int b = 0; b < (NR/16); ++b) {
              // acc[b][r] -> zmm(b*MR + r)
              vfmadd231ps(Zmm(b*MR + r), Zmm(baseB + t*(NR/16) + b), Zmm(30));
            }
          }
        }

        add(a_ptr, MR*UNROLL*4);
        add(b_ptr, NR*UNROLL*4);
        sub(kcnt, UNROLL);
        jmp(L_panel_loop, T_NEAR);

      L(L_panel_tail);
        cmp(kcnt, 0);
        je(L_panel_done, T_NEAR);

        if (BPF_U1 > 0) prefetcht0(ptr[b_ptr + BPF_U1]);
        if (APF_U1 > 0) prefetcht0(ptr[a_ptr + APF_U1]);

        // Load single B row (tail)
        for (int b = 0; b < (NR/16); ++b) {
          vmovaps(Zmm(MR*(NR/16) + b), ptr[b_ptr + b*16*4]);
        }
        for (int r = 0; r < MR; ++r) {
          vbroadcastss(Zmm(31), ptr[a_ptr + r*4]);
          for (int b = 0; b < (NR/16); ++b) {
            vfmadd231ps(Zmm(b*MR + r), Zmm(MR*(NR/16) + b), Zmm(31));
          }
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
      for (int b = 0; b < (NR/16); ++b) {
        vmovntps(ptr[c_row + (b*16)*4], Zmm(b*MR + r));
      }
    }
    jmp(L_done_all, T_NEAR);

    L(L_storeu_all);
    for (int r = 0; r < MR; ++r) {
      mov(c_row, C);
      if (r != 0) { mov(tmp, ldc_bytes); imul(tmp, tmp, r); add(c_row, tmp); }
      for (int b = 0; b < (NR/16); ++b) {
        vmovups(ptr[c_row + (b*16)*4], Zmm(b*MR + r));
      }
    }

    L(L_done_all);

    // epilogue
    pop(r15); pop(r14); pop(r13); pop(r12); pop(rbx);
    pop(rbp);
    ret();
  }
  Fn get() const { return getCode<Fn>(); }
};

// -------- JIT kernel factory (lazy singletons per config) -------------
template<int NR, int UNROLL, int KC_L1>
static typename Jit8xNR_Beta0<NR,UNROLL,KC_L1>::Fn get_jit_kernel() {
  static Jit8xNR_Beta0<NR,UNROLL,KC_L1> gen;
  return gen.get();
}

enum : int { UNR_MIN=1, UNR_MAX=8 };

template<int NR, int UNROLL>
static inline auto pick_by_kc(int kc)->typename Jit8xNR_Beta0<NR,UNROLL,768>::Fn {
#ifndef STRICT_NUMERICS
  switch (kc) {
    case 512: return get_jit_kernel<NR,UNROLL,512>();
    case 640: return get_jit_kernel<NR,UNROLL,640>();
    case 768: return get_jit_kernel<NR,UNROLL,768>();
    case 832: return get_jit_kernel<NR,UNROLL,832>();
    default:  return get_jit_kernel<NR,UNROLL,896>();
  }
#else
  constexpr int KCHuge = (1<<30);
  (void)kc;
  return get_jit_kernel<NR,UNROLL,KCHuge>();
#endif
}

// NR-specific dispatcher that ONLY instantiates feasible UNROLL values
static inline auto pick_jit(int NR, int UNROLL, int KC_L1)
  -> typename Jit8xNR_Beta0<NR_48,1,768>::Fn
{
  using Fn = typename Jit8xNR_Beta0<NR_48,1,768>::Fn;
  switch (NR) {
    case NR_48:
      switch (UNROLL) {
        case 1: return (Fn)pick_by_kc<NR_48,1>(KC_L1);
        case 2: return (Fn)pick_by_kc<NR_48,2>(KC_L1);
        default: return (Fn)pick_by_kc<NR_48,1>(KC_L1);
      }
    case NR_32:
      switch (UNROLL) {
        case 1: return (Fn)pick_by_kc<NR_32,1>(KC_L1);
        case 2: return (Fn)pick_by_kc<NR_32,2>(KC_L1);
        case 3: return (Fn)pick_by_kc<NR_32,3>(KC_L1);
        case 4: return (Fn)pick_by_kc<NR_32,4>(KC_L1);
        default: return (Fn)pick_by_kc<NR_32,2>(KC_L1);
      }
    case NR_16:
      switch (UNROLL) {
        case 1: return (Fn)pick_by_kc<NR_16,1>(KC_L1);
        case 2: return (Fn)pick_by_kc<NR_16,2>(KC_L1);
        case 3: return (Fn)pick_by_kc<NR_16,3>(KC_L1);
        case 4: return (Fn)pick_by_kc<NR_16,4>(KC_L1);
        case 5: return (Fn)pick_by_kc<NR_16,5>(KC_L1);
        case 6: return (Fn)pick_by_kc<NR_16,6>(KC_L1);
        case 7: return (Fn)pick_by_kc<NR_16,7>(KC_L1);
        case 8: return (Fn)pick_by_kc<NR_16,8>(KC_L1);
        default: return (Fn)pick_by_kc<NR_16,4>(KC_L1);
      }
    default:
      return (Fn)pick_by_kc<NR_48,1>(KC_L1);
  }
}

// =================== Config, tuner, and dispatcher ====================
struct KernelConfig {
  int NR;        // 16, 32, or 48
  int UNROLL;    // 1..8 (feasible combos only)
  int KC_L1;     // 512, 640, 768, 832, 896 (or huge if STRICT_NUMERICS)
};

static inline const char* cfg_str(const KernelConfig& c) {
  static thread_local char buf[64];
  std::snprintf(buf, sizeof(buf), "NR=%d, UNROLL=%d, KC_L1=%d", c.NR, c.UNROLL, c.KC_L1);
  return buf;
}

static KernelConfig g_cfg{};
static std::once_flag g_once;

static inline void select_default_cfg() {
#ifndef STRICT_NUMERICS
  g_cfg = {NR_48, 1, 768};
#else
  g_cfg = {NR_48, 1, (1<<30)};
#endif
}

static inline bool feasible_combo(int NR, int UNROLL) {
  int NB = NR / 16;
  // Keep zmm30/31 free: MR*NB + NB*UNROLL <= 30
  return (MR*NB + NB*UNROLL) <= 30 && UNROLL >= 1 && UNROLL <= 8;
}

// Forward decl
static void gemm_blocked_jit_impl(const float* A, int M, int K,
                                  const float* B, int N, float* C,
                                  float alpha, float beta,
                                  const KernelConfig& cfg);

// Autotune: try asymmetric, feasible grid on first call
static void autotune_once(const float* A, int M, int K,
                          const float* B, int N,
                          float* C)
{
#ifndef STRICT_NUMERICS
  static constexpr int KCs[] = {512, 640, 768, 832, 896};
#else
  static constexpr int KCs[] = {(1<<30)};
#endif
  static constexpr int NRs[] = {NR_48, NR_32, NR_16};

  double best_t = std::numeric_limits<double>::infinity();
  KernelConfig best{};
  bool found = false;

  // Scratch copies for timing (beta=0)
  const int ldA = K, ldB = N, ldC = N;
  size_t sizeA = (size_t)M * ldA;
  size_t sizeB = (size_t)K * ldB;
  size_t sizeC = (size_t)M * ldC;

  float* A_s = aligned_alloc64(sizeA);
  float* B_s = aligned_alloc64(sizeB);
  float* C_s = aligned_alloc64(sizeC);
  if (!A_s || !B_s || !C_s) {
    if (A_s) aligned_free64(A_s);
    if (B_s) aligned_free64(B_s);
    if (C_s) aligned_free64(C_s);
    select_default_cfg();
    return;
  }
  std::memcpy(A_s, A, sizeA * sizeof(float));
  std::memcpy(B_s, B, sizeB * sizeof(float));
  std::memset(C_s, 0, sizeC * sizeof(float));

  for (int NRv : NRs) {
    if (NRv == NR_48) {
      for (int UNv : {1,2}) {
        if (!feasible_combo(NRv, UNv)) continue;
        for (int KCv : KCs) {
          KernelConfig cfg{NRv, UNv, KCv};

          std::memset(C_s, 0, sizeC * sizeof(float));
          gemm_blocked_jit_impl(A_s, M, K, B_s, N, C_s, 1.0f, 0.0f, cfg);

          const int RUNS = 3;
          double tsum = 0.0;
          for (int r = 0; r < RUNS; ++r) {
            std::memset(C_s, 0, sizeC * sizeof(float));
            double t0 = omp_get_wtime();
            gemm_blocked_jit_impl(A_s, M, K, B_s, N, C_s, 1.0f, 0.0f, cfg);
            double t1 = omp_get_wtime();
            tsum += (t1 - t0);
          }
          double tavg = tsum / RUNS;
          if (tavg < best_t) { best_t = tavg; best = cfg; found = true; }
        }
      }
    } else if (NRv == NR_32) {
      for (int UNv : {1,2,3,4}) {
        if (!feasible_combo(NRv, UNv)) continue;
        for (int KCv : KCs) {
          KernelConfig cfg{NRv, UNv, KCv};

          std::memset(C_s, 0, sizeC * sizeof(float));
          gemm_blocked_jit_impl(A_s, M, K, B_s, N, C_s, 1.0f, 0.0f, cfg);

          const int RUNS = 3;
          double tsum = 0.0;
          for (int r = 0; r < RUNS; ++r) {
            std::memset(C_s, 0, sizeC * sizeof(float));
            double t0 = omp_get_wtime();
            gemm_blocked_jit_impl(A_s, M, K, B_s, N, C_s, 1.0f, 0.0f, cfg);
            double t1 = omp_get_wtime();
            tsum += (t1 - t0);
          }
          double tavg = tsum / RUNS;
          if (tavg < best_t) { best_t = tavg; best = cfg; found = true; }
        }
      }
    } else { // NR_16
      for (int UNv : {1,2,3,4,5,6,7,8}) {
        if (!feasible_combo(NRv, UNv)) continue;
        for (int KCv : KCs) {
          KernelConfig cfg{NRv, UNv, KCv};

          std::memset(C_s, 0, sizeC * sizeof(float));
          gemm_blocked_jit_impl(A_s, M, K, B_s, N, C_s, 1.0f, 0.0f, cfg);

          const int RUNS = 3;
          double tsum = 0.0;
          for (int r = 0; r < RUNS; ++r) {
            std::memset(C_s, 0, sizeC * sizeof(float));
            double t0 = omp_get_wtime();
            gemm_blocked_jit_impl(A_s, M, K, B_s, N, C_s, 1.0f, 0.0f, cfg);
            double t1 = omp_get_wtime();
            tsum += (t1 - t0);
          }
          double tavg = tsum / RUNS;
          if (tavg < best_t) { best_t = tavg; best = cfg; found = true; }
        }
      }
    }
  }

  if (found) g_cfg = best;
  else       select_default_cfg();

  aligned_free64(C_s);
  aligned_free64(B_s);
  aligned_free64(A_s);
}

// ================= Configurable dispatcher implementation =============
static inline int nb_blocks_for(int NR) { return NR/16; }

static void gemm_blocked_jit_impl(const float* A, int M, int K,
                                  const float* B, int N,
                                  float* C,
                                  float alpha, float beta,
                                  const KernelConfig& cfg)
{
  if (M <= 0 || N <= 0 || K <= 0) return;

  const int ldA = K, ldB = N, ldC = N;
  const int NRv  = cfg.NR;
  const int NBv  = nb_blocks_for(NRv);

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
  const int NT = (N + NRv - 1) / NRv;

  const size_t A_pack_elems = (size_t)MT * (size_t)K  * (size_t)MR;
  const size_t B_pack_elems = (size_t)NT * (size_t)K  * (size_t)NRv;
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

  // pack B (depends on NR choice)
#pragma omp parallel for schedule(static)
  for (int tn = 0; tn < NT; ++tn) {
    int j0 = tn * NRv;
    int nr_eff = std::min(NRv, N - j0);
    const float* B_src = B + j0;
    float* Bp = B_pack + (size_t)tn * K * NRv;
    if      (NRv == NR_48) pack_B_tile_Knr_NR<NR_48>(B_src, ldB, K, nr_eff, Bp);
    else if (NRv == NR_32) pack_B_tile_Knr_NR<NR_32>(B_src, ldB, K, nr_eff, Bp);
    else                   pack_B_tile_Knr_NR<NR_16>(B_src, ldB, K, nr_eff, Bp);
  }

  if (beta == 0.0f) {
    auto ker = pick_jit(cfg.NR, cfg.UNROLL, cfg.KC_L1);

#pragma omp parallel
    {
#pragma omp for schedule(static)
      for (int tn = 0; tn < NT; ++tn) {
        int j0 = tn * NRv;
        const float* Bp = B_pack + (size_t)tn * K * NRv;

        for (int tm = 0; tm < MT; ++tm) {
          int r0 = tm * MR;
          int mr_eff = std::min(MR, M - r0);
          int nr_eff = std::min(NRv, N - j0);
          float* C_tile = C + (size_t)r0 * ldC + j0;
          const float* Ap = A_pack + (size_t)tm * K * MR;

          if (mr_eff == MR && nr_eff == NRv) {
            bool aligned_segments = true;
            for (int b = 0; b < NBv; ++b)
              aligned_segments &= aligned64(C_tile + b*16);
            int stream_flag = aligned_segments ? 1 : 0;
            ker(Ap, Bp, C_tile, ldC, K, stream_flag);
          } else {
            microkernel_tail_scalar(Ap, Bp, C_tile, ldC, K, mr_eff, nr_eff, NRv, 0.0f);
          }
        }
      }
      _mm_sfence(); // drain NT stores once per thread
    }
  } else {
#pragma omp parallel for schedule(static)
    for (int tn = 0; tn < NT; ++tn) {
      int j0 = tn * NRv;
      const float* Bp = B_pack + (size_t)tn * K * NRv;

      for (int tm = 0; tm < MT; ++tm) {
        int r0 = tm * MR;
        int mr_eff = std::min(MR, M - r0);
        int nr_eff = std::min(NRv, N - j0);
        float* C_tile = C + (size_t)r0 * ldC + j0;
        const float* Ap = A_pack + (size_t)tm * K * MR;

        if (mr_eff == MR && nr_eff == NRv) {
          // KC here influences only prefetch cadence; 896 is a safe default
          if      (NRv == NR_48) beta_not_zero_fulltile_intrinsics<NR_48,896>(Ap, Bp, C_tile, ldC, K, beta);
          else if (NRv == NR_32) beta_not_zero_fulltile_intrinsics<NR_32,896>(Ap, Bp, C_tile, ldC, K, beta);
          else                   beta_not_zero_fulltile_intrinsics<NR_16,896>(Ap, Bp, C_tile, ldC, K, beta);
        } else {
          microkernel_tail_scalar(Ap, Bp, C_tile, ldC, K, mr_eff, nr_eff, NRv, beta);
        }
      }
    }
  }

  aligned_free64(B_pack);
  aligned_free64(A_pack);
}

// ============================== Public API ===========================
void gemm_blocked_jit(const float* A, int M, int K,
                      const float* B, int N,
                      float* C,
                      float alpha, float beta)
{
  static std::once_flag g_once;
  std::call_once(g_once, [&](){
    select_default_cfg();
    autotune_once(A, M, K, B, N, C);
    std::cout << "[GEMM] Tuned config: " << cfg_str(g_cfg) << std::endl;
  });

  gemm_blocked_jit_impl(A, M, K, B, N, C, alpha, beta, g_cfg);
}

} // namespace gemm
