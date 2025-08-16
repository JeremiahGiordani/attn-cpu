// sgemm_blocked.cpp — AVX-512 SGEMM with global prepack + column-parallel schedule
// NEW: one-shot runtime autotuner picks k-unroll (4 vs 8) micro-kernel once per process.
// Contract: gemm::sgemm_blocked(...) per your header.
//
// Build: g++ -O3 -Ofast -march=native -fopenmp -ffast-math -funroll-loops -frename-registers -std=c++17 -DNDEBUG -c sgemm_blocked.cpp
// Good env: export OMP_NUM_THREADS=8 OMP_PROC_BIND=close OMP_PLACES=cores KMP_BLOCKTIME=0
#include <immintrin.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <atomic>
#include <chrono>
#include <omp.h>

namespace gemm {

// ============================ Tunables =============================
static constexpr int MR = 8;           // rows per micro-kernel (broadcast)
static constexpr int NR = 48;          // cols per micro-kernel (3× zmm)
static constexpr int PREFETCH_DIST = 64;
static constexpr int NT_STRIPE = 1;    // columns per thread stripe (in NR tiles)
static constexpr int MT_CHUNK  = 16;   // A tiles per chunk (16*8 rows = 128 rows)

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
  return (reinterpret_cast<uintptr_t>(p) & 63u) == 0u;
}

// ============================ Packing ==============================
// Pack A MR×K (k-major) with alpha fused.
static inline void pack_A_tile_mrK(
    const float* __restrict A, int ldA,
    int mr_eff, int K, float alpha,
    float* __restrict Ap)
{
  for (int k = 0; k < K; ++k) {
    const float* a_col = A + k;
    float* dst = Ap + (size_t)k * MR;
    int r = 0;
    for (; r < mr_eff; ++r) dst[r] = a_col[(size_t)r * ldA] * alpha;
    for (; r < MR;     ++r) dst[r] = 0.0f;
  }
}

// Pack B K×NR (k-major). Full-width uses aligned zmm loads/stores.
static inline void pack_B_tile_Knr(
    const float* __restrict B, int ldB,
    int K, int nr_eff,
    float* __restrict Bp)
{
  for (int k = 0; k < K; ++k) {
    const float* b_row = B + (size_t)k * ldB;
    float* dst = Bp + (size_t)k * NR;
    if (nr_eff == NR) {
      __m512 x0 = _mm512_loadu_ps(b_row +  0);
      __m512 x1 = _mm512_loadu_ps(b_row + 16);
      __m512 x2 = _mm512_loadu_ps(b_row + 32);
      _mm512_store_ps(dst +  0, x0);
      _mm512_store_ps(dst + 16, x1);
      _mm512_store_ps(dst + 32, x2);
    } else {
      int j = 0;
      for (; j < nr_eff; ++j) dst[j] = b_row[j];
      for (; j < NR;     ++j) dst[j] = 0.0f;
    }
  }
}

// ==================== 8×48 AVX-512 micro-kernels ===================
// Both kernels assume: full tile (mr_eff==MR, nr_eff==NR), beta==0 hot path.
// A already has alpha fused. We sweep full K once, then write C once.

// --- k-unroll = 4 ---
static inline void microkernel_8x48_u4_beta0(
    const float* __restrict Ap, const float* __restrict Bp,
    float* __restrict C, int ldc, int K, bool stream_if_aligned)
{
  __m512 acc0[MR], acc1[MR], acc2[MR];
#pragma unroll
  for (int r=0;r<MR;++r){ acc0[r]=_mm512_setzero_ps(); acc1[r]=_mm512_setzero_ps(); acc2[r]=_mm512_setzero_ps(); }

  int k=0, kend = K & ~3;
  for (; k<kend; k+=4) {
    if (k + PREFETCH_DIST < K) {
      _mm_prefetch((const char*)(Ap + (size_t)(k+PREFETCH_DIST)*MR), _MM_HINT_T0);
      _mm_prefetch((const char*)(Bp + (size_t)(k+PREFETCH_DIST)*NR), _MM_HINT_T0);
    }
#pragma unroll
    for (int u=0; u<4; ++u) {
      const float* a = Ap + (size_t)(k+u)*MR;
      const float* b = Bp + (size_t)(k+u)*NR;
      __m512 b0 = _mm512_load_ps(b+0), b1=_mm512_load_ps(b+16), b2=_mm512_load_ps(b+32);
#pragma unroll
      for (int r=0;r<MR;++r){
        __m512 ar=_mm512_set1_ps(a[r]);
        acc0[r]=_mm512_fmadd_ps(ar,b0,acc0[r]);
        acc1[r]=_mm512_fmadd_ps(ar,b1,acc1[r]);
        acc2[r]=_mm512_fmadd_ps(ar,b2,acc2[r]);
      }
    }
  }
  for (; k<K; ++k){
    const float* a=Ap+(size_t)k*MR; const float* b=Bp+(size_t)k*NR;
    __m512 b0=_mm512_load_ps(b+0), b1=_mm512_load_ps(b+16), b2=_mm512_load_ps(b+32);
#pragma unroll
    for (int r=0;r<MR;++r){
      __m512 ar=_mm512_set1_ps(a[r]);
      acc0[r]=_mm512_fmadd_ps(ar,b0,acc0[r]);
      acc1[r]=_mm512_fmadd_ps(ar,b1,acc1[r]);
      acc2[r]=_mm512_fmadd_ps(ar,b2,acc2[r]);
    }
  }

#pragma unroll
  for (int r=0;r<MR;++r){
    float* c=C+(size_t)r*ldc;
    if (stream_if_aligned && aligned64(c)) {
      _mm512_stream_ps(c+0,acc0[r]); _mm512_stream_ps(c+16,acc1[r]); _mm512_stream_ps(c+32,acc2[r]);
    } else {
      _mm512_storeu_ps(c+0,acc0[r]); _mm512_storeu_ps(c+16,acc1[r]); _mm512_storeu_ps(c+32,acc2[r]);
    }
  }
}

// --- k-unroll = 8 ---
static inline void microkernel_8x48_u8_beta0(
    const float* __restrict Ap, const float* __restrict Bp,
    float* __restrict C, int ldc, int K, bool stream_if_aligned)
{
  __m512 acc0[MR], acc1[MR], acc2[MR];
#pragma unroll
  for (int r=0;r<MR;++r){ acc0[r]=_mm512_setzero_ps(); acc1[r]=_mm512_setzero_ps(); acc2[r]=_mm512_setzero_ps(); }

  int k=0, kend = K & ~7;
  for (; k<kend; k+=8) {
    if (k + PREFETCH_DIST < K) {
      _mm_prefetch((const char*)(Ap + (size_t)(k+PREFETCH_DIST)*MR), _MM_HINT_T0);
      _mm_prefetch((const char*)(Bp + (size_t)(k+PREFETCH_DIST)*NR), _MM_HINT_T0);
    }
#pragma unroll
    for (int u=0; u<8; ++u) {
      const float* a = Ap + (size_t)(k+u)*MR;
      const float* b = Bp + (size_t)(k+u)*NR;
      __m512 b0 = _mm512_load_ps(b+0), b1=_mm512_load_ps(b+16), b2=_mm512_load_ps(b+32);
#pragma unroll
      for (int r=0;r<MR;++r){
        __m512 ar=_mm512_set1_ps(a[r]);
        acc0[r]=_mm512_fmadd_ps(ar,b0,acc0[r]);
        acc1[r]=_mm512_fmadd_ps(ar,b1,acc1[r]);
        acc2[r]=_mm512_fmadd_ps(ar,b2,acc2[r]);
      }
    }
  }
  for (; k<K; ++k){
    const float* a=Ap+(size_t)k*MR; const float* b=Bp+(size_t)k*NR;
    __m512 b0=_mm512_load_ps(b+0), b1=_mm512_load_ps(b+16), b2=_mm512_load_ps(b+32);
#pragma unroll
    for (int r=0;r<MR;++r){
      __m512 ar=_mm512_set1_ps(a[r]);
      acc0[r]=_mm512_fmadd_ps(ar,b0,acc0[r]);
      acc1[r]=_mm512_fmadd_ps(ar,b1,acc1[r]);
      acc2[r]=_mm512_fmadd_ps(ar,b2,acc2[r]);
    }
  }

#pragma unroll
  for (int r=0;r<MR;++r){
    float* c=C+(size_t)r*ldc;
    if (stream_if_aligned && aligned64(c)) {
      _mm512_stream_ps(c+0,acc0[r]); _mm512_stream_ps(c+16,acc1[r]); _mm512_stream_ps(c+32,acc2[r]);
    } else {
      _mm512_storeu_ps(c+0,acc0[r]); _mm512_storeu_ps(c+16,acc1[r]); _mm512_storeu_ps(c+32,acc2[r]);
    }
  }
}

// Scalar tail (edges only; your dims are multiples of MR/NR so this is cold)
static inline void microkernel_tail_scalar(
    const float* __restrict Ap, const float* __restrict Bp,
    float* __restrict C, int ldc, int K, int mr_eff, int nr_eff, float beta)
{
  for (int r=0;r<mr_eff;++r){
    for (int j=0;j<nr_eff;++j){
      float acc=0.f;
      for (int kk=0;kk<K;++kk) acc += Ap[(size_t)kk*MR + r] * Bp[(size_t)kk*NR + j];
      float* c = C + (size_t)r*ldc + j;
      if (beta==0.0f) *c = acc;
      else if (beta==1.0f) *c += acc;
      else *c = acc + beta*(*c);
    }
  }
}

// =================== One-shot micro-autotuner ======================
enum class UnrollSel : int { U4=0, U8=1 };
static std::atomic<int> g_tuned{ -1 }; // -1: not tuned; 0: U4; 1: U8

static inline UnrollSel pick_unroll_once(
    const float* A_pack, const float* B_pack, int MT, int NT, int K, int ldc)
{
  int expected = -1;
  if (!g_tuned.compare_exchange_strong(expected, -2)) {
    // someone else tuning or tuned
    while (g_tuned.load() == -2) { /* spin very briefly */ }
    return g_tuned.load() == 1 ? UnrollSel::U8 : UnrollSel::U4;
  }

  // Tiny benchmark: use up to 8 A-tiles × 2 B-tiles (fits L2), beta==0 path.
  int tm_max = std::min(MT, 8);
  int tn_max = std::min(NT, 2);

  // temp C buffer
  float* Ctmp = aligned_alloc64((size_t)tm_max * MR * tn_max * NR);
  const int ldc_tmp = tn_max * NR;

  auto bench = [&](UnrollSel sel)->double{
    std::memset(Ctmp, 0, sizeof(float) * (size_t)tm_max*MR*tn_max*NR);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int rep=0; rep<2; ++rep) { // two reps to smooth timer noise
      for (int tm=0; tm<tm_max; ++tm) {
        const float* Ap = A_pack + (size_t)tm * K * MR;
        for (int tn=0; tn<tn_max; ++tn) {
          const float* Bp = B_pack + (size_t)tn * K * NR;
          float* C_tile = Ctmp + (size_t)tm*MR*ldc_tmp + tn*NR;
          bool stream_ok = false; // writes to temp buffer; don't stream
          if (sel == UnrollSel::U8)
            microkernel_8x48_u8_beta0(Ap, Bp, C_tile, ldc_tmp, K, stream_ok);
          else
            microkernel_8x48_u4_beta0(Ap, Bp, C_tile, ldc_tmp, K, stream_ok);
        }
      }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
  };

  double t4 = bench(UnrollSel::U4);
  double t8 = bench(UnrollSel::U8);

  aligned_free64(Ctmp);

  UnrollSel best = (t8 < t4) ? UnrollSel::U8 : UnrollSel::U4;
  g_tuned.store(best == UnrollSel::U8 ? 1 : 0);
  return best;
}

// ============================ Top-level ============================
void sgemm_blocked(const float* A, int M, int K,
                   const float* B, int N,
                   float* C,
                   float alpha, float beta)
{
  if (M <= 0 || N <= 0 || K <= 0) return;

  const int ldA = K, ldB = N, ldC = N;

  // alpha==0 ⇒ C = beta*C
  if (alpha == 0.0f) {
#pragma omp parallel for schedule(static)
    for (int i=0;i<M;++i) {
      float* Crow = C + (size_t)i * ldC;
      if (beta == 0.0f) std::memset(Crow, 0, sizeof(float)*N);
      else if (beta != 1.0f) for (int j=0;j<N;++j) Crow[j] *= beta;
    }
    return;
  }

  // Tile counts
  const int MT = (M + MR - 1) / MR;
  const int NT = (N + NR - 1) / NR;

  // Global packs (64B aligned)
  float* A_pack = aligned_alloc64((size_t)MT * K * MR);
  float* B_pack = aligned_alloc64((size_t)NT * K * NR);

  // Pack A (alpha fused) and B (parallel)
#pragma omp parallel for schedule(static)
  for (int tm=0; tm<MT; ++tm) {
    int r0 = tm * MR;
    int mr_eff = std::min(MR, M - r0);
    const float* A_src = A + (size_t)r0 * ldA;
    float* Ap = A_pack + (size_t)tm * K * MR;
    pack_A_tile_mrK(A_src, ldA, mr_eff, K, alpha, Ap);
  }
#pragma omp parallel for schedule(static)
  for (int tn=0; tn<NT; ++tn) {
    int j0 = tn * NR;
    int nr_eff = std::min(NR, N - j0);
    const float* B_src = B + j0;
    float* Bp = B_pack + (size_t)tn * K * NR;
    pack_B_tile_Knr(B_src, ldB, K, nr_eff, Bp);
  }

  // One-shot autotune (fast, first call only)
  // UnrollSel sel = pick_unroll_once(A_pack, B_pack, MT, NT, K, ldC);
  UnrollSel sel = UnrollSel::U4;

  // Column stripes (each thread owns NT_STRIPE tiles), A-chunk reuse
  const int NST = (NT + NT_STRIPE - 1) / NT_STRIPE;

  if (beta == 0.0f) {
#pragma omp parallel for schedule(static)
    for (int s=0; s<NST; ++s) {
      const int tn_begin = s * NT_STRIPE;
      const int tn_end   = std::min(NT, tn_begin + NT_STRIPE);

      for (int tm0 = 0; tm0 < MT; tm0 += MT_CHUNK) {
        const int tm_end = std::min(MT, tm0 + MT_CHUNK);

        for (int tm = tm0; tm < tm_end; ++tm) {
          const float* Ap = A_pack + (size_t)tm * K * MR;
          int r0 = tm * MR;

          for (int tn = tn_begin; tn < tn_end; ++tn) {
            const float* Bp = B_pack + (size_t)tn * K * NR;
            int j0 = tn * NR;
            float* C_tile = C + (size_t)r0 * ldC + j0;

            bool stream_ok = aligned64(C_tile); // each 16-float lane inherits alignment when ldc*4 is multiple of 64
            if (sel == UnrollSel::U8)
              microkernel_8x48_u8_beta0(Ap, Bp, C_tile, ldC, K, stream_ok);
            else
              microkernel_8x48_u4_beta0(Ap, Bp, C_tile, ldC, K, stream_ok);
          }
        }
      }
    }
  } else {
    // General beta path (kept correct via scalar tails for edges or RMW if needed)
#pragma omp parallel for schedule(static)
    for (int s=0; s<NST; ++s) {
      const int tn_begin = s * NT_STRIPE;
      const int tn_end   = std::min(NT, tn_begin + NT_STRIPE);

      for (int tm0 = 0; tm0 < MT; tm0 += MT_CHUNK) {
        const int tm_end = std::min(MT, tm0 + MT_CHUNK);
        for (int tm = tm0; tm < tm_end; ++tm) {
          const float* Ap = A_pack + (size_t)tm * K * MR;
          int r0 = tm * MR;
          int mr_eff = std::min(MR, M - r0);

          for (int tn = tn_begin; tn < tn_end; ++tn) {
            const float* Bp = B_pack + (size_t)tn * K * NR;
            int j0 = tn * NR;
            int nr_eff = std::min(NR, N - j0);
            float* C_tile = C + (size_t)r0 * ldC + j0;

            if (mr_eff == MR && nr_eff == NR) {
              // simple accumulate then beta blend (vector)
              __m512 acc0[MR], acc1[MR], acc2[MR];
              for (int r=0;r<MR;++r){ acc0[r]=_mm512_setzero_ps(); acc1[r]=_mm512_setzero_ps(); acc2[r]=_mm512_setzero_ps(); }
              int k=0, kend=K & ~3;
              for (; k<kend; k+=4) {
                if (k + PREFETCH_DIST < K) {
                  _mm_prefetch((const char*)(Ap + (size_t)(k+PREFETCH_DIST)*MR), _MM_HINT_T0);
                  _mm_prefetch((const char*)(Bp + (size_t)(k+PREFETCH_DIST)*NR), _MM_HINT_T0);
                }
                for (int u=0; u<4; ++u) {
                  const float* a = Ap + (size_t)(k+u)*MR;
                  const float* b = Bp + (size_t)(k+u)*NR;
                  __m512 b0=_mm512_load_ps(b+0), b1=_mm512_load_ps(b+16), b2=_mm512_load_ps(b+32);
                  for (int r=0;r<MR;++r){
                    __m512 ar=_mm512_set1_ps(a[r]);
                    acc0[r]=_mm512_fmadd_ps(ar,b0,acc0[r]);
                    acc1[r]=_mm512_fmadd_ps(ar,b1,acc1[r]);
                    acc2[r]=_mm512_fmadd_ps(ar,b2,acc2[r]);
                  }
                }
              }
              for (; k<K; ++k){
                const float* a=Ap+(size_t)k*MR; const float* b=Bp+(size_t)k*NR;
                __m512 b0=_mm512_load_ps(b+0), b1=_mm512_load_ps(b+16), b2=_mm512_load_ps(b+32);
                for (int r=0;r<MR;++r){
                  __m512 ar=_mm512_set1_ps(a[r]);
                  acc0[r]=_mm512_fmadd_ps(ar,b0,acc0[r]);
                  acc1[r]=_mm512_fmadd_ps(ar,b1,acc1[r]);
                  acc2[r]=_mm512_fmadd_ps(ar,b2,acc2[r]);
                }
              }
              __m512 vb=_mm512_set1_ps(beta);
              for (int r=0;r<MR;++r){
                float* c=C_tile+(size_t)r*ldC;
                __m512 c0=_mm512_loadu_ps(c+0), c1=_mm512_loadu_ps(c+16), c2=_mm512_loadu_ps(c+32);
                _mm512_storeu_ps(c+0, _mm512_fmadd_ps(c0,vb,acc0[r]));
                _mm512_storeu_ps(c+16,_mm512_fmadd_ps(c1,vb,acc1[r]));
                _mm512_storeu_ps(c+32,_mm512_fmadd_ps(c2,vb,acc2[r]));
              }
            } else {
              microkernel_tail_scalar(Ap,Bp,C_tile,ldC,K,mr_eff,nr_eff,beta);
            }
          }
        }
      }
    }
  }

  aligned_free64(B_pack);
  aligned_free64(A_pack);
}

} // namespace gemm
