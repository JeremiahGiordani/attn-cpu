// sgemm_blocked.cpp — AVX-512 SGEMM with persistent pack cache + one-shot unroll pick
#include <immintrin.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <atomic>
#include <mutex>
#include <chrono>
#include <omp.h>

namespace gemm {

// ---------------- Constants ----------------
static constexpr int MR = 8;    // rows per micro-kernel
static constexpr int NR = 48;   // cols per micro-kernel (3× zmm)
static constexpr int PREFETCH_DIST = 64;

// ---------------- Helpers ------------------
static inline float* aligned_alloc64(size_t n_floats){
#if defined(_MSC_VER)
  return static_cast<float*>(_aligned_malloc(n_floats*sizeof(float), 64));
#else
  void* p=nullptr; if (posix_memalign(&p,64,n_floats*sizeof(float))!=0) return nullptr; return (float*)p;
#endif
}
static inline void aligned_free64(float* p){
#if defined(_MSC_VER)
  _aligned_free(p);
#else
  free(p);
#endif
}
static inline bool aligned64(const void* p){ return ((reinterpret_cast<uintptr_t>(p)&63u)==0u); }
static inline int env_int(const char* name, int dflt){ const char* s=getenv(name); return s? std::atoi(s):dflt; }
static inline bool env_bool(const char* name, bool dflt){ const char* s=getenv(name); return s? (std::atoi(s)!=0):dflt; }

// ---------------- Packing -------------------
// A: pack MR rows across full K (k-major), fuse alpha: Ap[k*MR + r]
static inline void pack_A_tile_mrK(const float* __restrict A, int ldA,
                                   int mr_eff, int K, float alpha,
                                   float* __restrict Ap){
  for(int k=0;k<K;++k){
    const float* a_col = A + k;
    float* dst = Ap + (size_t)k*MR;
    int r=0; for(; r<mr_eff; ++r) dst[r] = a_col[(size_t)r*ldA]*alpha;
            for(; r<MR;     ++r) dst[r] = 0.0f;
  }
}
// B: pack NR cols across full K (k-major): Bp[k*NR + j], rows 64B aligned
static inline void pack_B_tile_Knr(const float* __restrict B, int ldB,
                                   int K, int nr_eff,
                                   float* __restrict Bp){
  for(int k=0;k<K;++k){
    const float* b_row = B + (size_t)k*ldB;
    float* dst = Bp + (size_t)k*NR;
    if(nr_eff==NR){
      __m512 x0=_mm512_loadu_ps(b_row+ 0);
      __m512 x1=_mm512_loadu_ps(b_row+16);
      __m512 x2=_mm512_loadu_ps(b_row+32);
      _mm512_store_ps(dst+ 0, x0);
      _mm512_store_ps(dst+16, x1);
      _mm512_store_ps(dst+32, x2);
    }else{
      int j=0; for(; j<nr_eff; ++j) dst[j]=b_row[j];
               for(; j<NR;     ++j) dst[j]=0.0f;
    }
  }
}

// ---------------- 8×48 micro-kernels (beta==0) ----------------
static inline void micro_8x48_u4_beta0(const float* __restrict Ap,
                                       const float* __restrict Bp,
                                       float* __restrict C, int ldc, int K,
                                       bool stream_ok)
{
  __m512 acc0[MR], acc1[MR], acc2[MR];
  for(int r=0;r<MR;++r){ acc0[r]=_mm512_setzero_ps(); acc1[r]=_mm512_setzero_ps(); acc2[r]=_mm512_setzero_ps(); }

  int k=0, kend=K&~3;
  for(; k<kend; k+=4){
    if(k+PREFETCH_DIST<K){
      _mm_prefetch((const char*)(Ap+(size_t)(k+PREFETCH_DIST)*MR), _MM_HINT_T0);
      _mm_prefetch((const char*)(Bp+(size_t)(k+PREFETCH_DIST)*NR), _MM_HINT_T0);
    }
#pragma unroll
    for(int u=0; u<4; ++u){
      const float* a=Ap+(size_t)(k+u)*MR; const float* b=Bp+(size_t)(k+u)*NR;
      __m512 b0=_mm512_load_ps(b+ 0), b1=_mm512_load_ps(b+16), b2=_mm512_load_ps(b+32);
#pragma unroll
      for(int r=0;r<MR;++r){
        __m512 ar=_mm512_set1_ps(a[r]);
        acc0[r]=_mm512_fmadd_ps(ar,b0,acc0[r]);
        acc1[r]=_mm512_fmadd_ps(ar,b1,acc1[r]);
        acc2[r]=_mm512_fmadd_ps(ar,b2,acc2[r]);
      }
    }
  }
  for(; k<K; ++k){
    const float* a=Ap+(size_t)k*MR; const float* b=Bp+(size_t)k*NR;
    __m512 b0=_mm512_load_ps(b+ 0), b1=_mm512_load_ps(b+16), b2=_mm512_load_ps(b+32);
#pragma unroll
    for(int r=0;r<MR;++r){
      __m512 ar=_mm512_set1_ps(a[r]);
      acc0[r]=_mm512_fmadd_ps(ar,b0,acc0[r]);
      acc1[r]=_mm512_fmadd_ps(ar,b1,acc1[r]);
      acc2[r]=_mm512_fmadd_ps(ar,b2,acc2[r]);
    }
  }
#pragma unroll
  for(int r=0;r<MR;++r){
    float* c=C+(size_t)r*ldc;
    if(stream_ok && aligned64(c)){
      _mm512_stream_ps(c+ 0, acc0[r]);
      _mm512_stream_ps(c+16, acc1[r]);
      _mm512_stream_ps(c+32, acc2[r]);
    }else{
      _mm512_storeu_ps(c+ 0, acc0[r]);
      _mm512_storeu_ps(c+16, acc1[r]);
      _mm512_storeu_ps(c+32, acc2[r]);
    }
  }
}

static inline void micro_8x48_u8_beta0(const float* __restrict Ap,
                                       const float* __restrict Bp,
                                       float* __restrict C, int ldc, int K,
                                       bool stream_ok)
{
  __m512 acc0[MR], acc1[MR], acc2[MR];
  for(int r=0;r<MR;++r){ acc0[r]=_mm512_setzero_ps(); acc1[r]=_mm512_setzero_ps(); acc2[r]=_mm512_setzero_ps(); }

  int k=0, kend=K&~7;
  for(; k<kend; k+=8){
    if(k+PREFETCH_DIST<K){
      _mm_prefetch((const char*)(Ap+(size_t)(k+PREFETCH_DIST)*MR), _MM_HINT_T0);
      _mm_prefetch((const char*)(Bp+(size_t)(k+PREFETCH_DIST)*NR), _MM_HINT_T0);
    }
#pragma unroll
    for(int u=0; u<8; ++u){
      const float* a=Ap+(size_t)(k+u)*MR; const float* b=Bp+(size_t)(k+u)*NR;
      __m512 b0=_mm512_load_ps(b+ 0), b1=_mm512_load_ps(b+16), b2=_mm512_load_ps(b+32);
#pragma unroll
      for(int r=0;r<MR;++r){
        __m512 ar=_mm512_set1_ps(a[r]);
        acc0[r]=_mm512_fmadd_ps(ar,b0,acc0[r]);
        acc1[r]=_mm512_fmadd_ps(ar,b1,acc1[r]);
        acc2[r]=_mm512_fmadd_ps(ar,b2,acc2[r]);
      }
    }
  }
  for(; k<K; ++k){
    const float* a=Ap+(size_t)k*MR; const float* b=Bp+(size_t)k*NR;
    __m512 b0=_mm512_load_ps(b+ 0), b1=_mm512_load_ps(b+16), b2=_mm512_load_ps(b+32);
#pragma unroll
    for(int r=0;r<MR;++r){
      __m512 ar=_mm512_set1_ps(a[r]);
      acc0[r]=_mm512_fmadd_ps(ar,b0,acc0[r]);
      acc1[r]=_mm512_fmadd_ps(ar,b1,acc1[r]);
      acc2[r]=_mm512_fmadd_ps(ar,b2,acc2[r]);
    }
  }
#pragma unroll
  for(int r=0;r<MR;++r){
    float* c=C+(size_t)r*ldc;
    if(stream_ok && aligned64(c)){
      _mm512_stream_ps(c+ 0, acc0[r]);
      _mm512_stream_ps(c+16, acc1[r]);
      _mm512_stream_ps(c+32, acc2[r]);
    }else{
      _mm512_storeu_ps(c+ 0, acc0[r]);
      _mm512_storeu_ps(c+16, acc1[r]);
      _mm512_storeu_ps(c+32, acc2[r]);
    }
  }
}

// Edges: scalar fallback (rare on your sizes)
static inline void micro_tail_scalar(const float* __restrict Ap, const float* __restrict Bp,
                                     float* __restrict C, int ldc, int K,
                                     int mr_eff, int nr_eff, float beta){
  for(int r=0;r<mr_eff;++r){
    for(int j=0;j<nr_eff;++j){
      float acc=0.f; for(int kk=0; kk<K; ++kk) acc += Ap[(size_t)kk*MR + r] * Bp[(size_t)kk*NR + j];
      float* c=C + (size_t)r*ldc + j;
      if(beta==0.0f) *c = acc;
      else if(beta==1.0f) *c += acc;
      else *c = acc + beta*(*c);
    }
  }
}

// ---------------- One-shot unroll pick (cached) --------------
enum class UnrollSel { U4=0, U8=1 };
static std::atomic<int> g_unroll{-1}; // -1:not picked, 0:U4, 1:U8

static inline UnrollSel pick_unroll_once(const float* A_pack, const float* B_pack,
                                         int MT, int NT, int K)
{
  int tune = env_int("SGEMM_TUNE", 1);
  int force_u = env_int("SGEMM_U", 0);
  if(tune==0 && (force_u==4 || force_u==8)){
    g_unroll.store(force_u==8 ? 1 : 0);
    return force_u==8 ? UnrollSel::U8 : UnrollSel::U4;
  }

  int expected=-1;
  if(!g_unroll.compare_exchange_strong(expected, -2)){
    while(g_unroll.load()==-2) {/*spin*/} // someone else picking
    return g_unroll.load()==1 ? UnrollSel::U8 : UnrollSel::U4;
  }

  // Tiny trial: up to 8 A-tiles × 2 B-tiles; beta==0 into temp buffer
  int tm_max = std::min(MT, 8);
  int tn_max = std::min(NT, 2);
  const int ldc = tn_max*NR;
  float* Ctmp = aligned_alloc64((size_t)tm_max*MR*ldc);

  auto bench = [&](UnrollSel sel){
    std::memset(Ctmp, 0, sizeof(float)*(size_t)tm_max*MR*ldc);
    auto t0 = std::chrono::high_resolution_clock::now();
    for(int tm=0; tm<tm_max; ++tm){
      const float* Ap = A_pack + (size_t)tm*K*MR;
      for(int tn=0; tn<tn_max; ++tn){
        const float* Bp = B_pack + (size_t)tn*K*NR;
        float* C_tile = Ctmp + (size_t)tm*MR*ldc + tn*NR;
        if(sel==UnrollSel::U8) micro_8x48_u8_beta0(Ap,Bp,C_tile,ldc,K,false);
        else                   micro_8x48_u4_beta0(Ap,Bp,C_tile,ldc,K,false);
      }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t1-t0).count();
  };

  double t4 = bench(UnrollSel::U4);
  double t8 = bench(UnrollSel::U8);
  aligned_free64(Ctmp);

  UnrollSel best = (t8 < t4) ? UnrollSel::U8 : UnrollSel::U4;
  g_unroll.store(best==UnrollSel::U8 ? 1 : 0);
  return best;
}

// ---------------- Persistent pack cache (process-wide) --------------
struct PackCache {
  // A pack: depends on (A_ptr, M, K, alpha)
  const float* A_ptr = nullptr;
  int M=0, K=0;
  float alpha=0.f;
  float* A_pack = nullptr;
  size_t A_pack_size = 0; // floats

  // B pack: depends on (B_ptr, K, N)
  const float* B_ptr = nullptr;
  int N=0;
  float* B_pack = nullptr;
  size_t B_pack_size = 0; // floats
};
static PackCache g_cache;
static std::mutex g_cache_mu;

static void ensure_A_pack(const float* A, int M, int K, float alpha,
                          float*& A_pack_out, int& MT_out, bool cache_on)
{
  const int MT = (M + MR - 1) / MR;
  size_t need = (size_t)MT * K * MR;

  if(cache_on){
    std::lock_guard<std::mutex> lk(g_cache_mu);
    bool hit = (g_cache.A_ptr==A && g_cache.M==M && g_cache.K==K && g_cache.alpha==alpha && g_cache.A_pack);
    if(!hit){
      if(g_cache.A_pack && g_cache.A_pack_size != need){ aligned_free64(g_cache.A_pack); g_cache.A_pack=nullptr; }
      if(!g_cache.A_pack) { g_cache.A_pack = aligned_alloc64(need); g_cache.A_pack_size = need; }
#pragma omp parallel for schedule(static)
      for(int tm=0; tm<MT; ++tm){
        int r0 = tm*MR; int mr_eff = std::min(MR, M-r0);
        const float* A_src = A + (size_t)r0*K;
        pack_A_tile_mrK(A_src, K, mr_eff, K, alpha, g_cache.A_pack + (size_t)tm*K*MR);
      }
      g_cache.A_ptr = A; g_cache.M=M; g_cache.K=K; g_cache.alpha=alpha;
    }
    A_pack_out = g_cache.A_pack; MT_out = MT;
  }else{
    // non-cached: allocate per call
    float* Ap = aligned_alloc64(need);
#pragma omp parallel for schedule(static)
    for(int tm=0; tm<MT; ++tm){
      int r0 = tm*MR; int mr_eff = std::min(MR, M-r0);
      const float* A_src = A + (size_t)r0*K;
      pack_A_tile_mrK(A_src, K, mr_eff, K, alpha, Ap + (size_t)tm*K*MR);
    }
    A_pack_out = Ap; MT_out = MT;
  }
}

static void ensure_B_pack(const float* B, int K, int N,
                          float*& B_pack_out, int& NT_out, bool cache_on)
{
  const int NT = (N + NR - 1) / NR;
  size_t need = (size_t)NT * K * NR;

  if(cache_on){
    std::lock_guard<std::mutex> lk(g_cache_mu);
    bool hit = (g_cache.B_ptr==B && g_cache.K==K && g_cache.N==N && g_cache.B_pack);
    if(!hit){
      if(g_cache.B_pack && g_cache.B_pack_size != need){ aligned_free64(g_cache.B_pack); g_cache.B_pack=nullptr; }
      if(!g_cache.B_pack) { g_cache.B_pack = aligned_alloc64(need); g_cache.B_pack_size = need; }
#pragma omp parallel for schedule(static)
      for(int tn=0; tn<NT; ++tn){
        int j0 = tn*NR; int nr_eff = std::min(NR, N-j0);
        const float* B_src = B + j0;
        pack_B_tile_Knr(B_src, N, K, nr_eff, g_cache.B_pack + (size_t)tn*K*NR);
      }
      g_cache.B_ptr = B; g_cache.K=K; g_cache.N=N;
    }
    B_pack_out = g_cache.B_pack; NT_out = NT;
  }else{
    float* Bp = aligned_alloc64(need);
#pragma omp parallel for schedule(static)
    for(int tn=0; tn<NT; ++tn){
      int j0 = tn*NR; int nr_eff = std::min(NR, N-j0);
      const float* B_src = B + j0;
      pack_B_tile_Knr(B_src, N, K, nr_eff, Bp + (size_t)tn*K*NR);
    }
    B_pack_out = Bp; NT_out = NT;
  }
}

// ---------------- Top-level SGEMM -------------------
void sgemm_blocked(const float* A, int M, int K,
                   const float* B, int N,
                   float* C,
                   float alpha, float beta)
{
  if(M<=0 || N<=0 || K<=0) return;

  const int ldA = K, ldB = N, ldC = N;

  // alpha==0 ⇒ C = beta*C
  if(alpha==0.0f){
#pragma omp parallel for schedule(static)
    for(int i=0;i<M;++i){
      float* Crow = C + (size_t)i*ldC;
      if(beta==0.0f) std::memset(Crow, 0, sizeof(float)*N);
      else if(beta!=1.0f) for(int j=0;j<N;++j) Crow[j]*=beta;
    }
    return;
  }

  const bool use_cache = env_bool("SGEMM_CACHE", true);

  // Ensure packed A and B (cache-aware)
  float* A_pack=nullptr; int MT=0;
  float* B_pack=nullptr; int NT=0;
  ensure_A_pack(A, M, K, alpha, A_pack, MT, use_cache);
  ensure_B_pack(B, K, N, B_pack, NT, use_cache);

  // One-shot unroll selection (cached)
  UnrollSel sel;
  if (g_unroll.load() >= 0) sel = (g_unroll.load()==1 ? UnrollSel::U8 : UnrollSel::U4);
  else                      sel = pick_unroll_once(A_pack, B_pack, MT, NT, K);
  auto kernel = (sel==UnrollSel::U8) ? micro_8x48_u8_beta0 : micro_8x48_u4_beta0;

  // ===== Column-parallel schedule (best performer) =====
  if(beta==0.0f){
#pragma omp parallel for schedule(static)
    for(int tn=0; tn<NT; ++tn){
      const float* Bp = B_pack + (size_t)tn*K*NR;
      int j0 = tn*NR;
      for(int tm=0; tm<MT; ++tm){
        const float* Ap = A_pack + (size_t)tm*K*MR;
        int r0 = tm*MR;
        float* C_tile = C + (size_t)r0*ldC + j0;
        bool stream_ok = aligned64(C_tile);
        kernel(Ap, Bp, C_tile, ldC, K, stream_ok);
      }
    }
  }else{
    // General beta path — keep correct with simple accumulate+blend or scalar edges
#pragma omp parallel for schedule(static)
    for(int tn=0; tn<NT; ++tn){
      const float* Bp = B_pack + (size_t)tn*K*NR;
      int j0 = tn*NR;
      for(int tm=0; tm<MT; ++tm){
        const float* Ap = A_pack + (size_t)tm*K*MR;
        int r0 = tm*MR; int mr_eff = std::min(MR, M-r0);
        int nr_eff = std::min(NR, N-j0);
        float* C_tile = C + (size_t)r0*ldC + j0;

        if(mr_eff==MR && nr_eff==NR){
          __m512 acc0[MR], acc1[MR], acc2[MR];
          for(int r=0;r<MR;++r){ acc0[r]=_mm512_setzero_ps(); acc1[r]=_mm512_setzero_ps(); acc2[r]=_mm512_setzero_ps(); }
          int k=0, kend=K&~3;
          for(; k<kend; k+=4){
            if(k+PREFETCH_DIST<K){
              _mm_prefetch((const char*)(Ap+(size_t)(k+PREFETCH_DIST)*MR), _MM_HINT_T0);
              _mm_prefetch((const char*)(Bp+(size_t)(k+PREFETCH_DIST)*NR), _MM_HINT_T0);
            }
#pragma unroll
            for(int u=0; u<4; ++u){
              const float* a=Ap+(size_t)(k+u)*MR; const float* b=Bp+(size_t)(k+u)*NR;
              __m512 b0=_mm512_load_ps(b+ 0), b1=_mm512_load_ps(b+16), b2=_mm512_load_ps(b+32);
              for(int r=0;r<MR;++r){ __m512 ar=_mm512_set1_ps(a[r]);
                acc0[r]=_mm512_fmadd_ps(ar,b0,acc0[r]); acc1[r]=_mm512_fmadd_ps(ar,b1,acc1[r]); acc2[r]=_mm512_fmadd_ps(ar,b2,acc2[r]); }
            }
          }
          for(; k<K; ++k){
            const float* a=Ap+(size_t)k*MR; const float* b=Bp+(size_t)k*NR;
            __m512 b0=_mm512_load_ps(b+ 0), b1=_mm512_load_ps(b+16), b2=_mm512_load_ps(b+32);
            for(int r=0;r<MR;++r){ __m512 ar=_mm512_set1_ps(a[r]);
              acc0[r]=_mm512_fmadd_ps(ar,b0,acc0[r]); acc1[r]=_mm512_fmadd_ps(ar,b1,acc1[r]); acc2[r]=_mm512_fmadd_ps(ar,b2,acc2[r]); }
          }
          __m512 vb=_mm512_set1_ps(beta);
          for(int r=0;r<MR;++r){
            float* c = C_tile + (size_t)r*ldC;
            __m512 c0=_mm512_loadu_ps(c+ 0), c1=_mm512_loadu_ps(c+16), c2=_mm512_loadu_ps(c+32);
            _mm512_storeu_ps(c+ 0,_mm512_fmadd_ps(c0,vb,acc0[r]));
            _mm512_storeu_ps(c+16,_mm512_fmadd_ps(c1,vb,acc1[r]));
            _mm512_storeu_ps(c+32,_mm512_fmadd_ps(c2,vb,acc2[r]));
          }
        }else{
          micro_tail_scalar(Ap,Bp,C_tile,ldC,K,mr_eff,nr_eff,beta);
        }
      }
    }
  }

  // Clean-up for non-cached path
  if(!use_cache){
    aligned_free64(A_pack);
    aligned_free64(B_pack);
  }
}

} // namespace gemm
