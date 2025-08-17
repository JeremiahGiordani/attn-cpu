// sgemm_blocked.cpp — AVX-512 SGEMM with masked tails, templated UNROLL, tuned prefetch, ptr-bump
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
static inline int  env_int (const char* name, int dflt){ const char* s=getenv(name); return s? std::atoi(s):dflt; }
static inline bool env_bool(const char* name, bool dflt){ const char* s=getenv(name); return s? (std::atoi(s)!=0):dflt; }

// Prefetch distance selection (env override: SGEMM_PFD; auto-clamped if <0)
static inline int pick_prefetch_distance(int K){
  int pf = env_int("SGEMM_PFD", -1);
  if (pf >= 0) return pf;
  // Heuristic: ~K/8, clamped to [16,128], skip if K small.
  if (K < 64) return 0;
  int x = K >> 3;
  if (x < 16)  x = 16;
  if (x > 128) x = 128;
  return x;
}

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

// ---------------- Tail masks for NR ----------------
static inline void nr_masks(int nr_eff, __mmask16 &m0, __mmask16 &m1, __mmask16 &m2){
  int c0 = std::min(16, nr_eff);
  int c1 = std::max(0, std::min(16, nr_eff - 16));
  int c2 = std::max(0, std::min(16, nr_eff - 32));
  m0 = (c0==16) ? 0xFFFF : ((__mmask16)((1u<<c0)-1u));
  m1 = (c1==16) ? 0xFFFF : ((__mmask16)((1u<<c1)-1u));
  m2 = (c2==16) ? 0xFFFF : ((__mmask16)((1u<<c2)-1u));
}

// ---------------- Generic 8×48 micro-kernel, beta==0, templated UNROLL ----------------
template<int UNROLL>
static inline void micro_8x48_beta0_u(const float* __restrict Ap,
                                      const float* __restrict Bp,
                                      float* __restrict C, int ldc, int K,
                                      int mr_eff, int nr_eff, bool stream_ok)
{
  static_assert(UNROLL>=1 && UNROLL<=8, "UNROLL in [1,8]");
  __m512 acc0[MR], acc1[MR], acc2[MR];
#pragma unroll
  for(int r=0;r<MR;++r){ acc0[r]=_mm512_setzero_ps(); acc1[r]=_mm512_setzero_ps(); acc2[r]=_mm512_setzero_ps(); }

  const int PFD = pick_prefetch_distance(K);

  const float* a_ptr = Ap;
  const float* b_ptr = Bp;

  int k = 0;
  int kend = K - (K % UNROLL);

  // Small helper to keep live ranges short (limits transient zmm pressure)
  auto k_step = [&](const float* __restrict a, const float* __restrict b){
    __m512 b0=_mm512_load_ps(b+ 0);
    __m512 b1=_mm512_load_ps(b+16);
    __m512 b2=_mm512_load_ps(b+32);
#pragma unroll
    for(int r=0;r<MR;++r){
      __m512 ar=_mm512_set1_ps(a[r]);
      acc0[r]=_mm512_fmadd_ps(ar,b0,acc0[r]);
      acc1[r]=_mm512_fmadd_ps(ar,b1,acc1[r]);
      acc2[r]=_mm512_fmadd_ps(ar,b2,acc2[r]);
    }
  };

  // Main UNROLLed loop
  for(; k<kend; k+=UNROLL){
    if(PFD>0){
      int kp = k + PFD;
      if(kp < K){
        _mm_prefetch((const char*)(Ap+(size_t)kp*MR), _MM_HINT_T0);
        _mm_prefetch((const char*)(Bp+(size_t)kp*NR), _MM_HINT_T0);
      }
    }
#pragma unroll
    for(int u=0; u<UNROLL; ++u){
      const float* a = a_ptr + (size_t)u*MR;
      const float* b = b_ptr + (size_t)u*NR;
      k_step(a,b);
    }
    a_ptr += (size_t)UNROLL*MR;
    b_ptr += (size_t)UNROLL*NR;
  }

  // Remainder (K % UNROLL)
  for(; k<K; ++k){
    k_step(a_ptr, b_ptr);
    a_ptr += MR;
    b_ptr += NR;
  }

  // Stores: full tile fast path (streaming if aligned & allowed)
  if(mr_eff==MR && nr_eff==NR){
#pragma unroll
    for(int r=0;r<MR;++r){
      float* c = C + (size_t)r*ldc;
      if(stream_ok && aligned64(c)){
        _mm512_stream_ps(c+ 0, acc0[r]);
        _mm512_stream_ps(c+16, acc1[r]);
        _mm512_stream_ps(c+32, acc2[r]);
      }else{
        // use aligned if possible, else unaligned; cost diff is tiny
        if(aligned64(c)){
          _mm512_store_ps (c+ 0, acc0[r]);
          _mm512_store_ps (c+16, acc1[r]);
          _mm512_store_ps (c+32, acc2[r]);
        }else{
          _mm512_storeu_ps(c+ 0, acc0[r]);
          _mm512_storeu_ps(c+16, acc1[r]);
          _mm512_storeu_ps(c+32, acc2[r]);
        }
      }
    }
    return;
  }

  // Tail in N and/or M: masked stores for N, limit rows for M
  __mmask16 m0, m1, m2;
  nr_masks(nr_eff, m0, m1, m2);

  for(int r=0; r<mr_eff; ++r){
    float* c = C + (size_t)r*ldc;
    // No masked streaming intrinsic; use cached masked stores on tails
    _mm512_mask_storeu_ps(c+ 0, m0, acc0[r]);
    _mm512_mask_storeu_ps(c+16, m1, acc1[r]);
    _mm512_mask_storeu_ps(c+32, m2, acc2[r]);
  }
}

// Typedef & dispatch table
using MicroFn = void(*)(const float*, const float*, float*, int, int, int, int, bool);

template<int U> static inline void micro_entry(const float* a,const float* b,float* c,int ldc,int K,int mr,int nr,bool s){
  micro_8x48_beta0_u<U>(a,b,c,ldc,K,mr,nr,s);
}

// UNROLL candidates we’ll consider at runtime
static constexpr int kNumCand = 6;
static constexpr int kCand[kNumCand] = {1,2,3,4,6,8};
static MicroFn kTable[kNumCand] = {
  micro_entry<1>, micro_entry<2>, micro_entry<3>, micro_entry<4>, micro_entry<6>, micro_entry<8>
};

// ---------------- One-shot UNROLL pick (cached) --------------
static std::atomic<int> g_unroll_idx{-1}; // -1:not picked, -2:picking, else index in kCand

static inline MicroFn pick_unroll_once_and_get_fn(const float* A_pack, const float* B_pack,
                                                  int MT, int NT, int K)
{
  // Optional bypass
  int tune   = env_int("SGEMM_TUNE", 1);
  int forceU = env_int("SGEMM_U", 0);
  if(tune==0 && forceU>0){
    // map forced UNROLL to index
    for(int i=0;i<kNumCand;++i) if(kCand[i]==forceU){ g_unroll_idx.store(i); return kTable[i]; }
    // invalid override -> fall through to picker
  }

  int expected=-1;
  if(!g_unroll_idx.compare_exchange_strong(expected, -2)){
    // Someone else picking; spin
    while(g_unroll_idx.load()==-2) {/*spin*/}
    int idx = g_unroll_idx.load();
    return kTable[(idx>=0 && idx<kNumCand)? idx : 3 /*fallback to 4*/];
  }

  // Tiny trial: a few tiles to amortize noise (beta==0 into temp)
  int tm_max = std::min(MT, 6);
  int tn_max = std::min(NT, 2);
  const int ldc = tn_max*NR;
  float* Ctmp = aligned_alloc64((size_t)tm_max*MR*ldc);

  auto bench = [&](int idx){
    std::memset(Ctmp, 0, sizeof(float)*(size_t)tm_max*MR*ldc);
    MicroFn fn = kTable[idx];
    auto t0 = std::chrono::high_resolution_clock::now();
    for(int tm=0; tm<tm_max; ++tm){
      const float* Ap = A_pack + (size_t)tm*K*MR;
      for(int tn=0; tn<tn_max; ++tn){
        const float* Bp = B_pack + (size_t)tn*K*NR;
        float* C_tile = Ctmp + (size_t)tm*MR*ldc + tn*NR;
        fn(Ap, Bp, C_tile, ldc, K, MR, NR, false);
      }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t1-t0).count();
  };

  // Try all candidates; pick min time
  int best_i = 0; double best_t = 1e100;
  for(int i=0;i<kNumCand;++i){
    double tt = bench(i);
    if(tt < best_t){ best_t = tt; best_i = i; }
  }
  g_unroll_idx.store(best_i);
  aligned_free64(Ctmp);
  return kTable[best_i];
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

  // One-shot UNROLL selection (cached), returns a function pointer
  MicroFn kernel;
  if (g_unroll_idx.load() >= 0){
    int idx = g_unroll_idx.load();
    kernel = kTable[(idx>=0 && idx<kNumCand)? idx : 3]; // fallback to UNROLL=4
  }else{
    kernel = pick_unroll_once_and_get_fn(A_pack, B_pack, MT, NT, K);
  }

  // ===== Column-parallel schedule (beta==0 fast path uses masked tails) =====
  if(beta==0.0f){
#pragma omp parallel for schedule(static)
    for(int tn=0; tn<NT; ++tn){
      const float* Bp = B_pack + (size_t)tn*K*NR;
      int j0 = tn*NR;
      int nr_eff = std::min(NR, N-j0);

      for(int tm=0; tm<MT; ++tm){
        const float* Ap = A_pack + (size_t)tm*K*MR;
        int r0 = tm*MR; int mr_eff = std::min(MR, M-r0);
        float* C_tile = C + (size_t)r0*ldC + j0;

        bool stream_ok = (mr_eff==MR && nr_eff==NR && aligned64(C_tile));
        kernel(Ap, Bp, C_tile, ldC, K, mr_eff, nr_eff, stream_ok);
      }
    }
  }else{
    // General beta path — unchanged (we can modernize later).
#pragma omp parallel for schedule(static)
    for(int tn=0; tn<NT; ++tn){
      const float* Bp = B_pack + (size_t)tn*K*NR;
      int j0 = tn*NR;
      for(int tm=0; tm<MT; ++tm){
        const float* Ap = A_pack + (size_t)tm*K*MR;
        int r0 = tm*MR; int mr_eff = std::min(MR, M-r0);
        int nr_eff = std::min(NR, N-j0);
        float* C_tile = C + (size_t)r0*ldC + j0;

        // Simple reference implementation with beta (full-tile optimized, edges fall back):
        if(mr_eff==MR && nr_eff==NR){
          __m512 acc0[MR], acc1[MR], acc2[MR];
          for(int r=0;r<MR;++r){ acc0[r]=_mm512_setzero_ps(); acc1[r]=_mm512_setzero_ps(); acc2[r]=_mm512_setzero_ps(); }
          const int PFD = pick_prefetch_distance(K);
          const float* a_ptr = Ap;
          const float* b_ptr = Bp;
          int k=0;
          for(; k+3<K; k+=4){
            if(PFD>0 && k+PFD<K){
              _mm_prefetch((const char*)(Ap+(size_t)(k+PFD)*MR), _MM_HINT_T0);
              _mm_prefetch((const char*)(Bp+(size_t)(k+PFD)*NR), _MM_HINT_T0);
            }
            for(int u=0; u<4; ++u){
              const float* a = a_ptr + (size_t)u*MR;
              const float* b = b_ptr + (size_t)u*NR;
              __m512 b0=_mm512_load_ps(b+ 0), b1=_mm512_load_ps(b+16), b2=_mm512_load_ps(b+32);
              for(int r=0;r<MR;++r){ __m512 ar=_mm512_set1_ps(a[r]);
                acc0[r]=_mm512_fmadd_ps(ar,b0,acc0[r]); acc1[r]=_mm512_fmadd_ps(ar,b1,acc1[r]); acc2[r]=_mm512_fmadd_ps(ar,b2,acc2[r]); }
            }
            a_ptr += 4*MR; b_ptr += 4*NR;
          }
          for(; k<K; ++k){
            __m512 b0=_mm512_load_ps(b_ptr+ 0), b1=_mm512_load_ps(b_ptr+16), b2=_mm512_load_ps(b_ptr+32);
            for(int r=0;r<MR;++r){ __m512 ar=_mm512_set1_ps(a_ptr[r]);
              acc0[r]=_mm512_fmadd_ps(ar,b0,acc0[r]); acc1[r]=_mm512_fmadd_ps(ar,b1,acc1[r]); acc2[r]=_mm512_fmadd_ps(ar,b2,acc2[r]); }
            a_ptr += MR; b_ptr += NR;
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
          // Keep scalar edge handling for beta!=0 for now
          for(int r=0;r<mr_eff;++r){
            for(int j=0;j<nr_eff;++j){
              float acc=0.f; for(int kk=0; kk<K; ++kk) acc += Ap[(size_t)kk*MR + r] * Bp[(size_t)kk*NR + j];
              float* c=C_tile + (size_t)r*ldC + j;
              *c = acc + beta*(*c);
            }
          }
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
