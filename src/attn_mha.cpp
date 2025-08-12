#include "attn.h"
#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace attn {

// Helpers to build Tensor views for [H,T,Dh] contiguous
static inline TensorF32 make_tensor3_const(const float *base, int H, int T,
                                           int Dh) {
    TensorF32 tv;
    tv.data = base;
    tv.H = H;
    tv.T = T;
    tv.Dh = Dh;
    tv.strideH = static_cast<int64_t>(T) * Dh;
    tv.strideT = static_cast<int64_t>(Dh);
    tv.strideDh = 1;
    return tv;
}
static inline MutableTensor3F32 make_tensor3_mut(float *base, int H, int T,
                                                 int Dh) {
    MutableTensor3F32 tv;
    tv.data = base;
    tv.H = H;
    tv.T = T;
    tv.Dh = Dh;
    tv.strideH = static_cast<int64_t>(T) * Dh;
    tv.strideT = static_cast<int64_t>(Dh);
    tv.strideDh = 1;
    return tv;
}

// y = x @ W^T + b
static void matmul_add_bias_T(const float *x, int M, int N, // x:[M,N]
                              const float *W, int K,
                              /*=N*/          // W:[K,N] (rows = outputs)
                              const float *b, // b:[K]
                              float *y) {     // y:[M,K]
    // For each output row k, y[:,k] = x @ W[k,:]^T + b[k]
    for (int m = 0; m < M; ++m) {
        const float *xr = x + m * N;
        float *yr = y + m * K;
        for (int k = 0; k < K; ++k) {
            const float *wk = W + k * N;
            float acc = b ? b[k] : 0.f;
            for (int n = 0; n < N; ++n)
                acc += xr[n] * wk[n];
            yr[k] = acc;
        }
    }
}

void mha_block_dense(const float *x, int T, int D, const float *W_in,
                     const float *b_in,                      // [3D,D],[3D]
                     const float *W_out, const float *b_out, // [D,D],[D]
                     int H, bool causal, float *y_out) {
    if (!x || !W_in || !b_in || !W_out || !b_out || !y_out)
        throw std::invalid_argument("Null pointer in mha_block_dense");
    if (T <= 0 || D <= 0 || H <= 0 || (D % H) != 0)
        throw std::invalid_argument("Invalid T/D/H; require D % H == 0");
    const int Dh = D / H;

    // 1) In-projection: [T,D] @ [3D,D]^T + b -> [T,3D]
    std::vector<float> q((size_t)T * D);
    std::vector<float> k((size_t)T * D);
    std::vector<float> v((size_t)T * D);

    const float *Wq = W_in + 0 * D * D;
    const float *Wk = W_in + 1 * D * D;
    const float *Wv = W_in + 2 * D * D;

    const float *bq = b_in + 0 * D;
    const float *bk = b_in + 1 * D;
    const float *bv = b_in + 2 * D;

    matmul_add_bias_T(x, T, D, Wq, D, bq, q.data());
    matmul_add_bias_T(x, T, D, Wk, D, bk, k.data());
    matmul_add_bias_T(x, T, D, Wv, D, bv, v.data());

    // 2) Reshape to heads: [T,D] -> [H,T,Dh] contiguous
    const size_t head_elems = static_cast<size_t>(H) * T * Dh;
    std::vector<float> Q(head_elems), K(head_elems), V(head_elems);

    for (int t = 0; t < T; ++t) {
  for (int h = 0; h < H; ++h) {
    const float* qsrc = q.data() + static_cast<size_t>(t) * D + h * Dh;
    const float* ksrc = k.data() + static_cast<size_t>(t) * D + h * Dh;
    const float* vsrc = v.data() + static_cast<size_t>(t) * D + h * Dh;

    float* qdst = Q.data() + (static_cast<size_t>(h) * T + t) * Dh;
    float* kdst = K.data() + (static_cast<size_t>(h) * T + t) * Dh;
    float* vdst = V.data() + (static_cast<size_t>(h) * T + t) * Dh;

    std::copy(qsrc, qsrc + Dh, qdst);
    std::copy(ksrc, ksrc + Dh, kdst);
    std::copy(vsrc, vsrc + Dh, vdst);
  }
}

    // 3) Core attention over the block: Q:[H,T,Dh], K/V same, O:[H,T,Dh]
    std::vector<float> O(head_elems);
    TensorF32 Qt = make_tensor3_const(Q.data(), H, T, Dh);
    TensorF32 Kt = make_tensor3_const(K.data(), H, T, Dh);
    TensorF32 Vt = make_tensor3_const(V.data(), H, T, Dh);
    auto Ot = make_tensor3_mut(O.data(), H, T, Dh);

    attn_block_dense(Qt, Kt, Vt, Ot, /*causal*/ causal);

    // 4) Concat heads back to [T,D]
    std::vector<float> Ocat(static_cast<size_t>(T) * D);
    for (int t = 0; t < T; ++t) {
        for (int h = 0; h < H; ++h) {
            const float *osrc = O.data() + (h * T + t) * Dh;
            float *odst = Ocat.data() + t * D + h * Dh;
            std::copy(osrc, osrc + Dh, odst);
        }
    }

    // 5) Output projection: [T,D] @ [D,D]^T + b -> [T,D]
    matmul_add_bias_T(/*x*/ Ocat.data(), /*M*/ T, /*N*/ D,
                      /*W*/ W_out, /*K*/ D,
                      /*b*/ b_out,
                      /*y*/ y_out);
}

} // namespace attn
