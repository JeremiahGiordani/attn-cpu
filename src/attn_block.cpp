#include "attn.h"
#include <algorithm>
#include <cmath>
#include <vector>

namespace attn {

static inline const float *q_ptr(const TensorF32 &Q, int h, int l) {
    return Q.data + h * Q.strideH + l * Q.strideT;
}
static inline const float *k_ptr(const TensorF32 &K, int h, int t) {
    return K.data + h * K.strideH + t * K.strideT;
}
static inline const float *v_ptr(const TensorF32 &V, int h, int t) {
    return V.data + h * V.strideH + t * V.strideT;
}
static inline float *o_ptr(const MutableTensor3F32 &O, int h, int l) {
    return O.data + h * O.strideH + l * O.strideT;
}

void attn_block_dense(const TensorF32 &Q, const TensorF32 &K,
                      const TensorF32 &V, MutableTensor3F32 &O, bool causal) {
    if (!Q.data || !K.data || !V.data || !O.data)
        throw std::invalid_argument("Null data pointer");
    if (Q.H <= 0 || Q.Dh <= 0 || K.H != Q.H || V.H != Q.H)
        throw std::invalid_argument("H mismatch");
    if (K.Dh != Q.Dh || V.Dh != Q.Dh)
        throw std::invalid_argument("Dh mismatch");
    if (K.T <= 0 || V.T != K.T)
        throw std::invalid_argument("K/V T mismatch");
    if (O.H != Q.H || O.Dh != Q.Dh || O.T != Q.T)
        throw std::invalid_argument("O must be [H,L,Dh] matching Q's [H,L,Dh]");
    if (causal && Q.T != K.T)
        throw std::invalid_argument("causal=true requires L==T");

    const int H = Q.H;
    const int Dh = Q.Dh;
    const int L = Q.T; // query length
    const int T = K.T; // key length

    const float inv_sqrt_dh = 1.0f / std::sqrt(static_cast<float>(Dh));
    std::vector<float> logits(T);

    for (int h = 0; h < H; ++h) {
        for (int l = 0; l < L; ++l) {
            const float *q = q_ptr(Q, h, l);
            // Pass 1: logits + max (respect causal mask if set)
            float m = -INFINITY;
            for (int t = 0; t < T; ++t) {
                if (causal && t > l) {
                    logits[t] = -INFINITY;
                    continue;
                }
                const float *k = k_ptr(K, h, t);
                float dot = 0.f;
                for (int d = 0; d < Dh; ++d)
                    dot += q[d * Q.strideDh] * k[d * K.strideDh];
                float lgt = dot * inv_sqrt_dh;
                logits[t] = lgt;
                if (lgt > m)
                    m = lgt;
            }
            // Pass 2: softmax + value mix
            float denom = 0.f;
            float *out = o_ptr(O, h, l);
            for (int d = 0; d < Dh; ++d)
                out[d * O.strideDh] = 0.f;

            for (int t = 0; t < T; ++t) {
                if (causal && t > l)
                    continue;
                float w = std::exp(logits[t] - m);
                denom += w;
                const float *v = v_ptr(V, h, t);
                for (int d = 0; d < Dh; ++d)
                    out[d * O.strideDh] += w * v[d * V.strideDh];
            }
            const float inv_denom = 1.f / denom;
            for (int d = 0; d < Dh; ++d)
                out[d * O.strideDh] *= inv_denom;
        }
    }
}

} // namespace attn
