#include "attn.h"
#include <algorithm>
#include <cmath>
#include <vector>

namespace attn {

static inline const float *ptr_Q(const TensorF32 &Q, int h) {
    return Q.data + h * Q.strideH;
}
static inline const float *ptr_K(const TensorF32 &K, int h, int t) {
    return K.data + h * K.strideH + t * K.strideT;
}
static inline const float *ptr_V(const TensorF32 &V, int h, int t) {
    return V.data + h * V.strideH + t * V.strideT;
}
static inline float *ptr_O(const MutableTensorF32 &O, int h) {
    return O.data + h * O.strideH;
}

void attn_step_dense(const TensorF32 &Q, const TensorF32 &K, const TensorF32 &V,
                     MutableTensorF32 &O) {
    validate_dense_shapes(Q, K, V, O);

    const int H = Q.H;
    const int Dh = Q.Dh;
    const int T = K.T;

    const float inv_sqrt_dh = 1.0f / std::sqrt(static_cast<float>(Dh));

    // Temporary storage for logits per head (size T).
    std::vector<float> logits(T);

    for (int h = 0; h < H; ++h) {
        const float *q = ptr_Q(Q, h);
        float *out = ptr_O(O, h);

        // Pass 1: compute logits and track max
        float m = -INFINITY;
        for (int t = 0; t < T; ++t) {
            const float *krow = ptr_K(K, h, t);
            // dot(q, krow)
            float dot = 0.0f;
            for (int d = 0; d < Dh; ++d) {
                dot += q[d * Q.strideDh] * krow[d * K.strideDh];
            }
            float l = dot * inv_sqrt_dh;
            logits[t] = l;
            if (l > m)
                m = l;
        }

        // Pass 2: compute weights and accumulate output
        float denom = 0.0f;
        // zero O
        for (int d = 0; d < Dh; ++d)
            out[d * O.strideDh] = 0.0f;

        for (int t = 0; t < T; ++t) {
            float w = std::exp(logits[t] - m);
            denom += w;
            const float *vrow = ptr_V(V, h, t);
            for (int d = 0; d < Dh; ++d) {
                out[d * O.strideDh] += w * vrow[d * V.strideDh];
            }
        }

        const float inv_denom = 1.0f / denom;
        for (int d = 0; d < Dh; ++d) {
            out[d * O.strideDh] *= inv_denom;
        }
    }
}

} // namespace attn
