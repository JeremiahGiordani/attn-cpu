#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>

namespace attn {

// Lightweight, non-owning views over float32 tensors.
// Layout assumption for K/V: contiguous in Dh (fastest), then T, then H.
struct TensorF32 {
  const float* data = nullptr;
  int H = 0, T = 0, Dh = 0;
  int64_t strideH = 0, strideT = 0, strideDh = 0; // in elements (not bytes)
};

struct MutableTensorF32 {
  float* data = nullptr;
  int H = 0, Dh = 0;
  int64_t strideH = 0, strideDh = 0; // in elements
};

struct MutableTensor3F32 {
  float* data = nullptr;
  int H = 0, T = 0, Dh = 0;
  int64_t strideH = 0, strideT = 0, strideDh = 0; // elements
};

// Dense attention (single decode step):
// Q: [H, Dh]
// K: [H, T, Dh]
// V: [H, T, Dh]
// O: [H, Dh]
//
// Numerically stable 2-pass softmax over T per head.
void attn_step_dense(const TensorF32& Q,
                     const TensorF32& K,
                     const TensorF32& V,
                     MutableTensorF32& O);

void attn_block_dense(const TensorF32& Q,
                      const TensorF32& K,
                      const TensorF32& V,
                      MutableTensor3F32& O,
                      bool causal);

void mha_block_dense(const float* x,   int T, int D,
                     const float* W_in, const float* b_in,   // [3D,D], [3D]
                     const float* W_out, const float* b_out, // [D,D],  [D]
                     int num_heads,
                     bool causal,
                     float* y_out);

// Minimal validation helper (throws std::invalid_argument)
inline void validate_dense_shapes(const TensorF32& Q,
                                  const TensorF32& K,
                                  const TensorF32& V,
                                  const MutableTensorF32& O) {
  if (!Q.data || !K.data || !V.data || !O.data) {
    throw std::invalid_argument("Null data pointer");
  }
  if (Q.H <= 0 || Q.Dh <= 0) {
    throw std::invalid_argument("Q has invalid H/Dh");
  }
  if (K.H != Q.H || V.H != Q.H) {
    throw std::invalid_argument("K/V H must match Q.H");
  }
  if (K.Dh != Q.Dh || V.Dh != Q.Dh) {
    throw std::invalid_argument("K/V Dh must match Q.Dh");
  }
  if (K.T <= 0 || V.T != K.T) {
    throw std::invalid_argument("K/V T must be positive and equal");
  }
  if (O.H != Q.H || O.Dh != Q.Dh) {
    throw std::invalid_argument("O must be [H, Dh] matching Q");
  }
  // Strides must be non-zero
  if (Q.strideH == 0 || Q.strideDh == 0 ||
      K.strideH == 0 || K.strideT == 0 || K.strideDh == 0 ||
      V.strideH == 0 || V.strideT == 0 || V.strideDh == 0 ||
      O.strideH == 0 || O.strideDh == 0) {
    throw std::invalid_argument("Zero stride encountered");
  }
}

} // namespace attn
