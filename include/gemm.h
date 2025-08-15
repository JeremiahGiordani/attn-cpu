#pragma once
#include <cstddef>

namespace gemm {

// Row-major, float32, contiguous.
// A: [M x K], B: [K x N], C: [M x N]
// Computes: C = alpha * (A @ B) + beta * C
// Tunables default to reasonable AVX-512 FP32 values; safe on non-AVX too (scalar loops inside).
void sgemm_blocked(const float* A, int M, int K,
                   const float* B, int N,
                   float* C,
                   float alpha = 1.0f, float beta = 0.0f,
                   int Mb = 192, int Nb = 144, int Kb = 256,
                   int mr = 16, int nr = 12, int ku = 4);

} // namespace gemm
