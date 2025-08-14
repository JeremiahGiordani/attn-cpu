#pragma once
#include "common.h"
#include "simd_math.h"
#include <limits>
#include <cmath>

namespace attn {
struct RunningSoftmax {
  float m = -std::numeric_limits<float>::infinity();
  float l = 0.f;
};
ATTN_ALWAYS_INLINE void update_running(RunningSoftmax& rs, float block_max, float block_sum) {
  const float m_new = (rs.m > block_max) ? rs.m : block_max;
  const float alpha = std::exp(rs.m - m_new);
  rs.l = rs.l * alpha + block_sum;
  rs.m = m_new;
}
} // namespace attn
