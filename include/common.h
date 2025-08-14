#pragma once
#include <cstdint>
#include <cstddef>
#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#if defined(__GNUC__)
# define ATTN_ALWAYS_INLINE __attribute__((always_inline)) inline
# define ATTN_LIKELY(x)    __builtin_expect(!!(x),1)
# define ATTN_UNLIKELY(x)  __builtin_expect(!!(x),0)
#else
# define ATTN_ALWAYS_INLINE inline
# define ATTN_LIKELY(x)    (x)
# define ATTN_UNLIKELY(x)  (x)
#endif
