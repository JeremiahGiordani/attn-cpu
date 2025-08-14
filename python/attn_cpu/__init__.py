# src/attn_cpu/__init__.py
from typing import Literal
import numpy as np
from ._attn_cpu import (  # type: ignore,
    mha_block_dense as _mha_block_dense,
)

def mha_block_dense(x: np.ndarray, W_in: np.ndarray, b_in: np.ndarray, W_out: np.ndarray, b_out: np.ndarray, num_heads: int, causal: bool = False) -> np.ndarray:
    return _mha_block_dense(x, W_in, b_in, W_out, b_out, num_heads, causal)
