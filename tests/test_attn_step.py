# tests/test_dense_vs_torch.py
import os
import sys
import math
import numpy as np
import torch
import torch.nn.functional as F

import attn_cpu


def numpy_from_torch(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().contiguous().numpy().astype(np.float32)


def run_one_case(H: int, Dh: int, T: int, seed: int = 0, atol=1e-5, rtol=1e-6):
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)

    # Shapes we agreed on:
    # Q: [H, Dh]
    # K: [H, T, Dh]
    # V: [H, T, Dh]
    Q = torch.randn(H, Dh, dtype=torch.float32, generator=rng)
    K = torch.randn(H, T, Dh, dtype=torch.float32, generator=rng)
    V = torch.randn(H, T, Dh, dtype=torch.float32, generator=rng)

    # --- Our implementation (NumPy arrays) ---
    Q_np = numpy_from_torch(Q)
    K_np = numpy_from_torch(K)
    V_np = numpy_from_torch(V)
    O_np = attn_cpu.attn_step_dense(Q_np, K_np, V_np)  # [H, Dh]

    # --- PyTorch reference using scaled_dot_product_attention ---
    # SDPA expects (B, H, L, Dh) for Q and (B, H, S, Dh) for K/V.
    q = Q.unsqueeze(0).unsqueeze(2)        # [1, H, 1, Dh]
    k = K.unsqueeze(0)                     # [1, H, T, Dh]
    v = V.unsqueeze(0)                     # [1, H, T, Dh]
    O_torch = F.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
    ).squeeze(0).squeeze(1)                # -> [H, Dh]

    O_ref = numpy_from_torch(O_torch)

    # Compare
    if not np.allclose(O_np, O_ref, atol=atol, rtol=rtol):
        max_abs = float(np.max(np.abs(O_np - O_ref)))
        max_rel = float(np.max(np.abs(O_np - O_ref) / (np.abs(O_ref) + 1e-12)))
        raise AssertionError(
            f"Mismatch for (H={H}, Dh={Dh}, T={T})  "
            f"max_abs={max_abs:.3e}, max_rel={max_rel:.3e}"
        )
    return True


def test_small_cases():
    # Quick sweep of lightweight shapes for CI-ish speed
    for H, Dh, T in [
        (1, 64, 1),
        (4, 64, 128),
        (8, 64, 1024),
        (8, 128, 512),
        (16, 96, 256),   # non-multiple-of-16 Dh
    ]:
        assert run_one_case(H, Dh, T, seed=42)


if __name__ == "__main__":
    # Heavier sweep when running as a script
    cases = [
        (1, 64, 1),
        (4, 64, 128),
        (8, 64, 1024),
        (8, 128, 1024),
        (8, 96, 2048),
        (16, 64, 4096),
    ]
    ok = 0
    for (H, Dh, T) in cases:
        run_one_case(H, Dh, T, seed=123)
        print(f"✔ Passed: H={H}, Dh={Dh}, T={T}")
        ok += 1
    print(f"\nAll good — {ok}/{len(cases)} cases passed.")
