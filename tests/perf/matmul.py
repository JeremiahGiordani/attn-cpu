#!/usr/bin/env python3
# test_matmul_avg.py
#
# Measure average runtime of several matrix multiplications
# over N_RUNS iterations, including JAX eager and JIT.

import os
import time
import torch
import numpy as np

# If you want to force JAX to CPU explicitly, set before importing jax:
# os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
import jax
import jax.numpy as jnp

from attn_cpu import (
    gemm,
    gemm_jit
)

# ----------------------------------------------------------------------
# Experiment parameters
# ----------------------------------------------------------------------
M = 2048
K = 1280
N = 960

# Benchmark dimensions:
N_RUNS = 20     # <-- number of repetitions
SEED    = 42
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Data generation (identical to original script)
# ----------------------------------------------------------------------
torch.manual_seed(SEED)
np.random.seed(SEED)

A = np.random.randn(M, K).astype(np.float32)
B = np.random.randn(K, N).astype(np.float32)

A_t = torch.tensor(A)
B_t = torch.tensor(B)

# JAX arrays
A_j = jnp.asarray(A)
B_j = jnp.asarray(B)

num_threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
torch.set_num_threads(num_threads)
torch.set_num_interop_threads(1)

# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------
def sync_result(x):
    """Synchronize any async work so wall times are accurate."""
    # JAX arrays have block_until_ready
    try:
        if hasattr(x, "block_until_ready"):
            x.block_until_ready()
    except Exception:
        pass
    # Torch CUDA safety (not expected here, but harmless)
    if hasattr(x, "device") and hasattr(x, "is_cuda") and x.is_cuda:
        torch.cuda.synchronize()

def timed_avg(fn, n=N_RUNS):
    total = 0.0
    for _ in range(n):
        t0 = time.perf_counter()
        out = fn()
        sync_result(out)
        t1 = time.perf_counter()
        total += (t1 - t0)
    return total / n   # seconds

# ----------------------------------------------------------------------
# Define callables we want to benchmark
# ----------------------------------------------------------------------
def torch_run():
    return torch.matmul(A_t, B_t)

def numpy_run():
    return A @ B

def cpu_run():
    # your custom backend
    return gemm(A, B)

def cpu_jit_run():
    # your custom backend
    return gemm_jit(A, B)

def jax_eager_run():
    return A_j @ B_j

# JIT-compiled matmul
@jax.jit
def jax_jit_matmul(a, b):
    return a @ b

def jax_jit_run():
    return jax_jit_matmul(A_j, B_j)

# ----------------------------------------------------------------------
# Correctness check (one-shot)
# ----------------------------------------------------------------------
print("=== Verifying correctness ===")
out_torch = torch_run().numpy()
out_numpy = numpy_run()
out_cpu   = cpu_run()
out_cpu_jit   = cpu_jit_run()
out_jax_eager = np.asarray(jax_eager_run())
# Warm up the JIT once to compile (don’t include this in timing)
_ = jax_jit_run().block_until_ready()
out_jax_jit = np.asarray(jax_jit_run())

print("Torch vs NumPy :", np.allclose(out_torch, out_numpy, atol=1e-4))
print("Torch vs GEMM  :", np.allclose(out_torch, out_cpu,   atol=1e-4))
print("Torch vs GEMM-JIT:", np.allclose(out_torch, out_cpu_jit,   atol=1e-4))
print("Torch vs JAX   :", np.allclose(out_torch, out_jax_eager, atol=1e-4))
print("Torch vs JAX-JIT:", np.allclose(out_torch, out_jax_jit, atol=1e-4))
print()

# ----------------------------------------------------------------------
# Benchmark
# ----------------------------------------------------------------------
print(f"=== Average runtime over {N_RUNS:,} runs ===")
print(f"=== M: {M:,}, K: {K:,}, N: {N} ===")
print(f"=== Threads: {num_threads} ===")

print(f"[PyTorch]       {timed_avg(torch_run)*1000:.3f} ms")
print(f"[NumPy]         {timed_avg(numpy_run)*1000:.3f} ms")
print(f"[GEMM]          {timed_avg(cpu_run)*1000:.3f} ms")
print(f"[GEMM JIT]      {timed_avg(cpu_jit_run)*1000:.3f} ms")
print(f"[JAX eager]     {timed_avg(jax_eager_run)*1000:.3f} ms")
# JIT warmup already done above; measure steady-state
print(f"[JAX jit]       {timed_avg(jax_jit_run)*1000:.3f} ms")
