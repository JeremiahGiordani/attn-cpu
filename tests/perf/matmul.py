#!/usr/bin/env python3
# test_matmul_avg.py
#
# Measure average runtime of several matrix-vector multiplications
# over N_RUNS iterations.
import os
import time
import torch
import numpy as np

from attn_cpu import (
    gemm,
)

# ----------------------------------------------------------------------
# Experiment parameters
# ----------------------------------------------------------------------
M = 2048
K = 1280
N = 960

# Benchmark dimensions:
N_RUNS     = 20     # <-- number of repetitions
SEED       = 42
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Data generation (identical to original script)
# ----------------------------------------------------------------------
torch.manual_seed(SEED)

A = np.random.randn(M, K).astype(np.float32)
B  = np.random.randn(K, N).astype(np.float32)

A_t   = torch.tensor(A)
B_t    = torch.tensor(B)

num_threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
torch.set_num_threads(num_threads)
torch.set_num_interop_threads(1)
num_threads = num_threads


# ----------------------------------------------------------------------
# Utility: average wall-time over n runs
# ----------------------------------------------------------------------
def timed_avg(fn, n=N_RUNS):
    total = 0.0
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
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
    return gemm(A, B)

# ----------------------------------------------------------------------
# Correctness check (one-shot)
# ----------------------------------------------------------------------
print("=== Verifying correctness ===")
out_torch  = torch_run().numpy()
print("Torch vs NumPy :", np.allclose(out_torch, numpy_run(), atol=1e-4))
print("Torch vs GEMM:", np.allclose(out_torch, cpu_run(), atol=1e-4))
print()

# ----------------------------------------------------------------------
# Benchmark
# ----------------------------------------------------------------------
print(f"=== Average runtime over {N_RUNS:,} runs ===")
print(f"=== Sparsity: M: {M:,}, K: {K:,}, N: {N} ===")
print(f"=== Threads: {num_threads} ===")
print(f"[PyTorch]      {timed_avg(torch_run)*1000:.3f} ms")
print(f"[NumPy]        {timed_avg(numpy_run)*1000:.3f} ms")
print(f"[GEMM]         {timed_avg(cpu_run)*1000:.3f} ms")