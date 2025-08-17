#!/usr/bin/env python3
import os
import subprocess

backends = ["torch", "numpy", "cpu", "jax", "jax_jit"]

env = os.environ.copy()
env["OMP_NUM_THREADS"] = "8"   # pin to 8 threads

print("=== Isolated Benchmarks ===")
for backend in backends:
    print(f"\n>>> Running {backend.upper()} benchmark...")
    result = subprocess.run(
        ["python", "tests/bench/bench_single.py", backend],
        capture_output=True,
        text=True,
        env=env,
    )
    print(result.stdout.strip())
