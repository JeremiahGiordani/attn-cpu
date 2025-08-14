#!/usr/bin/env python3
import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn

import attn_cpu  # ensure PYTHONPATH points to your build/ dir

def extract_in_out(mha: nn.MultiheadAttention):
    D = mha.embed_dim
    # PyTorch packs Q/K/V into in_proj_weight: [3D, D], in_proj_bias: [3D]
    W_in = mha.in_proj_weight.detach().cpu().numpy().astype(np.float32)
    b_in = mha.in_proj_bias.detach().cpu().numpy().astype(np.float32)
    W_out = mha.out_proj.weight.detach().cpu().numpy().astype(np.float32)
    b_out = mha.out_proj.bias.detach().cpu().numpy().astype(np.float32)
    return W_in, b_in, W_out, b_out

def make_attn_mask(T: int, causal: bool):
    if not causal:
        return None
    mask = torch.full((T, T), float("-inf"), dtype=torch.float32)
    return torch.triu(mask, diagonal=1)  # block future (upper triangle)

def bench(func, warmup: int, iters: int):
    # warmup
    for _ in range(warmup):
        func()
    # measured
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        func()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    mean_s = sum(times) / len(times)
    return mean_s, times

def main():
    p = argparse.ArgumentParser(description="Benchmark MHA: attn_cpu vs PyTorch (CPU)")
    p.add_argument("--D", type=int, default=128, help="Model dim")
    p.add_argument("--H", type=int, default=8, help="Num heads")
    p.add_argument("--T", type=int, default=1024, help="Sequence length (L=T)")
    p.add_argument("--causal", action="store_true", help="Use causal mask")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--threads", type=int, default=1, help="Torch CPU threads (0=leave default)")
    p.add_argument("--check", action="store_true", help="Check numeric closeness once before timing")
    args = p.parse_args()

    D, H, T = args.D, args.H, args.T

    D = 128
    H = 4
    T = 1024

    causal = args.causal
    causal = False


    assert D % H == 0, "D must be divisible by H"

    if args.threads > 0:
        # Keep it explicit/reproducible
        torch.set_num_threads(args.threads)
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
        os.environ["MKL_NUM_THREADS"] = str(args.threads)

    torch.manual_seed(0)
    rng = torch.Generator().manual_seed(0)

    mha = nn.MultiheadAttention(embed_dim=D, num_heads=H, batch_first=True, dropout=0.0, bias=True)
    x_t = torch.randn(1, T, D, dtype=torch.float32, generator=rng)  # [B=1,T,D]

    # Precompute everything needed for both paths (avoid conversions in timed loops)
    W_in, b_in, W_out, b_out = extract_in_out(mha)
    x_np = x_t[0].detach().cpu().contiguous().numpy().astype(np.float32)   # [T,D] for our binding
    attn_mask = make_attn_mask(T, causal)

    # Optional one-time correctness check (not timed)
    with torch.inference_mode():
        y_torch, _ = mha(x_t, x_t, x_t, attn_mask=attn_mask, need_weights=False)
    y_cpp = attn_cpu.mha_block_dense(x_np, W_in, b_in, W_out, b_out, H, causal)
    y_ref = y_torch[0].detach().cpu().numpy().astype(np.float32)
    max_abs = float(np.max(np.abs(y_cpp - y_ref)))
    ok = np.allclose(y_cpp, y_ref, atol=1e-5, rtol=1e-6)
    print(f"[check] allclose={ok}, max_abs={max_abs:.3e}")

    # Functions to time
    def run_attn_cpu():
        attn_cpu.mha_block_dense(x_np, W_in, b_in, W_out, b_out, H, causal)

    @torch.inference_mode()
    def run_torch():
        mha(x_t, x_t, x_t, attn_mask=attn_mask, need_weights=False)

    # Bench
    mean_torch_s,     times_torch   = bench(run_torch,    args.warmup, args.iters)

    mean_attn_cpu_s, times_attn_cpu = bench(run_attn_cpu, args.warmup, args.iters)

    # Report
    def ms(x): return x * 1000.0
    print("\n=== Benchmark Results (CPU) ===")
    print(f"Shape: D={D}, H={H}, T={T}, causal={causal}")
    print(f"Threads: torch={torch.get_num_threads()}, OMP={os.environ.get('OMP_NUM_THREADS','')}")
    print(f"Iterations: warmup={args.warmup}, measured={args.iters}")
    print(f"attn_cpu.mha_block_dense : {ms(mean_attn_cpu_s):8.3f} ms/iter  "
          f"(min {ms(min(times_attn_cpu)):.3f}, max {ms(max(times_attn_cpu)):.3f})")
    print(f"torch.nn.MultiheadAttention: {ms(mean_torch_s):8.3f} ms/iter  "
          f"(min {ms(min(times_torch)):.3f}, max {ms(max(times_torch)):.3f})")

if __name__ == "__main__":
    main()
