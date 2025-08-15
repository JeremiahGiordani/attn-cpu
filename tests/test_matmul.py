import numpy as np, torch
import attn_cpu as cpu

M, K, N = 128, 192, 96
A = np.random.randn(M, K).astype(np.float32)
B = np.random.randn(K, N).astype(np.float32)

C_cpu = cpu.gemm(A, B)          # your kernel
C_tch = (torch.from_numpy(A) @ torch.from_numpy(B)).numpy()

print(np.allclose(C_cpu, C_tch, atol=1e-5))
