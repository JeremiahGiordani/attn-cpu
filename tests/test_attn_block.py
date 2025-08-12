# tests/test_mha_block_vs_torch.py
import torch, numpy as np
import torch.nn as nn
from attn_cpu.mha_np import mha_forward_dense

def run_case(D=128, H=8, T=64, causal=False, seed=0):
    torch.manual_seed(seed)
    mha = nn.MultiheadAttention(embed_dim=D, num_heads=H, batch_first=True, dropout=0.0, bias=True)
    x = torch.randn(1, T, D, dtype=torch.float32)  # [B=1,T,D]

    # Extract fused weights (PyTorch stores in_proj as [3D, D])
    W_in = mha.in_proj_weight.detach().cpu().numpy()
    b_in = mha.in_proj_bias.detach().cpu().numpy()
    W_out = mha.out_proj.weight.detach().cpu().numpy()
    b_out = mha.out_proj.bias.detach().cpu().numpy()

    # Our path
    y_np = mha_forward_dense(x[0].cpu().numpy(), W_in, b_in, W_out, b_out, H, causal=causal)

    # Torch reference (self-attention; if causal=True use attn_mask)
    attn_mask = None
    if causal:
        attn_mask = torch.full((T, T), float("-inf"))
        attn_mask = torch.triu(attn_mask, diagonal=1)  # mask future

    y_torch, _ = mha(x, x, x, attn_mask=attn_mask, need_weights=False)
    y_ref = y_torch[0].detach().cpu().numpy()

    ok = np.allclose(y_np, y_ref, atol=1e-5, rtol=1e-6)
    print("OK:", ok, "max_abs:", np.max(np.abs(y_np - y_ref)))
    assert ok

if __name__ == "__main__":
    run_case(D=128, H=8, T=64, causal=False)
    run_case(D=128, H=8, T=64, causal=True)
