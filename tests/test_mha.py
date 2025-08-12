import numpy as np
import torch
import torch.nn as nn
import attn_cpu

def extract_in_out(mha: nn.MultiheadAttention):
    D = mha.embed_dim
    if hasattr(mha, "in_proj_weight") and mha.in_proj_weight is not None:
        W_in = mha.in_proj_weight.detach().cpu().numpy().astype(np.float32)
        b_in = mha.in_proj_bias.detach().cpu().numpy().astype(np.float32)
    else:
        # Separate q/k/v weights (rare for default MHA, but handle anyway)
        Wq = mha.q_proj_weight.detach().cpu().numpy().astype(np.float32)
        Wk = mha.k_proj_weight.detach().cpu().numpy().astype(np.float32)
        Wv = mha.v_proj_weight.detach().cpu().numpy().astype(np.float32)
        bq = mha.bias_k.detach().cpu().numpy().astype(np.float32) if mha.bias_k is not None else np.zeros(D, np.float32)
        bk = mha.bias_k.detach().cpu().numpy().astype(np.float32) if mha.bias_k is not None else np.zeros(D, np.float32)
        bv = mha.bias_v.detach().cpu().numpy().astype(np.float32) if mha.bias_v is not None else np.zeros(D, np.float32)
        W_in = np.concatenate([Wq, Wk, Wv], axis=0)
        b_in = np.concatenate([bq, bk, bv], axis=0)

    W_out = mha.out_proj.weight.detach().cpu().numpy().astype(np.float32)
    b_out = mha.out_proj.bias.detach().cpu().numpy().astype(np.float32)
    return W_in, b_in, W_out, b_out

def run_case(D=128, H=8, T=64, causal=False, seed=0):
    torch.manual_seed(seed)
    mha = nn.MultiheadAttention(embed_dim=D, num_heads=H, batch_first=True, dropout=0.0, bias=True)

    with torch.no_grad():
        mha.in_proj_bias.zero_()

    # Zero out the output projection bias
    with torch.no_grad():
        mha.out_proj.bias.zero_()
    x = torch.randn(1, T, D, dtype=torch.float32)  # [B=1,T,D]

    W_in, b_in, W_out, b_out = extract_in_out(mha)

    y_cpp = attn_cpu.mha_block_dense(
        x[0].cpu().numpy(), W_in, b_in, W_out, b_out, H, causal=causal
    )

    attn_mask = None
    if causal:
        attn_mask = torch.full((T, T), float("-inf"))
        attn_mask = torch.triu(attn_mask, diagonal=1)
    y_torch, _ = mha(x, x, x, attn_mask=attn_mask, need_weights=False)

    y_ref = y_torch[0].detach().cpu().numpy().astype(np.float32)
    ok = np.allclose(y_cpp, y_ref, atol=1e-5, rtol=1e-6)
    # print("torch shape:")
    # print(y_ref.shape)
    # print("Attn CPU shape")
    # print(y_cpp.shape)
    # print("torch ref:")
    # print(y_ref)
    # print("Attn CPU")
    # print(y_cpp)


    if ok:
        print("Correct:", ok)
    else:
        print("Correct:", ok, "max_abs:", float(np.max(np.abs(y_cpp - y_ref))))
    # assert ok

if __name__ == "__main__":
    run_case(D=128, H=2, T=65, causal=False, seed=42)
    run_case(D=128, H=2, T=66, causal=True,  seed=123)
