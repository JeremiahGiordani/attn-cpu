# python/mha_np.py
import numpy as np
import attn_cpu

def mha_forward_dense(x, W_in, b_in, W_out, b_out, num_heads, causal=False):
    """
    x:      [T, D]
    W_in:   [3D, D]   (PyTorch in_proj_weight)
    b_in:   [3D]
    W_out:  [D, D]    (PyTorch out_proj.weight)
    b_out:  [D]
    num_heads: H
    returns y: [T, D]
    """
    T, D = x.shape
    H = num_heads
    Dh = D // H
    assert D % H == 0

    # in-proj: [T,D] @ [D,3D]^T + b -> [T,3D]
    qkv = x @ W_in.T + b_in  # [T,3D]
    q, k, v = np.split(qkv, 3, axis=1)  # each [T,D]

    # reshape to heads: [T,D] -> [H,T,Dh]
    def to_heads(a):
        a = a.reshape(T, H, Dh)       # [T,H,Dh]
        return np.transpose(a, (1,0,2))  # [H,T,Dh]

    Q = to_heads(q).astype(np.float32, copy=False)
    K = to_heads(k).astype(np.float32, copy=False)
    V = to_heads(v).astype(np.float32, copy=False)

    # full block attention (L = T)
    O = attn_cpu.attn_block_dense(Q, K, V, causal=causal)  # [H,T,Dh]

    # concat heads back: [H,T,Dh] -> [T,D]
    O = np.transpose(O, (1,0,2)).reshape(T, D)

    # out-proj: [T,D] @ [D,D]^T + b
    y = O @ W_out.T + b_out
    return y.astype(np.float32, copy=False)
