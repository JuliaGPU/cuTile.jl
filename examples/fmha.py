#!/usr/bin/env python3
"""
Fused Multi-Head Attention (FMHA) example - cuTile Python

Based on cuTile Python's AttentionFMHA.py sample. Implements FlashAttention-2
style online softmax with tiling for Blackwell GPUs.
"""

import cupy as cp
import numpy as np
import cuda.tile as ct
import nvtx
from math import ceil, sqrt
from cuda.tile import RoundingMode as RMd

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]

INV_LOG_2 = 1.0 / np.log(2.0)


@ct.kernel(occupancy=2)
def fmha_kernel(Q, K, V, Out,
                qk_scale: float,
                input_pos: int,
                TILE_D: ConstInt,
                H: ConstInt,
                TILE_M: ConstInt,
                TILE_N: ConstInt,
                QUERY_GROUP_SIZE: ConstInt,
                CAUSAL: ConstBool,
                EVEN_K: ConstBool):
    """
    cuTile kernel for Fused Multi-Head Attention (FMHA).
    Computes attention output for a specific batch item and head,
    using tiling and online softmax.
    """
    # Map block IDs to batch and head indices
    bid_x = ct.bid(0)
    bid_y = ct.bid(1)
    batch_idx = bid_y // H
    head_idx = bid_y % H
    off_kv_h = head_idx // QUERY_GROUP_SIZE

    # Adjust qk_scale for exp2
    qk_scale = qk_scale * INV_LOG_2

    # Initialize offsets for current query tile (M-dimension)
    offs_m = bid_x * TILE_M + ct.arange(TILE_M, dtype=np.int32)
    offs_m += input_pos
    offs_m = offs_m[:, None]  # [TILE_M, 1]

    # Initialize local offsets for key/value tile (N-dimension)
    offs_n_tile = ct.arange(TILE_N, dtype=np.int32)
    offs_n_tile = offs_n_tile[None, :]  # [1, TILE_N]

    # Initialize online softmax accumulators in float32 for stability
    m_i = ct.full((TILE_M, 1), -np.inf, dtype=np.float32)
    l_i = ct.full((TILE_M, 1), 0.0, dtype=np.float32)
    acc = ct.full((TILE_M, TILE_D), 0.0, dtype=np.float32)

    # Load query tile
    q = ct.load(
        Q, index=(batch_idx, head_idx, bid_x, 0), shape=(1, 1, TILE_M, TILE_D)
    ).reshape((TILE_M, TILE_D))

    # Loop bounds
    m_end = input_pos + (bid_x + 1) * TILE_M
    k_seqlen = K.shape[2]
    if CAUSAL:
        mask_start = (input_pos + bid_x * TILE_M) // TILE_N
        mask_start = min(mask_start, k_seqlen // TILE_N)
        Tc = ct.cdiv(min(m_end, k_seqlen), TILE_N)
    else:
        Tc = ct.cdiv(k_seqlen, TILE_N)
        mask_start = k_seqlen // TILE_N

    # Loop over K, V blocks
    for j in range(0, Tc):
        # QK product
        k = ct.load(
            K, index=(batch_idx, off_kv_h, 0, j), shape=(1, 1, TILE_D, TILE_N),
            order=(0, 1, 3, 2),
            latency=2,
        ).reshape((TILE_D, TILE_N))
        qk = ct.full((TILE_M, TILE_N), 0., dtype=np.float32)
        qk = ct.mma(q, k, qk)

        # Causal masking
        if (CAUSAL or not EVEN_K) and j >= mask_start:
            offs_n = j * TILE_N + offs_n_tile
            mask = ct.full((TILE_M, TILE_N), True, dtype=np.bool_)
            if not EVEN_K:
                mask = mask & (offs_n < k_seqlen)
            if CAUSAL:
                mask = mask & (offs_m >= offs_n)
            mask = ct.where(mask, 0.0, -np.inf)
            qk += mask

        # Online softmax
        m_ij = max(m_i, ct.max(qk, axis=-1, keepdims=True) * qk_scale)
        qk = qk * qk_scale - m_ij

        p = ct.exp2(qk, flush_to_zero=True)
        l_ij = ct.sum(p, axis=-1, keepdims=True)
        alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha

        # PV product
        v = ct.load(
            V, index=(batch_idx, off_kv_h, j, 0), shape=(1, 1, TILE_N, TILE_D),
            latency=4,
        ).reshape((TILE_N, TILE_D))
        p = p.astype(Q.dtype)
        acc = ct.mma(p, v, acc)
        m_i = m_ij

    # Final normalization and store
    acc = ct.truediv(acc, l_i, flush_to_zero=True, rounding_mode=RMd.APPROX)
    acc = acc.reshape((1, 1, TILE_M, TILE_D)).astype(Out.dtype)
    ct.store(Out, index=(batch_idx, head_idx, bid_x, 0), tile=acc)


#=============================================================================
# Host-side wrapper
#=============================================================================

def cutile_fmha(Q, K, V, qk_scale=None, input_pos=0, tile_m=128, tile_n=128,
                query_group_size=1, causal=False):
    Batch, Heads, SeqLen_Q, D_k = Q.shape
    _, KV_Heads, SeqLen_KV, D_v = V.shape
    even_k = (SeqLen_KV % tile_n) == 0

    if qk_scale is None:
        qk_scale = 1.0 / sqrt(D_k)

    Out = cp.empty((Batch, Heads, SeqLen_Q, D_v), dtype=Q.dtype)

    grid_x = ceil(SeqLen_Q / tile_m)
    grid_y = Batch * Heads
    grid = (grid_x, grid_y, 1)
    stream = cp.cuda.get_current_stream()

    ct.launch(stream, grid, fmha_kernel, (
        Q, K, V, Out,
        qk_scale, input_pos,
        D_k, Heads, tile_m, tile_n,
        query_group_size, causal, even_k
    ))

    return Out


#=============================================================================
# Reference implementation
#=============================================================================

def ref_fmha(Q, K, V, qk_scale=None, causal=False):
    """Simple NumPy FMHA for correctness checking."""
    Q = cp.asnumpy(Q).astype(np.float32)
    K = cp.asnumpy(K).astype(np.float32)
    V = cp.asnumpy(V).astype(np.float32)

    B, H, M, D = Q.shape
    _, KH, N, _ = K.shape

    if qk_scale is None:
        qk_scale = 1.0 / sqrt(D)

    # Expand KV heads to match Q heads
    if KH != H:
        group = H // KH
        K = np.repeat(K, group, axis=1)
        V = np.repeat(V, group, axis=1)

    # QK^T
    scores = np.einsum('bhmd,bhnd->bhmn', Q, K) * qk_scale

    if causal:
        mask = np.triu(np.ones((M, N), dtype=bool), k=1)
        scores[:, :, mask] = -np.inf

    # Softmax
    scores_max = scores.max(axis=-1, keepdims=True)
    scores_max = np.where(np.isinf(scores_max), 0.0, scores_max)
    exp_scores = np.exp(scores - scores_max)
    attn = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

    # Weighted sum
    out = np.einsum('bhmn,bhnd->bhmd', attn, V)
    return out


#=============================================================================
# Example harness
#=============================================================================

def prepare(*, benchmark=False,
            batch=None, heads=None, seq_q=None, seq_kv=None,
            d_k=None, d_v=None, query_group_size=1,
            causal=True, tile_m=128, tile_n=128, dtype=np.float16):
    if batch is None:
        batch = 8 if benchmark else 2
    if heads is None:
        heads = 16 if benchmark else 8
    if seq_q is None:
        seq_q = 1024 if benchmark else 128
    if seq_kv is None:
        seq_kv = 1024 if benchmark else 128
    if d_k is None:
        d_k = 64
    if d_v is None:
        d_v = 64

    kv_heads = heads // query_group_size

    Q = (cp.random.random((batch, heads, seq_q, d_k), dtype=np.float32).astype(dtype) - 0.5)
    K = (cp.random.random((batch, kv_heads, seq_kv, d_k), dtype=np.float32).astype(dtype) - 0.5)
    V = (cp.random.random((batch, kv_heads, seq_kv, d_v), dtype=np.float32).astype(dtype) - 0.5)

    return {
        "Q": Q, "K": K, "V": V,
        "batch": batch, "heads": heads, "seq_q": seq_q, "seq_kv": seq_kv,
        "d_k": d_k, "d_v": d_v,
        "query_group_size": query_group_size,
        "causal": causal,
        "tile_m": tile_m, "tile_n": tile_n,
    }


def run(data, *, nruns=1, warmup=0):
    Q, K, V = data["Q"], data["K"], data["V"]
    causal = data["causal"]
    tile_m, tile_n = data["tile_m"], data["tile_n"]
    qgs = data["query_group_size"]

    stream = cp.cuda.get_current_stream()

    for _ in range(warmup):
        cutile_fmha(Q, K, V, tile_m=tile_m, tile_n=tile_n,
                    query_group_size=qgs, causal=causal)
    cp.cuda.runtime.deviceSynchronize()

    times = []
    out = None
    with nvtx.annotate("cuTile"):
        for i in range(nruns):
            with nvtx.annotate(f"run {i + 1}"):
                start = cp.cuda.Event()
                end = cp.cuda.Event()
                start.record(stream)
                out = cutile_fmha(Q, K, V, tile_m=tile_m, tile_n=tile_n,
                                  query_group_size=qgs, causal=causal)
                end.record(stream)
                end.synchronize()
                times.append(cp.cuda.get_elapsed_time(start, end))

    return {"out": out, "times": times}


def verify(data, result):
    expected = ref_fmha(data["Q"], data["K"], data["V"],
                        causal=data["causal"])
    actual = cp.asnumpy(result["out"]).astype(np.float32)
    assert np.allclose(actual, expected, rtol=1e-2, atol=1e-3), \
        f"FMHA incorrect! max diff: {np.max(np.abs(actual - expected))}"


def metric(data):
    B = data["batch"]
    H = data["heads"]
    M = data["seq_q"]
    N = data["seq_kv"]
    D = data["d_k"]
    # QK^T: 2*B*H*M*N*D, P@V: 2*B*H*M*N*D, total: 4*B*H*M*N*D
    flops = 4 * B * H * M * N * D
    return flops, "TFLOPS"


#=============================================================================
# Main
#=============================================================================

def test_fmha(batch, heads, seq_q, seq_kv, d_k, d_v, causal,
              tile_m=128, tile_n=128, query_group_size=1,
              dtype=np.float16, name=None):
    name = name or (f"fmha B={batch} H={heads} M={seq_q} N={seq_kv} "
                    f"D={d_k} causal={causal} {dtype.__name__}")
    print(f"--- {name} ---")
    data = prepare(batch=batch, heads=heads, seq_q=seq_q, seq_kv=seq_kv,
                   d_k=d_k, d_v=d_v, query_group_size=query_group_size,
                   causal=causal, tile_m=tile_m, tile_n=tile_n, dtype=dtype)
    result = run(data)
    verify(data, result)
    print("  passed")


def main():
    print("--- cuTile FMHA Examples ---\n")

    # Non-causal
    test_fmha(2, 8, 128, 128, 64, 64, causal=False)
    # Causal
    test_fmha(2, 8, 128, 128, 64, 64, causal=True)
    # Larger
    test_fmha(4, 16, 256, 256, 64, 64, causal=True)

    print("\n--- All FMHA examples completed ---")


if __name__ == "__main__":
    main()
