#!/usr/bin/env python3
"""
Fused Multi-Head Attention example - cuTile Python
Julia port equivalent with prepare/run/verify pattern for benchmarking.
"""

# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
import cuda.tile as ct
from cuda.tile import RoundingMode as RMd
from math import ceil, sqrt

from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import sdpa_kernel, SDPBackend

INV_LOG_2 = 1.0 / np.log(2)
ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]


@ct.kernel(occupancy=2)
def fmha_kernel(Q, K, V, Out,
                qk_scale: float,
                input_pos: int,
                D_K: ConstInt,   # Head dimension of Q and K
                D_V: ConstInt,   # Head dimension of V
                H: ConstInt,
                TILE_M: ConstInt,
                TILE_N: ConstInt,
                QUERY_GROUP_SIZE: ConstInt,
                CAUSAL: ConstBool,
                EVEN_K: ConstBool):
    """
    cuTile kernel for Fused Multi-Head Attention (FMHA).
    Computes attention output for a specific batch item and head, using tiling and online softmax.

    Layout: (Batch, Heads, SeqLen, D)
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
    offs_m = bid_x * TILE_M + ct.arange(TILE_M, dtype=np.int32)  # [TILE_M]
    offs_m += input_pos
    offs_m = offs_m[:, None]  # [TILE_M, 1]

    # Initialize local offsets for key/value tile (N-dimension)
    offs_n_tile = ct.arange(TILE_N, dtype=np.int32)  # [TILE_N]
    offs_n_tile = offs_n_tile[None, :]  # [1, TILE_N]

    # Initialize online softmax accumulators in float32 for stability
    m_i = ct.full((TILE_M, 1), -np.inf, dtype=np.float32)
    l_i = ct.full((TILE_M, 1), 0.0, dtype=np.float32)
    acc = ct.full((TILE_M, D_V), 0.0, dtype=np.float32)

    # Load query tile for this batch, head, and M-chunk
    q = ct.load(
        Q, index=(batch_idx, head_idx, bid_x, 0), shape=(1, 1, TILE_M, D_K)
    ).reshape((TILE_M, D_K))  # [TILE_M, D_K]

    # loop over k, v and update accumulator
    m_end = input_pos + (bid_x + 1) * TILE_M
    k_seqlen = K.shape[2]
    if CAUSAL:
        # when kv pos could exceed q pos
        mask_start = (input_pos + bid_x * TILE_M) // TILE_N
        # when kv pos could exceed k_seqlen
        mask_start = min(mask_start, k_seqlen // TILE_N)
        Tc = ct.cdiv(min(m_end, k_seqlen), TILE_N)
    else:
        Tc = ct.cdiv(k_seqlen, TILE_N)
        mask_start = k_seqlen // TILE_N

    # Loop over K, V blocks (N-dimension chunks)
    for j in range(0, Tc):
        # --- Compute QK product ---
        k = ct.load(
            K, index=(batch_idx, off_kv_h, 0, j), shape=(1, 1, D_K, TILE_N),
            order=(0, 1, 3, 2),
            latency=2,
        )
        k = k.reshape((D_K, TILE_N))  # [D_K, TILE_N]
        qk = ct.full((TILE_M, TILE_N), 0., dtype=np.float32)
        qk = ct.mma(q, k, qk)  # [TILE_M, TILE_N]

        # --- Apply Causal Masking ---
        if (CAUSAL or not EVEN_K) and j >= mask_start:
            offs_n = j * TILE_N + offs_n_tile
            mask = ct.full((TILE_M, TILE_N), True, dtype=np.bool)
            # out of bound mask
            if not EVEN_K:
                mask = mask & (offs_n < k_seqlen)
            # causal mask
            if CAUSAL:
                mask = mask & (offs_m >= offs_n)  # [TILE_M, TILE_N]
            mask = ct.where(mask, 0.0, -np.inf)  # [TILE_M, TILE_N]
            qk += mask

        # --- Online Softmax Update ---
        # Moving qk_scale multiplication after reduce_max is to improve performance.
        m_ij = max(m_i, ct.max(qk, axis=-1, keepdims=True) * qk_scale)
        qk = qk * qk_scale - m_ij  # [TILE_M, TILE_N]

        # attention weights
        p = ct.exp2(qk, flush_to_zero=True)  # [TILE_M, TILE_N]
        l_ij = ct.sum(p, axis=-1, keepdims=True)  # [TILE_M, 1]
        alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)  # [TILE_M, 1]
        # update m_i and l_i
        l_i = l_i * alpha + l_ij  # [TILE_M, 1]
        # scale acc
        acc = acc * alpha  # [TILE_M, D_V]

        # --- Compute PV product ---
        v = ct.load(
            V, index=(batch_idx, off_kv_h, j, 0), shape=(1, 1, TILE_N, D_V),
            latency=4,
        ).reshape((TILE_N, D_V))  # [TILE_N, D_V]
        p = p.astype(Q.dtype)
        acc = ct.mma(p, v, acc)  # [TILE_M, D_V]
        m_i = m_ij  # [TILE_M, 1]

    # --- Final Normalization and Store ---
    acc = ct.truediv(acc, l_i, flush_to_zero=True, rounding_mode=RMd.APPROX)
    acc = acc.reshape((1, 1, TILE_M, D_V)).astype(Out.dtype)
    ct.store(Out, index=(batch_idx, head_idx, bid_x, 0), tile=acc)


#=============================================================================
# Example harness
#=============================================================================

def prepare(*, benchmark: bool = False,
            D_k: int = 64,
            SeqLen_Q: int = None,
            Heads: int = 4,
            Batch: int = 4,
            D_v: int = None,
            SeqLen_KV: int = None,
            Heads_KV: int = None,
            causal: bool = False,
            dtype=torch.float32):
    """Allocate and initialize data for FMHA."""
    if SeqLen_Q is None:
        SeqLen_Q = 4096 if benchmark else 256
    if D_v is None:
        D_v = D_k
    if SeqLen_KV is None:
        SeqLen_KV = SeqLen_Q
    if Heads_KV is None:
        Heads_KV = Heads

    # Layout: (Batch, Heads, SeqLen, D)
    return {
        "Q": torch.randn(Batch, Heads, SeqLen_Q, D_k, dtype=dtype, device='cuda'),
        "K": torch.randn(Batch, Heads_KV, SeqLen_KV, D_k, dtype=dtype, device='cuda'),
        "V": torch.randn(Batch, Heads_KV, SeqLen_KV, D_v, dtype=dtype, device='cuda'),
        "Out": torch.empty(Batch, Heads, SeqLen_Q, D_v, dtype=dtype, device='cuda'),
        "D_k": D_k,
        "D_v": D_v,
        "SeqLen_Q": SeqLen_Q,
        "SeqLen_KV": SeqLen_KV,
        "Heads": Heads,
        "Heads_KV": Heads_KV,
        "Batch": Batch,
        "causal": causal,
    }


def run(data, *, tm: int = 64, tn: int = 64, nruns: int = 1, warmup: int = 0):
    """Run FMHA kernel with timing."""
    Q, K, V, Out = data["Q"], data["K"], data["V"], data["Out"]
    D_k, D_v = data["D_k"], data["D_v"]
    SeqLen_Q, SeqLen_KV = data["SeqLen_Q"], data["SeqLen_KV"]
    Heads, Heads_KV, Batch = data["Heads"], data["Heads_KV"], data["Batch"]
    causal = data["causal"]

    grid_x = ceil(SeqLen_Q / tm)
    grid_y = Heads * Batch
    grid = (grid_x, grid_y, 1)

    qk_scale = 1.0 / sqrt(D_k)
    input_pos = 0

    query_group_size, remainder = divmod(Heads, Heads_KV)
    assert remainder == 0, "Heads must be divisible by Heads_KV"

    even_k = (SeqLen_KV % tn) == 0

    stream = torch.cuda.current_stream()

    # Warmup
    for _ in range(warmup):
        ct.launch(stream, grid, fmha_kernel, (
            Q, K, V, Out,
            qk_scale, input_pos,
            D_k, D_v, Heads,
            tm, tn,
            query_group_size,
            causal, even_k
        ))
    torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(nruns):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        ct.launch(stream, grid, fmha_kernel, (
            Q, K, V, Out,
            qk_scale, input_pos,
            D_k, D_v, Heads,
            tm, tn,
            query_group_size,
            causal, even_k
        ))
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))  # ms

    return {"Out": Out, "times": times}


def torch_sdpa(Q, K, V, *, causal: bool = False, enable_gqa: bool = False):
    """Reference scaled dot-product attention using PyTorch."""
    # Use MATH backend as fallback (works with all dtypes)
    # cuDNN/Flash only support float16/bfloat16
    with sdpa_kernel(SDPBackend.MATH):
        return scaled_dot_product_attention(Q, K, V, is_causal=causal, enable_gqa=enable_gqa)


def verify(data, result):
    """Verify FMHA results against reference implementation."""
    Q, K, V = data["Q"], data["K"], data["V"]
    causal = data["causal"]
    Heads, Heads_KV = data["Heads"], data["Heads_KV"]

    enable_gqa = Heads != Heads_KV
    expected = torch_sdpa(Q, K, V, causal=causal, enable_gqa=enable_gqa)
    actual = result["Out"]

    max_diff = float(torch.max(torch.abs(actual - expected)))
    assert torch.allclose(actual, expected, rtol=1e-2, atol=1e-2), \
        f"FMHA mismatch! max diff: {max_diff}"


#=============================================================================
# Reference implementations for benchmarking
#=============================================================================

def run_others(data, *, nruns: int = 1, warmup: int = 0):
    """Run reference implementations for comparison."""
    results = {}
    Q, K, V = data["Q"], data["K"], data["V"]
    causal = data["causal"]
    Heads, Heads_KV = data["Heads"], data["Heads_KV"]
    enable_gqa = Heads != Heads_KV

    # PyTorch SDPA (uses cuDNN or Flash Attention)
    for _ in range(warmup):
        _ = torch_sdpa(Q, K, V, causal=causal, enable_gqa=enable_gqa)
    torch.cuda.synchronize()

    times_torch = []
    for _ in range(nruns):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = torch_sdpa(Q, K, V, causal=causal, enable_gqa=enable_gqa)
        end.record()
        torch.cuda.synchronize()
        times_torch.append(start.elapsed_time(end))
    results["PyTorch SDPA"] = times_torch

    return results


#=============================================================================
# Main
#=============================================================================

def test_attention(dtype, D_k, SeqLen_Q, Heads, Batch, D_v, SeqLen_KV, Heads_KV,
                   causal, tm, tn, name=None):
    """Test attention with given parameters."""
    if name is None:
        dtype_name = str(dtype).split('.')[-1]
        name = ", ".join([
            dtype_name,
            f"tile={tm}x{tn}",
            f"Q={D_k}x{SeqLen_Q}",
            f"K={D_k}x{SeqLen_KV}",
            f"V={D_v}x{SeqLen_KV}",
            f"Heads={Heads}/{Heads_KV}",
            f"Batch={Batch}",
            f"causal={causal}"
        ])
    print(f"--- {name} ---")
    data = prepare(
        D_k=D_k, SeqLen_Q=SeqLen_Q, Heads=Heads, Batch=Batch,
        D_v=D_v, SeqLen_KV=SeqLen_KV, Heads_KV=Heads_KV,
        causal=causal, dtype=dtype
    )
    result = run(data, tm=tm, tn=tn)
    verify(data, result)
    print("  passed")


def main():
    print("--- cuTile Fused Multi-Head Attention Examples ---\n")

    for dtype in (torch.float32, torch.float16):
        # basic
        test_attention(dtype, 64, 256, 8, 2, 64, 256, 8, False, 32, 32)
        test_attention(dtype, 64, 256, 8, 2, 64, 128, 4, False, 32, 64)
        test_attention(dtype, 64, 256, 8, 2, 64, 256, 8, True, 32, 32)

        # uneven seqlen
        test_attention(dtype, 64, 127, 4, 1, 64, 127, 4, False, 32, 32)
        test_attention(dtype, 64, 128, 4, 1, 64, 97, 2, False, 32, 32)

        # D_k != D_v
        test_attention(dtype, 64, 256, 8, 2, 64, 256, 8, False, 32, 32)

    print("\n--- All attention examples completed ---")


if __name__ == "__main__":
    main()
