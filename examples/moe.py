#!/usr/bin/env python3
"""
Mixture of Experts (MoE) example - cuTile Python

Fused MoE kernel that multiplies routed tokens by expert weights,
plus a SiLU-and-mul activation kernel. Based on cuTile Python's MoE.py sample.
"""

import cupy as cp
import numpy as np
import cuda.tile as ct
from math import ceil

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]


#=============================================================================
# Helper: 2D swizzle
#=============================================================================

def swizzle_2d(M, N, tm, tn, GROUP_SIZE_M):
    """Get the global IDs of the current block in a 1D grid."""
    bid = ct.bid(0)
    num_bid_m = ct.cdiv(M, tm)
    num_bid_n = ct.cdiv(N, tn)
    num_bid_in_group = GROUP_SIZE_M * num_bid_n
    group_id = bid // num_bid_in_group
    first_bid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_bid_m - first_bid_m, GROUP_SIZE_M)
    bid_m = first_bid_m + (bid % group_size_m)
    bid_n = (bid % num_bid_in_group) // group_size_m
    return bid_m, bid_n


#=============================================================================
# Kernels
#=============================================================================

@ct.kernel
def fused_moe_kernel(
    A, B, C,
    topk_weights,
    sorted_token_ids,
    sorted_expert_ids,
    num_token_replicas: int,
    mul_routed_weight: ConstBool,
    TILE_M: ConstInt,
    TILE_N: ConstInt,
    TILE_K: ConstInt,
):
    """
    Fused MoE kernel: multiplies routed tokens by their assigned expert weights.

    Token ids are sorted and padded so each expert processes a multiple of TILE_M tokens.
    """
    M = sorted_token_ids.shape[0]
    N = B.shape[1]
    K = B.shape[2]

    GROUP_SIZE_M = 8
    bid_m, bid_n = swizzle_2d(M, N, TILE_M, TILE_N, GROUP_SIZE_M)

    zero_pad = ct.PaddingMode.ZERO

    # Gather replicated/padded token indices for this block
    token_id_indices = bid_m * TILE_M + ct.arange(TILE_M, dtype=ct.int32)
    token_ids = ct.gather(sorted_token_ids, token_id_indices)

    # Collapse replica dimension to recover source row in A
    a_row_indices = token_ids // num_token_replicas

    # Each TILE_M block is homogeneous in expert assignment
    expert_id = ct.load(sorted_expert_ids, index=bid_m, shape=())

    accumulator = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
    for k in range(0, ct.cdiv(K, TILE_K)):
        a_col_indices = k * TILE_K + ct.arange(TILE_K, dtype=ct.int32)
        a = ct.gather(A, (a_row_indices[:, None], a_col_indices[None, :]))

        b = ct.load(B, (expert_id, k, bid_n), shape=(1, TILE_K, TILE_N),
                    order=(0, 2, 1), padding_mode=zero_pad).reshape((TILE_K, TILE_N))

        accumulator = ct.mma(a, b, accumulator)

    if mul_routed_weight:
        moe_weight = ct.gather(topk_weights, token_ids)
        accumulator = accumulator * moe_weight[:, None]

    # Scatter result back into C
    c_col_indices = bid_n * TILE_N + ct.arange(TILE_N, dtype=ct.int32)
    accumulator = ct.astype(accumulator, C.dtype)
    ct.scatter(C, (token_ids[:, None], c_col_indices[None, :]), accumulator)


@ct.kernel
def silu_and_mul_kernel(A, B, C, TILE_N: ConstInt):
    """Element-wise kernel: computes SiLU(A) * B."""
    bid_m = ct.bid(0)
    ta = ct.load(A, (bid_m, 0), (1, TILE_N)).astype(ct.float32)
    tb = ct.load(B, (bid_m, 0), (1, TILE_N)).astype(ct.float32)

    # Sigmoid(ta)
    denom = ct.add(1, ct.exp(-ta), flush_to_zero=True)
    sigmoid_ta = ct.truediv(1.0, denom, flush_to_zero=True, rounding_mode=ct.RoundingMode.APPROX)

    # SiLU(ta) * tb
    silu_ta = ct.mul(ta, sigmoid_ta, flush_to_zero=True)
    tc = ct.mul(silu_ta, tb, flush_to_zero=True)

    ct.store(C, (bid_m, 0), tc.astype(C.dtype))


#=============================================================================
# Host-side helpers
#=============================================================================

def next_power_of_2(n):
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    n += 1
    return n


def moe_align_tile_size(topk_ids, tile_m, num_experts):
    """
    Sort, replicate, and pad token indices by expert so every expert processes
    a TILE_M-aligned number of tokens.

    Returns:
        sorted_token_ids: 1D array of token-replica indices sorted by expert,
            padded with sentinel value (num_tokens * topk).
        sorted_expert_ids: Expert id for each TILE_M block.
    """
    num_tokens, topk = topk_ids.shape
    total_tokens = num_tokens * topk

    # Do everything on CPU (small metadata arrays)
    flat_expert_ids = cp.asnumpy(topk_ids.reshape(-1))
    sorted_token_indices_cpu = np.argsort(flat_expert_ids, kind='stable').astype(np.int32)

    expert_token_counts_cpu = np.bincount(flat_expert_ids.astype(np.int32), minlength=num_experts)
    expert_block_counts_cpu = (expert_token_counts_cpu - 1 + tile_m) // tile_m
    total_blocks = int(expert_block_counts_cpu.sum())

    sorted_token_ids_cpu = np.full((total_blocks * tile_m,), total_tokens, dtype=np.int32)
    sorted_expert_ids_cpu = np.zeros(total_blocks, dtype=np.int32)

    current_block = 0
    current_token = 0
    for expert_id in range(num_experts):
        token_count = int(expert_token_counts_cpu[expert_id])
        block_count = int(expert_block_counts_cpu[expert_id])

        sorted_expert_ids_cpu[current_block:current_block + block_count] = expert_id
        sorted_token_start = current_block * tile_m
        sorted_token_ids_cpu[sorted_token_start:sorted_token_start + token_count] = \
            sorted_token_indices_cpu[current_token:current_token + token_count]

        current_token += token_count
        current_block += block_count

    return cp.asarray(sorted_token_ids_cpu), cp.asarray(sorted_expert_ids_cpu)


def invoke_fused_moe_kernel(A, B, C, topk_weights, sorted_token_ids, sorted_expert_ids,
                            mul_routed_weight, num_token_replicas, tile_m, tile_n, tile_k):
    m = sorted_token_ids.shape[0]
    n = B.shape[1]
    grid = (ceil(m / tile_m) * ceil(n / tile_n),)
    topk_weights_flat = topk_weights.reshape(-1)
    C_flat = C.reshape(-1, C.shape[-1])
    stream = cp.cuda.get_current_stream()
    ct.launch(stream, grid, fused_moe_kernel,
              (A, B, C_flat, topk_weights_flat, sorted_token_ids, sorted_expert_ids,
               num_token_replicas, mul_routed_weight, tile_m, tile_n, tile_k))


def invoke_silu_and_mul_kernel(AB, C):
    A_half, B_half = cp.split(AB, 2, axis=-1)
    # Make contiguous copies since split returns views
    A_half = cp.ascontiguousarray(A_half)
    B_half = cp.ascontiguousarray(B_half)
    stream = cp.cuda.get_current_stream()
    ct.launch(stream, (AB.shape[0],), silu_and_mul_kernel,
              (A_half, B_half, C, next_power_of_2(C.shape[-1])))


def cutile_moe(hidden_states, w1, w2, topk_weights, topk_ids, tile_m, tile_n, tile_k):
    """Run full MoE forward pass: expert matmul -> SiLU activation -> expert matmul."""
    num_tokens, hidden_size = hidden_states.shape
    num_experts, _, intermediate_size = w2.shape
    _, topk = topk_ids.shape

    intermediate_cache1 = cp.zeros((num_tokens, topk, intermediate_size * 2), dtype=hidden_states.dtype)
    intermediate_cache2 = cp.zeros((num_tokens * topk, intermediate_size), dtype=hidden_states.dtype)
    intermediate_cache3 = cp.zeros((num_tokens, topk, hidden_size), dtype=hidden_states.dtype)

    sorted_token_ids, sorted_expert_ids = moe_align_tile_size(topk_ids, tile_m, num_experts)

    # First matmul: hidden_states @ w1^T -> gate+up projection
    invoke_fused_moe_kernel(
        hidden_states, w1, intermediate_cache1, topk_weights,
        sorted_token_ids, sorted_expert_ids,
        mul_routed_weight=False, num_token_replicas=topk,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k)

    # SiLU activation
    invoke_silu_and_mul_kernel(
        intermediate_cache1.reshape(-1, intermediate_cache1.shape[-1]),
        intermediate_cache2)

    # Second matmul: activated @ w2^T -> down projection (with routing weights)
    invoke_fused_moe_kernel(
        intermediate_cache2, w2, intermediate_cache3, topk_weights,
        sorted_token_ids, sorted_expert_ids,
        mul_routed_weight=True, num_token_replicas=1,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k)

    return cp.sum(intermediate_cache3, axis=1)


#=============================================================================
# Reference implementation
#=============================================================================

def ref_moe(hidden_states, w1, w2, topk_weights, topk_ids):
    """Naive NumPy MoE for correctness checking."""
    hidden_states = cp.asnumpy(hidden_states).astype(np.float32)
    w1 = cp.asnumpy(w1).astype(np.float32)
    w2 = cp.asnumpy(w2).astype(np.float32)
    topk_weights = cp.asnumpy(topk_weights).astype(np.float32)
    topk_ids = cp.asnumpy(topk_ids)

    num_tokens = hidden_states.shape[0]
    num_experts = w1.shape[0]
    _, topk = topk_ids.shape

    final = np.zeros_like(hidden_states)

    for expert_id in range(num_experts):
        # Find which (token, k) pairs route to this expert
        token_indices, k_indices = np.where(topk_ids == expert_id)
        if len(token_indices) == 0:
            continue

        tokens = hidden_states[token_indices]  # (count, hidden_size)
        gate_up = w1[expert_id]  # (intermediate*2, hidden_size)
        gate_proj = gate_up[:gate_up.shape[0] // 2]
        up_proj = gate_up[gate_up.shape[0] // 2:]
        down_proj = w2[expert_id]  # (hidden_size, intermediate)

        gate_out = tokens @ gate_proj.T
        up_out = tokens @ up_proj.T
        # SiLU activation
        silu_out = gate_out / (1.0 + np.exp(-gate_out)) * up_out
        expert_out = silu_out @ down_proj.T

        weights = topk_weights[token_indices, k_indices]
        weighted = expert_out * weights[:, None]

        np.add.at(final, token_indices, weighted)

    return final


#=============================================================================
# Example harness
#=============================================================================

def prepare(*, benchmark=False,
            num_tokens=None, hidden_size=None, num_experts=None,
            intermediate_size=None, topk=None, dtype=np.float16):
    if num_tokens is None:
        num_tokens = 256 if benchmark else 48
    if hidden_size is None:
        hidden_size = 1024 if benchmark else 512
    if num_experts is None:
        num_experts = 32 if benchmark else 64
    if intermediate_size is None:
        intermediate_size = 2048 if benchmark else 1024
    if topk is None:
        topk = 8

    # Use uniform random to avoid float64 intermediates from randn
    hidden_states = (cp.random.random((num_tokens, hidden_size), dtype=np.float32).astype(dtype) - 0.5)
    w1 = (cp.random.random((num_experts, intermediate_size * 2, hidden_size), dtype=np.float32).astype(dtype) - 0.5) * 0.2
    w2 = (cp.random.random((num_experts, hidden_size, intermediate_size), dtype=np.float32).astype(dtype) - 0.5) * 0.2

    # Unique expert IDs per token (no repeats within a row)
    topk_ids_cpu = np.stack([
        np.random.permutation(num_experts)[:topk] for _ in range(num_tokens)
    ])
    topk_ids = cp.asarray(topk_ids_cpu.astype(np.int64))

    # Softmax routing weights
    topk_weights = cp.random.randn(num_tokens, topk).astype(np.float32)
    topk_weights = cp.exp(topk_weights)
    topk_weights = topk_weights / topk_weights.sum(axis=1, keepdims=True)
    topk_weights = topk_weights.astype(dtype)

    return {
        "hidden_states": hidden_states,
        "w1": w1,
        "w2": w2,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "num_tokens": num_tokens,
        "hidden_size": hidden_size,
        "num_experts": num_experts,
        "intermediate_size": intermediate_size,
        "topk": topk,
        "tile_m": 128,
        "tile_n": 128,
        "tile_k": 64,
    }


def run(data, *, nruns=1, warmup=0):
    hs = data["hidden_states"]
    w1, w2 = data["w1"], data["w2"]
    tw, ti = data["topk_weights"], data["topk_ids"]
    tm, tn, tk = data["tile_m"], data["tile_n"], data["tile_k"]

    stream = cp.cuda.get_current_stream()

    for _ in range(warmup):
        cutile_moe(hs, w1, w2, tw, ti, tm, tn, tk)
    cp.cuda.runtime.deviceSynchronize()

    times = []
    for _ in range(nruns):
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record(stream)
        out = cutile_moe(hs, w1, w2, tw, ti, tm, tn, tk)
        end.record(stream)
        end.synchronize()
        times.append(cp.cuda.get_elapsed_time(start, end))

    return {"out": out, "times": times}


def verify(data, result):
    expected = ref_moe(data["hidden_states"], data["w1"], data["w2"],
                       data["topk_weights"], data["topk_ids"])
    actual = cp.asnumpy(result["out"]).astype(np.float32)
    assert np.allclose(actual, expected, rtol=1e-1, atol=1e-1), \
        f"MoE incorrect! max diff: {np.max(np.abs(actual - expected))}"


def metric(data):
    # Two matmuls per token per expert-selection:
    # matmul1: (num_tokens*topk, hidden) @ (hidden, intermediate*2) = 2*tokens*topk*hidden*intermediate*2
    # matmul2: (num_tokens*topk, intermediate) @ (intermediate, hidden) = 2*tokens*topk*intermediate*hidden
    # Total: 2 * num_tokens * topk * hidden * intermediate * 3  (gate+up = 2x, down = 1x)
    nt = data["num_tokens"]
    topk = data["topk"]
    hs = data["hidden_size"]
    inter = data["intermediate_size"]
    flops = 2 * nt * topk * hs * inter * 3
    return flops, "TFLOPS"


#=============================================================================
# Main
#=============================================================================

def test_moe(num_tokens, hidden_size, num_experts, intermediate_size, topk,
             tile_m=128, tile_n=128, tile_k=64, dtype=np.float16, name=None):
    name = name or (f"moe tokens={num_tokens}, hidden={hidden_size}, experts={num_experts}, "
                    f"inter={intermediate_size}, topk={topk}, {dtype.__name__}")
    print(f"--- {name} ---")
    data = prepare(num_tokens=num_tokens, hidden_size=hidden_size, num_experts=num_experts,
                   intermediate_size=intermediate_size, topk=topk, dtype=dtype)
    data["tile_m"] = tile_m
    data["tile_n"] = tile_n
    data["tile_k"] = tile_k
    result = run(data)
    verify(data, result)
    print("  passed")


def main():
    print("--- cuTile Mixture of Experts Examples ---\n")

    test_moe(48, 512, 64, 1024, 8)
    test_moe(128, 512, 32, 1024, 4)
    test_moe(64, 1024, 64, 2048, 8)

    print("\n--- All MoE examples completed ---")


if __name__ == "__main__":
    main()
