#!/usr/bin/env python3
"""
Matrix Multiplication example - cuTile Python
"""

import cupy as cp
import numpy as np
import cuda.tile as ct
from math import ceil

def swizzle_2d(M, N, tm, tn, GROUP_SIZE_M):
    """Get the global IDs of the current CUDA block in a 1D grid."""
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


@ct.kernel
def matmul_cutile_kernel(A, B, C, tm: ct.Constant[int], tn: ct.Constant[int], tk: ct.Constant[int]):
    GROUP_SIZE_M = 8
    M = A.shape[0]
    N = B.shape[1]
    bidx, bidy = swizzle_2d(M, N, tm, tn, GROUP_SIZE_M)

    num_tiles_k = ct.num_tiles(A, axis=1, shape=(tm, tk))
    accumulator = ct.full((tm, tn), 0, dtype=ct.float32)

    # Convert fp32 to tf32 for tensor cores
    dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype

    for k in range(num_tiles_k):
        a = ct.load(A, index=(bidx, k), shape=(tm, tk)).astype(dtype)
        b = ct.load(B, index=(k, bidy), shape=(tk, tn)).astype(dtype)
        accumulator = ct.mma(a, b, accumulator)

    accumulator = ct.astype(accumulator, C.dtype)
    ct.store(C, index=(bidx, bidy), tile=accumulator)


#=============================================================================
# Example harness
#=============================================================================

def prepare(*, benchmark: bool = False, M: int = None, N: int = None, K: int = None, dtype=np.float32):
    """Allocate and initialize data for matmul."""
    if M is None:
        M = 4096 if benchmark else 256
    if N is None:
        N = 4096 if benchmark else 256
    if K is None:
        K = 4096 if benchmark else 256
    return {
        "A": cp.random.randn(M, K).astype(dtype),
        "B": cp.random.randn(K, N).astype(dtype),
        "C": cp.empty((M, N), dtype=dtype),
        "M": M,
        "N": N,
        "K": K
    }


def run(data, *, tm: int = 64, tn: int = 64, tk: int = 64, nruns: int = 1, warmup: int = 0):
    """Run matmul kernel with timing."""
    A, B, C = data["A"], data["B"], data["C"]
    M, N = data["M"], data["N"]

    grid_m = ceil(M / tm)
    grid_n = ceil(N / tn)
    grid = (grid_m * grid_n, 1, 1)
    stream = cp.cuda.get_current_stream()

    # Warmup
    for _ in range(warmup):
        ct.launch(stream, grid, matmul_cutile_kernel, (A, B, C, tm, tn, tk))
    cp.cuda.runtime.deviceSynchronize()

    # Timed runs
    times = []
    for _ in range(nruns):
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record(stream)
        ct.launch(stream, grid, matmul_cutile_kernel, (A, B, C, tm, tn, tk))
        end.record(stream)
        end.synchronize()
        times.append(cp.cuda.get_elapsed_time(start, end))  # ms

    return {"C": C, "times": times}


def verify(data, result):
    """Verify matmul results."""
    expected = cp.asnumpy(data["A"]) @ cp.asnumpy(data["B"])
    # TF32 has reduced precision
    assert np.allclose(cp.asnumpy(result["C"]), expected, rtol=1e-1, atol=1e-1), \
        f"matmul incorrect! max diff: {np.max(np.abs(cp.asnumpy(result['C']) - expected))}"


#=============================================================================
# Reference implementations for benchmarking
#=============================================================================

def run_others(data, *, nruns: int = 1, warmup: int = 0):
    """Run reference implementations for comparison."""
    results = {}
    A, B = data["A"], data["B"]
    M, N = data["M"], data["N"]
    C_cupy = cp.zeros((M, N), dtype=A.dtype)

    stream = cp.cuda.get_current_stream()

    # CuPy matmul (uses cuBLAS)
    for _ in range(warmup):
        cp.matmul(A, B, out=C_cupy)
    cp.cuda.runtime.deviceSynchronize()

    times_cupy = []
    for _ in range(nruns):
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record(stream)
        cp.matmul(A, B, out=C_cupy)
        end.record(stream)
        end.synchronize()
        times_cupy.append(cp.cuda.get_elapsed_time(start, end))
    results["cuBLAS"] = times_cupy

    return results


#=============================================================================
# Main
#=============================================================================

def test_matmul(M, N, K, tm, tn, tk, dtype=np.float32, name=None):
    """Test matmul with given parameters."""
    name = name or f"matmul ({M}x{K}) @ ({K}x{N}), tiles={tm}x{tn}x{tk}, dtype={dtype.__name__}"
    print(f"--- {name} ---")
    data = prepare(M=M, N=N, K=K, dtype=dtype)
    result = run(data, tm=tm, tn=tn, tk=tk)
    verify(data, result)
    print("  passed")


def main():
    print("--- cuTile Matrix Multiplication Examples ---\n")

    test_matmul(256, 256, 256, 32, 32, 32)
    test_matmul(512, 512, 512, 64, 64, 64)
    test_matmul(256, 512, 128, 32, 32, 32)
    test_matmul(1024, 1024, 1024, 64, 64, 64)

    print("\n--- All matmul examples completed ---")


if __name__ == "__main__":
    main()
