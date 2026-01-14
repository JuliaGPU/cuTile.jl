#!/usr/bin/env python3
"""
Batch Matrix Multiplication example - cuTile Python
"""

import cupy as cp
import numpy as np
import cuda.tile as ct
from math import ceil

@ct.kernel
def batchmatmul_cutile_kernel(A, B, C, tm: ct.Constant[int], tn: ct.Constant[int], tk: ct.Constant[int]):
    """CuTile kernel for batch matrix multiplication
    A has shape (Batch, M, K), B has shape (Batch, K, N) and C has shape (Batch, M, N)
    Grid: (Batch, M_tiles, N_tiles)
    """
    pid_batch = ct.bid(0)
    bidx = ct.bid(1)
    bidy = ct.bid(2)

    num_k_tiles = ct.cdiv(A.shape[2], tk)
    accumulator = ct.full((tm, tn), 0.0, dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO

    for k in range(num_k_tiles):
        a = ct.load(A, index=(pid_batch, bidx, k), shape=(1, tm, tk), padding_mode=zero_pad)
        a = ct.reshape(a, (tm, tk))

        b = ct.load(B, index=(pid_batch, k, bidy), shape=(1, tk, tn), padding_mode=zero_pad)
        b = ct.reshape(b, (tk, tn))

        accumulator = ct.mma(a, b, acc=accumulator)

    result = ct.astype(accumulator, C.dtype)
    result_3d = ct.reshape(result, (1, tm, tn))
    ct.store(C, index=(pid_batch, bidx, bidy), tile=result_3d)


#=============================================================================
# Example harness
#=============================================================================

def prepare(*, benchmark: bool = False, Batch: int = None, M: int = None, K: int = None, N: int = None, dtype=np.float16):
    """Allocate and initialize data for batch matmul."""
    if Batch is None:
        Batch = 8 if benchmark else 4
    if M is None:
        M = 1024 if benchmark else 256
    if K is None:
        K = 512 if benchmark else 128
    if N is None:
        N = 2048 if benchmark else 256
    return {
        "A": cp.random.randn(Batch, M, K).astype(dtype),
        "B": cp.random.randn(Batch, K, N).astype(dtype),
        "C": cp.empty((Batch, M, N), dtype=dtype),
        "Batch": Batch,
        "M": M,
        "K": K,
        "N": N
    }


def run(data, *, tm: int = 64, tn: int = 64, tk: int = 64, nruns: int = 1, warmup: int = 0):
    """Run batch matmul kernel with timing."""
    A, B, C = data["A"], data["B"], data["C"]
    Batch, M, N = data["Batch"], data["M"], data["N"]

    grid = (Batch, ceil(M / tm), ceil(N / tn))
    stream = cp.cuda.get_current_stream()

    # Warmup
    for _ in range(warmup):
        ct.launch(stream, grid, batchmatmul_cutile_kernel, (A, B, C, tm, tn, tk))
    cp.cuda.runtime.deviceSynchronize()

    # Timed runs
    times = []
    for _ in range(nruns):
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record(stream)
        ct.launch(stream, grid, batchmatmul_cutile_kernel, (A, B, C, tm, tn, tk))
        end.record(stream)
        end.synchronize()
        times.append(cp.cuda.get_elapsed_time(start, end))  # ms

    return {"C": C, "times": times}


def verify(data, result):
    """Verify batch matmul results."""
    A_np = cp.asnumpy(data["A"]).astype(np.float32)
    B_np = cp.asnumpy(data["B"]).astype(np.float32)
    C_np = cp.asnumpy(result["C"]).astype(np.float32)
    Batch, M, N = data["Batch"], data["M"], data["N"]

    expected = np.zeros((Batch, M, N), dtype=np.float32)
    for b in range(Batch):
        expected[b] = A_np[b] @ B_np[b]
    assert np.allclose(C_np, expected, rtol=1e-1, atol=1e-1), \
        f"batchmatmul incorrect! max diff: {np.max(np.abs(C_np - expected))}"


#=============================================================================
# Reference implementations for benchmarking
#=============================================================================

def run_others(data, *, nruns: int = 1, warmup: int = 0):
    """Run reference implementations for comparison."""
    import torch

    results = {}
    A_cp, B_cp = data["A"], data["B"]
    Batch, M, N = data["Batch"], data["M"], data["N"]

    # PyTorch bmm
    A_torch = torch.as_tensor(A_cp, device='cuda')
    B_torch = torch.as_tensor(B_cp, device='cuda')
    C_torch = torch.zeros(Batch, M, N, dtype=A_torch.dtype, device='cuda')

    for _ in range(warmup):
        torch.bmm(A_torch, B_torch, out=C_torch)
    torch.cuda.synchronize()

    times_torch = []
    for _ in range(nruns):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        torch.bmm(A_torch, B_torch, out=C_torch)
        end.record()
        torch.cuda.synchronize()
        times_torch.append(start.elapsed_time(end))
    results["PyTorch bmm"] = times_torch

    return results


#=============================================================================
# Main
#=============================================================================

def test_batchmatmul(Batch, M, K, N, tm, tn, tk, dtype=np.float16, name=None):
    """Test batch matmul with given parameters."""
    name = name or f"batchmatmul ({Batch}x{M}x{K}) @ ({Batch}x{K}x{N}), tiles={tm}x{tn}x{tk}, dtype={dtype.__name__}"
    print(f"--- {name} ---")
    data = prepare(Batch=Batch, M=M, K=K, N=N, dtype=dtype)
    result = run(data, tm=tm, tn=tn, tk=tk)
    verify(data, result)
    print("  passed")


def main():
    print("--- cuTile Batch Matrix Multiplication Examples ---\n")

    test_batchmatmul(4, 256, 128, 256, 32, 32, 32, np.float32)
    test_batchmatmul(4, 512, 256, 512, 64, 64, 64, np.float32)
    test_batchmatmul(4, 512, 256, 1024, 128, 256, 64, np.float16)

    print("\n--- All batchmatmul examples completed ---")


if __name__ == "__main__":
    main()
