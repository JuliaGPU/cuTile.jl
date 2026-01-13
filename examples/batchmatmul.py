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


def run_batchmatmul(*, Batch: int = 4, M: int = 256, K: int = 128, N: int = 256,
                    tm: int = 128, tn: int = 256, tk: int = 64,
                    dtype=np.float16, A=None, B=None, C=None, validate: bool = False):
    """Run batch matrix multiplication. Returns (A, B, C) arrays for benchmarking."""
    if A is None:
        A = cp.random.randn(Batch, M, K).astype(dtype)
    else:
        Batch, M, K = A.shape
    if B is None:
        B = cp.random.randn(Batch, K, N).astype(dtype)
    else:
        Batch, K, N = B.shape
    if C is None:
        C = cp.zeros((Batch, M, N), dtype=dtype)

    grid = (Batch, ceil(M / tm), ceil(N / tn))
    stream = cp.cuda.get_current_stream()
    ct.launch(stream, grid, batchmatmul_cutile_kernel, (A, B, C, tm, tn, tk))

    if validate:
        cp.cuda.runtime.deviceSynchronize()
        A_np = cp.asnumpy(A).astype(np.float32)
        B_np = cp.asnumpy(B).astype(np.float32)
        C_np = cp.asnumpy(C).astype(np.float32)
        expected = np.zeros((Batch, M, N), dtype=np.float32)
        for b in range(Batch):
            expected[b] = A_np[b] @ B_np[b]
        assert np.allclose(C_np, expected, rtol=1e-1, atol=1e-1), \
            f"batchmatmul incorrect! max diff: {np.max(np.abs(C_np - expected))}"

    return A, B, C


def test_batchmatmul(Batch, M, K, N, tm, tn, tk, dtype=np.float16, name=None):
    """Test batch matmul with given parameters."""
    name = name or f"batchmatmul ({Batch}x{M}x{K}) @ ({Batch}x{K}x{N}), tiles={tm}x{tn}x{tk}, dtype={dtype.__name__}"
    print(f"--- {name} ---")
    run_batchmatmul(Batch=Batch, M=M, K=K, N=N, tm=tm, tn=tn, tk=tk, dtype=dtype, validate=True)
    print("  passed")


def main():
    print("--- cuTile Batch Matrix Multiplication Examples ---\n")

    test_batchmatmul(4, 256, 128, 256, 32, 32, 32, np.float32)
    test_batchmatmul(4, 512, 256, 512, 64, 64, 64, np.float32)
    test_batchmatmul(4, 512, 256, 1024, 128, 256, 64, np.float16)

    print("\n--- All batchmatmul examples completed ---")


if __name__ == "__main__":
    main()
