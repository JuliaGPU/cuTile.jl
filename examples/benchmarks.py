#!/usr/bin/env python3
"""
Comprehensive benchmarks for cuTile Python
Compares: CuPy, PyTorch, cuTile
Kernels: vadd, transpose, matmul
"""

import cupy as cp
import numpy as np
import torch
import cuda.tile as ct
from math import ceil

#=============================================================================
# Configuration
#=============================================================================

NRUNS = 10
WARMUP = 3

# Data sizes (must match Julia benchmarks)
VADD_SIZE = 2**25           # 134 MB
TRANSPOSE_DIM = 4096        # 4096x4096 = 67 MB
MATMUL_DIM = 2048           # 2048x2048x2048

# Tile sizes
VADD_TILE = 16
TRANSPOSE_TILE_M = 64
TRANSPOSE_TILE_N = 64
MATMUL_TM = 32
MATMUL_TN = 32
MATMUL_TK = 32

#=============================================================================
# Benchmark Utilities
#=============================================================================

class BenchmarkResult:
    def __init__(self, name: str, min_ms: float, mean_ms: float):
        self.name = name
        self.min_ms = min_ms
        self.mean_ms = mean_ms


def benchmark_cupy(f, nruns: int = NRUNS, warmup: int = WARMUP):
    """Benchmark a CuPy function using CUDA events."""
    stream = cp.cuda.get_current_stream()

    # Warmup
    for _ in range(warmup):
        f()
    cp.cuda.runtime.deviceSynchronize()

    # Benchmark
    times = []
    for _ in range(nruns):
        start = cp.cuda.Event()
        end = cp.cuda.Event()

        start.record(stream)
        f()
        end.record(stream)
        end.synchronize()

        elapsed_ms = cp.cuda.get_elapsed_time(start, end)
        times.append(elapsed_ms)

    return min(times), sum(times) / len(times)


def benchmark_torch(f, nruns: int = NRUNS, warmup: int = WARMUP):
    """Benchmark a PyTorch function using CUDA events."""
    # Warmup
    for _ in range(warmup):
        f()
    torch.cuda.synchronize()

    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(nruns):
        start_event.record()
        f()
        end_event.record()
        torch.cuda.synchronize()

        elapsed_ms = start_event.elapsed_time(end_event)
        times.append(elapsed_ms)

    return min(times), sum(times) / len(times)


def print_table(title: str, results: list, extra_col=None):
    """Print formatted benchmark results table."""
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)

    if extra_col:
        print(f"{'Implementation':<20}{'Min (ms)':<12}{'Mean (ms)':<12}{extra_col[0]}")
        print("-" * 60)
        for i, r in enumerate(results):
            print(f"{r.name:<20}{r.min_ms:<12.3f}{r.mean_ms:<12.3f}{extra_col[1][i]}")
    else:
        print(f"{'Implementation':<20}{'Min (ms)':<12}Mean (ms)")
        print("-" * 60)
        for r in results:
            print(f"{r.name:<20}{r.min_ms:<12.3f}{r.mean_ms:.3f}")
    print("-" * 60)


#=============================================================================
# Vector Addition
#=============================================================================

@ct.kernel
def vadd_cutile_kernel(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    tile_a = ct.load(a, index=(pid,), shape=(tile_size,))
    tile_b = ct.load(b, index=(pid,), shape=(tile_size,))
    result = tile_a + tile_b
    ct.store(c, index=(pid,), tile=result)


def benchmark_vadd():
    print("\nBenchmarking Vector Addition...")
    print(f"  Size: {VADD_SIZE} elements ({VADD_SIZE * 4 / 1e6} MB)")

    # CuPy arrays
    a_cp = cp.random.rand(VADD_SIZE).astype(np.float32)
    b_cp = cp.random.rand(VADD_SIZE).astype(np.float32)
    c_cp = cp.zeros(VADD_SIZE, dtype=np.float32)

    # PyTorch tensors (from same data)
    a_torch = torch.as_tensor(a_cp, device='cuda')
    b_torch = torch.as_tensor(b_cp, device='cuda')
    c_torch = torch.zeros(VADD_SIZE, dtype=torch.float32, device='cuda')

    expected = cp.asnumpy(a_cp) + cp.asnumpy(b_cp)
    results = []

    # CuPy
    def cupy_vadd():
        cp.add(a_cp, b_cp, out=c_cp)

    cupy_vadd()
    cp.cuda.runtime.deviceSynchronize()
    assert np.allclose(cp.asnumpy(c_cp), expected), "CuPy incorrect!"
    min_t, mean_t = benchmark_cupy(cupy_vadd)
    results.append(BenchmarkResult("CuPy", min_t, mean_t))

    # PyTorch
    def torch_vadd():
        torch.add(a_torch, b_torch, out=c_torch)

    torch_vadd()
    torch.cuda.synchronize()
    assert np.allclose(c_torch.cpu().numpy(), expected), "PyTorch incorrect!"
    min_t, mean_t = benchmark_torch(torch_vadd)
    results.append(BenchmarkResult("PyTorch", min_t, mean_t))

    # cuTile
    grid = (ct.cdiv(VADD_SIZE, VADD_TILE), 1, 1)
    stream = cp.cuda.get_current_stream()
    c_cp.fill(0)

    def cutile_vadd():
        ct.launch(stream, grid, vadd_cutile_kernel, (a_cp, b_cp, c_cp, VADD_TILE))

    cutile_vadd()
    cp.cuda.runtime.deviceSynchronize()
    assert np.allclose(cp.asnumpy(c_cp), expected), "cuTile incorrect!"
    min_t, mean_t = benchmark_cupy(cutile_vadd)
    results.append(BenchmarkResult("cuTile Python", min_t, mean_t))

    # Calculate bandwidth
    bytes_transferred = 3 * VADD_SIZE * 4  # 2 reads + 1 write, float32
    bandwidths = [f"{bytes_transferred / (r.min_ms / 1000) / 1e9:.1f} GB/s" for r in results]

    print_table("Vector Addition (Float32)", results, extra_col=("Bandwidth", bandwidths))
    return results


#=============================================================================
# Matrix Transpose
#=============================================================================

@ct.kernel
def transpose_cutile_kernel(input, output, tile_m: ct.Constant[int], tile_n: ct.Constant[int]):
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)
    tile = ct.load(input, index=(pid_m, pid_n), shape=(tile_m, tile_n))
    tile_t = ct.transpose(tile)
    ct.store(output, index=(pid_n, pid_m), tile=tile_t)


def benchmark_transpose():
    print("\nBenchmarking Matrix Transpose...")
    M, N = TRANSPOSE_DIM, TRANSPOSE_DIM
    print(f"  Size: {M}x{N} ({M * N * 4 / 1e6} MB)")

    # CuPy arrays
    input_cp = cp.random.rand(M, N).astype(np.float32)
    output_cp = cp.zeros((N, M), dtype=np.float32)

    # PyTorch tensors
    input_torch = torch.as_tensor(input_cp, device='cuda')
    output_torch = torch.zeros(N, M, dtype=torch.float32, device='cuda')

    expected = cp.asnumpy(input_cp).T
    results = []

    # CuPy
    def cupy_transpose():
        cp.copyto(output_cp, input_cp.T)

    cupy_transpose()
    cp.cuda.runtime.deviceSynchronize()
    assert np.allclose(cp.asnumpy(output_cp), expected), "CuPy incorrect!"
    min_t, mean_t = benchmark_cupy(cupy_transpose)
    results.append(BenchmarkResult("CuPy", min_t, mean_t))

    # PyTorch
    output_torch.fill_(0)
    def torch_transpose():
        output_torch.copy_(input_torch.T)

    torch_transpose()
    torch.cuda.synchronize()
    assert np.allclose(output_torch.cpu().numpy(), expected), "PyTorch incorrect!"
    min_t, mean_t = benchmark_torch(torch_transpose)
    results.append(BenchmarkResult("PyTorch", min_t, mean_t))

    # cuTile
    output_cp.fill(0)
    grid = (ct.cdiv(M, TRANSPOSE_TILE_M), ct.cdiv(N, TRANSPOSE_TILE_N), 1)
    stream = cp.cuda.get_current_stream()

    def cutile_transpose():
        ct.launch(stream, grid, transpose_cutile_kernel,
                  (input_cp, output_cp, TRANSPOSE_TILE_M, TRANSPOSE_TILE_N))

    cutile_transpose()
    cp.cuda.runtime.deviceSynchronize()
    assert np.allclose(cp.asnumpy(output_cp), expected), "cuTile incorrect!"
    min_t, mean_t = benchmark_cupy(cutile_transpose)
    results.append(BenchmarkResult("cuTile Python", min_t, mean_t))

    # Calculate bandwidth
    bytes_transferred = 2 * M * N * 4  # read + write, float32
    bandwidths = [f"{bytes_transferred / (r.min_ms / 1000) / 1e9:.1f} GB/s" for r in results]

    print_table("Matrix Transpose (Float32)", results, extra_col=("Bandwidth", bandwidths))
    return results


#=============================================================================
# Matrix Multiplication
#=============================================================================

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


def benchmark_matmul():
    print("\nBenchmarking Matrix Multiplication...")
    M, N, K = MATMUL_DIM, MATMUL_DIM, MATMUL_DIM
    print(f"  Size: {M}x{K} * {K}x{N}")

    # CuPy arrays (used for cuTile and cuBLAS)
    A_cp = cp.random.randn(M, K, dtype=np.float32)
    B_cp = cp.random.randn(K, N, dtype=np.float32)
    C_cp = cp.zeros((M, N), dtype=np.float32)

    # PyTorch tensors (from same data for fair comparison)
    torch.set_float32_matmul_precision("high")  # Enable TF32
    A_torch = torch.as_tensor(A_cp, device='cuda')
    B_torch = torch.as_tensor(B_cp, device='cuda')
    C_torch = torch.zeros(M, N, dtype=torch.float32, device='cuda')

    # Compute reference using CuPy (cuBLAS) for correctness checks
    # This avoids TF32 precision differences between PyTorch and CuPy
    C_ref_cp = cp.matmul(A_cp, B_cp)
    cp.cuda.runtime.deviceSynchronize()
    C_ref = cp.asnumpy(C_ref_cp)

    results = []
    flops = 2.0 * M * N * K

    # PyTorch
    def torch_matmul():
        torch.matmul(A_torch, B_torch, out=C_torch)

    torch_matmul()
    torch.cuda.synchronize()
    # PyTorch TF32 vs CuPy cuBLAS may differ, use relaxed tolerance
    assert np.allclose(C_torch.cpu().numpy(), C_ref, rtol=1e-1, atol=1e-1), "PyTorch incorrect!"
    min_t, mean_t = benchmark_torch(torch_matmul)
    results.append(BenchmarkResult("PyTorch", min_t, mean_t))

    # CuPy (uses cuBLAS) - this is the reference
    def cupy_matmul():
        cp.matmul(A_cp, B_cp, out=C_cp)

    cupy_matmul()
    cp.cuda.runtime.deviceSynchronize()
    min_t, mean_t = benchmark_cupy(cupy_matmul)
    results.append(BenchmarkResult("CuPy (cuBLAS)", min_t, mean_t))

    # cuTile
    C_cp.fill(0)
    grid_m = ceil(M / MATMUL_TM)
    grid_n = ceil(N / MATMUL_TN)
    grid = (grid_m * grid_n, 1, 1)
    stream = cp.cuda.get_current_stream()

    def cutile_matmul():
        ct.launch(stream, grid, matmul_cutile_kernel,
                  (A_cp, B_cp, C_cp, MATMUL_TM, MATMUL_TN, MATMUL_TK))

    cutile_matmul()
    cp.cuda.runtime.deviceSynchronize()
    # TF32 has reduced precision compared to FP32 cuBLAS
    assert np.allclose(cp.asnumpy(C_cp), C_ref, rtol=1e-1, atol=1e-1), "cuTile incorrect!"
    min_t, mean_t = benchmark_cupy(cutile_matmul)
    results.append(BenchmarkResult("cuTile Python", min_t, mean_t))

    # Calculate TFLOPS
    tflops_vals = [f"{flops / (r.min_ms * 1e-3) / 1e12:.2f} TFLOPS" for r in results]

    print_table("Matrix Multiplication (Float32, TF32 cores)", results, extra_col=("Performance", tflops_vals))
    return results


#=============================================================================
# Main
#=============================================================================

def main():
    print("=" * 60)
    print("  cuTile Python Comprehensive Benchmarks")
    print("=" * 60)
    print()
    print("Configuration:")
    print(f"  Runs: {NRUNS} (+ {WARMUP} warmup)")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print()

    vadd_results = benchmark_vadd()
    transpose_results = benchmark_transpose()
    matmul_results = benchmark_matmul()

    print()
    print("=" * 60)
    print("  Benchmark Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
