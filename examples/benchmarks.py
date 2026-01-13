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
import math
from math import ceil, log2

# Import kernels from example files
from vadd import vadd_cutile_kernel
from transpose import transpose_cutile_kernel
from matmul import matmul_cutile_kernel, swizzle_2d
from layernorm import layernorm_cutile_kernel
from batchmatmul import batchmatmul_cutile_kernel
from fft import fft_kernel, fft_make_twiddles

#=============================================================================
# Configuration
#=============================================================================

NRUNS = 10
WARMUP = 3

# Data sizes - large enough to saturate GPU and minimize launch overhead
VADD_SIZE = 2**27           # 512 MB (128M elements)
TRANSPOSE_DIM = 8192        # 8192x8192 = 268 MB
MATMUL_DIM = 4096           # 4096x4096x4096

# FFT sizes - must match Julia configuration
FFT_BATCH = 64
FFT_SIZE = 512
FFT_FACTORS = (8, 8, 8)
FFT_ATOM_PACKING_DIM = 2

# Tile sizes
VADD_TILE = 1024
TRANSPOSE_TILE_M = 64
TRANSPOSE_TILE_N = 64
MATMUL_TM = 64
MATMUL_TN = 64
MATMUL_TK = 64

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

# cuTile kernel: use vadd_cutile_kernel from vadd.py

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

# cuTile kernel: use transpose_cutile_kernel from transpose.py

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

# cuTile kernel: use matmul_cutile_kernel and swizzle_2d from matmul.py

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
# Layer Normalization
#=============================================================================

LAYERNORM_M = 4096
LAYERNORM_N = 4096
LAYERNORM_TILE_N = 1024
LAYERNORM_EPS = 1e-5

# Batch matmul sizes
BATCHMATMUL_BATCH = 8
BATCHMATMUL_M = 1024
BATCHMATMUL_K = 512
BATCHMATMUL_N = 2048
BATCHMATMUL_TM = 128
BATCHMATMUL_TN = 256
BATCHMATMUL_TK = 64

# cuTile kernel: use layernorm_cutile_kernel from layernorm.py

def benchmark_layernorm():
    print("\nBenchmarking Layer Normalization...")
    M, N = LAYERNORM_M, LAYERNORM_N
    print(f"  Size: {M}x{N} ({M * N * 4 / 1e6} MB)")

    # CuPy arrays
    X_cp = -2.3 + 0.5 * cp.random.randn(M, N).astype(np.float32)
    W_cp = cp.random.randn(N).astype(np.float32)
    B_cp = cp.random.randn(N).astype(np.float32)
    Y_cp = cp.zeros((M, N), dtype=np.float32)
    Mean_cp = cp.zeros(M, dtype=np.float32)
    Rstd_cp = cp.zeros(M, dtype=np.float32)

    # PyTorch tensors
    X_torch = torch.as_tensor(X_cp, device='cuda')
    W_torch = torch.as_tensor(W_cp, device='cuda')
    B_torch = torch.as_tensor(B_cp, device='cuda')
    Y_torch = torch.zeros(M, N, dtype=torch.float32, device='cuda')

    # Reference result
    X_np = cp.asnumpy(X_cp)
    W_np = cp.asnumpy(W_cp)
    B_np = cp.asnumpy(B_cp)
    expected_mean = np.mean(X_np, axis=1, keepdims=True)
    expected_var = np.mean((X_np - expected_mean) ** 2, axis=1, keepdims=True)
    expected_rstd = 1.0 / np.sqrt(expected_var + LAYERNORM_EPS)
    normalized = (X_np - expected_mean) * expected_rstd
    expected_Y = normalized * W_np + B_np

    results = []

    # PyTorch F.layer_norm
    def torch_layernorm():
        nonlocal Y_torch
        Y_torch = torch.nn.functional.layer_norm(X_torch, (N,), W_torch, B_torch, LAYERNORM_EPS)

    torch_layernorm()
    torch.cuda.synchronize()
    assert np.allclose(Y_torch.cpu().numpy(), expected_Y, rtol=1e-2, atol=1e-2), "PyTorch incorrect!"
    min_t, mean_t = benchmark_torch(torch_layernorm)
    results.append(BenchmarkResult("PyTorch", min_t, mean_t))

    # cuTile
    Y_cp.fill(0)
    Mean_cp.fill(0)
    Rstd_cp.fill(0)
    stream = cp.cuda.get_current_stream()

    def cutile_layernorm():
        ct.launch(stream, (M,), layernorm_cutile_kernel,
                  (X_cp, W_cp, B_cp, Y_cp, Mean_cp, Rstd_cp, LAYERNORM_EPS, LAYERNORM_TILE_N))

    cutile_layernorm()
    cp.cuda.runtime.deviceSynchronize()
    assert np.allclose(cp.asnumpy(Y_cp), expected_Y, rtol=1e-2, atol=1e-2), "cuTile incorrect!"
    min_t, mean_t = benchmark_cupy(cutile_layernorm)
    results.append(BenchmarkResult("cuTile Python", min_t, mean_t))

    # Calculate bandwidth (rough estimate: 3 reads of X + W + B, 1 write of Y)
    bytes_transferred = (3 * M * N + N + N + M * N) * 4
    bandwidths = [f"{bytes_transferred / (r.min_ms / 1000) / 1e9:.1f} GB/s" for r in results]

    print_table("Layer Normalization (Float32)", results, extra_col=("Bandwidth", bandwidths))
    return results


#=============================================================================
# Batch Matrix Multiplication
#=============================================================================

# cuTile kernel: use batchmatmul_cutile_kernel from batchmatmul.py

def benchmark_batchmatmul():
    print("\nBenchmarking Batch Matrix Multiplication...")
    Batch, M, K, N = BATCHMATMUL_BATCH, BATCHMATMUL_M, BATCHMATMUL_K, BATCHMATMUL_N
    print(f"  Size: ({Batch} x {M} x {K}) @ ({Batch} x {K} x {N}), Float16")

    # PyTorch tensors
    A_torch = torch.randn(Batch, M, K, dtype=torch.float16, device='cuda')
    B_torch = torch.randn(Batch, K, N, dtype=torch.float16, device='cuda')
    C_torch = torch.zeros(Batch, M, N, dtype=torch.float16, device='cuda')

    # CuPy arrays (from same data)
    A_cp = cp.asarray(A_torch)
    B_cp = cp.asarray(B_torch)
    C_cp = cp.zeros((Batch, M, N), dtype=np.float16)

    # Reference result (PyTorch bmm in fp32 for accuracy)
    C_ref = torch.bmm(A_torch.float(), B_torch.float()).cpu().numpy()

    results = []
    flops = 2.0 * Batch * M * N * K

    # PyTorch bmm
    def torch_bmm():
        torch.bmm(A_torch, B_torch, out=C_torch)

    torch_bmm()
    torch.cuda.synchronize()
    assert np.allclose(C_torch.float().cpu().numpy(), C_ref, rtol=1e-1, atol=1e-1), "PyTorch incorrect!"
    min_t, mean_t = benchmark_torch(torch_bmm)
    results.append(BenchmarkResult("PyTorch bmm", min_t, mean_t))

    # cuTile
    C_cp.fill(0)
    grid = (Batch, ceil(M / BATCHMATMUL_TM), ceil(N / BATCHMATMUL_TN))
    stream = cp.cuda.get_current_stream()

    def cutile_bmm():
        ct.launch(stream, grid, batchmatmul_cutile_kernel,
                  (A_cp, B_cp, C_cp, BATCHMATMUL_TM, BATCHMATMUL_TN, BATCHMATMUL_TK))

    cutile_bmm()
    cp.cuda.runtime.deviceSynchronize()
    assert np.allclose(cp.asnumpy(C_cp).astype(np.float32), C_ref, rtol=1e-1, atol=1e-1), "cuTile incorrect!"
    min_t, mean_t = benchmark_cupy(cutile_bmm)
    results.append(BenchmarkResult("cuTile Python", min_t, mean_t))

    # Calculate TFLOPS
    tflops_vals = [f"{flops / (r.min_ms * 1e-3) / 1e12:.2f} TFLOPS" for r in results]

    print_table("Batch Matrix Multiplication (Float16)", results, extra_col=("Performance", tflops_vals))
    return results


#=============================================================================
# FFT (3-stage Cooley-Tukey)
#=============================================================================

# cuTile kernel: use fft_kernel and fft_make_twiddles from fft.py

def benchmark_fft():
    print("\nBenchmarking FFT...")
    BS, N = FFT_BATCH, FFT_SIZE
    F0, F1, F2 = FFT_FACTORS
    D = FFT_ATOM_PACKING_DIM
    print(f"  Size: {BS} batches × {N} FFT ({BS * N * 8 / 1e6} MB)")

    # PyTorch complex input
    input_torch = torch.randn(BS, N, dtype=torch.complex64, device='cuda')

    # Reference result
    reference = torch.fft.fft(input_torch, dim=-1)
    torch.cuda.synchronize()

    results = []

    # Pre-compute everything outside timing loop
    x_ri = torch.view_as_real(input_torch)
    x_packed = x_ri.reshape(BS, N * 2 // D, D).contiguous()
    W0, W1, W2, T0, T1 = fft_make_twiddles(FFT_FACTORS, input_torch.real.dtype, input_torch.device)
    y_packed = torch.empty_like(x_packed)
    grid = (BS, 1, 1)

    # Kernel launch function
    def fft_launch():
        ct.launch(torch.cuda.current_stream(), grid, fft_kernel,
                  (x_packed, y_packed, W0, W1, W2, T0, T1, N, F0, F1, F2, BS, D))

    # Verify correctness
    fft_launch()
    torch.cuda.synchronize()
    output = torch.view_as_complex(y_packed.reshape(BS, N, 2))
    assert torch.allclose(output, reference, rtol=1e-3, atol=1e-3), "cuTile FFT incorrect!"

    min_t, mean_t = benchmark_torch(fft_launch)
    results.append(BenchmarkResult("cuTile Python", min_t, mean_t))

    # Calculate GFLOPS (5 * N * log2(N) ops per complex FFT)
    flops_per_fft = 5.0 * N * log2(N)
    total_flops = BS * flops_per_fft
    gflops = [f"{total_flops / (r.min_ms * 1e-3) / 1e9:.1f} GFLOPS" for r in results]

    print_table("FFT (ComplexF32)", results, extra_col=("Performance", gflops))
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
    layernorm_results = benchmark_layernorm()
    batchmatmul_results = benchmark_batchmatmul()
    fft_results = benchmark_fft()

    print()
    print("=" * 60)
    print("  Benchmark Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
