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
from math import ceil, log2

# Import prepare/run/verify functions from example files
from vadd import vadd_prepare, vadd_run, vadd_verify
from transpose import transpose_prepare, transpose_run, transpose_verify
from matmul import matmul_prepare, matmul_run, matmul_verify
from layernorm import layernorm_prepare, layernorm_run, layernorm_verify
from batchmatmul import batchmatmul_prepare, batchmatmul_run, batchmatmul_verify
from fft import fft_prepare, fft_run, fft_verify

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

# Tile sizes
VADD_TILE = 1024
TRANSPOSE_TILE_M = 64
TRANSPOSE_TILE_N = 64
MATMUL_TM = 64
MATMUL_TN = 64
MATMUL_TK = 64

# Layer norm sizes
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

def benchmark_vadd():
    print("\nBenchmarking Vector Addition...")
    print(f"  Size: {VADD_SIZE} elements ({VADD_SIZE * 4 / 1e6} MB)")

    # CuPy arrays for CuPy/PyTorch benchmarks
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

    # cuTile - use prepare/run/verify pattern
    data = vadd_prepare(shape=(VADD_SIZE,), dtype=np.float32)
    # Copy expected data for apples-to-apples comparison
    data["a"] = a_cp
    data["b"] = b_cp
    data["c"] = cp.zeros(VADD_SIZE, dtype=np.float32)

    result = vadd_run(data, tile=VADD_TILE, nruns=NRUNS, warmup=WARMUP)
    vadd_verify(data, result)
    min_t, mean_t = min(result["times"]), sum(result["times"]) / len(result["times"])
    results.append(BenchmarkResult("cuTile Python", min_t, mean_t))

    # Calculate bandwidth
    bytes_transferred = 3 * VADD_SIZE * 4  # 2 reads + 1 write, float32
    bandwidths = [f"{bytes_transferred / (r.min_ms / 1000) / 1e9:.1f} GB/s" for r in results]

    print_table("Vector Addition (Float32)", results, extra_col=("Bandwidth", bandwidths))
    return results


#=============================================================================
# Matrix Transpose
#=============================================================================

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

    # cuTile - use prepare/run/verify pattern
    data = transpose_prepare(M=M, N=N, dtype=np.float32)
    # Copy input for apples-to-apples comparison
    data["input"] = input_cp
    data["output"] = cp.zeros((N, M), dtype=np.float32)

    result = transpose_run(data, tile_m=TRANSPOSE_TILE_M, tile_n=TRANSPOSE_TILE_N,
                           nruns=NRUNS, warmup=WARMUP)
    transpose_verify(data, result)
    min_t, mean_t = min(result["times"]), sum(result["times"]) / len(result["times"])
    results.append(BenchmarkResult("cuTile Python", min_t, mean_t))

    # Calculate bandwidth
    bytes_transferred = 2 * M * N * 4  # read + write, float32
    bandwidths = [f"{bytes_transferred / (r.min_ms / 1000) / 1e9:.1f} GB/s" for r in results]

    print_table("Matrix Transpose (Float32)", results, extra_col=("Bandwidth", bandwidths))
    return results


#=============================================================================
# Matrix Multiplication
#=============================================================================

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

    # cuTile - use prepare/run/verify pattern
    data = matmul_prepare(M=M, N=N, K=K, dtype=np.float32)
    # Copy input for apples-to-apples comparison
    data["A"] = A_cp
    data["B"] = B_cp
    data["C"] = cp.zeros((M, N), dtype=np.float32)

    result = matmul_run(data, tm=MATMUL_TM, tn=MATMUL_TN, tk=MATMUL_TK,
                        nruns=NRUNS, warmup=WARMUP)
    matmul_verify(data, result)
    min_t, mean_t = min(result["times"]), sum(result["times"]) / len(result["times"])
    results.append(BenchmarkResult("cuTile Python", min_t, mean_t))

    # Calculate TFLOPS
    tflops_vals = [f"{flops / (r.min_ms * 1e-3) / 1e12:.2f} TFLOPS" for r in results]

    print_table("Matrix Multiplication (Float32, TF32 cores)", results, extra_col=("Performance", tflops_vals))
    return results


#=============================================================================
# Layer Normalization
#=============================================================================

def benchmark_layernorm():
    print("\nBenchmarking Layer Normalization...")
    M, N = LAYERNORM_M, LAYERNORM_N
    print(f"  Size: {M}x{N} ({M * N * 4 / 1e6} MB)")

    # cuTile - prepare data
    data = layernorm_prepare(M=M, N=N, eps=LAYERNORM_EPS, dtype=np.float32)

    # Extract CuPy/NumPy arrays for other benchmarks
    X_cp = data["X"]
    W_cp = data["W"]
    B_cp = data["B"]

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

    # cuTile - use prepare/run/verify pattern (unified fwd+bwd)
    result = layernorm_run(data, tile_n=LAYERNORM_TILE_N, nruns=NRUNS, warmup=WARMUP)
    layernorm_verify(data, result)

    # Forward pass timing
    min_t_fwd = min(result["times_fwd"])
    mean_t_fwd = sum(result["times_fwd"]) / len(result["times_fwd"])
    results.append(BenchmarkResult("cuTile Fwd", min_t_fwd, mean_t_fwd))

    # Backward pass timing
    min_t_bwd = min(result["times_bwd"])
    mean_t_bwd = sum(result["times_bwd"]) / len(result["times_bwd"])

    # Calculate bandwidth (rough estimate: 3 reads of X + W + B, 1 write of Y)
    bytes_transferred = (3 * M * N + N + N + M * N) * 4
    bandwidths = [f"{bytes_transferred / (r.min_ms / 1000) / 1e9:.1f} GB/s" for r in results]

    print_table("Layer Normalization Forward (Float32)", results, extra_col=("Bandwidth", bandwidths))

    # Print backward results separately
    bwd_results = [BenchmarkResult("cuTile Bwd", min_t_bwd, mean_t_bwd)]
    print_table("Layer Normalization Backward (Float32)", bwd_results)

    return results


#=============================================================================
# Batch Matrix Multiplication
#=============================================================================

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

    # cuTile - use prepare/run/verify pattern
    data = batchmatmul_prepare(Batch=Batch, M=M, K=K, N=N, dtype=np.float16)
    # Copy input for apples-to-apples comparison
    data["A"] = A_cp
    data["B"] = B_cp
    data["C"] = cp.zeros((Batch, M, N), dtype=np.float16)

    result = batchmatmul_run(data, tm=BATCHMATMUL_TM, tn=BATCHMATMUL_TN, tk=BATCHMATMUL_TK,
                              nruns=NRUNS, warmup=WARMUP)
    batchmatmul_verify(data, result)
    min_t, mean_t = min(result["times"]), sum(result["times"]) / len(result["times"])
    results.append(BenchmarkResult("cuTile Python", min_t, mean_t))

    # Calculate TFLOPS
    tflops_vals = [f"{flops / (r.min_ms * 1e-3) / 1e12:.2f} TFLOPS" for r in results]

    print_table("Batch Matrix Multiplication (Float16)", results, extra_col=("Performance", tflops_vals))
    return results


#=============================================================================
# FFT (3-stage Cooley-Tukey)
#=============================================================================

def benchmark_fft():
    print("\nBenchmarking FFT...")
    BS, N = FFT_BATCH, FFT_SIZE
    print(f"  Size: {BS} batches × {N} FFT ({BS * N * 8 / 1e6} MB)")

    # cuTile - use prepare/run/verify pattern
    data = fft_prepare(batch=BS, size=N, factors=FFT_FACTORS)

    # Reference result using torch
    reference = torch.fft.fft(data["input"], dim=-1)
    torch.cuda.synchronize()

    results = []

    # cuTile FFT
    result = fft_run(data, nruns=NRUNS, warmup=WARMUP)
    fft_verify(data, result)
    min_t, mean_t = min(result["times"]), sum(result["times"]) / len(result["times"])
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
