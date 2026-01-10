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


@ct.kernel
def layernorm_cutile_kernel(X, W, B, Y, Mean, Rstd, eps: ct.Constant[float], TILE_N: ct.Constant[int]):
    bid_m = ct.bid(0)
    num_tiles = ct.num_tiles(X, axis=1, shape=(1, TILE_N))
    N = X.shape[1]

    # Compute mean
    mean = ct.full((1, TILE_N), 0, dtype=ct.float32)
    for j in range(num_tiles):
        tx = ct.load(X, index=(bid_m, j), shape=(1, TILE_N), padding_mode=ct.PaddingMode.ZERO)
        mean += tx
    mean = ct.sum(mean, axis=1) / N
    ct.store(Mean, index=(bid_m,), tile=mean)

    # Compute variance
    var = ct.full((1, TILE_N), 0, dtype=ct.float32)
    for j in range(num_tiles):
        tx = ct.load(X, index=(bid_m, j), shape=(1, TILE_N), padding_mode=ct.PaddingMode.ZERO)
        mask = (j * TILE_N + ct.arange(TILE_N, dtype=ct.int32)) < N
        centered_tx = ct.where(mask, tx - mean, 0)
        var += centered_tx ** 2
    var = ct.sum(var, axis=1) / N
    rstd = 1 / ct.sqrt(var + eps)
    ct.store(Rstd, index=(bid_m,), tile=rstd)

    # Normalize and apply affine transformation
    for j in range(num_tiles):
        tx = ct.load(X, index=(bid_m, j), shape=(1, TILE_N), padding_mode=ct.PaddingMode.ZERO)
        tw = ct.load(W, index=(j,), shape=(TILE_N,), padding_mode=ct.PaddingMode.ZERO)
        tb = ct.load(B, index=(j,), shape=(TILE_N,), padding_mode=ct.PaddingMode.ZERO)
        ty = (tx - mean) * rstd
        ty = ty * tw + tb
        ct.store(Y, index=(bid_m, j), tile=ty.astype(Y.dtype))


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

@ct.kernel
def fft_kernel(x_packed_in, y_packed_out,
               W0, W1, W2, T0, T1,
               N: ct.Constant[int], F0: ct.Constant[int], F1: ct.Constant[int], F2: ct.Constant[int],
               BS: ct.Constant[int], D: ct.Constant[int]):
    """cuTile kernel for 3-stage Cooley-Tukey FFT."""
    F0F1 = F0 * F1
    F1F2 = F1 * F2
    F0F2 = F0 * F2

    bid = ct.bid(0)

    # Load input, reshape to separate real/imag
    X_ri = ct.reshape(ct.load(x_packed_in, index=(bid, 0, 0), shape=(BS, N * 2 // D, D)), (BS, N, 2))
    X_r = ct.reshape(ct.extract(X_ri, index=(0, 0, 0), shape=(BS, N, 1)), (BS, F0, F1, F2))
    X_i = ct.reshape(ct.extract(X_ri, index=(0, 0, 1), shape=(BS, N, 1)), (BS, F0, F1, F2))

    # Load W matrices (rotation matrices)
    W0_ri = ct.reshape(ct.load(W0, index=(0, 0, 0), shape=(F0, F0, 2)), (F0, F0, 2))
    W0_r = ct.reshape(ct.extract(W0_ri, index=(0, 0, 0), shape=(F0, F0, 1)), (1, F0, F0))
    W0_i = ct.reshape(ct.extract(W0_ri, index=(0, 0, 1), shape=(F0, F0, 1)), (1, F0, F0))

    W1_ri = ct.reshape(ct.load(W1, index=(0, 0, 0), shape=(F1, F1, 2)), (F1, F1, 2))
    W1_r = ct.reshape(ct.extract(W1_ri, index=(0, 0, 0), shape=(F1, F1, 1)), (1, F1, F1))
    W1_i = ct.reshape(ct.extract(W1_ri, index=(0, 0, 1), shape=(F1, F1, 1)), (1, F1, F1))

    W2_ri = ct.reshape(ct.load(W2, index=(0, 0, 0), shape=(F2, F2, 2)), (F2, F2, 2))
    W2_r = ct.reshape(ct.extract(W2_ri, index=(0, 0, 0), shape=(F2, F2, 1)), (1, F2, F2))
    W2_i = ct.reshape(ct.extract(W2_ri, index=(0, 0, 1), shape=(F2, F2, 1)), (1, F2, F2))

    # Load T matrices (twiddle factors)
    T0_ri = ct.reshape(ct.load(T0, index=(0, 0, 0), shape=(F0, F1F2, 2)), (F0, F1F2, 2))
    T0_r = ct.reshape(ct.extract(T0_ri, index=(0, 0, 0), shape=(F0, F1F2, 1)), (N, 1))
    T0_i = ct.reshape(ct.extract(T0_ri, index=(0, 0, 1), shape=(F0, F1F2, 1)), (N, 1))

    T1_ri = ct.reshape(ct.load(T1, index=(0, 0, 0), shape=(F1, F2, 2)), (F1, F2, 2))
    T1_r = ct.reshape(ct.extract(T1_ri, index=(0, 0, 0), shape=(F1, F2, 1)), (F1F2, 1))
    T1_i = ct.reshape(ct.extract(T1_ri, index=(0, 0, 1), shape=(F1, F2, 1)), (F1F2, 1))

    # CT0: Contract over F0 dimension
    X_r = ct.reshape(X_r, (BS, F0, F1F2))
    X_i = ct.reshape(X_i, (BS, F0, F1F2))
    X_r_ = ct.reshape(ct.matmul(W0_r, X_r) - ct.matmul(W0_i, X_i), (BS, N, 1))
    X_i_ = ct.reshape(ct.matmul(W0_i, X_r) + ct.matmul(W0_r, X_i), (BS, N, 1))

    # Twiddle & Permute 0
    X_r = T0_r * X_r_ - T0_i * X_i_
    X_i = T0_i * X_r_ + T0_r * X_i_
    X_r = ct.permute(ct.reshape(X_r, (BS, F0, F1, F2)), (0, 2, 3, 1))
    X_i = ct.permute(ct.reshape(X_i, (BS, F0, F1, F2)), (0, 2, 3, 1))

    # CT1: Contract over F1 dimension
    X_r = ct.reshape(X_r, (BS, F1, F0F2))
    X_i = ct.reshape(X_i, (BS, F1, F0F2))
    X_r_ = ct.reshape(ct.matmul(W1_r, X_r) - ct.matmul(W1_i, X_i), (BS, F1F2, F0))
    X_i_ = ct.reshape(ct.matmul(W1_i, X_r) + ct.matmul(W1_r, X_i), (BS, F1F2, F0))

    # Twiddle & Permute 1
    X_r = T1_r * X_r_ - T1_i * X_i_
    X_i = T1_i * X_r_ + T1_r * X_i_
    X_r = ct.permute(ct.reshape(X_r, (BS, F1, F2, F0)), (0, 2, 3, 1))
    X_i = ct.permute(ct.reshape(X_i, (BS, F1, F2, F0)), (0, 2, 3, 1))

    # CT2: Contract over F2 dimension
    X_r = ct.reshape(X_r, (BS, F2, F0F1))
    X_i = ct.reshape(X_i, (BS, F2, F0F1))
    X_r_ = ct.matmul(W2_r, X_r) - ct.matmul(W2_i, X_i)
    X_i_ = ct.matmul(W2_i, X_r) + ct.matmul(W2_r, X_i)

    # Final Permutation
    X_r = ct.permute(ct.reshape(X_r_, (BS, F2, F0, F1)), (0, 1, 3, 2))
    X_i = ct.permute(ct.reshape(X_i_, (BS, F2, F0, F1)), (0, 1, 3, 2))
    X_r = ct.reshape(X_r, (BS, N, 1))
    X_i = ct.reshape(X_i, (BS, N, 1))

    # Concatenate and Store
    Y_ri = ct.reshape(ct.cat((X_r, X_i), axis=-1), (BS, N * 2 // D, D))
    ct.store(y_packed_out, index=(bid, 0, 0), tile=Y_ri)


def fft_twiddles(rows: int, cols: int, factor: int, device, precision):
    """Generate DFT twiddle factors."""
    I, J = torch.meshgrid(torch.arange(rows, device=device),
                          torch.arange(cols, device=device), indexing='ij')
    W_complex = torch.exp(-2 * math.pi * 1j * (I * J) / factor)
    return torch.view_as_real(W_complex).to(precision).contiguous()


def fft_make_twiddles(factors, precision, device):
    """Generate W and T matrices for FFT."""
    F0, F1, F2 = factors
    N = F0 * F1 * F2
    F1F2 = F1 * F2
    W0 = fft_twiddles(F0, F0, F0, device, precision)
    W1 = fft_twiddles(F1, F1, F1, device, precision)
    W2 = fft_twiddles(F2, F2, F2, device, precision)
    T0 = fft_twiddles(F0, F1F2, N, device, precision)
    T1 = fft_twiddles(F1, F2, F1F2, device, precision)
    return (W0, W1, W2, T0, T1)


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
