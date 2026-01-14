#!/usr/bin/env python3
"""
FFT (3-stage Cooley-Tukey) example - cuTile Python
"""

import torch
import math
import cuda.tile as ct

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


#=============================================================================
# Example harness
#=============================================================================

def prepare(*, benchmark: bool = False, batch: int = None, size: int = None, factors: tuple = None, atom_packing_dim: int = 2):
    """Allocate and initialize data for FFT."""
    if batch is None:
        batch = 64 if benchmark else 2
    if factors is None:
        factors = (8, 8, 8) if benchmark else (2, 2, 2)
    F0, F1, F2 = factors
    N = F0 * F1 * F2
    if size is None:
        size = N
    assert size == N, f"size ({size}) must equal product of factors ({N})"
    D = atom_packing_dim

    input_data = torch.randn(batch, N, dtype=torch.complex64, device='cuda')

    # Pre-compute twiddles
    W0, W1, W2, T0, T1 = fft_make_twiddles(factors, input_data.real.dtype, input_data.device)

    # Pack input
    x_ri = torch.view_as_real(input_data)
    x_packed = x_ri.reshape(batch, N * 2 // D, D).contiguous()
    y_packed = torch.empty_like(x_packed)

    return {
        "input": input_data,
        "x_packed": x_packed,
        "y_packed": y_packed,
        "W0": W0, "W1": W1, "W2": W2, "T0": T0, "T1": T1,
        "factors": factors,
        "batch": batch,
        "N": N,
        "D": D
    }


def run(data, *, nruns: int = 1, warmup: int = 0):
    """Run FFT kernel with timing."""
    x_packed = data["x_packed"]
    y_packed = data["y_packed"]
    W0, W1, W2, T0, T1 = data["W0"], data["W1"], data["W2"], data["T0"], data["T1"]
    F0, F1, F2 = data["factors"]
    batch, N, D = data["batch"], data["N"], data["D"]

    grid = (batch, 1, 1)

    # Warmup
    for _ in range(warmup):
        ct.launch(torch.cuda.current_stream(), grid, fft_kernel,
                  (x_packed, y_packed, W0, W1, W2, T0, T1, N, F0, F1, F2, batch, D))
    torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(nruns):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        ct.launch(torch.cuda.current_stream(), grid, fft_kernel,
                  (x_packed, y_packed, W0, W1, W2, T0, T1, N, F0, F1, F2, batch, D))
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))  # ms

    output = torch.view_as_complex(y_packed.reshape(batch, N, 2))

    return {"output": output, "times": times}


def verify(data, result):
    """Verify FFT results."""
    reference = torch.fft.fft(data["input"], dim=-1)
    assert torch.allclose(result["output"], reference, rtol=1e-3, atol=1e-3), \
        f"FFT incorrect! max diff: {torch.max(torch.abs(result['output'] - reference))}"


#=============================================================================
# Reference implementations for benchmarking
#=============================================================================

def run_others(data, *, nruns: int = 1, warmup: int = 0):
    """Run reference implementations for comparison."""
    results = {}
    input_data = data["input"]

    # PyTorch FFT (uses cuFFT)
    for _ in range(warmup):
        torch.fft.fft(input_data, dim=-1)
    torch.cuda.synchronize()

    times_torch = []
    for _ in range(nruns):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        torch.fft.fft(input_data, dim=-1)
        end.record()
        torch.cuda.synchronize()
        times_torch.append(start.elapsed_time(end))
    results["cuFFT"] = times_torch

    return results


#=============================================================================
# Main
#=============================================================================

def test_fft(batch, size, factors, name=None):
    """Test FFT with given parameters."""
    name = name or f"fft batch={batch}, size={size}, factors={factors}"
    print(f"--- {name} ---")
    data = prepare(batch=batch, size=size, factors=factors)
    result = run(data)
    verify(data, result)
    print("  passed")


def main():
    print("--- cuTile FFT Examples ---\n")

    test_fft(64, 512, (8, 8, 8))
    test_fft(32, 512, (8, 8, 8))

    print("\n--- All FFT examples completed ---")


if __name__ == "__main__":
    main()
