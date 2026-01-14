# FFT Example - Julia port of cuTile Python's FFT.py sample
#
# This implements a 3-stage Cooley-Tukey FFT decomposition. The FFT of size N is decomposed
# as N = F0 * F1 * F2, allowing efficient tensor factorization.
#
# Key difference from Python: Julia uses column-major storage, so reshape dimensions are
# swapped and right-multiply (X @ W) is used instead of left-multiply (W @ X) to process
# rows instead of columns, achieving the same strided element access pattern.
#
# SPDX-License-Identifier: Apache-2.0

using CUDA
import cuTile as ct
using Test
using FFTW

# FFT kernel - 3-stage Cooley-Tukey decomposition (column-major version)
#
# The key insight: In Python row-major, reshape (F0, F1F2) puts stride-F1F2 elements
# in columns. In Julia column-major, reshape (F1F2, F0) puts stride-F0 elements in rows.
# We use right-multiply X @ W instead of W @ X to process rows instead of columns.
#
# Input/output layout: (D, BS, N2D) where D=2 for real/imag interleaving.
function fft_kernel(
    x_packed_in::ct.TileArray{Float32, 3},   # Input (D, BS, N2D) - natural Julia complex layout
    y_packed_out::ct.TileArray{Float32, 3},  # Output (D, BS, N2D)
    W0::ct.TileArray{Float32, 3},            # W0 (F0, F0, 2) DFT matrix
    W1::ct.TileArray{Float32, 3},            # W1 (F1, F1, 2)
    W2::ct.TileArray{Float32, 3},            # W2 (F2, F2, 2)
    T0::ct.TileArray{Float32, 3},            # T0 (F1F2, F0, 2) twiddle factors
    T1::ct.TileArray{Float32, 3},            # T1 (F0F2, F1, 2) twiddle factors
    n_const::ct.Constant{Int},
    f0_const::ct.Constant{Int},
    f1_const::ct.Constant{Int},
    f2_const::ct.Constant{Int},
    f0f1_const::ct.Constant{Int},
    f1f2_const::ct.Constant{Int},
    f0f2_const::ct.Constant{Int},
    bs_const::ct.Constant{Int},
    d_const::ct.Constant{Int},
    n2d_const::ct.Constant{Int}
)
    # Extract constant values
    N = n_const[]
    F0 = f0_const[]
    F1 = f1_const[]
    F2 = f2_const[]
    F0F1 = f0f1_const[]
    F1F2 = f1f2_const[]
    F0F2 = f0f2_const[]
    BS = bs_const[]
    D = d_const[]
    N2D = n2d_const[]

    bid = ct.bid(1)

    # --- Load Input Data ---
    # Input is (D, BS, N2D) where D=2 for real/imag. Load and reshape to (2, BS, N).
    X_ri = ct.reshape(ct.load(x_packed_in, (1, bid, 1), (D, BS, N2D)), (2, BS, N))

    # Split real and imaginary parts (extract from first dimension)
    X_r = ct.reshape(ct.extract(X_ri, (1, 1, 1), (1, BS, N)), (BS, F1F2, F0))
    X_i = ct.reshape(ct.extract(X_ri, (2, 1, 1), (1, BS, N)), (BS, F1F2, F0))

    # --- Load DFT Matrices ---
    # W0 (F0 x F0) - for right-multiply X @ W0
    W0_ri = ct.reshape(ct.load(W0, (1, 1, 1), (F0, F0, 2)), (F0, F0, 2))
    W0_r = ct.broadcast_to(ct.reshape(ct.extract(W0_ri, (1, 1, 1), (F0, F0, 1)), (1, F0, F0)), (BS, F0, F0))
    W0_i = ct.broadcast_to(ct.reshape(ct.extract(W0_ri, (1, 1, 2), (F0, F0, 1)), (1, F0, F0)), (BS, F0, F0))

    # W1 (F1 x F1)
    W1_ri = ct.reshape(ct.load(W1, (1, 1, 1), (F1, F1, 2)), (F1, F1, 2))
    W1_r = ct.broadcast_to(ct.reshape(ct.extract(W1_ri, (1, 1, 1), (F1, F1, 1)), (1, F1, F1)), (BS, F1, F1))
    W1_i = ct.broadcast_to(ct.reshape(ct.extract(W1_ri, (1, 1, 2), (F1, F1, 1)), (1, F1, F1)), (BS, F1, F1))

    # W2 (F2 x F2)
    W2_ri = ct.reshape(ct.load(W2, (1, 1, 1), (F2, F2, 2)), (F2, F2, 2))
    W2_r = ct.broadcast_to(ct.reshape(ct.extract(W2_ri, (1, 1, 1), (F2, F2, 1)), (1, F2, F2)), (BS, F2, F2))
    W2_i = ct.broadcast_to(ct.reshape(ct.extract(W2_ri, (1, 1, 2), (F2, F2, 1)), (1, F2, F2)), (BS, F2, F2))

    # --- Load Twiddle Factors ---
    # T0 (F1F2, F0) - note swapped from Python's (F0, F1F2)
    T0_ri = ct.reshape(ct.load(T0, (1, 1, 1), (F1F2, F0, 2)), (F1F2, F0, 2))
    T0_r = ct.reshape(ct.extract(T0_ri, (1, 1, 1), (F1F2, F0, 1)), (1, N))
    T0_i = ct.reshape(ct.extract(T0_ri, (1, 1, 2), (F1F2, F0, 1)), (1, N))

    # T1 (F0F2, F1) - note swapped from Python's (F1, F2)
    T1_ri = ct.reshape(ct.load(T1, (1, 1, 1), (F0F2, F1, 2)), (F0F2, F1, 2))
    T1_r = ct.reshape(ct.extract(T1_ri, (1, 1, 1), (F0F2, F1, 1)), (1, F0F2 * F1))
    T1_i = ct.reshape(ct.extract(T1_ri, (1, 1, 2), (F0F2, F1, 1)), (1, F0F2 * F1))

    # --- Stage 0: F0-point DFT ---
    # X is (BS, F1F2, F0), W0 is (BS, F0, F0)
    # Right-multiply: X @ W0 processes each row (F1F2 rows, each with F0 elements)
    # Each row has elements at stride F1F2 in the original array - exactly what we need!
    X_r_ = X_r * W0_r - X_i * W0_i  # (BS, F1F2, F0) @ (BS, F0, F0) → (BS, F1F2, F0)
    X_i_ = X_r * W0_i + X_i * W0_r

    # --- Twiddle & Permute 0 ---
    # Reshape to (BS, N) for element-wise twiddle multiply
    X_r_flat = ct.reshape(X_r_, (BS, N))
    X_i_flat = ct.reshape(X_i_, (BS, N))
    X_r2 = T0_r .* X_r_flat .- T0_i .* X_i_flat
    X_i2 = T0_i .* X_r_flat .+ T0_r .* X_i_flat

    # Reshape and permute for stage 1
    # Current logical layout after reshape (BS, F1F2, F0): data at (bs, f1*F2+f2, f0)
    # Reshape to (BS, F2, F1, F0) then permute to (BS, F0F2, F1) for stage 1
    X_r3 = ct.reshape(X_r2, (BS, F2, F1, F0))
    X_i3 = ct.reshape(X_i2, (BS, F2, F1, F0))
    X_r4 = ct.permute(X_r3, (1, 2, 4, 3))  # (BS, F2, F0, F1)
    X_i4 = ct.permute(X_i3, (1, 2, 4, 3))
    X_r5 = ct.reshape(X_r4, (BS, F0F2, F1))
    X_i5 = ct.reshape(X_i4, (BS, F0F2, F1))

    # --- Stage 1: F1-point DFT ---
    # X is (BS, F0F2, F1), W1 is (BS, F1, F1)
    X_r6 = X_r5 * W1_r - X_i5 * W1_i
    X_i6 = X_r5 * W1_i + X_i5 * W1_r

    # --- Twiddle & Permute 1 ---
    X_r_flat2 = ct.reshape(X_r6, (BS, N))
    X_i_flat2 = ct.reshape(X_i6, (BS, N))
    X_r7 = T1_r .* X_r_flat2 .- T1_i .* X_i_flat2
    X_i7 = T1_i .* X_r_flat2 .+ T1_r .* X_i_flat2

    # Reshape and permute for stage 2
    X_r8 = ct.reshape(X_r7, (BS, F2, F0, F1))
    X_i8 = ct.reshape(X_i7, (BS, F2, F0, F1))
    X_r9 = ct.permute(X_r8, (1, 3, 4, 2))  # (BS, F0, F1, F2)
    X_i9 = ct.permute(X_i8, (1, 3, 4, 2))
    X_r10 = ct.reshape(X_r9, (BS, F0F1, F2))
    X_i10 = ct.reshape(X_i9, (BS, F0F1, F2))

    # --- Stage 2: F2-point DFT ---
    # X is (BS, F0F1, F2), W2 is (BS, F2, F2)
    X_r11 = X_r10 * W2_r - X_i10 * W2_i
    X_i11 = X_r10 * W2_i + X_i10 * W2_r

    # --- Final Output ---
    # After stage 2, data is in (BS, F0F1, F2) layout
    # Reshape to (BS, F0, F1, F2) - output is already in frequency order
    X_r_final = ct.reshape(X_r11, (1, BS, N))
    X_i_final = ct.reshape(X_i11, (1, BS, N))

    # --- Concatenate and Store ---
    Y_ri = ct.reshape(ct.cat((X_r_final, X_i_final), 1), (D, BS, N2D))
    ct.store(y_packed_out, (1, bid, 1), Y_ri)

    return
end

# Helper: Generate DFT matrix W_n^{ij} = exp(-2πi * ij / n)
function dft_matrix(size::Int)
    W = zeros(ComplexF32, size, size)
    for i in 0:size-1, j in 0:size-1
        W[i+1, j+1] = exp(-2π * im * i * j / size)
    end
    result = zeros(Float32, size, size, 2)
    result[:, :, 1] = Float32.(real.(W))
    result[:, :, 2] = Float32.(imag.(W))
    return result
end

# Generate twiddle factors T0 for column-major layout (F1F2, F0)
# In Julia column-major, position (j, i) in (F1F2, F0) has linear index j + i*F1F2
# This corresponds to Python's position (i, j) in (F0, F1F2) with linear index i*F1F2 + j
# The twiddle value is ω_N^{i * j}
function make_twiddles_T0(F0::Int, F1F2::Int, N::Int)
    T0 = zeros(Float32, F1F2, F0, 2)
    for j in 0:F1F2-1, i in 0:F0-1
        val = exp(-2π * im * i * j / N)
        T0[j+1, i+1, 1] = Float32(real(val))
        T0[j+1, i+1, 2] = Float32(imag(val))
    end
    return T0
end

# Generate twiddle factors T1 for column-major layout (F0F2, F1)
# After stage 0 and permute, data is in (F0F2, F1) layout
# The twiddle value is ω_{F1F2}^{j * k} where j is F1 index and k is F2 index within F0F2
function make_twiddles_T1(F0::Int, F1::Int, F2::Int)
    F0F2 = F0 * F2
    F1F2 = F1 * F2
    T1 = zeros(Float32, F0F2, F1, 2)
    for k in 0:F0F2-1, j in 0:F1-1
        # k encodes (f0, f2) = (k ÷ F2, k % F2) after permute
        f2 = k % F2
        val = exp(-2π * im * j * f2 / F1F2)
        T1[k+1, j+1, 1] = Float32(real(val))
        T1[k+1, j+1, 2] = Float32(imag(val))
    end
    return T1
end

# Generate all W and T matrices for column-major algorithm
function make_twiddles(factors::NTuple{3, Int})
    F0, F1, F2 = factors
    N = F0 * F1 * F2
    F1F2 = F1 * F2

    # DFT matrices (same for row/column-major since symmetric)
    W0 = dft_matrix(F0)
    W1 = dft_matrix(F1)
    W2 = dft_matrix(F2)

    # Column-major twiddle factors
    T0 = make_twiddles_T0(F0, F1F2, N)
    T1 = make_twiddles_T1(F0, F1, F2)

    return (W0, W1, W2, T0, T1)
end

#=============================================================================
 Example harness
=============================================================================#

function prepare(; benchmark::Bool=false,
                  batch::Int=benchmark ? 64 : 2,
                  n::Int=benchmark ? 512 : 8,
                  factors::NTuple{3,Int}=benchmark ? (8, 8, 8) : (2, 2, 2),
                  atom_packing_dim::Int=2)
    @assert factors[1] * factors[2] * factors[3] == n "Factors must multiply to N"
    @assert (n * 2) % atom_packing_dim == 0 "N*2 must be divisible by atom_packing_dim"

    CUDA.seed!(42)
    input = CUDA.randn(ComplexF32, batch, n)

    # Pre-compute twiddles (one-time CPU cost)
    W0, W1, W2, T0, T1 = make_twiddles(factors)
    W0_gpu = CuArray(W0)
    W1_gpu = CuArray(W1)
    W2_gpu = CuArray(W2)
    T0_gpu = CuArray(T0)
    T1_gpu = CuArray(T1)

    # Pack input
    D = atom_packing_dim
    N2D = n * 2 ÷ D
    x_packed = reinterpret(reshape, Float32, input)
    y_packed = CuArray{Float32}(undef, D, batch, N2D)

    return (;
        input, x_packed, y_packed,
        W0_gpu, W1_gpu, W2_gpu, T0_gpu, T1_gpu,
        factors, batch, n, D, N2D
    )
end

function run(data; nruns::Int=1, warmup::Int=0)
    (; x_packed, y_packed, W0_gpu, W1_gpu, W2_gpu, T0_gpu, T1_gpu,
       factors, batch, n, D, N2D) = data

    F0, F1, F2 = factors
    F0F1 = F0 * F1
    F1F2 = F1 * F2
    F0F2 = F0 * F2
    grid = (batch, 1, 1)

    CUDA.@sync for _ in 1:warmup
        ct.launch(fft_kernel, grid,
                  x_packed, y_packed,
                  W0_gpu, W1_gpu, W2_gpu, T0_gpu, T1_gpu,
                  ct.Constant(n), ct.Constant(F0), ct.Constant(F1), ct.Constant(F2),
                  ct.Constant(F0F1), ct.Constant(F1F2), ct.Constant(F0F2),
                  ct.Constant(batch), ct.Constant(D), ct.Constant(N2D))
    end

    times = Float64[]
    for _ in 1:nruns
        t = CUDA.@elapsed ct.launch(fft_kernel, grid,
                                    x_packed, y_packed,
                                    W0_gpu, W1_gpu, W2_gpu, T0_gpu, T1_gpu,
                                    ct.Constant(n), ct.Constant(F0), ct.Constant(F1), ct.Constant(F2),
                                    ct.Constant(F0F1), ct.Constant(F1F2), ct.Constant(F0F2),
                                    ct.Constant(batch), ct.Constant(D), ct.Constant(N2D))
        push!(times, t * 1000)  # ms
    end

    # Unpack output
    y_complex = reinterpret(reshape, ComplexF32, y_packed)
    output = copy(y_complex)

    return (; output, times)
end

function verify(data, result)
    reference = FFTW.fft(Array(data.input), 2)
    @assert isapprox(Array(result.output), reference, rtol=1e-4)
end

#=============================================================================
 Reference implementations for benchmarking
=============================================================================#

function run_others(data; nruns::Int=1, warmup::Int=0)
    (; input, batch, n) = data
    results = Dict{String, Vector{Float64}}()

    output_cufft = similar(input)

    # CUFFT via CUDA.CUFFT
    CUDA.@sync for _ in 1:warmup
        CUDA.CUFFT.fft!(copy(input), 2)
    end
    times_cufft = Float64[]
    for _ in 1:nruns
        input_copy = copy(input)
        t = CUDA.@elapsed CUDA.CUFFT.fft!(input_copy, 2)
        push!(times_cufft, t * 1000)
    end
    results["cuFFT"] = times_cufft

    return results
end

#=============================================================================
 Main
=============================================================================#

function main()
    println("--- Running cuTile FFT Example ---")

    # Configuration
    BATCH_SIZE = 2
    FFT_SIZE = 8
    FFT_FACTORS = (2, 2, 2)
    ATOM_PACKING_DIM = 2

    println("  Configuration:")
    println("    FFT Size (N): $FFT_SIZE")
    println("    Batch Size: $BATCH_SIZE")
    println("    FFT Factors: $FFT_FACTORS")
    println("    Atom Packing Dim: $ATOM_PACKING_DIM")

    # Use prepare/run/verify pattern
    data = prepare(; batch=BATCH_SIZE, n=FFT_SIZE, factors=FFT_FACTORS, atom_packing_dim=ATOM_PACKING_DIM)
    println("\nInput data shape: $(size(data.input)), dtype: $(eltype(data.input))")

    result = run(data)
    println("cuTile FFT Output shape: $(size(result.output)), dtype: $(eltype(result.output))")

    verify(data, result)
    println("\n✓ Correctness check PASSED")

    println("\n--- cuTile FFT example execution complete ---")
end

isinteractive() || main()
