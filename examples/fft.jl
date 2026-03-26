# FFT Example - Julia port of cuTile Python's FFT.py sample
#
# This implements a 3-stage Cooley-Tukey FFT decomposition. The FFT of size N is decomposed
# as N = F0 * F1 * F2, allowing efficient tensor factorization.
#
# All shapes are the reverse of Python's row-major shapes, so that the memory layout is
# identical. No extra permutedims are needed for batch dimension shuffling.
#
# SPDX-License-Identifier: Apache-2.0

using CUDA
import cuTile as ct
using Test
using FFTW

# FFT kernel - 3-stage Cooley-Tukey decomposition
#
# Python row-major shape (A, B, C) ↔ Julia col-major shape (C, B, A) — same memory layout.
# Python left-multiply W @ X ↔ Julia right-multiply X * W (batch dims trailing).
# Python ct.permute(x, (0,2,3,1)) ↔ Julia permutedims(x, (3,1,2,4)).
function fft_kernel(
    x_packed_in::ct.TileArray{Float32, 3},   # Input (D, N2D, BS)
    y_packed_out::ct.TileArray{Float32, 3},  # Output (D, N2D, BS)
    W0::ct.TileArray{Float32, 3},            # W0 (2, F0, F0) DFT matrix
    W1::ct.TileArray{Float32, 3},            # W1 (2, F1, F1)
    W2::ct.TileArray{Float32, 3},            # W2 (2, F2, F2)
    T0::ct.TileArray{Float32, 3},            # T0 (2, F1F2, F0) twiddle factors
    T1::ct.TileArray{Float32, 3},            # T1 (2, F2, F1) twiddle factors
    n_const::Int,
    f0_const::Int,
    f1_const::Int,
    f2_const::Int,
    f0f1_const::Int,
    f1f2_const::Int,
    f0f2_const::Int,
    bs_const::Int,
    d_const::Int,
    n2d_const::Int
)
    N = n_const
    F0 = f0_const
    F1 = f1_const
    F2 = f2_const
    F0F1 = f0f1_const
    F1F2 = f1f2_const
    F0F2 = f0f2_const
    BS = bs_const
    D = d_const
    N2D = n2d_const

    bid = ct.bid(1)

    # --- Load Input Data ---
    # Input is (D, N2D, BS). Load and reshape to (2, N, BS).
    X_ri = reshape(ct.load(x_packed_in; index=(Int32(1), Int32(1), bid), shape=(D, N2D, BS)), (2, N, BS))

    # Split real and imaginary parts, reshape to 4D factored form
    X_r = reshape(ct.extract(X_ri, (1, 1, 1), (1, N, BS)), (F2, F1, F0, BS))
    X_i = reshape(ct.extract(X_ri, (2, 1, 1), (1, N, BS)), (F2, F1, F0, BS))

    # --- Load DFT Matrices ---
    # W0 (F0 × F0): trailing batch dim 1 for broadcast in batched matmul
    W0_ri = reshape(ct.load(W0; index=(1, 1, 1), shape=(2, F0, F0)), (2, F0, F0))
    W0_r = reshape(ct.extract(W0_ri, (1, 1, 1), (1, F0, F0)), (F0, F0, 1))
    W0_i = reshape(ct.extract(W0_ri, (2, 1, 1), (1, F0, F0)), (F0, F0, 1))

    # W1 (F1 × F1)
    W1_ri = reshape(ct.load(W1; index=(1, 1, 1), shape=(2, F1, F1)), (2, F1, F1))
    W1_r = reshape(ct.extract(W1_ri, (1, 1, 1), (1, F1, F1)), (F1, F1, 1))
    W1_i = reshape(ct.extract(W1_ri, (2, 1, 1), (1, F1, F1)), (F1, F1, 1))

    # W2 (F2 × F2)
    W2_ri = reshape(ct.load(W2; index=(1, 1, 1), shape=(2, F2, F2)), (2, F2, F2))
    W2_r = reshape(ct.extract(W2_ri, (1, 1, 1), (1, F2, F2)), (F2, F2, 1))
    W2_i = reshape(ct.extract(W2_ri, (2, 1, 1), (1, F2, F2)), (F2, F2, 1))

    # --- Load Twiddle Factors ---
    # T0 (2, F1F2, F0) → flatten to (1, N) for element-wise multiply
    T0_ri = reshape(ct.load(T0; index=(1, 1, 1), shape=(2, F1F2, F0)), (2, F1F2, F0))
    T0_r = reshape(ct.extract(T0_ri, (1, 1, 1), (1, F1F2, F0)), (1, N))
    T0_i = reshape(ct.extract(T0_ri, (2, 1, 1), (1, F1F2, F0)), (1, N))

    # T1 (2, F2, F1) → flatten to (1, F1F2) for element-wise multiply
    T1_ri = reshape(ct.load(T1; index=(1, 1, 1), shape=(2, F2, F1)), (2, F2, F1))
    T1_r = reshape(ct.extract(T1_ri, (1, 1, 1), (1, F2, F1)), (1, F1F2))
    T1_i = reshape(ct.extract(T1_ri, (2, 1, 1), (1, F2, F1)), (1, F1F2))

    # --- Stage 0: F0-point DFT ---
    # X: (F1F2, F0, BS) × W0: (F0, F0, 1) → (F1F2, F0, BS)
    X_r = reshape(X_r, (F1F2, F0, BS))
    X_i = reshape(X_i, (F1F2, F0, BS))
    X_r_ = reshape(X_r * W0_r - X_i * W0_i, (1, N, BS))
    X_i_ = reshape(X_r * W0_i + X_i * W0_r, (1, N, BS))

    # --- Twiddle & Permute 0 ---
    X_r2 = T0_r .* X_r_ .- T0_i .* X_i_
    X_i2 = T0_i .* X_r_ .+ T0_r .* X_i_

    # Reshape to 4D factored form and permute for stage 1
    X_r3 = permutedims(reshape(X_r2, (F2, F1, F0, BS)), (3, 1, 2, 4))  # → (F0, F2, F1, BS)
    X_i3 = permutedims(reshape(X_i2, (F2, F1, F0, BS)), (3, 1, 2, 4))

    # --- Stage 1: F1-point DFT ---
    # Merge (F0, F2) → F0F2; X: (F0F2, F1, BS) × W1: (F1, F1, 1) → (F0F2, F1, BS)
    X_r4 = reshape(X_r3, (F0F2, F1, BS))
    X_i4 = reshape(X_i3, (F0F2, F1, BS))
    X_r5 = reshape(X_r4 * W1_r - X_i4 * W1_i, (F0, F1F2, BS))
    X_i5 = reshape(X_r4 * W1_i + X_i4 * W1_r, (F0, F1F2, BS))

    # --- Twiddle & Permute 1 ---
    X_r6 = T1_r .* X_r5 .- T1_i .* X_i5
    X_i6 = T1_i .* X_r5 .+ T1_r .* X_i5

    # Reshape to 4D and permute for stage 2
    X_r7 = permutedims(reshape(X_r6, (F0, F2, F1, BS)), (3, 1, 2, 4))  # → (F1, F0, F2, BS)
    X_i7 = permutedims(reshape(X_i6, (F0, F2, F1, BS)), (3, 1, 2, 4))

    # --- Stage 2: F2-point DFT ---
    # Merge (F1, F0) → F0F1; X: (F0F1, F2, BS) × W2: (F2, F2, 1) → (F0F1, F2, BS)
    X_r8 = reshape(X_r7, (F0F1, F2, BS))
    X_i8 = reshape(X_i7, (F0F1, F2, BS))
    X_r9 = X_r8 * W2_r - X_i8 * W2_i
    X_i9 = X_r8 * W2_i + X_i8 * W2_r

    # --- Final permute ---
    X_r10 = permutedims(reshape(X_r9, (F1, F0, F2, BS)), (2, 1, 3, 4))  # → (F0, F1, F2, BS)
    X_i10 = permutedims(reshape(X_i9, (F1, F0, F2, BS)), (2, 1, 3, 4))

    # --- Concatenate and Store ---
    X_r_final = reshape(X_r10, (1, N, BS))
    X_i_final = reshape(X_i10, (1, N, BS))
    Y_ri = reshape(ct.cat((X_r_final, X_i_final), 1), (D, N2D, BS))
    ct.store(y_packed_out; index=(Int32(1), Int32(1), bid), tile=Y_ri)

    return
end

# Helper: Generate DFT matrix W_n^{ij} = exp(-2πi * ij / n)
# Stored as (2, size, size) — reversed from Python's (size, size, 2).
# DFT matrices are symmetric in i,j so no transpose needed.
function dft_matrix(size::Int)
    W = zeros(ComplexF32, size, size)
    for i in 0:size-1, j in 0:size-1
        W[i+1, j+1] = exp(-2π * im * i * j / size)
    end
    result = zeros(Float32, 2, size, size)
    result[1, :, :] = Float32.(real.(W))
    result[2, :, :] = Float32.(imag.(W))
    return result
end

# Generate twiddle factors T0: (2, F1F2, F0) — reversed from Python's (F0, F1F2, 2)
function make_twiddles_T0(F0::Int, F1F2::Int, N::Int)
    T0 = zeros(Float32, 2, F1F2, F0)
    for i in 0:F0-1, j in 0:F1F2-1
        val = exp(-2π * im * i * j / N)
        T0[1, j+1, i+1] = Float32(real(val))
        T0[2, j+1, i+1] = Float32(imag(val))
    end
    return T0
end

# Generate twiddle factors T1: (2, F2, F1) — reversed from Python's (F1, F2, 2)
function make_twiddles_T1(F1::Int, F2::Int, F1F2::Int)
    T1 = zeros(Float32, 2, F2, F1)
    for j in 0:F1-1, k in 0:F2-1
        val = exp(-2π * im * j * k / F1F2)
        T1[1, k+1, j+1] = Float32(real(val))
        T1[2, k+1, j+1] = Float32(imag(val))
    end
    return T1
end

# Generate all W and T matrices
function make_twiddles(factors::NTuple{3, Int})
    F0, F1, F2 = factors
    N = F0 * F1 * F2
    F1F2 = F1 * F2

    W0 = dft_matrix(F0)
    W1 = dft_matrix(F1)
    W2 = dft_matrix(F2)
    T0 = make_twiddles_T0(F0, F1F2, N)
    T1 = make_twiddles_T1(F1, F2, F1F2)

    return (W0, W1, W2, T0, T1)
end

#=============================================================================
 Example harness
=============================================================================#

function prepare(; benchmark::Bool=false,
                  batch::Int=benchmark ? 64 : 2,
                  factors::NTuple{3,Int}=benchmark ? (8, 8, 8) : (2, 2, 2),
                  atom_packing_dim::Int=min(64, 2 * prod(factors)))
    n = prod(factors)
    @assert (n * 2) % atom_packing_dim == 0 "N*2 must be divisible by atom_packing_dim"

    CUDA.seed!(42)
    input = CUDA.randn(ComplexF32, n, batch)

    W0, W1, W2, T0, T1 = make_twiddles(factors)
    W0_gpu = CuArray(W0)
    W1_gpu = CuArray(W1)
    W2_gpu = CuArray(W2)
    T0_gpu = CuArray(T0)
    T1_gpu = CuArray(T1)

    D = atom_packing_dim
    N2D = n * 2 ÷ D
    # Pack complex input as (D, N2D, batch) Float32 — matches Python's (batch, N2D, D) row-major.
    # When D=2, reinterpret gives (2, n, batch) directly. For D>2, reshape the flat layout.
    x_ri = reinterpret(reshape, Float32, input)  # (2, n, batch)
    x_packed = D == 2 ? x_ri : reshape(x_ri, D, N2D, batch)
    y_packed = CuArray{Float32}(undef, D, N2D, batch)

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

    # Unpack output: (D, N2D, batch) → (2, n, batch) → ComplexF32(n, batch)
    y_ri = D == 2 ? y_packed : reshape(y_packed, 2, n, batch)
    y_complex = reinterpret(reshape, ComplexF32, y_ri)
    output = copy(y_complex)

    return (; output, times)
end

function verify(data, result)
    # FFT along dim 1 (the n dimension)
    reference = FFTW.fft(Array(data.input), 1)
    @assert isapprox(Array(result.output), reference, rtol=1e-4)
end

#=============================================================================
 Reference implementations for benchmarking
=============================================================================#

function run_others(data; nruns::Int=1, warmup::Int=0)
    (; input, batch, n) = data
    results = Dict{String, Vector{Float64}}()

    CUDA.@sync for _ in 1:warmup
        CUDA.CUFFT.fft!(copy(input), 1)
    end
    times_cufft = Float64[]
    for _ in 1:nruns
        input_copy = copy(input)
        t = CUDA.@elapsed CUDA.CUFFT.fft!(input_copy, 1)
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

    data = prepare()
    println("  Configuration:")
    println("    FFT Size (N): $(data.n)")
    println("    Batch Size: $(data.batch)")
    println("    FFT Factors: $(data.factors)")
    println("    Atom Packing Dim: $(data.D)")
    println("\nInput data shape: $(size(data.input)), dtype: $(eltype(data.input))")

    result = run(data)
    println("cuTile FFT Output shape: $(size(result.output)), dtype: $(eltype(result.output))")

    verify(data, result)
    println("\n✓ Correctness check PASSED")

    println("\n--- cuTile FFT example execution complete ---")
end

isinteractive() || main()
