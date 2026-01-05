# FFT Example - Julia port of cuTile Python's FFT.py sample
#
# This implements a 3-stage Cooley-Tukey FFT decomposition. The FFT of size N is decomposed
# as N = F0 * F1 * F2, allowing efficient tensor factorization.
#
# SPDX-License-Identifier: Apache-2.0

using CUDA
import cuTile as ct
using Test
using FFTW

# FFT kernel - 3-stage Cooley-Tukey decomposition
# Matches the Python cuTile FFT kernel in res/cutile-python/samples/FFT.py
#
# Uses ct.Constant{Int} parameters with [] access to get compile-time constants.
# The Constant type is Constant{T,V} where T is the type and V is the value.
# Access via const_param[] extracts V as a compile-time constant.
function fft_kernel(
    x_packed_in::ct.TileArray{Float32, 3},   # Input (BS, N2D, D) packed
    y_packed_out::ct.TileArray{Float32, 3},  # Output (BS, N2D, D) packed
    W0::ct.TileArray{Float32, 3},            # W0 (F0, F0, 2) real/imag interleaved
    W1::ct.TileArray{Float32, 3},            # W1 (F1, F1, 2)
    W2::ct.TileArray{Float32, 3},            # W2 (F2, F2, 2)
    T0::ct.TileArray{Float32, 3},            # T0 (F0, F1*F2, 2)
    T1::ct.TileArray{Float32, 3},            # T1 (F1, F2, 2)
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
    # Load packed input, reshape to (BS, N, 2) to separate real/imag
    X_ri = ct.reshape(ct.load(x_packed_in, (bid, 1, 1), (BS, N2D, D)), (BS, N, 2))

    # Split real and imaginary parts using slice indices (1-indexed)
    # Shape (BS, N, 2) → (BS, N, 1), then reshape to (BS, F0, F1, F2)
    X_r = ct.reshape(ct.extract(X_ri, (1, 1, 1), (BS, N, 1)), (BS, F0, F1, F2))
    X_i = ct.reshape(ct.extract(X_ri, (1, 1, 2), (BS, N, 1)), (BS, F0, F1, F2))

    # --- Load Rotation (W) and Twiddle (T) Matrices ---
    # W0 (F0 x F0) - broadcast to (BS, F0, F0) for batched matmul
    W0_ri = ct.reshape(ct.load(W0, (1, 1, 1), (F0, F0, 2)), (F0, F0, 2))
    W0_r = ct.broadcast_to(ct.reshape(ct.extract(W0_ri, (1, 1, 1), (F0, F0, 1)), (1, F0, F0)), (BS, F0, F0))
    W0_i = ct.broadcast_to(ct.reshape(ct.extract(W0_ri, (1, 1, 2), (F0, F0, 1)), (1, F0, F0)), (BS, F0, F0))

    # W1 (F1 x F1) - broadcast to (BS, F1, F1)
    W1_ri = ct.reshape(ct.load(W1, (1, 1, 1), (F1, F1, 2)), (F1, F1, 2))
    W1_r = ct.broadcast_to(ct.reshape(ct.extract(W1_ri, (1, 1, 1), (F1, F1, 1)), (1, F1, F1)), (BS, F1, F1))
    W1_i = ct.broadcast_to(ct.reshape(ct.extract(W1_ri, (1, 1, 2), (F1, F1, 1)), (1, F1, F1)), (BS, F1, F1))

    # W2 (F2 x F2) - broadcast to (BS, F2, F2)
    W2_ri = ct.reshape(ct.load(W2, (1, 1, 1), (F2, F2, 2)), (F2, F2, 2))
    W2_r = ct.broadcast_to(ct.reshape(ct.extract(W2_ri, (1, 1, 1), (F2, F2, 1)), (1, F2, F2)), (BS, F2, F2))
    W2_i = ct.broadcast_to(ct.reshape(ct.extract(W2_ri, (1, 1, 2), (F2, F2, 1)), (1, F2, F2)), (BS, F2, F2))

    # T0 (F0 x F1F2)
    T0_ri = ct.reshape(ct.load(T0, (1, 1, 1), (F0, F1F2, 2)), (F0, F1F2, 2))
    T0_r = ct.reshape(ct.extract(T0_ri, (1, 1, 1), (F0, F1F2, 1)), (N, 1))
    T0_i = ct.reshape(ct.extract(T0_ri, (1, 1, 2), (F0, F1F2, 1)), (N, 1))

    # T1 (F1 x F2)
    T1_ri = ct.reshape(ct.load(T1, (1, 1, 1), (F1, F2, 2)), (F1, F2, 2))
    T1_r = ct.reshape(ct.extract(T1_ri, (1, 1, 1), (F1, F2, 1)), (F1F2, 1))
    T1_i = ct.reshape(ct.extract(T1_ri, (1, 1, 2), (F1, F2, 1)), (F1F2, 1))

    # --- CT0: Contract over F0 dimension ---
    X_r = ct.reshape(X_r, (BS, F0, F1F2))
    X_i = ct.reshape(X_i, (BS, F0, F1F2))
    # Complex matmul: (A+iB)(C+iD) = (AC-BD) + i(AD+BC)
    X_r_ = ct.reshape(ct.matmul(W0_r, X_r) - ct.matmul(W0_i, X_i), (BS, N, 1))
    X_i_ = ct.reshape(ct.matmul(W0_i, X_r) + ct.matmul(W0_r, X_i), (BS, N, 1))

    # --- Twiddle & Permute 0 ---
    X_r2 = T0_r .* X_r_ .- T0_i .* X_i_
    X_i2 = T0_i .* X_r_ .+ T0_r .* X_i_
    X_r3 = ct.permute(ct.reshape(X_r2, (BS, F0, F1, F2)), (1, 3, 4, 2))
    X_i3 = ct.permute(ct.reshape(X_i2, (BS, F0, F1, F2)), (1, 3, 4, 2))

    # --- CT1: Contract over F1 dimension ---
    X_r4 = ct.reshape(X_r3, (BS, F1, F0F2))
    X_i4 = ct.reshape(X_i3, (BS, F1, F0F2))
    X_r5 = ct.reshape(ct.matmul(W1_r, X_r4) - ct.matmul(W1_i, X_i4), (BS, F1F2, F0))
    X_i5 = ct.reshape(ct.matmul(W1_i, X_r4) + ct.matmul(W1_r, X_i4), (BS, F1F2, F0))

    # --- Twiddle & Permute 1 ---
    X_r6 = T1_r .* X_r5 .- T1_i .* X_i5
    X_i6 = T1_i .* X_r5 .+ T1_r .* X_i5
    X_r7 = ct.permute(ct.reshape(X_r6, (BS, F1, F2, F0)), (1, 3, 4, 2))
    X_i7 = ct.permute(ct.reshape(X_i6, (BS, F1, F2, F0)), (1, 3, 4, 2))

    # --- CT2: Contract over F2 dimension ---
    X_r8 = ct.reshape(X_r7, (BS, F2, F0F1))
    X_i8 = ct.reshape(X_i7, (BS, F2, F0F1))
    X_r9 = ct.matmul(W2_r, X_r8) - ct.matmul(W2_i, X_i8)
    X_i9 = ct.matmul(W2_i, X_r8) + ct.matmul(W2_r, X_i8)

    # --- Final Permutation ---
    X_r_out = ct.permute(ct.reshape(X_r9, (BS, F2, F0, F1)), (1, 2, 4, 3))
    X_i_out = ct.permute(ct.reshape(X_i9, (BS, F2, F0, F1)), (1, 2, 4, 3))
    X_r_final = ct.reshape(X_r_out, (BS, N, 1))
    X_i_final = ct.reshape(X_i_out, (BS, N, 1))

    # --- Concatenate and Store ---
    Y_ri = ct.reshape(ct.cat((X_r_final, X_i_final), -1), (BS, N2D, D))
    ct.store(y_packed_out, (bid, 1, 1), Y_ri)

    return
end

# Helper: Generate DFT twiddle factors W_n^{ij} = exp(-2πi * ij / n)
function twiddles(rows::Int, cols::Int, factor::Int)
    W = zeros(ComplexF32, rows, cols)
    for i in 0:rows-1, j in 0:cols-1
        W[i+1, j+1] = exp(-2π * im * i * j / factor)
    end
    # Return as (rows, cols, 2) with real/imag interleaved
    result = zeros(Float32, rows, cols, 2)
    result[:, :, 1] = Float32.(real.(W))
    result[:, :, 2] = Float32.(imag.(W))
    return result
end

# Generate all W and T matrices
function make_twiddles(factors::NTuple{3, Int})
    F0, F1, F2 = factors
    N = F0 * F1 * F2
    F1F2 = F1 * F2

    W0 = twiddles(F0, F0, F0)
    W1 = twiddles(F1, F1, F1)
    W2 = twiddles(F2, F2, F2)
    T0 = twiddles(F0, F1F2, N)
    T1 = twiddles(F1, F2, F1F2)

    return (W0, W1, W2, T0, T1)
end

# Main FFT function
function cutile_fft(x::CuMatrix{ComplexF32}, factors::NTuple{3, Int}; atom_packing_dim::Int=2)
    BS = size(x, 1)
    N = size(x, 2)
    F0, F1, F2 = factors

    @assert F0 * F1 * F2 == N "Factors must multiply to N"
    @assert (N * 2) % atom_packing_dim == 0 "N*2 must be divisible by atom_packing_dim"

    D = atom_packing_dim

    # Generate W and T matrices
    W0, W1, W2, T0, T1 = make_twiddles(factors)

    # Upload to GPU
    W0_gpu = CuArray(W0)
    W1_gpu = CuArray(W1)
    W2_gpu = CuArray(W2)
    T0_gpu = CuArray(T0)
    T1_gpu = CuArray(T1)

    # Pack input: complex (BS, N) → real (BS, N, 2) → packed (BS, N*2/D, D)
    x_cpu = Array(x)
    x_ri = zeros(Float32, BS, N, 2)
    x_ri[:, :, 1] = Float32.(real.(x_cpu))
    x_ri[:, :, 2] = Float32.(imag.(x_cpu))
    x_packed = CuArray(reshape(x_ri, BS, N * 2 ÷ D, D))

    # Allocate output
    y_packed = CUDA.zeros(Float32, BS, N * 2 ÷ D, D)

    # Launch kernel
    F0F1 = F0 * F1
    F1F2 = F1 * F2
    F0F2 = F0 * F2
    N2D = N * 2 ÷ D
    grid = (BS, 1, 1)
    ct.launch(fft_kernel, grid,
              x_packed, y_packed,
              W0_gpu, W1_gpu, W2_gpu, T0_gpu, T1_gpu,
              ct.Constant(N), ct.Constant(F0), ct.Constant(F1), ct.Constant(F2),
              ct.Constant(F0F1), ct.Constant(F1F2), ct.Constant(F0F2),
              ct.Constant(BS), ct.Constant(D), ct.Constant(N2D))

    # Unpack output
    y_ri = reshape(Array(y_packed), BS, N, 2)
    y_complex = ComplexF32.(y_ri[:, :, 1]) .+ im .* ComplexF32.(y_ri[:, :, 2])

    return CuArray(y_complex)
end

# Validation and example
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

    # Create sample input
    CUDA.seed!(42)
    input_complex = CUDA.randn(ComplexF32, BATCH_SIZE, FFT_SIZE)

    println("\nInput data shape: $(size(input_complex)), dtype: $(eltype(input_complex))")

    # Perform FFT using cuTile kernel
    output_cutile = cutile_fft(input_complex, FFT_FACTORS; atom_packing_dim=ATOM_PACKING_DIM)

    println("cuTile FFT Output shape: $(size(output_cutile)), dtype: $(eltype(output_cutile))")

    # Verify against reference (FFTW)
    input_cpu = Array(input_complex)
    reference_output = FFTW.fft(input_cpu, 2)

    output_cpu = Array(output_cutile)

    if isapprox(output_cpu, reference_output, rtol=1e-4)
        println("\n✓ Correctness check PASSED")
    else
        max_diff = maximum(abs.(output_cpu .- reference_output))
        println("\n✗ Correctness check FAILED - max difference: $max_diff")
        println("\nExpected (first 4):")
        println(reference_output[1, 1:4])
        println("\nGot (first 4):")
        println(output_cpu[1, 1:4])
    end

    println("\n--- cuTile FFT example execution complete ---")
end

# Run validation
main()
