# Batch matrix multiplication example - Julia port of cuTile Python's BatchMatMul.py sample
#
# SPDX-License-Identifier: Apache-2.0
#
# Uses Julia-idiomatic batch-last ordering: A(M, K, Batch), B(K, N, Batch), C(M, N, Batch)
# This provides optimal memory access with Julia's column-major layout.

using CUDA
import cuTile as ct

# Batch matrix multiplication kernel
# A: (M, K, Batch), B: (K, N, Batch), C: (M, N, Batch)
# Grid: (M_tiles, N_tiles, Batch)
function batch_matmul_kernel(A::ct.TileArray{T,3}, B::ct.TileArray{T,3}, C::ct.TileArray{T,3},
                             tm::ct.Constant{Int}, tn::ct.Constant{Int},
                             tk::ct.Constant{Int}) where {T}
    # Grid dimensions (1-indexed)
    bid_m = ct.bid(1)      # M tile index
    bid_n = ct.bid(2)      # N tile index
    pid_batch = ct.bid(3)  # Batch index

    # Number of K tiles to iterate over
    K = A.sizes[2]
    num_k = cld(K, Int32(tk[]))

    # Initialize accumulator with Float32 for precision
    acc = ct.full((tm[], tn[]), zero(Float32), Float32)

    # K reduction loop
    k = Int32(1)
    while k <= num_k
        # Load 3D tiles: (tm, tk, 1) and (tk, tn, 1)
        a = ct.load(A, (bid_m, k, pid_batch), (tm[], tk[], 1);
                    padding_mode=ct.PaddingMode.Zero)
        b = ct.load(B, (k, bid_n, pid_batch), (tk[], tn[], 1);
                    padding_mode=ct.PaddingMode.Zero)

        # Reshape 3D tiles to 2D for mma
        a_2d = ct.reshape(a, (tm[], tk[]))
        b_2d = ct.reshape(b, (tk[], tn[]))

        # Convert to TF32 for tensor cores (Float32 inputs only)
        if T === Float32
            a_2d = convert(ct.Tile{ct.TFloat32}, a_2d)
            b_2d = convert(ct.Tile{ct.TFloat32}, b_2d)
        end

        acc = muladd(a_2d, b_2d, acc)
        k += Int32(1)
    end

    # Convert to output type, reshape to 3D, and store
    result = convert(ct.Tile{T}, acc)
    result_3d = ct.reshape(result, (tm[], tn[], 1))
    ct.store(C, (bid_m, bid_n, pid_batch), result_3d)

    return nothing
end

#=============================================================================
 Example harness
=============================================================================#

function prepare(; benchmark::Bool=false,
                  M::Int=benchmark ? 1024 : 256,
                  K::Int=benchmark ? 512 : 128,
                  N::Int=benchmark ? 2048 : 256,
                  Batch::Int=benchmark ? 8 : 4,
                  T::DataType=Float32)
    return (;
        A = CUDA.rand(T, M, K, Batch),
        B = CUDA.rand(T, K, N, Batch),
        C = CuArray{T}(undef, M, N, Batch),
        M, K, N, Batch
    )
end

function run(data; tm::Int=64, tn::Int=64, tk::Int=64, nruns::Int=1, warmup::Int=0)
    (; A, B, C, M, N, Batch) = data
    grid = (cld(M, tm), cld(N, tn), Batch)

    CUDA.@sync for _ in 1:warmup
        ct.launch(batch_matmul_kernel, grid, A, B, C,
                  ct.Constant(tm), ct.Constant(tn), ct.Constant(tk))
    end

    times = Float64[]
    for _ in 1:nruns
        t = CUDA.@elapsed ct.launch(batch_matmul_kernel, grid, A, B, C,
                                    ct.Constant(tm), ct.Constant(tn), ct.Constant(tk))
        push!(times, t * 1000)  # ms
    end

    return (; C, times)
end

function verify(data, result)
    (; A, B, M, N, Batch) = data
    A_cpu = Array(A)
    B_cpu = Array(B)
    expected = similar(A_cpu, M, N, Batch)
    for b in 1:Batch
        expected[:, :, b] = A_cpu[:, :, b] * B_cpu[:, :, b]
    end
    @assert isapprox(Array(result.C), expected; rtol=1e-2) "max diff: $(maximum(abs.(Array(result.C) - expected)))"
end

#=============================================================================
 Reference implementations for benchmarking
=============================================================================#

function run_others(data; nruns::Int=1, warmup::Int=0)
    (; A, B, M, N, Batch) = data
    results = Dict{String, Vector{Float64}}()

    C_cublas = similar(A, M, N, Batch)

    # cuBLAS batched gemm via CUBLAS.gemm_strided_batched!
    CUDA.@sync for _ in 1:warmup
        CUDA.CUBLAS.gemm_strided_batched!('N', 'N', one(eltype(A)), A, B, zero(eltype(A)), C_cublas)
    end
    times_cublas = Float64[]
    for _ in 1:nruns
        t = CUDA.@elapsed CUDA.CUBLAS.gemm_strided_batched!('N', 'N', one(eltype(A)), A, B, zero(eltype(A)), C_cublas)
        push!(times_cublas, t * 1000)
    end
    results["cuBLAS batched"] = times_cublas

    return results
end

#=============================================================================
 Main
=============================================================================#

function test_batch_matmul(::Type{T}, M, K, N, Batch, tm, tn, tk; name=nothing) where T
    name = something(name, "batch_matmul ($M x $K x $Batch) @ ($K x $N x $Batch), $T, tiles=$tm x $tn x $tk")
    println("--- $name ---")
    data = prepare(; M, K, N, Batch, T)
    result = run(data; tm, tn, tk)
    verify(data, result)
    println("  passed")
end

function main()
    println("--- cuTile Batch Matrix Multiplication Examples ---\n")

    # Float32 tests with smaller tile sizes
    test_batch_matmul(Float32, 256, 128, 256, 4, 32, 32, 32)
    test_batch_matmul(Float32, 512, 256, 512, 4, 64, 64, 64)

    # Float16 tests - can use larger tiles for tensor cores
    test_batch_matmul(Float16, 512, 256, 1024, 4, 128, 256, 64)

    # Non-square matrices
    test_batch_matmul(Float32, 256, 512, 128, 2, 32, 32, 32)

    println("\n--- All batch matmul examples completed ---")
end

isinteractive() || main()
