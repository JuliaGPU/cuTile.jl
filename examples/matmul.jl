# Matrix multiplication example - Julia port of cuTile Python's MatMul.py sample
#
# SPDX-License-Identifier: Apache-2.0

using CUDA
using LinearAlgebra
import cuTile as ct

# 2D swizzle for better L2 cache locality
# Groups blocks to access nearby memory regions together
@inline function swizzle_2d(M, N, tm, tn, GROUP_SIZE_M, bid)
    num_bid_m = cld(M, Int32(tm))
    num_bid_n = cld(N, Int32(tn))
    num_bid_in_group = Int32(GROUP_SIZE_M) * num_bid_n
    group_id = fld(bid, num_bid_in_group)
    first_bid_m = group_id * Int32(GROUP_SIZE_M)
    group_size_m = min(num_bid_m - first_bid_m, Int32(GROUP_SIZE_M))
    bid_m = first_bid_m + rem(bid, group_size_m)
    bid_n = fld(rem(bid, num_bid_in_group), group_size_m)
    return bid_m, bid_n
end

# Matrix multiplication kernel with K reduction loop and 2D swizzle
# C = A @ B where A is (M, K), B is (K, N), C is (M, N)
function matmul_kernel(A::ct.TileArray{T,2}, B::ct.TileArray{T,2}, C::ct.TileArray{T,2},
                       tm::ct.Constant{Int}, tn::ct.Constant{Int}, tk::ct.Constant{Int}) where {T}
    # Use 1D grid with swizzle for better cache locality
    bid = ct.bid(1)
    M = size(A, 1)
    N = size(B, 2)
    # swizzle_2d expects 0-indexed bid, returns 0-indexed tile coords
    bid_m_0, bid_n_0 = swizzle_2d(M, N, tm[], tn[], 8, bid - Int32(1))
    # Convert to 1-indexed tile coordinates
    bid_m = bid_m_0 + Int32(1)
    bid_n = bid_n_0 + Int32(1)

    # Number of K tiles to iterate over
    num_k = ct.num_tiles(A, 2, (tm[], tk[]))

    # Initialize accumulator with Float32 for precision
    acc = ct.full((tm[], tn[]), zero(Float32), Float32)

    # K reduction loop - accumulate partial products
    # NOTE: Uses while-loop pattern. Native `for k in 0:n` syntax generates complex
    # iterator protocol IR that doesn't map cleanly to ForOp. Use while-loops for now.
    k = Int32(1)
    while k <= num_k
        # Load and convert to TF32 for tensor cores (Float32 only)
        # padding_mode=Zero ensures out-of-bounds reads return zero (for non-aligned dimensions)
        a = ct.load(A, (bid_m, k), (tm[], tk[]); padding_mode=ct.PaddingMode.Zero)
        b = ct.load(B, (k, bid_n), (tk[], tn[]); padding_mode=ct.PaddingMode.Zero)
        if T === Float32
            a = convert(ct.Tile{ct.TFloat32}, a)
            b = convert(ct.Tile{ct.TFloat32}, b)
        end
        acc = muladd(a, b, acc)
        k += Int32(1)
    end

    # Convert accumulator to output type and store
    ct.store(C, (bid_m, bid_n), convert(ct.Tile{T}, acc))

    return nothing
end

#=============================================================================
 Example harness
=============================================================================#

function prepare(; benchmark::Bool=false,
                  M::Int=benchmark ? 4096 : 256,
                  N::Int=benchmark ? 4096 : 256,
                  K::Int=benchmark ? 4096 : 256,
                  T::DataType=Float32)
    return (;
        A = CUDA.rand(T, M, K),
        B = CUDA.rand(T, K, N),
        C = CuArray{T}(undef, M, N),
        M, N, K
    )
end

function run(data; tm::Int=64, tn::Int=64, tk::Int=64, nruns::Int=1, warmup::Int=0)
    (; A, B, C, M, N, K) = data
    grid = cld(M, tm) * cld(N, tn)

    CUDA.@sync for _ in 1:warmup
        ct.launch(matmul_kernel, grid, A, B, C,
                  ct.Constant(tm), ct.Constant(tn), ct.Constant(tk))
    end

    times = Float64[]
    for _ in 1:nruns
        t = CUDA.@elapsed ct.launch(matmul_kernel, grid, A, B, C,
                                    ct.Constant(tm), ct.Constant(tn), ct.Constant(tk))
        push!(times, t * 1000)  # ms
    end

    return (; C, times)
end

function verify(data, result)
    expected = Array(data.A) * Array(data.B)
    @assert isapprox(Array(result.C), expected; rtol=1e-2) "max diff: $(maximum(abs.(Array(result.C) - expected)))"
end

#=============================================================================
 Reference implementations for benchmarking
=============================================================================#

function run_others(data; nruns::Int=1, warmup::Int=0)
    (; A, B) = data
    results = Dict{String, Vector{Float64}}()

    C_gpuarrays = similar(A, size(A, 1), size(B, 2))

    # GPUArrays (uses cuBLAS under the hood via LinearAlgebra.mul!)
    CUDA.@sync for _ in 1:warmup
        mul!(C_gpuarrays, A, B)
    end
    times_gpuarrays = Float64[]
    for _ in 1:nruns
        t = CUDA.@elapsed mul!(C_gpuarrays, A, B)
        push!(times_gpuarrays, t * 1000)
    end
    results["cuBLAS"] = times_gpuarrays

    return results
end

#=============================================================================
 Main
=============================================================================#

function test_matmul(::Type{T}, M, N, K, tm, tn, tk; name=nothing) where T
    name = something(name, "matmul ($M x $K) @ ($K x $N), $T, tiles=$tm x $tn x $tk")
    println("--- $name ---")
    data = prepare(; M, N, K, T)
    result = run(data; tm, tn, tk)
    verify(data, result)
    println("  passed")
end

function main(T=Float32)
    println("--- cuTile Matrix Multiplication Examples ---\n")

    # Small matrices
    test_matmul(T, 256, 256, 256, 32, 32, 32)
    test_matmul(T, 512, 512, 512, 64, 64, 64)

    # Non-square matrices
    test_matmul(T, 256, 512, 128, 32, 32, 32)
    test_matmul(T, 512, 256, 384, 64, 64, 64)

    # Larger matrices
    test_matmul(T, 1024, 1024, 1024, 32, 32, 32)

    println("\n--- All matmul examples completed ---")
end

isinteractive() || main()
