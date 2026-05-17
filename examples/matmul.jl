# Matrix multiplication example - Julia port of cuTile Python's MatMul.py sample
#
# SPDX-License-Identifier: Apache-2.0

using CUDACore, NVTX
import cuRAND, cuBLAS
using LinearAlgebra
using cuTile: cuTile
import cuTile as ct

# 2D swizzle for better L2 cache locality. Takes a 1-indexed bid and
# returns 1-indexed (bid_m, bid_n). Modular arithmetic is done on the
# 0-indexed bid internally; the conversion is contained in this helper.
function swizzle_2d(M, N, tm, tn, GROUP_SIZE_M, bid)
    num_bid_m = cld(M, Int32(tm))
    num_bid_n = cld(N, Int32(tn))
    num_bid_in_group = Int32(GROUP_SIZE_M) * num_bid_n
    bid0 = bid - Int32(1)
    group_id = fld(bid0, num_bid_in_group)
    first_bid_m = group_id * Int32(GROUP_SIZE_M)
    group_size_m = min(num_bid_m - first_bid_m, Int32(GROUP_SIZE_M))
    bid_m = first_bid_m + rem(bid0, group_size_m) + Int32(1)
    bid_n = fld(rem(bid0, num_bid_in_group), group_size_m) + Int32(1)
    return bid_m, bid_n
end

# Matrix multiplication kernel with K reduction loop and 2D swizzle
# C = A @ B where A is (M, K), B is (K, N), C is (M, N)
function matmul_kernel(A::ct.TileArray{T,2}, B::ct.TileArray{T,2}, C::ct.TileArray{T,2},
                       tm::Int, tn::Int, tk::Int) where {T}
    ct.@compiler_options num_ctas=ct.ByTarget(v"10.0" => 2)
    # Use 1D grid with swizzle for better cache locality
    bid = ct.bid(1)
    M = size(A, 1)
    N = size(B, 2)
    bid_m, bid_n = swizzle_2d(M, N, tm, tn, 8, bid)

    # Number of K tiles to iterate over
    num_k = ct.num_tiles(A, 2, (tm, tk))

    # Initialize accumulator with Float32 for precision
    acc = zeros(Float32, tm, tn)

    # K reduction loop - accumulate partial products
    for k in Int32(1):num_k
        # Load and convert to TF32 for tensor cores (Float32 only)
        # padding_mode=Zero ensures out-of-bounds reads return zero (for non-aligned dimensions)
        a = ct.load(A; index=(bid_m, k), shape=(tm, tk), padding_mode=ct.PaddingMode.Zero)
        b = ct.load(B; index=(k, bid_n), shape=(tk, tn), padding_mode=ct.PaddingMode.Zero)
        if T === Float32
            a = convert(ct.Tile{ct.TFloat32}, a)
            b = convert(ct.Tile{ct.TFloat32}, b)
        end
        acc = muladd(a, b, acc)
    end

    # Convert accumulator to output type and store
    ct.store(C; index=(bid_m, bid_n), tile=convert(ct.Tile{T}, acc))

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
        A = cuRAND.rand(T, M, K),
        B = cuRAND.rand(T, K, N),
        C = CuArray{T}(undef, M, N),
        M, N, K
    )
end

function run(data; tm::Int=64, tn::Int=64, tk::Int=64, nruns::Int=1, warmup::Int=0)
    (; A, B, C, M, N, K) = data
    grid = cld(M, tm) * cld(N, tn)

    @cuda backend=cuTile blocks=grid matmul_kernel(A, B, C, ct.Constant(tm), ct.Constant(tn), ct.Constant(tk))

    times = Float64[]
    NVTX.@range "cuTile" begin
        for i in 1:nruns
            NVTX.@range "run $i" begin
                t = CUDACore.@elapsed @cuda backend=cuTile blocks=grid matmul_kernel(A, B, C, ct.Constant(tm), ct.Constant(tn), ct.Constant(tk))
                push!(times, t * 1000)  # ms
            end
        end
    end

    return (; C, times)
end

function verify(data, result)
    expected = Array(data.A) * Array(data.B)
    @assert isapprox(Array(result.C), expected; rtol=1e-2) "max diff: $(maximum(abs.(Array(result.C) - expected)))"
end

function metric(data)
    # 2*M*N*K FLOPs (multiply-add = 2 ops)
    return 2 * data.M * data.N * data.K, "TFLOPS"
end

#=============================================================================
 Reference implementations for benchmarking
=============================================================================#

function run_others(data; nruns::Int=1, warmup::Int=0)
    (; A, B) = data
    results = Dict{String, Vector{Float64}}()

    C_gpuarrays = similar(A, size(A, 1), size(B, 2))

    # GPUArrays (uses cuBLAS under the hood via LinearAlgebra.mul!)
    CUDACore.@sync for _ in 1:warmup
        mul!(C_gpuarrays, A, B)
    end
    times_gpuarrays = Float64[]
    NVTX.@range "cuBLAS" begin
        for i in 1:nruns
            NVTX.@range "run $i" begin
                t = CUDACore.@elapsed mul!(C_gpuarrays, A, B)
                push!(times_gpuarrays, t * 1000)
            end
        end
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

function main()
    println("--- cuTile Matrix Multiplication Examples ---\n")

    test_matmul(Float32, 256, 256, 256, 32, 32, 32)
    test_matmul(Float32, 512, 512, 512, 64, 64, 64)
    test_matmul(Float32, 256, 512, 128, 32, 32, 32)
    test_matmul(Float32, 1024, 1024, 1024, 64, 64, 64)

    println("\n--- All matmul examples completed ---")
end

if !isinteractive() && abspath(PROGRAM_FILE) == @__FILE__
    main()
end
