# Matrix transpose example - Julia port of cuTile Python's Transpose.py sample
#
# SPDX-License-Identifier: Apache-2.0

using CUDA, NVTX
using cuTile: cuTile
import cuTile as ct

# Transpose kernel with TileArray and constant tile sizes
# TileArray carries size/stride metadata, Constant parameters are ghost types
function transpose_kernel(x::ct.TileArray{T,2}, y::ct.TileArray{T,2},
                          tm::Int, tn::Int) where {T}
    bidx = ct.bid(1)
    bidy = ct.bid(2)
    input_tile = ct.load(x; index=(bidx, bidy), shape=(tm, tn))
    transposed_tile = transpose(input_tile)
    ct.store(y; index=(bidy, bidx), tile=transposed_tile)
    return
end

#=============================================================================
 Example harness
=============================================================================#

function prepare(; benchmark::Bool=false,
                  m::Int=benchmark ? 8192 : 1024,
                  n::Int=benchmark ? 8192 : 512,
                  T::DataType=Float32)
    x = cuRAND.rand(T, m, n)
    return (;
        x,
        y = similar(x, n, m),
        m, n
    )
end

function run(data; tm::Int=64, tn::Int=64, nruns::Int=1, warmup::Int=0)
    (; x, y, m, n) = data
    grid = (cld(m, tm), cld(n, tn))

    CUDA.@sync for _ in 1:warmup
        @cuda backend=cuTile blocks=grid transpose_kernel(x, y, ct.Constant(tm), ct.Constant(tn))
    end

    times = Float64[]
    NVTX.@range "cuTile" begin
        for i in 1:nruns
            NVTX.@range "run $i" begin
                t = CUDA.@elapsed @cuda backend=cuTile blocks=grid transpose_kernel(x, y, ct.Constant(tm), ct.Constant(tn))
                push!(times, t * 1000)  # ms
            end
        end
    end

    return (; y, times)
end

function verify(data, result)
    @assert Array(result.y) ≈ transpose(Array(data.x))
end

function metric(data)
    T = eltype(data.x)
    # 1 read + 1 write
    return 2 * data.m * data.n * sizeof(T), "GB/s"
end

#=============================================================================
 Reference implementations for benchmarking
=============================================================================#

# Simple SIMT transpose kernel (naive, no shared memory)
function simt_naive_kernel(x, y, m, n)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if i <= m && j <= n
        @inbounds y[j, i] = x[i, j]
    end
    return
end

function run_others(data; nruns::Int=1, warmup::Int=0)
    (; x, m, n) = data
    results = Dict{String, Vector{Float64}}()

    y_gpuarrays = similar(x, n, m)
    y_simt = similar(x, n, m)

    # GPUArrays (permutedims)
    CUDA.@sync for _ in 1:warmup
        permutedims!(y_gpuarrays, x, (2, 1))
    end
    times_gpuarrays = Float64[]
    NVTX.@range "GPUArrays" begin
        for i in 1:nruns
            NVTX.@range "run $i" begin
                t = CUDA.@elapsed permutedims!(y_gpuarrays, x, (2, 1))
                push!(times_gpuarrays, t * 1000)
            end
        end
    end
    results["GPUArrays"] = times_gpuarrays

    # SIMT naive kernel
    threads = (16, 16)
    blocks = (cld(m, threads[1]), cld(n, threads[2]))
    CUDA.@sync for _ in 1:warmup
        @cuda threads=threads blocks=blocks simt_naive_kernel(x, y_simt, m, n)
    end
    times_simt = Float64[]
    NVTX.@range "SIMT naive" begin
        for i in 1:nruns
            NVTX.@range "run $i" begin
                t = CUDA.@elapsed @cuda threads=threads blocks=blocks simt_naive_kernel(x, y_simt, m, n)
                push!(times_simt, t * 1000)
            end
        end
    end
    results["SIMT naive"] = times_simt

    return results
end

#=============================================================================
 Main
=============================================================================#

function test_transpose(::Type{T}, m, n, tm, tn; name=nothing) where T
    name = something(name, "transpose ($m x $n, $T, tiles=$tm x $tn)")
    println("--- $name ---")
    data = prepare(; m, n, T)
    result = run(data; tm, tn)
    verify(data, result)
    println("✓ passed")
end

function main()
    println("--- cuTile Matrix Transposition Examples ---\n")

    test_transpose(Float32, 256, 256, 32, 32)
    test_transpose(Float32, 512, 512, 64, 64)
    test_transpose(Float32, 256, 512, 32, 64)
    test_transpose(Float32, 1024, 1024, 64, 64)

    println("\n--- All transpose examples completed ---")
end

if !isinteractive() && abspath(PROGRAM_FILE) == @__FILE__
    main()
end
