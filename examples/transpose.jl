# Matrix transpose example - Julia port of cuTile Python's Transpose.py sample
#
# SPDX-License-Identifier: Apache-2.0

using CUDA
import cuTile as ct

# Transpose kernel with TileArray and constant tile sizes
# TileArray carries size/stride metadata, Constant parameters are ghost types
function transpose_kernel(x::ct.TileArray{T,2}, y::ct.TileArray{T,2},
                          tm::ct.Constant{Int}, tn::ct.Constant{Int}) where {T}
    bidx = ct.bid(1)
    bidy = ct.bid(2)
    input_tile = ct.load(x, (bidx, bidy), (tm[], tn[]))
    transposed_tile = ct.transpose(input_tile)
    ct.store(y, (bidy, bidx), transposed_tile)
    return
end

#=============================================================================
 Example harness
=============================================================================#

function prepare(; benchmark::Bool=false,
                  m::Int=benchmark ? 8192 : 1024,
                  n::Int=benchmark ? 8192 : 512,
                  T::DataType=Float32)
    x = CUDA.rand(T, m, n)
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
        ct.launch(transpose_kernel, grid, x, y,
                  ct.Constant(tm), ct.Constant(tn))
    end

    times = Float64[]
    for _ in 1:nruns
        t = CUDA.@elapsed ct.launch(transpose_kernel, grid, x, y,
                                    ct.Constant(tm), ct.Constant(tn))
        push!(times, t * 1000)  # ms
    end

    return (; y, times)
end

function verify(data, result)
    @assert Array(result.y) ≈ transpose(Array(data.x))
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
    for _ in 1:nruns
        t = CUDA.@elapsed permutedims!(y_gpuarrays, x, (2, 1))
        push!(times_gpuarrays, t * 1000)
    end
    results["GPUArrays"] = times_gpuarrays

    # SIMT naive kernel
    threads = (16, 16)
    blocks = (cld(m, threads[1]), cld(n, threads[2]))
    CUDA.@sync for _ in 1:warmup
        @cuda threads=threads blocks=blocks simt_naive_kernel(x, y_simt, m, n)
    end
    times_simt = Float64[]
    for _ in 1:nruns
        t = CUDA.@elapsed @cuda threads=threads blocks=blocks simt_naive_kernel(x, y_simt, m, n)
        push!(times_simt, t * 1000)
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

    # Float32 tests (like Python's test case 2)
    test_transpose(Float32, 1024, 512, 32, 32)
    test_transpose(Float32, 1024, 512, 64, 64)

    # Float64 tests
    test_transpose(Float64, 1024, 512, 32, 32)
    test_transpose(Float64, 512, 1024, 64, 64)

    # Float16 tests (like Python's test case 1 with 128x128 tiles)
    test_transpose(Float16, 1024, 512, 128, 128)
    test_transpose(Float16, 1024, 1024, 64, 64)

    println("\n--- All transpose examples completed ---")
end

isinteractive() || main()
