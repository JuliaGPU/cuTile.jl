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

# Run transpose - for benchmarking or programmatic use
function run_transpose(; m::Int, n::Int, tm::Int, tn::Int, T::DataType=Float32,
                        x::Union{CuArray,Nothing}=nothing,
                        y::Union{CuArray,Nothing}=nothing,
                        validate::Bool=false)
    x = something(x, CUDA.rand(T, m, n))
    y = something(y, CUDA.zeros(T, n, m))
    grid_x = cld(m, tm)
    grid_y = cld(n, tn)
    ct.launch(transpose_kernel, (grid_x, grid_y), x, y,
              ct.Constant(tm), ct.Constant(tn))
    if validate
        @assert Array(y) ≈ transpose(Array(x))
    end
    return (; x, y)
end

function test_transpose(::Type{T}, m, n, tm, tn; name=nothing) where T
    name = something(name, "transpose ($m x $n, $T, tiles=$tm x $tn)")
    println("--- $name ---")
    run_transpose(; m, n, tm, tn, T, validate=true)
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
