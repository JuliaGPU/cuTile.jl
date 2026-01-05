# Vector/Matrix addition example - Julia port of cuTile Python's VectorAddition.py sample
#
# SPDX-License-Identifier: Apache-2.0

using CUDA
import cuTile as ct

# 1D kernel with TileArray and constant tile size
# TileArray carries size/stride metadata, Constant is a ghost type
function vec_add_kernel_1d(a::ct.TileArray{T,1}, b::ct.TileArray{T,1}, c::ct.TileArray{T,1},
                           tile::ct.Constant{Int}) where {T}
    bid = ct.bid(1)
    a_tile = ct.load(a, bid, (tile[],))
    b_tile = ct.load(b, bid, (tile[],))
    ct.store(c, bid, a_tile + b_tile)
    return
end

# 2D kernel with TileArray and constant tile sizes
function vec_add_kernel_2d(a::ct.TileArray{T,2}, b::ct.TileArray{T,2}, c::ct.TileArray{T,2},
                           tile_x::ct.Constant{Int}, tile_y::ct.Constant{Int}) where {T}
    bid_x = ct.bid(1)
    bid_y = ct.bid(2)
    a_tile = ct.load(a, (bid_x, bid_y), (tile_x[], tile_y[]))
    b_tile = ct.load(b, (bid_x, bid_y), (tile_x[], tile_y[]))
    ct.store(c, (bid_x, bid_y), a_tile + b_tile)
    return
end

function test_add_1d(::Type{T}, n, tile; name=nothing) where T
    name = something(name, "1D vec_add ($n elements, $T, tile=$tile)")
    println("--- $name ---")
    a, b = CUDA.rand(T, n), CUDA.rand(T, n)
    c = CUDA.zeros(T, n)

    # Launch with ct.launch - CuArrays are auto-converted to TileArray
    # Constant parameters are ghost types - filtered out at launch time
    ct.launch(vec_add_kernel_1d, cld(n, tile), a, b, c, ct.Constant(tile))

    @assert Array(c) ≈ Array(a) + Array(b)
    println("✓ passed")
end

function test_add_2d(::Type{T}, m, n, tile_x, tile_y; name=nothing) where T
    name = something(name, "2D vec_add ($m x $n, $T, tiles=$tile_x x $tile_y)")
    println("--- $name ---")
    a, b = CUDA.rand(T, m, n), CUDA.rand(T, m, n)
    c = CUDA.zeros(T, m, n)

    # Launch with ct.launch - CuArrays are auto-converted to TileArray
    ct.launch(vec_add_kernel_2d, (cld(m, tile_x), cld(n, tile_y)), a, b, c,
              ct.Constant(tile_x), ct.Constant(tile_y))

    @assert Array(c) ≈ Array(a) + Array(b)
    println("✓ passed")
end

# 1D kernel using gather/scatter (explicit index-based memory access)
# This demonstrates the gather/scatter API for cases where you need
# explicit control over indices (e.g., for non-contiguous access patterns)
function vec_add_kernel_1d_gather(a::ct.TileArray{T,1}, b::ct.TileArray{T,1}, c::ct.TileArray{T,1},
                                   tile::ct.Constant{Int}) where {T}
    bid = ct.bid(1)
    # Create index tile for this block's elements
    offsets = ct.arange((tile[],), Int32)
    base = ct.Tile((bid - Int32(1)) * Int32(tile[]))
    indices = ct.broadcast_to(base, (tile[],)) .+ offsets

    # Gather, add, scatter
    a_tile = ct.gather(a, indices)
    b_tile = ct.gather(b, indices)
    sum_tile = a_tile + b_tile
    ct.scatter(c, indices, sum_tile)
    return
end

function test_add_1d_gather(::Type{T}, n, tile; name=nothing) where T
    name = something(name, "1D vec_add gather ($n elements, $T, tile=$tile)")
    println("--- $name ---")
    a, b = CUDA.rand(T, n), CUDA.rand(T, n)
    c = CUDA.zeros(T, n)

    ct.launch(vec_add_kernel_1d_gather, cld(n, tile), a, b, c, ct.Constant(tile))

    @assert Array(c) ≈ Array(a) + Array(b)
    println("✓ passed")
end

function main()
    println("--- cuTile Vector/Matrix Addition Examples ---\n")

    # 1D tests with Float32
    test_add_1d(Float32, 1_024_000, 1024)
    test_add_1d(Float32, 2^20, 512)

    # 1D tests with Float64
    test_add_1d(Float64, 2^18, 512)

    # 1D tests with Float16
    test_add_1d(Float16, 1_024_000, 1024)

    # 2D tests with Float32
    test_add_2d(Float32, 2048, 1024, 32, 32)
    test_add_2d(Float32, 1024, 2048, 64, 64)

    # 2D tests with Float64
    test_add_2d(Float64, 1024, 512, 32, 32)

    # 2D tests with Float16
    test_add_2d(Float16, 1024, 1024, 64, 64)

    # 1D gather/scatter tests with Float32
    # Uses explicit index-based memory access instead of tiled loads/stores
    test_add_1d_gather(Float32, 1_024_000, 1024)
    test_add_1d_gather(Float32, 2^20, 512)

    println("\n--- All addition examples completed ---")
end

isinteractive() || main()
