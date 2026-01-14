# Vector/Matrix addition example - Julia port of cuTile Python's VectorAddition.py sample
#
# SPDX-License-Identifier: Apache-2.0

using CUDA
import cuTile as ct

# 1D kernel
function vec_add_kernel_1d(a::ct.TileArray{T,1}, b::ct.TileArray{T,1}, c::ct.TileArray{T,1},
                           tile::ct.Constant{Int}) where {T}
    bid = ct.bid(1)
    a_tile = ct.load(a, bid, (tile[],))
    b_tile = ct.load(b, bid, (tile[],))
    ct.store(c, bid, a_tile + b_tile)
    return
end

# 2D kernel
function vec_add_kernel_2d(a::ct.TileArray{T,2}, b::ct.TileArray{T,2}, c::ct.TileArray{T,2},
                           tile_x::ct.Constant{Int}, tile_y::ct.Constant{Int}) where {T}
    bid_x = ct.bid(1)
    bid_y = ct.bid(2)
    a_tile = ct.load(a, (bid_x, bid_y), (tile_x[], tile_y[]))
    b_tile = ct.load(b, (bid_x, bid_y), (tile_x[], tile_y[]))
    ct.store(c, (bid_x, bid_y), a_tile + b_tile)
    return
end

# 1D kernel using gather/scatter
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


#=============================================================================
# Example harness
=============================================================================#

function prepare(; benchmark::Bool=false,
                  shape::Tuple=benchmark ? (2^27,) : (1_024_000,),
                  use_gather::Bool=false, T::DataType=Float32)
    a = CUDA.rand(T, shape...)
    return (;
        a,
        b = CUDA.rand(T, shape...),
        c = similar(a),
        shape,
        use_gather
    )
end

function run(data; tile::Union{Int, Tuple{Int,Int}}=1024, nruns::Int=1, warmup::Int=0)
    (; a, b, c, shape, use_gather) = data

    if length(shape) == 2
        # 2D case
        m, n = shape
        tile_x, tile_y = tile isa Tuple ? tile : (tile, tile)
        grid = (cld(m, tile_x), cld(n, tile_y))

        CUDA.@sync for _ in 1:warmup
            ct.launch(vec_add_kernel_2d, grid, a, b, c,
                      ct.Constant(tile_x), ct.Constant(tile_y))
        end

        times = Float64[]
        for _ in 1:nruns
            t = CUDA.@elapsed ct.launch(vec_add_kernel_2d, grid, a, b, c,
                                        ct.Constant(tile_x), ct.Constant(tile_y))
            push!(times, t * 1000)  # ms
        end
    else
        # 1D case
        n = shape[1]
        tile_val = tile isa Tuple ? tile[1] : tile
        grid = cld(n, tile_val)
        kernel = use_gather ? vec_add_kernel_1d_gather : vec_add_kernel_1d

        CUDA.@sync for _ in 1:warmup
            ct.launch(kernel, grid, a, b, c, ct.Constant(tile_val))
        end

        times = Float64[]
        for _ in 1:nruns
            t = CUDA.@elapsed ct.launch(kernel, grid, a, b, c, ct.Constant(tile_val))
            push!(times, t * 1000)  # ms
        end
    end

    return (; c, times)
end

function verify(data, result)
    @assert Array(result.c) ≈ Array(data.a) + Array(data.b)
end


#=============================================================================
# Reference implementations for benchmarking
=============================================================================#

# Simple SIMT kernel for comparison
function simt_kernel(a, b, c, n)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= n
        @inbounds c[i] = a[i] + b[i]
    end
    return
end

function run_others(data; nruns::Int=1, warmup::Int=0)
    (; a, b, c, shape) = data
    results = Dict{String, Vector{Float64}}()

    if length(shape) == 1
        n = shape[1]
        c_gpuarrays = similar(c)
        c_simt = similar(c)

        # GPUArrays (broadcasting)
        CUDA.@sync for _ in 1:warmup
            c_gpuarrays .= a .+ b
        end
        times_gpuarrays = Float64[]
        for _ in 1:nruns
            t = CUDA.@elapsed c_gpuarrays .= a .+ b
            push!(times_gpuarrays, t * 1000)
        end
        results["GPUArrays"] = times_gpuarrays

        # SIMT kernel
        threads = 256
        blocks = cld(n, threads)
        CUDA.@sync for _ in 1:warmup
            @cuda threads=threads blocks=blocks simt_kernel(a, b, c_simt, n)
        end
        times_simt = Float64[]
        for _ in 1:nruns
            t = CUDA.@elapsed @cuda threads=threads blocks=blocks simt_kernel(a, b, c_simt, n)
            push!(times_simt, t * 1000)
        end
        results["SIMT"] = times_simt
    end

    return results
end


#=============================================================================
# Main
=============================================================================#

function test_vadd(shape, tile; use_gather::Bool=false, T::DataType=Float32, name=nothing)
    if name === nothing
        if length(shape) == 2
            name = "2D vec_add ($(shape[1]) x $(shape[2]), $T, tile=$tile)"
        elseif use_gather
            name = "1D vec_add gather ($(shape[1]) elements, $T, tile=$tile)"
        else
            name = "1D vec_add ($(shape[1]) elements, $T, tile=$tile)"
        end
    end
    println("--- $name ---")
    data = prepare(; shape, use_gather, T)
    result = run(data; tile)
    verify(data, result)
    println("  passed")
end

function main()
    println("--- cuTile Vector/Matrix Addition Examples ---\n")

    # 1D tests with Float32
    test_vadd((1_024_000,), 1024)
    test_vadd((2^20,), 512)

    # 1D tests with Float64
    test_vadd((2^18,), 512; T=Float64)

    # 1D tests with Float16
    test_vadd((1_024_000,), 1024; T=Float16)

    # 2D tests with Float32
    test_vadd((2048, 1024), (32, 32))
    test_vadd((1024, 2048), (64, 64))

    # 2D tests with Float64
    test_vadd((1024, 512), (32, 32); T=Float64)

    # 2D tests with Float16
    test_vadd((1024, 1024), (64, 64); T=Float16)

    # 1D gather/scatter tests with Float32
    # Uses explicit index-based memory access instead of tiled loads/stores
    test_vadd((1_024_000,), 1024; use_gather=true)
    test_vadd((2^20,), 512; use_gather=true)

    println("\n--- All addition examples completed ---")
end

isinteractive() || main()
