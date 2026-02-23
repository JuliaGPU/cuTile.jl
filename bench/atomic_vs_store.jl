#!/usr/bin/env julia
#
# Benchmark: ct.store vs ct.atomic_add (tile-space)
#
# 1D: Compares atomic_add vs store when there's no contention.
# 2D: Compares flattened 1D (N*N) vs native 2D (NxN) tile-space atomic_add.
#
# Usage:
#   julia --project=test bench/atomic_vs_store.jl

using cuTile
import cuTile as ct
using CUDA

# --- 1D Kernels ---

function store_1d_kernel(dst::ct.TileArray{Float32,1}, TILE::Int)
    bid = ct.bid(1)
    tile = ct.full((TILE,), 1.0f0, Float32)
    ct.store(dst, bid, tile)
    return
end

function atomic_add_1d_kernel(dst::ct.TileArray{Float32,1}, TILE::Int)
    bid = ct.bid(1)
    tile = ct.full((TILE,), 1.0f0, Float32)
    ct.atomic_add(dst, bid, tile)
    return
end

# --- 2D Kernels ---

# Flat: treat N*N as a 1D array, one tile per block
function atomic_add_flat_kernel(dst::ct.TileArray{Float32,1}, TILE::Int)
    bid = ct.bid(1)
    tile = ct.full((TILE,), 1.0f0, Float32)
    ct.atomic_add(dst, bid, tile)
    return
end

# Native 2D: NxN tile-space index
function atomic_add_2d_kernel(dst::ct.TileArray{Float32,2}, TILE_R::Int, NCOLS::Int)
    bid = ct.bid(1)
    # bid is linear over the 2D tile grid; convert to (row_tile, col_tile)
    row = (bid - Int32(1)) ÷ NCOLS + Int32(1)
    col = (bid - Int32(1)) % NCOLS + Int32(1)
    tile = ct.full((TILE_R, TILE_R), 1.0f0, Float32)
    ct.atomic_add(dst, (row, col), tile)
    return
end

# --- Benchmark harness ---

function bench(f, grid, args...; warmup=5, iters=100, reset=nothing, kwargs...)
    for _ in 1:warmup
        reset !== nothing && reset()
        ct.launch(f, grid, args...; kwargs...)
    end
    CUDA.synchronize()

    times = Float64[]
    for _ in 1:iters
        reset !== nothing && reset()
        CUDA.synchronize()
        t = CUDA.@elapsed begin
            ct.launch(f, grid, args...; kwargs...)
        end
        push!(times, t)
    end

    sort!(times)
    trim = max(1, iters ÷ 10)
    trimmed = times[trim+1:end-trim]

    return (
        median = trimmed[length(trimmed) ÷ 2] * 1e6,
        mean   = sum(trimmed) / length(trimmed) * 1e6,
        min    = times[1] * 1e6,
        max    = times[end] * 1e6,
    )
end

function print_result(label, t; reference=nothing)
    line = "  $(rpad(label, 14))$(lpad(round(t.median, digits=2), 8))μs  (min $(round(t.min, digits=2))μs)"
    if reference !== nothing
        ratio = t.median / reference.median
        line *= "  $(round(ratio, digits=2))x"
    end
    println(line)
end

# --- 1D benchmark ---

function bench_1d()
    TILE = 128
    println("=" ^ 60)
    println("1D: ct.store vs ct.atomic_add (no contention)")
    println("   Each block writes its own tile of $TILE Float32s")
    println("=" ^ 60)
    println()

    for n_tiles in [64, 256, 1024, 4096, 16384, 65536]
        n = n_tiles * TILE

        dst_store = CUDA.zeros(Float32, n)
        dst_atomic = CUDA.zeros(Float32, n)

        # Correctness
        ct.launch(store_1d_kernel, n_tiles, dst_store, ct.Constant(TILE))
        ct.launch(atomic_add_1d_kernel, n_tiles, dst_atomic, ct.Constant(TILE))
        CUDA.synchronize()
        @assert all(Array(dst_store) .== 1.0f0)
        @assert all(Array(dst_atomic) .== 1.0f0)

        t_store = bench(store_1d_kernel, n_tiles, dst_store, ct.Constant(TILE))
        t_atomic = bench(atomic_add_1d_kernel, n_tiles, dst_atomic, ct.Constant(TILE);
                         reset=() -> CUDA.fill!(dst_atomic, 0))

        println("$(lpad(n_tiles, 4)) tiles × $TILE = $(lpad(n, 7)) elements")
        print_result("store", t_store)
        print_result("atomic_add", t_atomic; reference=t_store)
        println()
    end
end

# --- 2D benchmark ---

function bench_2d()
    println("=" ^ 60)
    println("2D: flat 1D (N*N) vs native 2D (NxN) atomic_add")
    println("   Same total elements, different indexing strategies")
    println("=" ^ 60)
    println()

    for (tile_r, grid_r) in [(8, 8), (8, 16), (16, 16), (16, 32), (32, 32), (32, 64), (32, 128), (64, 64), (64, 128)]
        n_rows = tile_r * grid_r
        n_cols = n_rows
        n = n_rows * n_cols
        n_tiles_flat = n ÷ (tile_r * tile_r)   # total tiles when flattened
        n_col_tiles = n_cols ÷ tile_r

        dst_flat = CUDA.zeros(Float32, n)
        dst_2d = CUDA.zeros(Float32, n_rows, n_cols)

        # Correctness
        ct.launch(atomic_add_flat_kernel, n_tiles_flat, dst_flat, ct.Constant(tile_r * tile_r))
        ct.launch(atomic_add_2d_kernel, n_tiles_flat, dst_2d,
                  ct.Constant(tile_r), ct.Constant(n_col_tiles))
        CUDA.synchronize()
        @assert all(Array(dst_flat) .== 1.0f0) "flat failed at $(n_rows)×$(n_cols)"
        @assert all(Array(dst_2d) .== 1.0f0) "2D failed at $(n_rows)×$(n_cols)"

        t_flat = bench(atomic_add_flat_kernel, n_tiles_flat, dst_flat, ct.Constant(tile_r * tile_r);
                       reset=() -> CUDA.fill!(dst_flat, 0))
        t_2d = bench(atomic_add_2d_kernel, n_tiles_flat, dst_2d,
                     ct.Constant(tile_r), ct.Constant(n_col_tiles);
                     reset=() -> CUDA.fill!(dst_2d, 0))

        println("$(n_rows)×$(n_cols) = $(lpad(n, 7)) elements  ($(tile_r)×$(tile_r) tiles, $(n_tiles_flat) blocks)")
        print_result("flat 1D", t_flat)
        print_result("native 2D", t_2d; reference=t_flat)
        println()
    end
end

# --- 3D trailing singleton benchmark ---

# NxN tile in a 2D array
function atomic_add_2d_ref_kernel(dst::ct.TileArray{Float32,2}, TILE_R::Int, NCOLS::Int)
    bid = ct.bid(1)
    row = (bid - Int32(1)) ÷ NCOLS + Int32(1)
    col = (bid - Int32(1)) % NCOLS + Int32(1)
    tile = ct.full((TILE_R, TILE_R), 1.0f0, Float32)
    ct.atomic_add(dst, (row, col), tile)
    return
end

# NxNx1 tile in a 3D array (trailing singleton)
function atomic_add_3d_singleton_kernel(dst::ct.TileArray{Float32,3}, TILE_R::Int, NCOLS::Int)
    bid = ct.bid(1)
    row = (bid - Int32(1)) ÷ NCOLS + Int32(1)
    col = (bid - Int32(1)) % NCOLS + Int32(1)
    tile = ct.full((TILE_R, TILE_R, 1), 1.0f0, Float32)
    ct.atomic_add(dst, (row, col, Int32(1)), tile)
    return
end

function bench_trailing_singleton()
    println("=" ^ 60)
    println("Trailing singleton: 2D (NxN) vs 3D (NxNx1) atomic_add")
    println("   Same data, extra singleton dimension in 3D")
    println("=" ^ 60)
    println()

    for (tile_r, grid_r) in [(8, 8), (8, 16), (16, 16), (16, 32), (32, 32), (32, 64), (32, 128), (64, 64), (64, 128)]
        n_rows = tile_r * grid_r
        n_cols = n_rows
        n = n_rows * n_cols
        n_tiles = n ÷ (tile_r * tile_r)
        n_col_tiles = n_cols ÷ tile_r

        dst_2d = CUDA.zeros(Float32, n_rows, n_cols)
        dst_3d = CUDA.zeros(Float32, n_rows, n_cols, 1)

        # Correctness
        ct.launch(atomic_add_2d_ref_kernel, n_tiles, dst_2d,
                  ct.Constant(tile_r), ct.Constant(n_col_tiles))
        ct.launch(atomic_add_3d_singleton_kernel, n_tiles, dst_3d,
                  ct.Constant(tile_r), ct.Constant(n_col_tiles))
        CUDA.synchronize()
        @assert all(Array(dst_2d) .== 1.0f0) "2D failed"
        @assert all(Array(dst_3d) .== 1.0f0) "3D singleton failed"

        t_2d = bench(atomic_add_2d_ref_kernel, n_tiles, dst_2d,
                     ct.Constant(tile_r), ct.Constant(n_col_tiles);
                     reset=() -> CUDA.fill!(dst_2d, 0))
        t_3d = bench(atomic_add_3d_singleton_kernel, n_tiles, dst_3d,
                     ct.Constant(tile_r), ct.Constant(n_col_tiles);
                     reset=() -> CUDA.fill!(dst_3d, 0))

        println("$(n_rows)×$(n_cols) = $(lpad(n, 7)) elements  ($(tile_r)×$(tile_r) tiles, $(n_tiles) blocks)")
        print_result("2D (NxN)", t_2d)
        print_result("3D (NxNx1)", t_3d; reference=t_2d)
        println()
    end
end

# --- Run ---

bench_1d()
bench_2d()
bench_trailing_singleton()
