# @view / slice example — mirrors cuTile Python's test_array_slice.py sample
#
# Showcases Julia's standard `@view` macro applied to a `TileArray`. Each
# kernel takes a subrange of an array and operates on it as if it were a
# smaller TileArray. cuTile Python calls this `Array.slice(axis, start, stop)`.
#
# Slicing is most useful when per-block start/stop are computed at runtime
# (e.g. ragged segments): `@view A[start:stop, :]` lets the kernel work in
# local coordinates without threading a global offset everywhere.
#
# SPDX-License-Identifier: Apache-2.0

using CUDA
import cuTile as ct

#=============================================================================
 1D static slice copy: y[start:stop] ← x[start:stop]
=============================================================================#

function slice_copy_1d(x::ct.TileArray{T,1}, y::ct.TileArray{T,1},
                      start::Int32, stop::Int32, TILE::Int) where {T}
    sub_x = @view x[start:stop]
    sub_y = @view y[start:stop]
    tile = ct.load(sub_x, 1, (TILE,))
    ct.store(sub_y, 1, tile)
    return
end

#=============================================================================
 Ragged 2D copy: per-segment slicing along axis 0
 B[indptr[seg], :] .. B[indptr[seg+1]-1, :] ← A[same rows, :]
=============================================================================#

function ragged_copy_2d(A::ct.TileArray{T,2}, B::ct.TileArray{T,2},
                        indptr::ct.TileArray{Int32,1},
                        TILE_M::Int, TILE_N::Int) where {T}
    seg_id = ct.bid(1)
    j = ct.bid(2)

    # Per-segment row range from indptr (Julia 1-indexed: [start, stop] inclusive).
    start = indptr[seg_id]
    stop = indptr[seg_id + Int32(1)] - Int32(1)

    sub_A = @view A[start:stop, :]
    sub_B = @view B[start:stop, :]

    m = stop - start + Int32(1)
    num_m_tiles = cld(m, Int32(TILE_M))
    for i in Int32(1):num_m_tiles
        tile = ct.load(sub_A, (i, j), (TILE_M, TILE_N))
        ct.store(sub_B, (i, j), tile)
    end
    return
end

#=============================================================================
 Example harness
=============================================================================#

function prepare_1d(T::DataType=Float32, n::Int=32)
    return (;
        x = CUDA.rand(T, n),
        y = CUDA.zeros(T, n),
        n,
    )
end

function prepare_ragged(T::DataType=Float32)
    # Ragged segments along axis 0 with tile-aligned boundaries.
    # indptr is Julia 1-indexed: segment k covers rows [indptr[k], indptr[k+1]-1].
    M, N = 12, 16
    A = CUDA.rand(T, M, N)
    B = CUDA.zeros(T, M, N)
    indptr = CuArray(Int32[1, 5, 9, 13])   # segments: 1..4, 5..8, 9..12
    return (; A, B, indptr, M, N, num_segments=3)
end

function run_1d(data; start::Int32=Int32(4), stop::Int32=Int32(12), TILE::Int=8)
    (; x, y, n) = data
    ct.launch(slice_copy_1d, 1, x, y, start, stop, ct.Constant(TILE))
    return Array(y)
end

function run_ragged(data; TILE_M::Int=4, TILE_N::Int=8)
    (; A, B, indptr, M, N, num_segments) = data
    grid = (num_segments, cld(N, TILE_N))
    ct.launch(ragged_copy_2d, grid, A, B, indptr, ct.Constant(TILE_M), ct.Constant(TILE_N))
    return Array(B)
end

function verify_1d(data, result)
    expected = zeros(eltype(data.x), data.n)
    # @view x[4:12] in Julia 1-indexed = start:stop range (5..12 Julia ↔ 4..11 0-indexed)
    expected[5:12] .= Array(data.x)[5:12]
    @assert isapprox(result, expected) "1D slice copy mismatch"
end

function verify_ragged(data, result)
    expected = Array(data.A)   # ragged copy covers the whole array
    @assert isapprox(result, expected) "ragged slice copy mismatch"
end

function main()
    println("--- cuTile @view / slice examples ---\n")

    # 1D static slice: @view x[5:12] (Julia 1-indexed inclusive).
    # Converted to 0-indexed half-open [4, 12) for the intrinsic.
    data_1d = prepare_1d()
    result_1d = run_1d(data_1d; start=Int32(5), stop=Int32(12), TILE=8)
    verify_1d(data_1d, result_1d)
    println("  1D slice: passed")

    # 2D ragged copy: each block slices A[indptr[k]:indptr[k+1], :].
    data_rag = prepare_ragged()
    result_rag = run_ragged(data_rag)
    verify_ragged(data_rag, result_rag)
    println("  2D ragged slice: passed")

    println("\n--- All slice examples completed ---")
end

isinteractive() || main()
