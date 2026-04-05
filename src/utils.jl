using Base.Cartesian: @nexprs, @ntuple, @nloops

#=============================================================================
 Grid and tile sizing helpers (used by broadcast and mapreduce)
=============================================================================#

"""
    _flatten_grid(grid::NTuple{N,Int}) -> (launch_grid, overflow)

Pack an N-dimensional tile grid into at most 3 CUDA grid dimensions.
Dims 1–2 pass through; dims 3+ are multiplied into a single axis, with the
per-dimension sizes returned as `overflow` for [`_unflatten_bids`](@ref) to
unpack inside the kernel.
"""
function _flatten_grid(grid::NTuple{N,Int}) where N
    launch_grid = N <= 3 ? grid : (grid[1], grid[2], prod(grid[i] for i in 3:N))
    overflow = N > 3 ? grid[3:end] : ()
    return launch_grid, overflow
end

"""
    _unflatten_bids(::Val{N}, overflow_grids) -> NTuple{N}

Inverse of [`_flatten_grid`](@ref): recover N-dimensional block IDs inside a
kernel from the ≤3-dimensional CUDA launch grid.  Dims 1–2 map directly to
`bid(1)` and `bid(2)`; dims 3+ are unpacked from `bid(3)` using the
`overflow_grids` metadata produced by `_flatten_grid`.
"""
@inline @generated function _unflatten_bids(::Val{N}, overflow_grids) where N
    quote
        $(N > 2 ? :(_rem = bid(3) - Int32(1)) : nothing)
        @nexprs $N d -> if d <= 2
            bid_d = bid(d)
        elseif d == $N && d > 2
            bid_d = _rem + Int32(1)
        else
            bid_d = rem(_rem, Int32(overflow_grids[d - 2])) + Int32(1)
            _rem = fld(_rem, Int32(overflow_grids[d - 2]))
        end
        @ntuple $N d -> bid_d
    end
end

"""
    _compute_tile_sizes(dest_size; budget=4096)

Distribute a total element budget greedily across dimensions, skipping singletons.
Each tile dimension is a power of 2, capped by the array size in that dimension.
"""
function _compute_tile_sizes(dest_size::NTuple{N,Int}; budget::Int=4096) where N
    _compute_tile_sizes(dest_size, 1:N; budget)
end

"""
    _compute_tile_sizes(input_size, dim_order; budget=4096)

Distribute tile budget greedily in the given dimension order.
Dimensions not in `dim_order` get tile size 1.
"""
function _compute_tile_sizes(input_size::NTuple{N,Int}, dim_order; budget::Int=4096) where N
    ts = ones(Int, N)
    remaining = budget
    for i in dim_order
        s = input_size[i]
        s <= 1 && continue
        t = prevpow(2, min(remaining, s))
        ts[i] = t
        remaining = remaining ÷ t
        remaining < 2 && break
    end
    return NTuple{N,Int}(ts)
end

