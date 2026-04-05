using Base.Cartesian: @nexprs, @ntuple, inlineanonymous

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

#=============================================================================
 @nwhileloops — while-loop variant of Base.Cartesian.@nloops
=============================================================================#

"""
    @nwhileloops N condexpr [preexpr [postexpr]] body

Generate N nested `while` loops, analogous to `Base.Cartesian.@nloops` but
using `while` instead of `for`.  This is needed for loops with non-unit strides,
where Julia's `for i in start:step:stop` generates `StepRange` construction IR
that is too complex for cuTile's codegen (ArgumentError handling, overflow_case,
checked_srem_int).  Unit-step loops should use native `for i in start:stop`.

`condexpr` and the optional `preexpr`/`postexpr` are `d->` anonymous functions
specialized per dimension with Cartesian `_d` suffix naming.  If you want just
a post-expression, supply `nothing` for the pre-expression.

# Example
```julia
@nwhileloops 2 d->(idx_d <= n_d) d->(idx_d = start_d) d->(idx_d += stride[d]) begin
    # innermost body
end
```
generates:
```julia
idx_2 = start_2
while idx_2 <= n_2
    idx_1 = start_1
    while idx_1 <= n_1
        # innermost body
        idx_1 += stride[1]
    end
    idx_2 += stride[2]
end
```
"""
macro nwhileloops(N, condexpr, args...)
    _nwhileloops(N, condexpr, args...)
end

function _nwhileloops(N::Int, condexpr::Expr, args::Expr...)
    if !(1 <= length(args) <= 3)
        throw(ArgumentError("expected 1 to 3 trailing arguments (body, or pre+body, or pre+post+body), got $(length(args))"))
    end
    body = args[end]
    ex = Expr(:escape, body)
    for d in 1:N
        cond = esc(inlineanonymous(condexpr, d))
        preexpr = length(args) > 1 ? esc(inlineanonymous(args[1], d)) : nothing
        postexpr = length(args) > 2 ? esc(inlineanonymous(args[2], d)) : nothing
        ex = quote
            $preexpr
            while $cond
                $ex
                $postexpr
            end
        end
    end
    ex
end
