import Base.Broadcast: BroadcastStyle, Broadcasted
import CUDACore: CuArrayStyle

## broadcast style

struct TiledStyle{N} <: BroadcastStyle end
TiledStyle{M}(::Val{N}) where {N,M} = TiledStyle{N}()

BroadcastStyle(::Type{<:Tiled{A}}) where A = TiledStyle{ndims(A)}()

# TiledStyle wins over DefaultArrayStyle and CuArrayStyle
BroadcastStyle(::TiledStyle{N}, ::Base.Broadcast.DefaultArrayStyle{M}) where {N,M} = TiledStyle{max(N,M)}()
BroadcastStyle(::TiledStyle{N}, ::TiledStyle{M}) where {N,M} = TiledStyle{max(N,M)}()
BroadcastStyle(::TiledStyle{N}, ::CuArrayStyle{M}) where {N,M} = TiledStyle{max(N,M)}()


## broadcast interface

function Base.Broadcast.materialize!(dest::Tiled, bc::Broadcasted)
    arr = parent(dest)
    _tiled_broadcast!(arr, bc)
    return arr
end

function Base.copy(bc::Broadcasted{TiledStyle{N}}) where N
    arr = @something _find_tiled_array(bc) error("tiled broadcast requires at least one Tiled() argument")
    ElType = Base.Broadcast.combine_eltypes(bc.f, bc.args)
    dest = similar(arr, ElType, axes(bc))
    _tiled_broadcast!(dest, bc)
    return dest
end

"""Find the first underlying array from a Tiled leaf in a Broadcasted tree."""
_find_tiled_array(t::Tiled) = parent(t)
_find_tiled_array(x) = nothing
function _find_tiled_array(bc::Broadcasted)
    for arg in bc.args
        arr = _find_tiled_array(arg)
        arr !== nothing && return arr
    end
    return nothing
end


## kernel wrapper

"""
    Extruded{K}(arr::AbstractTileArray)

Array leaf of a `Broadcasted` tree. Like `Base.Broadcast.Extruded`, `K` flags
the dimensions along which the leaf has extent, except that the mask lives in
the type domain because it determines the static shape of the tile to load:
flagged dimensions cover the tile, the others are loaded 1-wide at index 1 and
expanded by the device-side tile broadcast.
"""
struct Extruded{K, A}
    x::A
end
Extruded{K}(x::A) where {K, A} = Extruded{K, A}(x)

# Counterpart of `Base.Broadcast.preprocess`: convert the array leaves of a
# Broadcasted tree into Extruded TileArrays, given the destination rank.
preprocess(t::Tiled, rank::Val) = preprocess(parent(t), rank)
preprocess(arr::AbstractArray{<:Any, 0}, rank::Val) = preprocess(reshape(arr, 1), rank)
function preprocess(arr::AbstractArray, ::Val{N}) where N
    keeps = ntuple(d -> size(arr, d) != 1, Val(N))
    # The typeassert rejects leaves without device storage, like ranges,
    # which `cuTileconvert` passes through unchanged.
    Extruded{keeps}(cuTileconvert(arr)::AbstractTileArray)
end
preprocess(bc::Broadcasted, rank::Val) =
    Broadcasted{Nothing}(bc.f, map(arg -> preprocess(arg, rank), bc.args), nothing)
preprocess(x, rank::Val) = x

function _tiled_broadcast!(dest::AbstractArray{T,N}, bc::Broadcasted) where {T, N}
    # Match Base semantics: size-1 dimensions expand, other mismatches throw.
    Base.Broadcast.check_broadcast_axes(axes(dest), bc.args...)
    isempty(dest) && return

    dest_ta = cuTileconvert(dest)
    tiled_bc = preprocess(bc, Val(N))

    ts = _compute_tile_sizes(size(dest))
    grid = ntuple(i -> cld(size(dest, i), ts[i]), N)
    launch_grid, overflow = _flatten_grid(grid)

    launch(broadcast_kernel, launch_grid, dest_ta, tiled_bc,
           Constant(ts), Constant(overflow))
end

# The kernel operates on tiles of rank >= 1, so run 0-dim broadcasts (both
# destinations, above, and leaves, in `preprocess`) as 1-element vectors.
_tiled_broadcast!(dest::AbstractArray{<:Any,0}, bc::Broadcasted) =
    _tiled_broadcast!(reshape(dest, 1), bc)


## kernel

# Evaluate one node of the Broadcasted tree for the current block, like
# `Base.Broadcast._broadcast_getindex` does per element. A leaf's rank may be
# lower than the destination's (trailing dimensions); `load` completes the
# tile shape with trailing singleton dimensions.
@generated function _eval_bc(leaf::Extruded{K}, bid, tile_size) where K
    M = ndims(fieldtype(leaf, :x))
    shape = [K[d] ? :(tile_size[$d]) : 1 for d in 1:length(K)]
    index = [K[d] ? :(bid[$d]) : :(Int32(1)) for d in 1:M]
    return :(cuTile.load(leaf.x, ($(index...),), ($(shape...),)))
end
@inline _eval_bc(x::Number, bid, tile_size) = x
@inline function _eval_bc(bc::Broadcasted, bid, tile_size)
    args = map(arg -> _eval_bc(arg, bid, tile_size), bc.args)
    # Use broadcast to get element-wise semantics (not direct call, which
    # would dispatch to e.g. matmul for * on tiles)
    broadcast(_bc_func(bc.f), args...)
end

@inline _bc_func(f) = f
@inline _bc_func(::Constant{Type{T}, T}) where {T} = T

@generated function broadcast_kernel(dest::AbstractTileArray{T, N}, bc, tile_size, overflow_grids) where {T, N}
    quote
        bids = _unflatten_bids(Val{$N}(), overflow_grids)
        result = _eval_bc(bc, bids, tile_size)
        # The result can be smaller than the tile (all leaves size-1 in some
        # dimension) or even a scalar (scalar-only right-hand side).
        store(dest, bids, broadcast_to(convert(Tile{$T}, result), tile_size))
        return
    end
end


## broadcasting macro

# Walk dotted AST, wrap value-position leaves in Tiled()
_wrap_tiled(x) = x  # literals pass through
_wrap_tiled(s::Symbol) = :($Tiled($s))
function _wrap_tiled(ex::Expr)
    if ex.head === :.=
        Expr(:.=, _wrap_tiled(ex.args[1]), _wrap_tiled(ex.args[2]))
    elseif ex.head === :. && length(ex.args) == 2 &&
           ex.args[2] isa Expr && ex.args[2].head === :tuple
        # f.(args...) — wrap args, NOT function position
        new_args = map(_wrap_tiled, ex.args[2].args)
        Expr(:., ex.args[1], Expr(:tuple, new_args...))
    else
        Expr(ex.head, map(_wrap_tiled, ex.args)...)
    end
end

"""
    @. expr

Like `Base.@.` but wraps every value-position leaf in `Tiled()`, routing
the broadcast through cuTile kernels.

    using cuTile; const ct = cuTile
    ct.@. C = A + sin(B)
    # equivalent to: Tiled(C) .= Tiled(A) .+ sin.(Tiled(B))

For in-place assignment, the expression evaluates to the original destination
array (`C` above), not its `Tiled` wrapper.
"""
macro __dot__(ex)
    esc(_wrap_tiled(Base.Broadcast.__dot__(ex)))
end
