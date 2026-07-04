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
    BroadcastLeaf{P}(arr::TileArray)

Kernel-side wrapper for an array leaf whose shape differs from the broadcast
destination's. `P` is an `NTuple{N,Bool}` over the destination dims, flagging
dims where the leaf is size 1 (or absent) and must be loaded 1-wide at index 1,
to be expanded to the full tile shape by the device-side tile broadcast.
"""
struct BroadcastLeaf{P, A}
    arr::A
end
BroadcastLeaf{P}(arr::A) where {P, A} = BroadcastLeaf{P, A}(arr)

# Reject leaves the kernel cannot address: a host Array would compile fine but
# dereference a CPU pointer on device (illegal memory access, poisoning the
# CUDA context), and ranges have no pointer at all.
function _check_device_leaf(arr::AbstractArray)
    root = arr
    while (p = parent(root)) !== root
        root = p
    end
    if root isa Array || root isa AbstractRange
        throw(ArgumentError("cuTile broadcast requires device arrays, got $(typeof(arr))"))
    end
end

_to_tiled_bc(t::Tiled, dsize) = _to_tiled_bc(parent(t), dsize)
# 0-dim leaves become 1-element vectors; the kernel path assumes rank >= 1.
_to_tiled_bc(arr::AbstractArray{<:Any,0}, dsize::Dims) = _to_tiled_bc(reshape(arr, 1), dsize)
function _to_tiled_bc(arr::AbstractArray, dsize::Dims{N}) where N
    _check_device_leaf(arr)
    ta = cuTileconvert(arr)
    ndims(arr) == N && size(arr) == dsize && return ta
    P = ntuple(d -> size(arr, d) == 1 && dsize[d] > 1, N)
    return BroadcastLeaf{P}(ta)
end
_to_tiled_bc(x::Number, dsize) = x
_to_tiled_bc(x, dsize) = x  # fallback for other types
function _to_tiled_bc(bc::Broadcasted, dsize)
    new_args = map(arg -> _to_tiled_bc(arg, dsize), bc.args)
    Broadcasted{Nothing}(bc.f, new_args, nothing)
end

# 0-dim destinations: the kernel path needs at least one dimension, so run as
# a 1-element 1-D broadcast (0-dim leaves load rank-0 tiles and expand).
function _tiled_broadcast!(dest::AbstractArray{T,0}, bc::Broadcasted) where T
    Base.Broadcast.check_broadcast_axes(axes(dest), bc.args...)
    _tiled_broadcast!(reshape(dest, 1), bc)
end

function _tiled_broadcast!(dest::AbstractArray{T,N}, bc::Broadcasted) where {T, N}
    # Reject shapes Base broadcasting would reject (throws DimensionMismatch);
    # size-1 dims are legal and expanded per-leaf in the kernel.
    Base.Broadcast.check_broadcast_axes(axes(dest), bc.args...)
    isempty(dest) && return

    _check_device_leaf(dest)
    dest_ta = cuTileconvert(dest)
    tiled_bc = _to_tiled_bc(bc, size(dest))

    ts = _compute_tile_sizes(size(dest))
    grid = ntuple(i -> cld(size(dest, i), ts[i]), N)
    launch_grid, overflow = _flatten_grid(grid)

    launch(broadcast_kernel, launch_grid, dest_ta, tiled_bc,
           Constant(ts), Constant(overflow))
end


## kernel

@inline _eval_bc(arr::AbstractTileArray, bid, tile_size) = cuTile.load(arr, bid, tile_size)
@inline _eval_bc(x::Number, bid, tile_size) = x

@inline _bc_func(f) = f
@inline _bc_func(::Constant{Type{T}, T}) where {T} = T

# Shape-mismatched leaf: dims flagged in P load a 1-wide slice at index 1;
# the tile-level broadcast in `broadcast(bc.f, ...)` expands them to the
# common tile shape. The leaf's rank M may differ from the destination rank
# (trailing dims); `load` pads/squeezes trailing singleton dims to match.
@generated function _eval_bc(leaf::BroadcastLeaf{P}, bid, tile_size) where P
    A = fieldtype(leaf, :arr)
    M, N = ndims(A), length(P)
    shape = [P[d] ? 1 : :(tile_size[$d]) for d in 1:N]
    index = [(d > N || P[d]) ? :(Int32(1)) : :(bid[$d]) for d in 1:M]
    return :(cuTile.load(leaf.arr, ($(index...),), ($(shape...),)))
end

@inline function _eval_bc(bc::Broadcasted, bid, tile_size)
    args = _eval_bc_args(bc.args, bid, tile_size)
    # Use broadcast to get element-wise semantics (not direct call, which
    # would dispatch to e.g. matmul for * on tiles)
    broadcast(_bc_func(bc.f), args...)
end

@inline _eval_bc_args(::Tuple{}, bid, tile_size) = ()
@inline _eval_bc_args(args::Tuple, bid, tile_size) =
    (_eval_bc(args[1], bid, tile_size), _eval_bc_args(Base.tail(args), bid, tile_size)...)

# Convert the broadcast result to a dest-eltype, dest-shaped tile. A `Number`
# result comes from a scalar-only RHS (e.g. `Tiled(C) .= 0`); a smaller tile
# arises when every leaf is size-1 in some dim.
@inline _bc_result(x::Number, ::Type{T}, tile_size) where {T} =
    broadcast_to(Tile(T(x)), tile_size)
@inline _bc_result(t::Tile, ::Type{T}, tile_size) where {T} =
    broadcast_to(convert(Tile{T}, t), tile_size)

@generated function broadcast_kernel(dest::AbstractTileArray{T, N}, bc, tile_size, overflow_grids) where {T, N}
    quote
        bids = _unflatten_bids(Val{$N}(), overflow_grids)
        result = _eval_bc(bc, bids, tile_size)
        store(dest, bids, _bc_result(result, $T, tile_size))
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
