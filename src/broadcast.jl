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

_to_tiled_bc(arr::AbstractArray) = cuTileconvert(arr)
_to_tiled_bc(t::Tiled) = _to_tiled_bc(parent(t))
_to_tiled_bc(x::Number) = x
_to_tiled_bc(x) = x  # fallback for other types
function _to_tiled_bc(bc::Broadcasted)
    new_args = map(_to_tiled_bc, bc.args)
    Broadcasted{Nothing}(bc.f, new_args, nothing)
end

function _tiled_broadcast!(dest::AbstractArray{T,N}, bc::Broadcasted) where {T, N}
    dest_ta = _to_tiled_bc(dest)
    tiled_bc = _to_tiled_bc(bc)

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

@inline function _eval_bc(bc::Broadcasted, bid, tile_size)
    args = _eval_bc_args(bc.args, bid, tile_size)
    # Use broadcast to get element-wise semantics (not direct call, which
    # would dispatch to e.g. matmul for * on tiles)
    broadcast(_bc_func(bc.f), args...)
end

@inline _eval_bc_args(::Tuple{}, bid, tile_size) = ()
@inline _eval_bc_args(args::Tuple, bid, tile_size) =
    (_eval_bc(args[1], bid, tile_size), _eval_bc_args(Base.tail(args), bid, tile_size)...)

@generated function broadcast_kernel(dest::AbstractTileArray{T, N}, bc, tile_size, overflow_grids) where {T, N}
    quote
        bids = _unflatten_bids(Val{$N}(), overflow_grids)
        result = _eval_bc(bc, bids, tile_size)
        store(dest, bids, convert(Tile{$T}, result))
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
"""
macro __dot__(ex)
    esc(_wrap_tiled(Base.Broadcast.__dot__(ex)))
end
