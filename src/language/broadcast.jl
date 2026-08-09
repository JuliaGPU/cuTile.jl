# Broadcasting Infrastructure for Tiles
#
# Defines the broadcast style and shape computation for Tile types.
# All broadcasted operations are materialized via copy.

import Base.Broadcast: BroadcastStyle, Broadcasted, broadcastable, broadcast_shape


#=============================================================================
 Custom BroadcastStyle for Tiles
=============================================================================#

struct TileStyle <: BroadcastStyle end
Base.Broadcast.BroadcastStyle(::Type{<:Tile}) = TileStyle()

# When combining TileStyle with itself, return TileStyle
Base.Broadcast.BroadcastStyle(::TileStyle, ::TileStyle) = TileStyle()

# When combining TileStyle with scalars, TileStyle wins
Base.Broadcast.BroadcastStyle(::TileStyle, ::Base.Broadcast.DefaultArrayStyle{0}) = TileStyle()

# Tiles are already broadcastable - return as-is
Base.Broadcast.broadcastable(t::Tile) = t


# Base's `broadcastable` wraps scalar-like arguments in Ref: functions and
# Vals via the fallback (e.g. `x .^ 2` lowers to `broadcasted(literal_pow,
# ^, x, Val(2))`) and Types as `Ref{Type{T}}`. The Refs pass through the
# machinery below and never reach codegen: with all uses inlined and
# non-escaping, SROA eliminates the mutable allocation.


#=============================================================================
 Broadcast materialization via copy
=============================================================================#

# Tile is a ghost type with no storage, so axes/size are meaningless.
# Skip instantiate (which calls axes) by returning the Broadcasted as-is.
@inline Base.Broadcast.instantiate(bc::Broadcasted{TileStyle}) = bc

# Recursively materialize nested Broadcasted nodes,
# promote scalars to Tiles, broadcast to a common shape, then apply f.
# This handles all element-wise operations: scalar @overlay methods provide
# the implementation for overlaid ops, while Julia's native scalar functions
# (compiled to Core intrinsics) handle the rest. Mixed-type and type-changing
# operations (comparisons, ifelse) are supported by the mixed-type map methods
# in operations.jl.
@inline function Base.copy(bc::Broadcasted{TileStyle})
    args = _materialize_args(bc.args)
    promoted = _promote_to_tiles(args...)
    S = _broadcast_shapes(promoted...)
    broadcasted = _broadcast_all(S, promoted...)
    _apply_broadcast(bc.f, broadcasted...)
end

# Recursively materialize nested Broadcasted nodes into concrete Tiles.
# Unlike standard Julia broadcast (which fuses by keeping lazy Broadcasted
# nodes and indexing element-by-element in one loop), cuTile must eagerly
# materialize because Tile IR operates on whole tiles — there is no
# element-wise indexing. Two separate IR ops (e.g. mulf then addf) IS the
# correct output. The intermediate from_scalar/to_scalar pairs between
# stages are zero-cost (just CGVal type reinterpretation at codegen time).
@inline _materialize_arg(x) = x
@inline _materialize_arg(bc::Broadcasted{TileStyle}) = copy(bc)
@inline _materialize_args(::Tuple{}) = ()
@inline _materialize_args(args::Tuple) =
    (_materialize_arg(args[1]), _materialize_args(Base.tail(args))...)

# Promote scalars to the first same-category Tile element type. This keeps
# integer and floating literals from widening narrower tiles; unrelated Tile
# arguments such as an `ifelse` condition are skipped.
@inline _promote_to_tiles(args...) = _promote_to_tiles(args, args)
@inline _promote_to_tiles(::Tuple{}, ::Tuple) = ()
@inline function _promote_to_tiles(args::Tuple, all::Tuple)
    (_promote_to_tile(args[1], all), _promote_to_tiles(Base.tail(args), all)...)
end

@inline _promote_to_tile(a::Tile, ::Tuple) = a
@inline _promote_to_tile(a::Base.RefValue, ::Tuple) = a
@inline function _promote_to_tile(a::T, args::Tuple) where {T <: Number}
    U = _loose_scalar_type(T, args)
    Tile(convert(U, a))
end

@inline _loose_scalar_type(::Type{T}, ::Tuple{}) where {T} = T
@inline function _loose_scalar_type(::Type{T}, args::Tuple{A, Vararg}) where {T, A<:Tile}
    U = eltype(A)
    if (T <: AbstractFloat && U <: AbstractFloat) ||
       (T <: Integer && T !== Bool && U <: Integer && U !== Bool) ||
       (T === Bool && U === Bool)
        return U
    end
    _loose_scalar_type(T, Base.tail(args))
end
@inline _loose_scalar_type(::Type{T}, args::Tuple{Any, Vararg}) where {T} =
    _loose_scalar_type(T, Base.tail(args))

# Compute combined broadcast shape across all Tile arguments via tuple peeling.
# Shape is always a tuple TYPE (e.g., Tuple{16, 32}). Convert to value for broadcast_shape.
# Base.RefValue arguments are skipped — they have no shape.
@inline _tile_shape(t::Tile) = size(t)
@inline _broadcast_shapes(t::Tile) = _tile_shape(t)
@inline _broadcast_shapes(t::Tile, rest...) =
    broadcast_shape(_tile_shape(t), _broadcast_shapes(rest...))
@inline _broadcast_shapes(::Base.RefValue, rest...) = _broadcast_shapes(rest...)
@inline _broadcast_shapes(::Base.RefValue) = ()

# Broadcast all tiles to shape S via tuple peeling.
# Base.RefValue arguments pass through unchanged.
@inline _broadcast_all(S::Tuple) = ()
@inline _broadcast_all(S::Tuple, a::Tile, rest...) =
    (broadcast_to(a, S), _broadcast_all(S, rest...)...)
@inline _broadcast_all(S::Tuple, a::Base.RefValue, rest...) =
    (a, _broadcast_all(S, rest...)...)

# Convert args to scalars, apply f, wrap result back into a Tile.
@inline function _apply_broadcast(f, args...)
    Intrinsics.from_scalar(f(map(_to_scalar, args)...), _result_shape(args...))
end

# Reinterpret arguments as scalars for broadcast application: Tiles via
# to_scalar, Refs via their contents. The Ref{Type{T}} method recovers the
# Type from the type parameter, mirroring Base's `_broadcast_getindex`.
@inline _to_scalar(t::Tile) = Intrinsics.to_scalar(t)
@inline _to_scalar(r::Base.RefValue) = r[]
@inline _to_scalar(::Base.RefValue{Type{T}}) where T = T

# Result shape comes from the first Tile argument; after _broadcast_all,
# every Tile already has the common shape.
@inline _result_shape(t::Tile{<:Any,S}, rest...) where S = S
@inline _result_shape(::Base.RefValue, rest...) = _result_shape(rest...)
