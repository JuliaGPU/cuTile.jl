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


#=============================================================================
 Ghost wrapper for Type values in broadcasting
=============================================================================#

# Replaces Julia's RefValue{Type{T}} wrapping which the cuTile compiler can't construct.
# The value is encoded in the type parameter — no runtime representation needed.
struct TypeRef{T} end

Base.Broadcast.BroadcastStyle(::Type{<:TypeRef}) = Base.Broadcast.DefaultArrayStyle{0}()
Base.Broadcast.broadcastable(a::TypeRef) = a

# Scalar-carrying broadcast arguments without a tile shape: TypeRef, plus
# Base.RefValue from Julia's `broadcastable` fallback (e.g. `x .^ 2` lowers to
# `broadcasted(literal_pow, ^, x, Val(2))`, wrapping `^` and `Val(2)` in Ref).
# The Ref never reaches codegen: with all uses inlined and non-escaping, SROA
# eliminates the mutable allocation.
const ScalarRef = Union{TypeRef, Base.RefValue}


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

# Promote Number arguments to 0-dimensional Tiles. Each Number is wrapped
# using its own type (e.g., 0.0f0 → Tile(Float32(0.0))), preserving the
# type that Julia's broadcast promotion chose. This avoids the pitfall of
# using the first Tile's eltype (which could be Bool for ifelse conditions).
# ScalarRef arguments pass through unchanged — they carry no tile shape.
@inline _promote_to_tiles() = ()
@inline _promote_to_tiles(a::Tile, rest...) = (a, _promote_to_tiles(rest...)...)
@inline _promote_to_tiles(a::T, rest...) where {T <: Number} =
    (Tile(a), _promote_to_tiles(rest...)...)
@inline _promote_to_tiles(a::ScalarRef, rest...) = (a, _promote_to_tiles(rest...)...)

# Compute combined broadcast shape across all Tile arguments via tuple peeling.
# Shape is always a tuple TYPE (e.g., Tuple{16, 32}). Convert to value for broadcast_shape.
# ScalarRef arguments are skipped — they have no shape.
@inline _tile_shape(t::Tile) = size(t)
@inline _broadcast_shapes(t::Tile) = _tile_shape(t)
@inline _broadcast_shapes(t::Tile, rest...) =
    broadcast_shape(_tile_shape(t), _broadcast_shapes(rest...))
@inline _broadcast_shapes(::ScalarRef, rest...) = _broadcast_shapes(rest...)
@inline _broadcast_shapes(::ScalarRef) = ()

# Broadcast all tiles to shape S via tuple peeling.
# ScalarRef arguments pass through unchanged.
@inline _broadcast_all(S::Tuple) = ()
@inline _broadcast_all(S::Tuple, a::Tile, rest...) =
    (broadcast_to(a, S), _broadcast_all(S, rest...)...)
@inline _broadcast_all(S::Tuple, a::ScalarRef, rest...) =
    (a, _broadcast_all(S, rest...)...)

# Convert args to scalars, apply f, wrap result back into a Tile.
@inline function _apply_broadcast(f, args...)
    Intrinsics.from_scalar(f(map(_to_scalar, args)...), _result_shape(args...))
end

# Reinterpret arguments as scalars for broadcast application: Tiles via
# to_scalar, TypeRefs via their type parameter, Refs via their contents.
@inline _to_scalar(t::Tile) = Intrinsics.to_scalar(t)
@inline _to_scalar(::TypeRef{T}) where T = T
@inline _to_scalar(r::Base.RefValue) = r[]

# Result shape comes from the first Tile argument; after _broadcast_all,
# every Tile already has the common shape.
@inline _result_shape(t::Tile{<:Any,S}, rest...) where S = S
@inline _result_shape(::ScalarRef, rest...) = _result_shape(rest...)
