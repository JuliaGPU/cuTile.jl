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
# operations (comparisons, ifelse) need nothing extra — `f` decides the result
# element type. `Base.map` (operations.jl) enters the same path at
# `_apply_broadcast`, its tiles already sharing a shape.
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
# Base.RefValue arguments pass through unchanged — they carry no tile shape.
@inline _promote_to_tiles() = ()
@inline _promote_to_tiles(a::Tile, rest...) = (a, _promote_to_tiles(rest...)...)
@inline _promote_to_tiles(a::T, rest...) where {T <: Number} =
    (Tile(a), _promote_to_tiles(rest...)...)
@inline _promote_to_tiles(a::Base.RefValue, rest...) = (a, _promote_to_tiles(rest...)...)

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
#
# Restricted floats (FP8/FP4/TFloat32) are gated here rather than by shadowing
# the upstream scalar methods one by one: kernels can only ever obtain a
# restricted-float scalar through this function (and `map`), so this is the one
# choke point that covers every present and future upstream method. Only
# conversion, selection and comparison are let through; everything else —
# arithmetic, math functions, user lambdas — is rejected before dispatch, so no
# upstream fallback implementation is ever consulted.
@inline function _apply_broadcast(f, args...)
    if _restricted_args(args...)
        _restricted_broadcast(f, args...)
    else
        _broadcast_scalars(f, args...)
    end
end

@inline _broadcast_scalars(f, args...) =
    Intrinsics.from_scalar(f(map(_to_scalar, args)...), _result_shape(args...))

# Does any Tile argument have a restricted float element type? Tuple peeling
# rather than `any` with a closure: the argument tuple is heterogeneous (Tiles
# and Refs). Folds to `false` at inference time for arithmetic element types,
# keeping the common path branch-free. `is_restricted_float` is called directly,
# not through `invokelatest`: kernel inference sees the extension methods, and
# the fold depends on it.
@inline _restricted_args() = false
@inline _restricted_args(a::Tile, rest...) =
    is_restricted_float(eltype(a)) || _restricted_args(rest...)
@inline _restricted_args(a, rest...) = _restricted_args(rest...)

const ComparisonOps = Union{typeof(<), typeof(<=), typeof(>), typeof(>=),
                            typeof(==), typeof(!=), typeof(isless)}

# Explicit element-type conversion (`Float32.(tile)`, `convert.(Float32, tile)`,
# and `convert(Tile{T}, tile)` via `map`) is the sanctioned escape hatch: pass it
# through to the constructor overlays, which lower it to a single `ftof`.
@inline _restricted_broadcast(f::Union{Type,typeof(convert)}, args...) =
    _broadcast_scalars(f, args...)

# `ifelse` selects between unmodified values and lowers via `Core.ifelse`, so no
# upstream restricted-float method is involved.
@inline _restricted_broadcast(f::typeof(ifelse), args...) =
    _broadcast_scalars(f, args...)

# Comparisons stay available (as they do in cuTile Python), but Tile IR has no
# native fp8/fp4 comparison: upcast the restricted operands to Float32 and
# re-apply. That is exact and injective for every restricted format (NaN → NaN,
# ±0 preserved), so the result matches the host's ordering. Going through the
# upcast rather than the upstream scalar `<` also keeps their implementation
# details (bit tricks, `isnan` guards) out of the kernel.
@inline _restricted_broadcast(f::ComparisonOps, args...) =
    _apply_broadcast(f, map(_upcast_restricted, args)...)

# Everything else — arithmetic, math functions, user lambdas — is rejected.
# Upstream implements scalar arithmetic on these formats as a Float32 round-trip,
# which would otherwise compile into a silent ftof/op/ftof with an extra rounding
# per operation, and no hint that a cast happened.
@inline _restricted_broadcast(f, args...) = throw(ArgumentError(RESTRICTED_ARITHMETIC_MESSAGE))

@inline _upcast_restricted(a::Tile{T}) where {T} =
    is_restricted_float(T) ? convert(Tile{Float32}, a) : a
@inline _upcast_restricted(a) = a

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
