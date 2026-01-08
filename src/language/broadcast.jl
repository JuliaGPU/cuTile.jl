# Broadcasting Infrastructure for Tiles
#
# Enables Julia's broadcasting syntax (a .+ b, a .< b, etc.) for Tile types.
# Broadcasting with different shapes is handled by automatic broadcast_to calls.

import Base.Broadcast: BroadcastStyle, Broadcasted, broadcastable

#=============================================================================
 Broadcast Shape Computation
=============================================================================#

"""
    broadcast_shape(s1::Tuple, s2::Tuple) -> Tuple

Compute the broadcast shape from two tile shapes using NumPy-style broadcasting rules.
- Shapes are compared from right to left (trailing dimensions)
- Dimensions are compatible if they're equal or one of them is 1
- Missing dimensions are treated as 1

This is a pure function that Julia's const-prop can evaluate at compile time.

# Examples
```julia
broadcast_shape((128,), (1, 128))   # => (1, 128)
broadcast_shape((1,), (128,))       # => (128,)
broadcast_shape((4, 1), (1, 8))     # => (4, 8)
broadcast_shape((16, 32), (16, 32)) # => (16, 32)
```
"""
@inline function broadcast_shape(s1::Tuple, s2::Tuple)
    max_ndim = max(length(s1), length(s2))
    ntuple(max_ndim) do i
        # Index from the right (trailing dimensions)
        idx1 = length(s1) - max_ndim + i
        idx2 = length(s2) - max_ndim + i
        d1 = idx1 > 0 ? s1[idx1] : 1
        d2 = idx2 > 0 ? s2[idx2] : 1
        # Check compatibility
        (d1 == d2 || d1 == 1 || d2 == 1) || error("Shapes $s1 and $s2 are not broadcastable")
        max(d1, d2)
    end
end

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
 Broadcasted Arithmetic Operators
=============================================================================#

# Tile-Tile arithmetic (a .+ b becomes broadcasted(+, a, b))
# Float operations
for (op, intrinsic) in ((:+, :addf), (:-, :subf), (:*, :mulf), (:/, :divf))
    @eval @inline function Base.Broadcast.broadcasted(::TileStyle, ::typeof($op), a::Tile{T,S1}, b::Tile{T,S2}) where {T<:AbstractFloat,S1,S2}
        S = broadcast_shape(S1, S2)
        Intrinsics.$intrinsic(broadcast_to(a, S), broadcast_to(b, S))
    end
end
# Integer operations (no division)
for (op, intrinsic) in ((:+, :addi), (:-, :subi), (:*, :muli))
    @eval @inline function Base.Broadcast.broadcasted(::TileStyle, ::typeof($op), a::Tile{T,S1}, b::Tile{T,S2}) where {T<:Integer,S1,S2}
        S = broadcast_shape(S1, S2)
        Intrinsics.$intrinsic(broadcast_to(a, S), broadcast_to(b, S))
    end
end
# Power (float only)
@inline function Base.Broadcast.broadcasted(::TileStyle, ::typeof(^), a::Tile{T,S1}, b::Tile{T,S2}) where {T<:AbstractFloat,S1,S2}
    S = broadcast_shape(S1, S2)
    Intrinsics.pow(broadcast_to(a, S), broadcast_to(b, S))
end

# Tile-Scalar arithmetic (tile .+ scalar, scalar .+ tile) - Float
# Scalar is broadcast to match tile shape before calling intrinsic
for (op, intrinsic) in ((:+, :addf), (:-, :subf), (:*, :mulf), (:/, :divf))
    @eval begin
        @inline Base.Broadcast.broadcasted(::TileStyle, ::typeof($op), a::Tile{T,S}, b::Number) where {T<:AbstractFloat,S} =
            Intrinsics.$intrinsic(a, broadcast_to(Tile(T(b)), S))
        @inline Base.Broadcast.broadcasted(::TileStyle, ::typeof($op), a::Number, b::Tile{T,S}) where {T<:AbstractFloat,S} =
            Intrinsics.$intrinsic(broadcast_to(Tile(T(a)), S), b)
    end
end
# Tile-Scalar arithmetic - Integer (no division)
for (op, intrinsic) in ((:+, :addi), (:-, :subi), (:*, :muli))
    @eval begin
        @inline Base.Broadcast.broadcasted(::TileStyle, ::typeof($op), a::Tile{T,S}, b::Number) where {T<:Integer,S} =
            Intrinsics.$intrinsic(a, broadcast_to(Tile(T(b)), S))
        @inline Base.Broadcast.broadcasted(::TileStyle, ::typeof($op), a::Number, b::Tile{T,S}) where {T<:Integer,S} =
            Intrinsics.$intrinsic(broadcast_to(Tile(T(a)), S), b)
    end
end
@inline Base.Broadcast.broadcasted(::TileStyle, ::typeof(^), a::Tile{T,S}, b::Number) where {T<:AbstractFloat,S} =
    Intrinsics.pow(a, broadcast_to(Tile(T(b)), S))
@inline Base.Broadcast.broadcasted(::TileStyle, ::typeof(^), a::Number, b::Tile{T,S}) where {T<:AbstractFloat,S} =
    Intrinsics.pow(broadcast_to(Tile(T(a)), S), b)

#=============================================================================
 Direct Scalar-Tile Operators (non-dotted syntax: tile * scalar)
=============================================================================#

# Scalar-tile operators (broadcast scalar to match tile shape)
@inline Base.:(*)(a::Tile{T, S}, b::Number) where {T <: AbstractFloat, S} = Intrinsics.mulf(a, broadcast_to(Tile(T(b)), S))
@inline Base.:(*)(a::Number, b::Tile{T, S}) where {T <: AbstractFloat, S} = Intrinsics.mulf(broadcast_to(Tile(T(a)), S), b)
@inline Base.:(/)(a::Tile{T, S}, b::Number) where {T <: AbstractFloat, S} = Intrinsics.divf(a, broadcast_to(Tile(T(b)), S))

#=============================================================================
 Comparison Operators
=============================================================================#

# Integer tile comparisons (same shape)
@inline Base.:(<)(a::Tile{T, S}, b::Tile{T, S}) where {T<:Integer, S} =
    Intrinsics.cmpi(a, b, CmpLessThan, SignednessSigned)
@inline Base.:(>)(a::Tile{T, S}, b::Tile{T, S}) where {T<:Integer, S} =
    Intrinsics.cmpi(a, b, CmpGreaterThan, SignednessSigned)
@inline Base.:(<=)(a::Tile{T, S}, b::Tile{T, S}) where {T<:Integer, S} =
    Intrinsics.cmpi(a, b, CmpLessThanOrEqual, SignednessSigned)
@inline Base.:(>=)(a::Tile{T, S}, b::Tile{T, S}) where {T<:Integer, S} =
    Intrinsics.cmpi(a, b, CmpGreaterThanOrEqual, SignednessSigned)
@inline Base.:(==)(a::Tile{T, S}, b::Tile{T, S}) where {T<:Integer, S} =
    Intrinsics.cmpi(a, b, CmpEqual, SignednessSigned)
@inline Base.:(!=)(a::Tile{T, S}, b::Tile{T, S}) where {T<:Integer, S} =
    Intrinsics.cmpi(a, b, CmpNotEqual, SignednessSigned)

# Float tile comparisons (same shape)
@inline Base.:(<)(a::Tile{T, S}, b::Tile{T, S}) where {T<:AbstractFloat, S} =
    Intrinsics.cmpf(a, b, CmpLessThan)
@inline Base.:(>)(a::Tile{T, S}, b::Tile{T, S}) where {T<:AbstractFloat, S} =
    Intrinsics.cmpf(a, b, CmpGreaterThan)
@inline Base.:(<=)(a::Tile{T, S}, b::Tile{T, S}) where {T<:AbstractFloat, S} =
    Intrinsics.cmpf(a, b, CmpLessThanOrEqual)
@inline Base.:(>=)(a::Tile{T, S}, b::Tile{T, S}) where {T<:AbstractFloat, S} =
    Intrinsics.cmpf(a, b, CmpGreaterThanOrEqual)
@inline Base.:(==)(a::Tile{T, S}, b::Tile{T, S}) where {T<:AbstractFloat, S} =
    Intrinsics.cmpf(a, b, CmpEqual)
@inline Base.:(!=)(a::Tile{T, S}, b::Tile{T, S}) where {T<:AbstractFloat, S} =
    Intrinsics.cmpf(a, b, CmpNotEqual)

# Broadcasting versions - different shapes
@inline Base.:(<)(a::Tile{T, S1}, b::Tile{T, S2}) where {T, S1, S2} =
    broadcast_to(a, broadcast_shape(S1, S2)) < broadcast_to(b, broadcast_shape(S1, S2))
@inline Base.:(>)(a::Tile{T, S1}, b::Tile{T, S2}) where {T, S1, S2} =
    broadcast_to(a, broadcast_shape(S1, S2)) > broadcast_to(b, broadcast_shape(S1, S2))
@inline Base.:(<=)(a::Tile{T, S1}, b::Tile{T, S2}) where {T, S1, S2} =
    broadcast_to(a, broadcast_shape(S1, S2)) <= broadcast_to(b, broadcast_shape(S1, S2))
@inline Base.:(>=)(a::Tile{T, S1}, b::Tile{T, S2}) where {T, S1, S2} =
    broadcast_to(a, broadcast_shape(S1, S2)) >= broadcast_to(b, broadcast_shape(S1, S2))
@inline Base.:(==)(a::Tile{T, S1}, b::Tile{T, S2}) where {T, S1, S2} =
    broadcast_to(a, broadcast_shape(S1, S2)) == broadcast_to(b, broadcast_shape(S1, S2))
@inline Base.:(!=)(a::Tile{T, S1}, b::Tile{T, S2}) where {T, S1, S2} =
    broadcast_to(a, broadcast_shape(S1, S2)) != broadcast_to(b, broadcast_shape(S1, S2))

# Mixed-type integer comparisons - promote to common type
@inline _promote_cmp(op, a::Tile{T1, S1}, b::Tile{T2, S2}, ::Type{T}, ::Val{S}) where {T1, T2, S1, S2, T, S} =
    op(astype(broadcast_to(a, S), T), astype(broadcast_to(b, S), T))

for op in (:<, :>, :<=, :>=, :(==), :(!=))
    @eval @inline function Base.$op(a::Tile{T1, S1}, b::Tile{T2, S2}) where {T1<:Integer, T2<:Integer, S1, S2}
        _promote_cmp($op, a, b, promote_type(T1, T2), Val(broadcast_shape(S1, S2)))
    end
end

#=============================================================================
 Logical Operations
=============================================================================#

"""
Element-wise logical AND for boolean tiles.
"""
@inline Base.:(&)(a::Tile{Bool, S}, b::Tile{Bool, S}) where {S} =
    Intrinsics.andi(a, b)

#=============================================================================
 Broadcasted Comparison Operators
=============================================================================#

# Tile-Tile comparisons (uses Base overloads with broadcasting)
for op in (:<, :>, :<=, :>=, :(==), :(!=))
    @eval @inline Base.Broadcast.broadcasted(::TileStyle, ::typeof($op), a::Tile{T,S1}, b::Tile{T,S2}) where {T,S1,S2} =
        $op(a, b)
end

# Mixed-type integer comparisons (delegate to Base overloads which handle promotion)
for op in (:<, :>, :<=, :>=, :(==), :(!=))
    @eval @inline Base.Broadcast.broadcasted(::TileStyle, ::typeof($op), a::Tile{T1,S1}, b::Tile{T2,S2}) where {T1<:Integer,T2<:Integer,S1,S2} =
        $op(a, b)
end

# Tile-Scalar comparisons (convert scalar to 0D tile, then broadcast)
for op in (:<, :>, :<=, :>=, :(==), :(!=))
    @eval begin
        @inline Base.Broadcast.broadcasted(::TileStyle, ::typeof($op), a::Tile{T,S}, b::Number) where {T,S} =
            $op(a, Tile(T(b)))
        @inline Base.Broadcast.broadcasted(::TileStyle, ::typeof($op), a::Number, b::Tile{T,S}) where {T,S} =
            $op(Tile(T(a)), b)
    end
end

#=============================================================================
 Broadcasted Math Functions
=============================================================================#

# Unary math functions - broadcast calls the intrinsic
for fn in (:exp, :exp2, :log, :log2, :sqrt)
    @eval @inline Base.Broadcast.broadcasted(::TileStyle, ::typeof($fn), a::Tile{T,S}) where {T<:AbstractFloat,S} =
        Intrinsics.$fn(a)
end

# rsqrt isn't in Base, so we define it and its broadcast handler
rsqrt(x::Number) = 1 / sqrt(x)
@inline Base.Broadcast.broadcasted(::TileStyle, ::typeof(rsqrt), a::Tile{T,S}) where {T<:AbstractFloat,S} =
    Intrinsics.rsqrt(a)
