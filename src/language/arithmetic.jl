# Arithmetic operations


## scalar arithmetic

# Most scalar arithmetic operations are NOT overlaid — Julia's type inference
# inlines Base functions down to Core.Intrinsics (e.g., Base.:-(x::Int32, y::Int32)
# → Core.Intrinsics.sub_int(x, y)), and the normalize pass converts those to
# cuTile Intrinsics after structurization.
#
# Overlays are only needed for operations where Julia's implementation is complex
# (expands to many intrinsics, branches, or function calls) and we need to replace
# the entire tree with a single cuTile Intrinsic.

# integer division (checked_sdiv_int / checked_srem_int in Julia — complex)
@overlay Base.div(x::T, y::T) where {T <: Signed} = Intrinsics.divi(x, y, Signedness.Signed)
@overlay Base.div(x::T, y::T) where {T <: Unsigned} = Intrinsics.divi(x, y, Signedness.Unsigned)
@overlay Base.div(x::T, y::T, ::typeof(RoundToZero)) where {T <: Signed} = Intrinsics.divi(x, y, Signedness.Signed)
@overlay Base.div(x::T, y::T, ::typeof(RoundToZero)) where {T <: Unsigned} = Intrinsics.divi(x, y, Signedness.Unsigned)
@overlay Base.div(x::T, y::T, ::typeof(RoundDown)) where {T <: Signed} = Intrinsics.fldi(x, y, Signedness.Signed)
@overlay Base.div(x::T, y::T, ::typeof(RoundDown)) where {T <: Unsigned} = Intrinsics.divi(x, y, Signedness.Unsigned)
@overlay Base.div(x::T, y::T, ::typeof(RoundUp)) where {T <: Signed} = Intrinsics.cldi(x, y, Signedness.Signed)
@overlay Base.div(x::T, y::T, ::typeof(RoundUp)) where {T <: Unsigned} = Intrinsics.cldi(x, y, Signedness.Unsigned)
@overlay Base.rem(x::T, y::T) where {T <: Signed} = Intrinsics.remi(x, y, Signedness.Signed)
@overlay Base.rem(x::T, y::T) where {T <: Unsigned} = Intrinsics.remi(x, y, Signedness.Unsigned)

for T in (:ScalarInt, :ScalarFloat)
    @eval @overlay function Base.mod(x::$T, y::$T)
        value = rem(x, y)
        zero_value = zero(x)
        needs_fix = ((value < zero_value) != (y < zero_value)) & (value != zero_value)
        ifelse(needs_fix, value + y, value)
    end
end

# floor division on floats — `fld(x, y)` and `div(x, y, RoundDown)`
@overlay Base.div(x::T, y::T, ::typeof(RoundDown)) where {T <: ScalarFloat} = Intrinsics.floor(Intrinsics.divf(x, y))

# float power (expands to dozens of intrinsics in Julia — complex)
@overlay Base.:^(x::T, y::T) where {T <: ScalarFloat} = Intrinsics.pow(x, y)

# Use `pow` for accuracy and restore the sign from the integer exponent.
@overlay function Base.:^(x::T, y::ScalarInt) where {T <: ScalarFloat}
    magnitude = Intrinsics.pow(abs(x), T(y))
    odd = (y & one(y)) != zero(y)
    ifelse(signbit(x) & odd, -magnitude, magnitude)
end

# integer != (Julia expands to not_int(===) — 2 ops; overlay gives 1 op)
@overlay Base.:(!=)(x::T, y::T) where {T <: ScalarInt} = Intrinsics.cmpi(x, y, ComparisonPredicate.NotEqual, Signedness.Signed)

# float != with NaN-correct semantics: Julia's `!=` on IEEEFloats lowers via
# `ne_float` (canonicalized to `cmpf(NotEqual, Unordered)`), but non-IEEEFloat
# scalars (BFloat16, TFloat32) take detours that can lose `Unordered`. Force
# the unordered predicate uniformly so `NaN != NaN` returns `true`.
@overlay Base.:(!=)(x::T, y::T) where {T <: ScalarFloat} = Intrinsics.cmpf(x, y, ComparisonPredicate.NotEqual, ComparisonOrdering.Unordered)

# shifts (Julia's << includes range checking, bitcast, branching — complex)
@overlay Base.:<<(x::ScalarInt, y::Integer) = Intrinsics.shli(x, y)
@overlay Base.:>>(x::Signed, y::Integer) = Intrinsics.shri(x, y, Signedness.Signed)
@overlay Base.:>>(x::Unsigned, y::Integer) = Intrinsics.shri(x, y, Signedness.Unsigned)
@overlay Base.:>>>(x::ScalarInt, y::Integer) = Intrinsics.shri(x, y, Signedness.Unsigned)


## tile arithmetic

public divmod

"""
    divmod(x, y) -> (q, r)

Floored quotient and remainder, i.e. `(fld(x, y), mod(x, y))`. Accepts integer
scalars or integer tiles of matching shape, and follows Julia's sign
conventions: the remainder takes the sign of the divisor.
"""
@inline divmod(x::T, y::T) where {T<:Integer} = (div(x, y, RoundDown), mod(x, y))
@inline divmod(x::Tile{T,S}, y::Tile{T,S}) where {T<:Integer, S} =
    (div.(x, y, RoundDown), mod.(x, y))

# direct operators (same shape required)
@inline Base.:(+)(a::Tile{T, S}, b::Tile{T, S}) where {T <: AbstractFloat, S} = Intrinsics.addf(a, b)
@inline Base.:(+)(a::Tile{T, S}, b::Tile{T, S}) where {T <: Integer, S} = Intrinsics.addi(a, b)
@inline Base.:(-)(a::Tile{T, S}, b::Tile{T, S}) where {T <: AbstractFloat, S} = Intrinsics.subf(a, b)
@inline Base.:(-)(a::Tile{T, S}, b::Tile{T, S}) where {T <: Integer, S} = Intrinsics.subi(a, b)

@inline Base.:(-)(a::Tile{T}) where {T <: AbstractFloat} = Intrinsics.negf(a)
@inline Base.:(-)(a::Tile{T}) where {T <: Integer} = Intrinsics.negi(a)

# All other tile arithmetic (*, -, /, ^, comparisons, ifelse, etc.) is handled
# by the generic Broadcast.copy → map path: scalar @overlay methods or Julia's
# native implementations provide the element-wise logic, and map handles
# broadcasting + to_scalar/from_scalar wrapping.

# mul_hi (high bits of integer multiply — no Core.Intrinsic equivalent).
# Tile IR's `mulhii` is unsigned-only; signed inputs are rejected at codegen.
@static if VERSION >= v"1.13-"
    using Base: mul_hi
    @overlay Base.mul_hi(x::T, y::T) where {T <: Integer} = Intrinsics.mulhii(x, y)
else
    @inline mul_hi(x::T, y::T) where {T <: Integer} = Intrinsics.mulhii(x, y)
end


## mixed arithmetic

# direct operators (tile * scalar, tile / scalar)
@inline Base.:(*)(a::Tile{T}, b::Number) where {T <: AbstractFloat} = Intrinsics.mulf(a, broadcast_to(Tile(T(b)), size(a)))
@inline Base.:(*)(a::Number, b::Tile{T}) where {T <: AbstractFloat} = Intrinsics.mulf(broadcast_to(Tile(T(a)), size(b)), b)
@inline Base.:(/)(a::Tile{T}, b::Number) where {T <: AbstractFloat} = Intrinsics.divf(a, broadcast_to(Tile(T(b)), size(a)))
