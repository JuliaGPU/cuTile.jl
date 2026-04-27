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

# float power (expands to dozens of intrinsics in Julia — complex)
@overlay Base.:^(x::T, y::T) where {T <: ScalarFloat} = Intrinsics.pow(x, y)

# float-by-integer power. Base's `^(::Float, ::Int)` calls `power_by_squaring`,
# whose `trailing_zeros` loop emits `cttz_int` (no Tile IR equivalent) for
# runtime exponents and refuses to const-fold for literal exponents on 1.11.
@overlay function Base.:^(x::Float32, y::Int64)
    y == -1 && return inv(x)
    y == 0 && return one(x)
    y == 1 && return x
    y == 2 && return x*x
    y == 3 && return x*x*x
    x ^ Float32(y)
end
@overlay function Base.:^(x::Float64, y::Int64)
    y == -1 && return inv(x)
    y == 0 && return one(x)
    y == 1 && return x
    y == 2 && return x*x
    y == 3 && return x*x*x
    x ^ Float64(y)
end

# integer != (Julia expands to not_int(===) — 2 ops; overlay gives 1 op)
@overlay Base.:(!=)(x::T, y::T) where {T <: ScalarInt} = Intrinsics.cmpi(x, y, ComparisonPredicate.NotEqual, Signedness.Signed)

# shifts (Julia's << includes range checking, bitcast, branching — complex)
@overlay Base.:<<(x::ScalarInt, y::Integer) = Intrinsics.shli(x, y)
@overlay Base.:>>(x::Signed, y::Integer) = Intrinsics.shri(x, y, Signedness.Signed)
@overlay Base.:>>(x::Unsigned, y::Integer) = Intrinsics.shri(x, y, Signedness.Unsigned)
@overlay Base.:>>>(x::ScalarInt, y::Integer) = Intrinsics.shri(x, y, Signedness.Unsigned)


## tile arithmetic

# direct operators (same shape required)
@inline Base.:(+)(a::Tile{T, S}, b::Tile{T, S}) where {T <: AbstractFloat, S} = Intrinsics.addf(a, b)
@inline Base.:(+)(a::Tile{T, S}, b::Tile{T, S}) where {T <: Integer, S} = Intrinsics.addi(a, b)
@inline Base.:(-)(a::Tile{T, S}, b::Tile{T, S}) where {T <: AbstractFloat, S} = Intrinsics.subf(a, b)
@inline Base.:(-)(a::Tile{T, S}, b::Tile{T, S}) where {T <: Integer, S} = Intrinsics.subi(a, b)

# All other tile arithmetic (*, -, /, ^, comparisons, ifelse, etc.) is handled
# by the generic Broadcast.copy → map path: scalar @overlay methods or Julia's
# native implementations provide the element-wise logic, and map handles
# broadcasting + to_scalar/from_scalar wrapping.

# mul_hi (high bits of integer multiply — no Core.Intrinsic equivalent)
@static if VERSION >= v"1.13-"
    using Base: mul_hi
    @overlay Base.mul_hi(x::T, y::T) where {T <: Signed} = Intrinsics.mulhii(x, y, Signedness.Signed)
    @overlay Base.mul_hi(x::T, y::T) where {T <: Unsigned} = Intrinsics.mulhii(x, y, Signedness.Unsigned)
else
    @inline mul_hi(x::T, y::T) where {T <: Signed} = Intrinsics.mulhii(x, y, Signedness.Signed)
    @inline mul_hi(x::T, y::T) where {T <: Unsigned} = Intrinsics.mulhii(x, y, Signedness.Unsigned)
end


## mixed arithmetic

# direct operators (tile * scalar, tile / scalar)
@inline Base.:(*)(a::Tile{T}, b::Number) where {T <: AbstractFloat} = Intrinsics.mulf(a, broadcast_to(Tile(T(b)), size(a)))
@inline Base.:(*)(a::Number, b::Tile{T}) where {T <: AbstractFloat} = Intrinsics.mulf(broadcast_to(Tile(T(a)), size(b)), b)
@inline Base.:(/)(a::Tile{T}, b::Number) where {T <: AbstractFloat} = Intrinsics.divf(a, broadcast_to(Tile(T(b)), size(a)))
