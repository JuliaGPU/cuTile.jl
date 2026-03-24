# Arithmetic operations


## scalar arithmetic

# NOTE: some integer arithmetic operations are NOT overlaid because
#       the IRStructurizer needs to see them to convert `while` loops into `for` loops.

# integer
#@overlay Base.:+(x::T, y::T) where {T <: ScalarInt} = Intrinsics.addi(x, y)
@overlay Base.:-(x::T, y::T) where {T <: ScalarInt} = Intrinsics.subi(x, y)
@overlay Base.:*(x::T, y::T) where {T <: ScalarInt} = Intrinsics.muli(x, y)
@overlay Base.:-(x::ScalarInt) = Intrinsics.negi(x)
# div with default rounding (toward zero)
@overlay Base.div(x::T, y::T) where {T <: Signed} = Intrinsics.divi(x, y, Signedness.Signed)
@overlay Base.div(x::T, y::T) where {T <: Unsigned} = Intrinsics.divi(x, y, Signedness.Unsigned)

# div with explicit RoundToZero
@overlay Base.div(x::T, y::T, ::typeof(RoundToZero)) where {T <: Signed} = Intrinsics.divi(x, y, Signedness.Signed)
@overlay Base.div(x::T, y::T, ::typeof(RoundToZero)) where {T <: Unsigned} = Intrinsics.divi(x, y, Signedness.Unsigned)

# fld uses div with RoundDown
# Note: for unsigned, floor division equals truncating division (values are non-negative)
@overlay Base.div(x::T, y::T, ::typeof(RoundDown)) where {T <: Signed} = Intrinsics.fldi(x, y, Signedness.Signed)
@overlay Base.div(x::T, y::T, ::typeof(RoundDown)) where {T <: Unsigned} = Intrinsics.divi(x, y, Signedness.Unsigned)

# cld uses div with RoundUp
@overlay Base.div(x::T, y::T, ::typeof(RoundUp)) where {T <: Signed} = Intrinsics.cldi(x, y, Signedness.Signed)
@overlay Base.div(x::T, y::T, ::typeof(RoundUp)) where {T <: Unsigned} = Intrinsics.cldi(x, y, Signedness.Unsigned)
@overlay Base.rem(x::T, y::T) where {T <: Signed} = Intrinsics.remi(x, y, Signedness.Signed)
@overlay Base.rem(x::T, y::T) where {T <: Unsigned} = Intrinsics.remi(x, y, Signedness.Unsigned)

# float
@overlay Base.:+(x::T, y::T) where {T <: ScalarFloat} = Intrinsics.addf(x, y)
@overlay Base.:-(x::T, y::T) where {T <: ScalarFloat} = Intrinsics.subf(x, y)
@overlay Base.:*(x::T, y::T) where {T <: ScalarFloat} = Intrinsics.mulf(x, y)
@overlay Base.:/(x::T, y::T) where {T <: ScalarFloat} = Intrinsics.divf(x, y)
@overlay Base.:-(x::ScalarFloat) = Intrinsics.negf(x)
@overlay Base.:^(x::T, y::T) where {T <: ScalarFloat} = Intrinsics.pow(x, y)

# comparison (integer)
@overlay Base.:(==)(x::T, y::T) where {T <: ScalarInt} = Intrinsics.cmpi(x, y, ComparisonPredicate.Equal, Signedness.Signed)
@overlay Base.:(!=)(x::T, y::T) where {T <: ScalarInt} = Intrinsics.cmpi(x, y, ComparisonPredicate.NotEqual, Signedness.Signed)
#@overlay Base.:<(x::T, y::T) where {T <: Signed} = Intrinsics.cmpi(x, y, ComparisonPredicate.LessThan, Signedness.Signed)
#@overlay Base.:<(x::T, y::T) where {T <: Unsigned} = Intrinsics.cmpi(x, y, ComparisonPredicate.LessThan, Signedness.Unsigned)
#@overlay Base.:<=(x::T, y::T) where {T <: Signed} = Intrinsics.cmpi(x, y, ComparisonPredicate.LessThanOrEqual, Signedness.Signed)
#@overlay Base.:<=(x::T, y::T) where {T <: Unsigned} = Intrinsics.cmpi(x, y, ComparisonPredicate.LessThanOrEqual, Signedness.Unsigned)
@overlay Base.:>(x::T, y::T) where {T <: Signed} = Intrinsics.cmpi(y, x, ComparisonPredicate.LessThan, Signedness.Signed)
@overlay Base.:>(x::T, y::T) where {T <: Unsigned} = Intrinsics.cmpi(y, x, ComparisonPredicate.LessThan, Signedness.Unsigned)
@overlay Base.:>=(x::T, y::T) where {T <: Signed} = Intrinsics.cmpi(y, x, ComparisonPredicate.LessThanOrEqual, Signedness.Signed)
@overlay Base.:>=(x::T, y::T) where {T <: Unsigned} = Intrinsics.cmpi(y, x, ComparisonPredicate.LessThanOrEqual, Signedness.Unsigned)

# comparison (float)
@overlay Base.:<(x::T, y::T) where {T <: ScalarFloat} = Intrinsics.cmpf(x, y, ComparisonPredicate.LessThan)
@overlay Base.:<=(x::T, y::T) where {T <: ScalarFloat} = Intrinsics.cmpf(x, y, ComparisonPredicate.LessThanOrEqual)
@overlay Base.:>(x::T, y::T) where {T <: ScalarFloat} = Intrinsics.cmpf(x, y, ComparisonPredicate.GreaterThan)
@overlay Base.:>=(x::T, y::T) where {T <: ScalarFloat} = Intrinsics.cmpf(x, y, ComparisonPredicate.GreaterThanOrEqual)
@overlay Base.:(==)(x::T, y::T) where {T <: ScalarFloat} = Intrinsics.cmpf(x, y, ComparisonPredicate.Equal)
@overlay Base.:(!=)(x::T, y::T) where {T <: ScalarFloat} = Intrinsics.cmpf(x, y, ComparisonPredicate.NotEqual)

@overlay Base.ifelse(cond::Bool, x::T, y::T) where {T} = Intrinsics.select(cond, x, y)

# bitwise
@overlay Base.:&(x::T, y::T) where {T <: ScalarInt} = Intrinsics.andi(x, y)
@overlay Base.:|(x::T, y::T) where {T <: ScalarInt} = Intrinsics.ori(x, y)
@overlay Base.:&(x::Bool, y::Bool) = Intrinsics.andi(x, y)
@overlay Base.:|(x::Bool, y::Bool) = Intrinsics.ori(x, y)
@overlay Base.xor(x::T, y::T) where {T <: ScalarInt} = Intrinsics.xori(x, y)
@overlay Base.:~(x::T) where {T <: Signed} = Intrinsics.xori(x, T(-1))
@overlay Base.:~(x::T) where {T <: Unsigned} = Intrinsics.xori(x, ~T(0))
@overlay Base.:!(x::Bool) = Intrinsics.xori(x, true)
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

# mul_hi (high bits of integer multiply)
# Base.mul_hi added in Julia 1.13; before that, use ct.mul_hi
# Scalar overlays let the generic copy→map path handle tile broadcasting.
@static if VERSION >= v"1.13-"
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
