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
@overlay Base.div(x::T, y::T) where {T <: Signed} = Intrinsics.divi(x, y, SignednessSigned)
@overlay Base.div(x::T, y::T) where {T <: Unsigned} = Intrinsics.divi(x, y, SignednessUnsigned)

# div with explicit RoundToZero
@overlay Base.div(x::T, y::T, ::typeof(RoundToZero)) where {T <: Signed} = Intrinsics.divi(x, y, SignednessSigned)
@overlay Base.div(x::T, y::T, ::typeof(RoundToZero)) where {T <: Unsigned} = Intrinsics.divi(x, y, SignednessUnsigned)

# fld uses div with RoundDown
# Note: for unsigned, floor division equals truncating division (values are non-negative)
@overlay Base.div(x::T, y::T, ::typeof(RoundDown)) where {T <: Signed} = Intrinsics.fldi(x, y, SignednessSigned)
@overlay Base.div(x::T, y::T, ::typeof(RoundDown)) where {T <: Unsigned} = Intrinsics.divi(x, y, SignednessUnsigned)

# cld uses div with RoundUp
@overlay Base.div(x::T, y::T, ::typeof(RoundUp)) where {T <: Signed} = Intrinsics.cldi(x, y, SignednessSigned)
@overlay Base.div(x::T, y::T, ::typeof(RoundUp)) where {T <: Unsigned} = Intrinsics.cldi(x, y, SignednessUnsigned)
@overlay Base.rem(x::T, y::T) where {T <: Signed} = Intrinsics.remi(x, y, SignednessSigned)
@overlay Base.rem(x::T, y::T) where {T <: Unsigned} = Intrinsics.remi(x, y, SignednessUnsigned)

# float
@overlay Base.:+(x::T, y::T) where {T <: ScalarFloat} = Intrinsics.addf(x, y)
@overlay Base.:-(x::T, y::T) where {T <: ScalarFloat} = Intrinsics.subf(x, y)
@overlay Base.:*(x::T, y::T) where {T <: ScalarFloat} = Intrinsics.mulf(x, y)
@overlay Base.:/(x::T, y::T) where {T <: ScalarFloat} = Intrinsics.divf(x, y)
@overlay Base.:-(x::ScalarFloat) = Intrinsics.negf(x)
@overlay Base.:^(x::T, y::T) where {T <: ScalarFloat} = Intrinsics.pow(x, y)

# comparison (integer)
@overlay Base.:(==)(x::T, y::T) where {T <: ScalarInt} = Intrinsics.cmpi(x, y, CmpEqual, SignednessSigned)
@overlay Base.:(!=)(x::T, y::T) where {T <: ScalarInt} = Intrinsics.cmpi(x, y, CmpNotEqual, SignednessSigned)
#@overlay Base.:<(x::T, y::T) where {T <: Signed} = Intrinsics.cmpi(x, y, CmpLessThan, SignednessSigned)
#@overlay Base.:<(x::T, y::T) where {T <: Unsigned} = Intrinsics.cmpi(x, y, CmpLessThan, SignednessUnsigned)
#@overlay Base.:<=(x::T, y::T) where {T <: Signed} = Intrinsics.cmpi(x, y, CmpLessThanOrEqual, SignednessSigned)
#@overlay Base.:<=(x::T, y::T) where {T <: Unsigned} = Intrinsics.cmpi(x, y, CmpLessThanOrEqual, SignednessUnsigned)
@overlay Base.:>(x::T, y::T) where {T <: Signed} = Intrinsics.cmpi(y, x, CmpLessThan, SignednessSigned)
@overlay Base.:>(x::T, y::T) where {T <: Unsigned} = Intrinsics.cmpi(y, x, CmpLessThan, SignednessUnsigned)
@overlay Base.:>=(x::T, y::T) where {T <: Signed} = Intrinsics.cmpi(y, x, CmpLessThanOrEqual, SignednessSigned)
@overlay Base.:>=(x::T, y::T) where {T <: Unsigned} = Intrinsics.cmpi(y, x, CmpLessThanOrEqual, SignednessUnsigned)

# comparison (float)
@overlay Base.:<(x::T, y::T) where {T <: ScalarFloat} = Intrinsics.cmpf(x, y, CmpLessThan)
@overlay Base.:<=(x::T, y::T) where {T <: ScalarFloat} = Intrinsics.cmpf(x, y, CmpLessThanOrEqual)
@overlay Base.:>(x::T, y::T) where {T <: ScalarFloat} = Intrinsics.cmpf(x, y, CmpGreaterThan)
@overlay Base.:>=(x::T, y::T) where {T <: ScalarFloat} = Intrinsics.cmpf(x, y, CmpGreaterThanOrEqual)
@overlay Base.:(==)(x::T, y::T) where {T <: ScalarFloat} = Intrinsics.cmpf(x, y, CmpEqual)
@overlay Base.:(!=)(x::T, y::T) where {T <: ScalarFloat} = Intrinsics.cmpf(x, y, CmpNotEqual)

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
@overlay Base.:>>(x::Signed, y::Integer) = Intrinsics.shri(x, y, SignednessSigned)
@overlay Base.:>>(x::Unsigned, y::Integer) = Intrinsics.shri(x, y, SignednessUnsigned)
@overlay Base.:>>>(x::ScalarInt, y::Integer) = Intrinsics.shri(x, y, SignednessUnsigned)


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
    @overlay Base.mul_hi(x::T, y::T) where {T <: Signed} = Intrinsics.mulhii(x, y, SignednessSigned)
    @overlay Base.mul_hi(x::T, y::T) where {T <: Unsigned} = Intrinsics.mulhii(x, y, SignednessUnsigned)
else
    @inline mul_hi(x::T, y::T) where {T <: Signed} = Intrinsics.mulhii(x, y, SignednessSigned)
    @inline mul_hi(x::T, y::T) where {T <: Unsigned} = Intrinsics.mulhii(x, y, SignednessUnsigned)
end


## mixed arithmetic

# direct operators (tile * scalar, tile / scalar)
@inline Base.:(*)(a::Tile{T}, b::Number) where {T <: AbstractFloat} = Intrinsics.mulf(a, broadcast_to(Tile(T(b)), size(a)))
@inline Base.:(*)(a::Number, b::Tile{T}) where {T <: AbstractFloat} = Intrinsics.mulf(broadcast_to(Tile(T(a)), size(b)), b)
@inline Base.:(/)(a::Tile{T}, b::Number) where {T <: AbstractFloat} = Intrinsics.divf(a, broadcast_to(Tile(T(b)), size(a)))
