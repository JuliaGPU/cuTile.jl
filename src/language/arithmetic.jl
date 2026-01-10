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
@overlay Base.mod1(x::T, y::T) where {T <: ScalarInt} = (m = mod(x, y); m == zero(m) ? y : m)

# float
@overlay Base.:+(x::T, y::T) where {T <: ScalarFloat} = Intrinsics.addf(x, y)
@overlay Base.:-(x::T, y::T) where {T <: ScalarFloat} = Intrinsics.subf(x, y)
@overlay Base.:*(x::T, y::T) where {T <: ScalarFloat} = Intrinsics.mulf(x, y)
@overlay Base.:/(x::T, y::T) where {T <: ScalarFloat} = Intrinsics.divf(x, y)
@overlay Base.:-(x::ScalarFloat) = Intrinsics.negf(x)

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

# bitwise
@overlay Base.:&(x::T, y::T) where {T <: ScalarInt} = Intrinsics.andi(x, y)
@overlay Base.:|(x::T, y::T) where {T <: ScalarInt} = Intrinsics.ori(x, y)
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

# broadcasted arithmetic (float)
for (op, intrinsic) in ((:+, :addf), (:-, :subf), (:*, :mulf), (:/, :divf), (:max, :maxf), (:min, :minf))
    @eval @inline function Base.Broadcast.broadcasted(::TileStyle, ::typeof($op), a::Tile{T,S1}, b::Tile{T,S2}) where {T<:AbstractFloat,S1,S2}
        S = broadcast_shape(S1, S2)
        Intrinsics.$intrinsic(broadcast_to(a, S), broadcast_to(b, S))
    end
end

# broadcasted arithmetic (integer)
for (op, intrinsic) in ((:+, :addi), (:-, :subi), (:*, :muli))
    @eval @inline function Base.Broadcast.broadcasted(::TileStyle, ::typeof($op), a::Tile{T,S1}, b::Tile{T,S2}) where {T<:Integer,S1,S2}
        S = broadcast_shape(S1, S2)
        Intrinsics.$intrinsic(broadcast_to(a, S), broadcast_to(b, S))
    end
end

# broadcasted power
@inline function Base.Broadcast.broadcasted(::TileStyle, ::typeof(^), a::Tile{T,S1}, b::Tile{T,S2}) where {T<:AbstractFloat,S1,S2}
    S = broadcast_shape(S1, S2)
    Intrinsics.pow(broadcast_to(a, S), broadcast_to(b, S))
end

# broadcasted comparison
@inline _cmp_intrinsic(a::Tile{T,S}, b::Tile{T,S}, pred) where {T<:Integer, S} =
    Intrinsics.cmpi(a, b, pred, SignednessSigned)
@inline _cmp_intrinsic(a::Tile{T,S}, b::Tile{T,S}, pred) where {T<:AbstractFloat, S} =
    Intrinsics.cmpf(a, b, pred)

for (op, pred) in ((:<, :CmpLessThan), (:>, :CmpGreaterThan),
                   (:<=, :CmpLessThanOrEqual), (:>=, :CmpGreaterThanOrEqual),
                   (:(==), :CmpEqual), (:(!=), :CmpNotEqual))
    @eval @inline function Base.Broadcast.broadcasted(::TileStyle, ::typeof($op), a::Tile{T,S1}, b::Tile{T,S2}) where {T,S1,S2}
        S = broadcast_shape(S1, S2)
        _cmp_intrinsic(broadcast_to(a, S), broadcast_to(b, S), $pred)
    end
end

# mixed-type integer comparison (promote to common type)
for (op, pred) in ((:<, :CmpLessThan), (:>, :CmpGreaterThan),
                   (:<=, :CmpLessThanOrEqual), (:>=, :CmpGreaterThanOrEqual),
                   (:(==), :CmpEqual), (:(!=), :CmpNotEqual))
    @eval @inline function Base.Broadcast.broadcasted(::TileStyle, ::typeof($op), a::Tile{T1,S1}, b::Tile{T2,S2}) where {T1<:Integer,T2<:Integer,S1,S2}
        T = promote_type(T1, T2)
        S = broadcast_shape(S1, S2)
        _cmp_intrinsic(astype(broadcast_to(a, S), T), astype(broadcast_to(b, S), T), $pred)
    end
end

# broadcasted logical
@inline Base.Broadcast.broadcasted(::TileStyle, ::typeof(&), a::Tile{Bool,S1}, b::Tile{Bool,S2}) where {S1,S2} =
    Intrinsics.andi(broadcast_to(a, broadcast_shape(S1, S2)), broadcast_to(b, broadcast_shape(S1, S2)))

# mul_hi (high bits of integer multiply)
# Base.mul_hi added in Julia 1.13; before that, use ct.mul_hi
# Like cld/fld, requires broadcast for element-wise tile application: mul_hi.(a, b)
@static if VERSION >= v"1.13-"
    # Broadcasted for tiles (scalar uses Base.mul_hi directly):
    @inline Base.Broadcast.broadcasted(::TileStyle, ::typeof(Base.mul_hi), a::Tile{T,S1}, b::Tile{T,S2}) where {T<:Signed,S1,S2} =
        Intrinsics.mulhii(broadcast_to(a, broadcast_shape(S1, S2)), broadcast_to(b, broadcast_shape(S1, S2)), SignednessSigned)
    @inline Base.Broadcast.broadcasted(::TileStyle, ::typeof(Base.mul_hi), a::Tile{T,S1}, b::Tile{T,S2}) where {T<:Unsigned,S1,S2} =
        Intrinsics.mulhii(broadcast_to(a, broadcast_shape(S1, S2)), broadcast_to(b, broadcast_shape(S1, S2)), SignednessUnsigned)
else
    # Scalar definition (uses intrinsic for compilation to Tile IR)
    @inline mul_hi(x::T, y::T) where {T <: Signed} = Intrinsics.mulhii(x, y, SignednessSigned)
    @inline mul_hi(x::T, y::T) where {T <: Unsigned} = Intrinsics.mulhii(x, y, SignednessUnsigned)
    # Broadcasted for tiles:
    @inline Base.Broadcast.broadcasted(::TileStyle, ::typeof(mul_hi), a::Tile{T,S1}, b::Tile{T,S2}) where {T<:Signed,S1,S2} =
        Intrinsics.mulhii(broadcast_to(a, broadcast_shape(S1, S2)), broadcast_to(b, broadcast_shape(S1, S2)), SignednessSigned)
    @inline Base.Broadcast.broadcasted(::TileStyle, ::typeof(mul_hi), a::Tile{T,S1}, b::Tile{T,S2}) where {T<:Unsigned,S1,S2} =
        Intrinsics.mulhii(broadcast_to(a, broadcast_shape(S1, S2)), broadcast_to(b, broadcast_shape(S1, S2)), SignednessUnsigned)
end


## mixed arithmetic

# direct operators (tile * scalar, tile / scalar)
@inline Base.:(*)(a::Tile{T, S}, b::Number) where {T <: AbstractFloat, S} = Intrinsics.mulf(a, broadcast_to(Tile(T(b)), S))
@inline Base.:(*)(a::Number, b::Tile{T, S}) where {T <: AbstractFloat, S} = Intrinsics.mulf(broadcast_to(Tile(T(a)), S), b)
@inline Base.:(/)(a::Tile{T, S}, b::Number) where {T <: AbstractFloat, S} = Intrinsics.divf(a, broadcast_to(Tile(T(b)), S))

# broadcasted arithmetic (float)
for (op, intrinsic) in ((:+, :addf), (:-, :subf), (:*, :mulf), (:/, :divf), (:max, :maxf), (:min, :minf))
    @eval begin
        @inline Base.Broadcast.broadcasted(::TileStyle, ::typeof($op), a::Tile{T,S}, b::Number) where {T<:AbstractFloat,S} =
            Intrinsics.$intrinsic(a, broadcast_to(Tile(T(b)), S))
        @inline Base.Broadcast.broadcasted(::TileStyle, ::typeof($op), a::Number, b::Tile{T,S}) where {T<:AbstractFloat,S} =
            Intrinsics.$intrinsic(broadcast_to(Tile(T(a)), S), b)
    end
end

# broadcasted arithmetic (integer)
for (op, intrinsic) in ((:+, :addi), (:-, :subi), (:*, :muli))
    @eval begin
        @inline Base.Broadcast.broadcasted(::TileStyle, ::typeof($op), a::Tile{T,S}, b::Number) where {T<:Integer,S} =
            Intrinsics.$intrinsic(a, broadcast_to(Tile(T(b)), S))
        @inline Base.Broadcast.broadcasted(::TileStyle, ::typeof($op), a::Number, b::Tile{T,S}) where {T<:Integer,S} =
            Intrinsics.$intrinsic(broadcast_to(Tile(T(a)), S), b)
    end
end

# broadcasted power
@inline Base.Broadcast.broadcasted(::TileStyle, ::typeof(^), a::Tile{T,S}, b::Number) where {T<:AbstractFloat,S} =
    Intrinsics.pow(a, broadcast_to(Tile(T(b)), S))
@inline Base.Broadcast.broadcasted(::TileStyle, ::typeof(^), a::Number, b::Tile{T,S}) where {T<:AbstractFloat,S} =
    Intrinsics.pow(broadcast_to(Tile(T(a)), S), b)

# broadcasted comparison
for (op, pred) in ((:<, :CmpLessThan), (:>, :CmpGreaterThan),
                   (:<=, :CmpLessThanOrEqual), (:>=, :CmpGreaterThanOrEqual),
                   (:(==), :CmpEqual), (:(!=), :CmpNotEqual))
    @eval begin
        @inline Base.Broadcast.broadcasted(::TileStyle, ::typeof($op), a::Tile{T,S}, b::Number) where {T,S} =
            _cmp_intrinsic(a, broadcast_to(Tile(T(b)), S), $pred)
        @inline Base.Broadcast.broadcasted(::TileStyle, ::typeof($op), a::Number, b::Tile{T,S}) where {T,S} =
            _cmp_intrinsic(broadcast_to(Tile(T(a)), S), b, $pred)
    end
end
