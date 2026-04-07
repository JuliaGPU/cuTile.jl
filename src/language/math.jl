# Math operations

public rsqrt, exp2


## scalar math

# unary
rsqrt(x::T) where {T <: AbstractFloat} = Intrinsics.rsqrt(x)
for fn in (:ceil, :floor, :exp, :exp2, :log, :log2, :sqrt,
           :sin, :cos, :tan, :sinh, :cosh, :tanh)
    @eval @overlay Base.$fn(x::T) where {T <: ScalarFloat} = Intrinsics.$fn(x)
end

@overlay Base.abs(x::T) where {T <: ScalarFloat} = Intrinsics.absf(x)
@overlay Base.abs(x::T) where {T <: Signed} = Intrinsics.absi(x)

@overlay Base.isnan(x::T) where {T <: ScalarFloat} = Intrinsics.cmpf(x, x, ComparisonPredicate.NotEqual, ComparisonOrdering.Unordered)

@overlay Base.rem(x::T, y::T) where {T <: ScalarFloat} = Intrinsics.remf(x, y)

@overlay Base.fma(x::T, y::T, z::T) where {T <: ScalarFloat} = Intrinsics.fma(x, y, z)

# max/min
@overlay Base.max(x::T, y::T) where {T <: Signed} = Intrinsics.maxi(x, y, Signedness.Signed)
@overlay Base.max(x::T, y::T) where {T <: Unsigned} = Intrinsics.maxi(x, y, Signedness.Unsigned)
@overlay Base.min(x::T, y::T) where {T <: Signed} = Intrinsics.mini(x, y, Signedness.Signed)
@overlay Base.min(x::T, y::T) where {T <: Unsigned} = Intrinsics.mini(x, y, Signedness.Unsigned)
@overlay Base.max(x::T, y::T) where {T <: ScalarFloat} = Intrinsics.maxf(x, y)
@overlay Base.min(x::T, y::T) where {T <: ScalarFloat} = Intrinsics.minf(x, y)


## tile math

# All tile broadcasting (unary, binary, ternary) is handled by the generic
# Broadcast.copy → map path: scalar @overlay methods provide the element-wise
# implementation, and map handles broadcasting + to_scalar/from_scalar.
# No explicit broadcasted methods needed here.
