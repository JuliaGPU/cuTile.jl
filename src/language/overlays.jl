# Method Overlay Infrastructure for cuTile Compilation
#
# Defines the @overlay macro and type conversion overlays.
# Arithmetic and math overlays are defined in arithmetic.jl and math.jl.

macro overlay(ex)
    esc(:(Base.Experimental.@consistent_overlay cuTileMethodTable Base.@assume_effects :foldable $ex))
end


#=============================================================================
 Type Conversions
=============================================================================#

# Type tuples for metaprogramming specific overlays
# Generic overlays don't take precedence over Core's Int64(x::BuiltinInts) etc.
const SignedInts = (Int8, Int16, Int32, Int64)
const UnsignedInts = (UInt8, UInt16, UInt32, UInt64)
const Floats = (Float16, Float32, Float64)

# Integer to integer (specific type pairs for promotion/truncation)
for T in SignedInts, S in SignedInts
    T === S && continue
    if sizeof(T) > sizeof(S)
        @eval @overlay $T(x::$S) = Intrinsics.exti(x, $T, SignednessSigned)
    else
        @eval @overlay $T(x::$S) = Intrinsics.trunci(x, $T)
    end
end

for T in UnsignedInts, S in UnsignedInts
    T === S && continue
    if sizeof(T) > sizeof(S)
        @eval @overlay $T(x::$S) = Intrinsics.exti(x, $T, SignednessUnsigned)
    else
        @eval @overlay $T(x::$S) = Intrinsics.trunci(x, $T)
    end
end

# Bool to integer (zero-extend: false→0, true→1)
for T in (SignedInts..., UnsignedInts...)
    @eval @overlay $T(x::Bool) = Intrinsics.exti(x, $T, SignednessUnsigned)
end

# Integer extension/truncation (via rem) - T and S both used in body
@overlay Base.rem(x::T, ::Type{S}) where {T <: Signed, S <: Signed} =
    sizeof(S) > sizeof(T) ? Intrinsics.exti(x, S, SignednessSigned) :
    sizeof(S) < sizeof(T) ? Intrinsics.trunci(x, S) : x

@overlay Base.rem(x::T, ::Type{S}) where {T <: Unsigned, S <: Unsigned} =
    sizeof(S) > sizeof(T) ? Intrinsics.exti(x, S, SignednessUnsigned) :
    sizeof(S) < sizeof(T) ? Intrinsics.trunci(x, S) : x

# Float to float (specific type pairs)
for T in Floats, S in Floats
    T === S && continue
    @eval @overlay $T(x::$S) = Intrinsics.ftof(x, $T)
end

# Integer to float (specific type pairs)
for F in Floats
    for I in SignedInts
        @eval @overlay $F(x::$I) = Intrinsics.itof(x, $F, SignednessSigned)
    end
    for I in UnsignedInts
        @eval @overlay $F(x::$I) = Intrinsics.itof(x, $F, SignednessUnsigned)
    end
    @eval @overlay $F(x::Bool) = Intrinsics.itof(x, $F, SignednessUnsigned)
end

# Float to integer (via unsafe_trunc)
for F in Floats
    for I in SignedInts
        @eval @overlay Base.unsafe_trunc(::Type{$I}, x::$F) = Intrinsics.ftoi(x, $I, SignednessSigned)
    end
    for I in UnsignedInts
        @eval @overlay Base.unsafe_trunc(::Type{$I}, x::$F) = Intrinsics.ftoi(x, $I, SignednessUnsigned)
    end
end
