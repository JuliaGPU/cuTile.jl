# Method Overlay Infrastructure for cuTile Compilation
#
# Defines the @overlay macro and type conversion overlays.
# Arithmetic and math overlays are defined in arithmetic.jl and math.jl.

macro overlay(ex)
    esc(:(Base.Experimental.@consistent_overlay cuTileMethodTable Base.@assume_effects :foldable $ex))
end


#=============================================================================
 Broadcasting
=============================================================================#

# Route Type values through TypeRef instead of RefValue (which can't be constructed in Tile IR).
@overlay Base.Broadcast.broadcastable(::Type{T}) where T = TypeRef{T}()


#=============================================================================
 Type Conversions
=============================================================================#

# Type tuples for metaprogramming specific overlays
# Generic overlays don't take precedence over Core's Int64(x::BuiltinInts) etc.
const SignedInts = (Int8, Int16, Int32, Int64)
const UnsignedInts = (UInt8, UInt16, UInt32, UInt64)
const Floats = (Float16, BFloat16, Float32, TFloat32, Float64)

# Integer to integer (specific type pairs for promotion/truncation)
for T in SignedInts, S in SignedInts
    T === S && continue
    if sizeof(T) > sizeof(S)
        @eval @overlay $T(x::$S) = Intrinsics.exti(x, $T, Signedness.Signed)
    else
        @eval @overlay $T(x::$S) = Intrinsics.trunci(x, $T)
    end
end

for T in UnsignedInts, S in UnsignedInts
    T === S && continue
    if sizeof(T) > sizeof(S)
        @eval @overlay $T(x::$S) = Intrinsics.exti(x, $T, Signedness.Unsigned)
    else
        @eval @overlay $T(x::$S) = Intrinsics.trunci(x, $T)
    end
end

# Bool to integer (zero-extend: false→0, true→1)
for T in (SignedInts..., UnsignedInts...)
    @eval @overlay $T(x::Bool) = Intrinsics.exti(x, $T, Signedness.Unsigned)
end

# Integer extension/truncation (via rem) - T and S both used in body
@overlay Base.rem(x::T, ::Type{S}) where {T <: Signed, S <: Signed} =
    sizeof(S) > sizeof(T) ? Intrinsics.exti(x, S, Signedness.Signed) :
    sizeof(S) < sizeof(T) ? Intrinsics.trunci(x, S) : x

@overlay Base.rem(x::T, ::Type{S}) where {T <: Unsigned, S <: Unsigned} =
    sizeof(S) > sizeof(T) ? Intrinsics.exti(x, S, Signedness.Unsigned) :
    sizeof(S) < sizeof(T) ? Intrinsics.trunci(x, S) : x

# Float to float
for T in Floats, S in Floats
    T === S && continue
    @eval @overlay $T(x::$S) = Intrinsics.ftof(x, $T)
end

# Integer to float
for F in Floats
    for I in SignedInts
        @eval @overlay $F(x::$I) = Intrinsics.itof(x, $F, Signedness.Signed)
    end
    for I in UnsignedInts
        @eval @overlay $F(x::$I) = Intrinsics.itof(x, $F, Signedness.Unsigned)
    end
    @eval @overlay $F(x::Bool) = Intrinsics.itof(x, $F, Signedness.Unsigned)
end

# Float to integer (via unsafe_trunc)
for F in Floats
    for I in SignedInts
        @eval @overlay Base.unsafe_trunc(::Type{$I}, x::$F) = Intrinsics.ftoi(x, $I, Signedness.Signed)
    end
    for I in UnsignedInts
        @eval @overlay Base.unsafe_trunc(::Type{$I}, x::$F) = Intrinsics.ftoi(x, $I, Signedness.Unsigned)
    end
end

# Float to integer (round with RoundToZero)
for F in Floats, I in (SignedInts..., UnsignedInts...)
    @eval @overlay function Base.round(::Type{$I}, x::$F, ::Base.Rounding.RoundingMode{:ToZero})
        # TODO: assert that x is within bounds etc
        unsafe_trunc($I, x)
    end
end

# Float to integer (direct constructor)
for F in Floats
    for I in SignedInts
        @eval @overlay function $I(x::$F)
            # TODO: assert that x is within bounds etc
            unsafe_trunc($I, x)
        end
    end
    for I in UnsignedInts
        @eval @overlay function $I(x::$F)
            # TODO: assert that x is within bounds etc
            unsafe_trunc($I, x)
        end
    end
end


#=============================================================================
 Tile Constructors
=============================================================================#

# Base.fill/zeros/ones return Tiles in kernel context, matching Julia's standard API.
# Marked non-foldable because they return differently-typed objects.
Base.Experimental.@consistent_overlay cuTileMethodTable @inline Base.fill(v, dims::NTuple{N, Int}) where {N} =
    _full(v, typeof(v), dims)
Base.Experimental.@consistent_overlay cuTileMethodTable @inline Base.zeros(::Type{T}, dims::NTuple{N, Int}) where {T, N} =
    _full(zero(T), T, dims)
Base.Experimental.@consistent_overlay cuTileMethodTable @inline Base.ones(::Type{T}, dims::NTuple{N, Int}) where {T, N} =
    _full(one(T), T, dims)
