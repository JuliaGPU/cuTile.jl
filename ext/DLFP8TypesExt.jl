module DLFP8TypesExt

import cuTile as ct
import DLFP8Types

using DLFP8Types: Float8_E4M3FN, Float8_E5M2

function ct.julia_to_tile_dtype!(table::ct.TypeTable, ::Type{Float8_E4M3FN})
    return ct.F8E4M3FN(table)
end

function ct.julia_to_tile_dtype!(table::ct.TypeTable, ::Type{Float8_E5M2})
    return ct.F8E5M2(table)
end

# Non-scaled `mma`/`matmul` (`cuda_tile.mmaf`) accepts f8e4m3fn and f8e5m2
# operands with an f16 or f32 accumulator (f16 first/preferred), mirroring
# cuda-tile's mmaf type table and cutile-python's `_mma_supported_dtypes`.
ct.mma_allowed_acc_dtypes(::Type{Float8_E4M3FN}) = (Float16, Float32)
ct.mma_allowed_acc_dtypes(::Type{Float8_E5M2})   = (Float16, Float32)

# `fast_acc` (lower-precision MMA accumulation) is an FP8-only throughput hint.
ct.mma_supports_fast_acc(::Type{Float8_E4M3FN}) = true
ct.mma_supports_fast_acc(::Type{Float8_E5M2})   = true

# Float ↔ FP8 scalar constructor overlays (for map/convert dispatch)
const FP8Types = (Float8_E4M3FN, Float8_E5M2)
const StandardFloats = (Float16, ct.BFloat16, Float32, ct.TFloat32, Float64)

for F8 in FP8Types
    # Standard float → FP8
    for F in StandardFloats
        @eval Base.Experimental.@consistent_overlay ct.cuTileMethodTable Base.@assume_effects :foldable $F8(x::$F) = ct.Intrinsics.ftof(x, $F8)
    end
    # FP8 → standard float
    for F in StandardFloats
        @eval Base.Experimental.@consistent_overlay ct.cuTileMethodTable Base.@assume_effects :foldable $F(x::$F8) = ct.Intrinsics.ftof(x, $F)
    end
    # FP8 → FP8
    for F8b in FP8Types
        F8 === F8b && continue
        @eval Base.Experimental.@consistent_overlay ct.cuTileMethodTable Base.@assume_effects :foldable $F8(x::$F8b) = ct.Intrinsics.ftof(x, $F8)
    end
end

# FP8 is a storage / tensor-core operand format, not an arithmetic type: the
# Tile IR elementwise float ops only accept f16/bf16/f32/f64. Registered for
# every `FP8` subtype, not just the two with a Tile IR dtype, so the blocking
# overlays below cover exactly the methods DLFP8Types defines.
ct.is_restricted_float(::Type{<:DLFP8Types.FP8}) = true

# DLFP8Types implements scalar arithmetic as a Float32 round-trip
# (`T(op(Float32(a), Float32(b)))`, src/DLFP8Types.jl). Combined with our `ftof`
# constructor overlays above, a kernel-side `f8_tile .+ f8_tile` would compile
# silently into ftof → addf → ftof: an implicit upcast with an extra rounding
# per operation, and no hint that a cast happened. Shadow those methods so the
# broadcast path reports the same error as the tile-level operators.
#
# Plain `@overlay`, not `@consistent_overlay`: throwing is deliberately
# inconsistent with the shadowed method. Host-side FP8 arithmetic is untouched.
for op in (:+, :-, :*, :/, :\, :^)
    @eval Base.Experimental.@overlay ct.cuTileMethodTable Base.$op(a::T, b::T) where {T<:DLFP8Types.FP8} =
        ct.check_arithmetic(Base.$op, T)
end
for op in (:sin, :cos, :tan, :asin, :acos, :atan, :sinh, :cosh, :tanh, :asinh,
           :acosh, :atanh, :exp, :exp2, :exp10, :expm1, :log, :log2, :log10,
           :sqrt, :cbrt, :log1p)
    @eval Base.Experimental.@overlay ct.cuTileMethodTable Base.$op(a::T) where {T<:DLFP8Types.FP8} =
        ct.check_arithmetic(Base.$op, T)
end
# Unary negation is an exact sign-bit flip upstream, so it would compile — but
# the tile-level `-(::Tile{T})` is blocked (it lowers to `negf`), and `-x` and
# `(-).(x)` disagreeing on the same tile is worse than rejecting both.
Base.Experimental.@overlay ct.cuTileMethodTable Base.:(-)(a::T) where {T<:DLFP8Types.FP8} =
    ct.check_arithmetic(Base.:(-), T)

end
