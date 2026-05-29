module MicrofloatsExt

import cuTile as ct
import Microfloats

using Microfloats: Float8_E4M3FN, Float8_E5M2, Float8_E8M0FNU, Float4_E2M1FN

# Microfloats spells the Tile-IR-mapped types as: Float8_E4M3FN (NanOnlyAllOnes),
# Float8_E5M2 (IEEE), Float8_E8M0FNU (since v13.2), Float4_E2M1FN (since v13.3).
# Other Microfloats variants (Float8_E4M3 IEEE, Float8_E3M4, Float6_*) have no
# Tile IR equivalent and intentionally fall through to the unsupported-type error.

ct.julia_to_tile_dtype!(table::ct.TypeTable, ::Type{Float8_E4M3FN}) = ct.F8E4M3FN(table)
ct.julia_to_tile_dtype!(table::ct.TypeTable, ::Type{Float8_E5M2})   = ct.F8E5M2(table)
ct.julia_to_tile_dtype!(table::ct.TypeTable, ::Type{Float8_E8M0FNU}) = ct.F8E8M0FNU(table)
ct.julia_to_tile_dtype!(table::ct.TypeTable, ::Type{Float4_E2M1FN})  = ct.F4E2M1FN(table)

# Microfloats are byte-storage primitives, so cuTile's default
# `bitwidth` (8 * sizeof) over-counts the sub-byte formats.
ct.bitwidth(::Type{T}) where {T<:Microfloats.Microfloat} = Microfloats.bitwidth(T)

# E8M0FNU has no sign bit and represents a power of two; tileiras rejects
# nearest-even on f32→E8M0FNU (only `zero` and `positive_inf` are valid).
ct.ftof_rounding_mode(::Type{Float8_E8M0FNU}) = ct.RoundingMode.Zero

# Non-scaled `mma`/`matmul` (`cuda_tile.mmaf`) accepts f8e4m3fn and f8e5m2
# operands with an f16 or f32 accumulator — f16 first/preferred, mirroring
# cuda-tile's mmaf type table and cutile-python's `_mma_supported_dtypes`. The
# other Microfloats formats (E8M0FNU, Float4_E2M1FN) are only valid as
# scaled-mma operands/scales, so they stay absent and fall through to `mma`'s
# unsupported-dtype error.
ct.mma_allowed_acc_dtypes(::Type{Float8_E4M3FN}) = (Float16, Float32)
ct.mma_allowed_acc_dtypes(::Type{Float8_E5M2})   = (Float16, Float32)

# Float ↔ microfloat scalar constructor overlays (for map/convert dispatch).
# Mirrors DLFP8TypesExt: route to `Intrinsics.ftof` so kernel-side conversions
# lower to the FToFOp Tile IR intrinsic instead of Microfloats' Float32-fallback
# software path.
const MicrofloatTypes = (Float8_E4M3FN, Float8_E5M2, Float8_E8M0FNU, Float4_E2M1FN)
const StandardFloats = (Float16, ct.BFloat16, Float32, ct.TFloat32, Float64)

for MF in MicrofloatTypes
    # standard float → microfloat
    for F in StandardFloats
        @eval Base.Experimental.@consistent_overlay ct.cuTileMethodTable Base.@assume_effects :foldable $MF(x::$F) = ct.Intrinsics.ftof(x, $MF)
    end
    # microfloat → standard float
    for F in StandardFloats
        @eval Base.Experimental.@consistent_overlay ct.cuTileMethodTable Base.@assume_effects :foldable $F(x::$MF) = ct.Intrinsics.ftof(x, $F)
    end
    # microfloat → microfloat
    for MFb in MicrofloatTypes
        MF === MFb && continue
        @eval Base.Experimental.@consistent_overlay ct.cuTileMethodTable Base.@assume_effects :foldable $MF(x::$MFb) = ct.Intrinsics.ftof(x, $MF)
    end
end

end
