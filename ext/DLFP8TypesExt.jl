module DLFP8TypesExt

import cuTile as ct

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

end
