# Pass Pipeline
#
# Defines all IR passes and their execution order. Rewrite-based passes are
# defined inline here; complex imperative passes live in their own files
# (alias_analysis.jl, token_order.jl, dce.jl) and are called from run_passes!.

#=============================================================================
 IR Normalization (rewrite)
=============================================================================#

# Lowers Julia Core intrinsics and builtins to cuTile Intrinsics.
# Core intrinsics appear in the SCI either because:
# - IRStructurizer introduces them for control flow (loop bounds, increments)
# - Julia's type inference inlined Base functions down to Core intrinsics
#   (e.g., Base.:-(x::Int32, y::Int32) → Core.Intrinsics.sub_int(x, y))

const NORMALIZE_RULES = RewriteRule[
    # Integer arithmetic
    @rewrite Core.Intrinsics.add_int(~x, ~y) => Intrinsics.addi(~x, ~y)
    @rewrite Core.Intrinsics.sub_int(~x, ~y) => Intrinsics.subi(~x, ~y)
    @rewrite Core.Intrinsics.mul_int(~x, ~y) => Intrinsics.muli(~x, ~y)
    @rewrite Core.Intrinsics.neg_int(~x)     => Intrinsics.negi(~x)

    # Integer comparison
    @rewrite Core.Intrinsics.slt_int(~x, ~y) =>
            Intrinsics.cmpi(~x, ~y, $(ComparisonPredicate.LessThan), $(Signedness.Signed))
    @rewrite Core.Intrinsics.sle_int(~x, ~y) =>
            Intrinsics.cmpi(~x, ~y, $(ComparisonPredicate.LessThanOrEqual), $(Signedness.Signed))
    @rewrite Core.Intrinsics.ult_int(~x, ~y) =>
            Intrinsics.cmpi(~x, ~y, $(ComparisonPredicate.LessThan), $(Signedness.Unsigned))

    # Bitwise
    @rewrite Core.Intrinsics.and_int(~x, ~y) => Intrinsics.andi(~x, ~y)
    @rewrite Core.Intrinsics.or_int(~x, ~y)  => Intrinsics.ori(~x, ~y)
    @rewrite Core.Intrinsics.xor_int(~x, ~y) => Intrinsics.xori(~x, ~y)

    # not_int: xori with all-ones constant (type-dependent)
    @rewrite Core.Intrinsics.not_int(~x::Bool)   => Intrinsics.xori(~x, $(true))
    @rewrite Core.Intrinsics.not_int(~x::Int32)  => Intrinsics.xori(~x, $(Int32(-1)))
    @rewrite Core.Intrinsics.not_int(~x::Int64)  => Intrinsics.xori(~x, $(Int64(-1)))
    @rewrite Core.Intrinsics.not_int(~x::UInt32) => Intrinsics.xori(~x, $(~UInt32(0)))
    @rewrite Core.Intrinsics.not_int(~x::UInt64) => Intrinsics.xori(~x, $(~UInt64(0)))

    # Float arithmetic
    @rewrite Core.Intrinsics.add_float(~x, ~y) => Intrinsics.addf(~x, ~y)
    @rewrite Core.Intrinsics.sub_float(~x, ~y) => Intrinsics.subf(~x, ~y)
    @rewrite Core.Intrinsics.mul_float(~x, ~y) => Intrinsics.mulf(~x, ~y)
    @rewrite Core.Intrinsics.div_float(~x, ~y) => Intrinsics.divf(~x, ~y)
    @rewrite Core.Intrinsics.neg_float(~x)     => Intrinsics.negf(~x)

    # Float comparison
    @rewrite Core.Intrinsics.lt_float(~x, ~y) =>
            Intrinsics.cmpf(~x, ~y, $(ComparisonPredicate.LessThan))
    @rewrite Core.Intrinsics.le_float(~x, ~y) =>
            Intrinsics.cmpf(~x, ~y, $(ComparisonPredicate.LessThanOrEqual))
    @rewrite Core.Intrinsics.eq_float(~x, ~y) =>
            Intrinsics.cmpf(~x, ~y, $(ComparisonPredicate.Equal))
    @rewrite Core.Intrinsics.ne_float(~x, ~y) =>
            Intrinsics.cmpf(~x, ~y, $(ComparisonPredicate.NotEqual))

    # Builtins
    @rewrite (===)(~x, ~y) =>
            Intrinsics.cmpi(~x, ~y, $(ComparisonPredicate.Equal), $(Signedness.Signed))
    @rewrite Core.ifelse(~c, ~x, ~y) => Intrinsics.select(~c, ~x, ~y)
]

normalize_pass!(sci::StructuredIRCode) = rewrite_patterns!(sci, NORMALIZE_RULES)

#=============================================================================
 Scalar View Elimination (rewrite)
=============================================================================#

# Removes redundant to_scalar(from_scalar(x, S)) chains from Julia's broadcast
# system. Transparent op tracing handles intermediate broadcasts.

const SVE_RULES = RewriteRule[
    @rewrite Intrinsics.to_scalar(Intrinsics.from_scalar(~x, ~_)) => ~x
]

scalar_view_elim_pass!(sci::StructuredIRCode) = rewrite_patterns!(sci, SVE_RULES)

#=============================================================================
 FMA Fusion (rewrite)
=============================================================================#

# mul+add/sub → fma to reduce register pressure.
# Mirrors cuTile Python's fuse_mul_addsub in rewrite_patterns.py.
#
# Two rule variants per pattern: 2-arg (default RM/FTZ from normalization) and
# 4-arg (explicit RM/FTZ). Repeated binds ~rm/~ftz enforce consistency between
# mul and add/sub — mismatched flags cause the pattern match to fail, preventing
# incorrect fusion.

const FMA_RULES = RewriteRule[
    # Default RM/FTZ (2-arg forms from normalization)
    @rewrite Intrinsics.addf(one_use(Intrinsics.mulf(~x, ~y)), ~z) =>
            Intrinsics.fma(~x, ~y, ~z)
    @rewrite Intrinsics.addf(~z, one_use(Intrinsics.mulf(~x, ~y))) =>
            Intrinsics.fma(~x, ~y, ~z)
    @rewrite Intrinsics.subf(one_use(Intrinsics.mulf(~x, ~y)), ~z) =>
            Intrinsics.fma(~x, ~y, Intrinsics.negf(~z))

    # Explicit RM/FTZ: repeated ~rm/~ftz binds require mul and add/sub to agree
    @rewrite Intrinsics.addf(one_use(Intrinsics.mulf(~x, ~y, ~rm, ~ftz)), ~z, ~rm, ~ftz) =>
            Intrinsics.fma(~x, ~y, ~z, ~rm, ~ftz)
    @rewrite Intrinsics.addf(~z, one_use(Intrinsics.mulf(~x, ~y, ~rm, ~ftz)), ~rm, ~ftz) =>
            Intrinsics.fma(~x, ~y, ~z, ~rm, ~ftz)
    @rewrite Intrinsics.subf(one_use(Intrinsics.mulf(~x, ~y, ~rm, ~ftz)), ~z, ~rm, ~ftz) =>
            Intrinsics.fma(~x, ~y, Intrinsics.negf(~z), ~rm, ~ftz)
]

fma_fusion_pass!(sci::StructuredIRCode) = rewrite_patterns!(sci, FMA_RULES)

#=============================================================================
 Pass Pipeline
=============================================================================#

"""
    run_passes!(sci::StructuredIRCode)

Run the full pass pipeline on a StructuredIRCode. Called for both kernel
and subprogram compilation.
"""
function run_passes!(sci::StructuredIRCode)
    # Rewrite passes (order matters: normalize before optimize, SVE before FMA)
    normalize_pass!(sci)
    index_lower_pass!(sci)
    scalar_view_elim_pass!(sci)
    fma_fusion_pass!(sci)

    # Memory ordering
    alias_result = alias_analysis_pass!(sci)
    token_order_pass!(sci, alias_result)

    # Cleanup
    dce_pass!(sci)
end
