# Pass Pipeline
#
# Defines all IR passes and their execution order. Rewrite-based passes are
# defined inline here; complex imperative passes live in their own files
# (alias_analysis.jl, token_order.jl, dce.jl) and are called from run_passes!.

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
 Algebraic Simplification (rewrite)
=============================================================================#

# Cancel inverse addi/subi pairs: x+c-c → x, x-c+c → x.
# Repeated ~c binds enforce that both operands are the same value.

const ALGEBRA_RULES = RewriteRule[
    @rewrite Intrinsics.subi(Intrinsics.addi(~x, ~c), ~c) => ~x
    @rewrite Intrinsics.addi(Intrinsics.subi(~x, ~c), ~c) => ~x
]

algebra_pass!(sci::StructuredIRCode) = rewrite_patterns!(sci, ALGEBRA_RULES)

#=============================================================================
 Combined Rule Set
=============================================================================#

const OPTIMIZATION_RULES = RewriteRule[
    ALGEBRA_RULES...,
    FMA_RULES...,
]

#=============================================================================
 Pass Pipeline
=============================================================================#

"""
    run_passes!(sci::StructuredIRCode)

Run the full pass pipeline on a StructuredIRCode. Called for both kernel
and subprogram compilation.
"""
function run_passes!(sci::StructuredIRCode)
    canonicalize!(sci)

    rewrite_patterns!(sci, OPTIMIZATION_RULES)

    alias_result = alias_analysis_pass!(sci)
    token_order_pass!(sci, alias_result)

    dce_pass!(sci)
end
