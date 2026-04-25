# Pass Pipeline
#
# Defines all IR passes and their execution order. Rewrite-based passes
# (`@rewrite` rule sets) are defined inline here; larger transforms live
# alongside this file under `transform/`, and dataflow analyses live under
# `analysis/`. `run_passes!` orchestrates the whole sequence.

#=============================================================================
 Print Fusion (rewrite)
=============================================================================#

# Fuse format_string (from string interpolation overlay) into print_tko.
# Julia lowers `print("hello $x")` → `print(string("hello ", x))`, which our
# overlays compile to `print_tko(format_string("hello ", x), "\n")`.
# This rule inlines format_string's args into the print_tko call.

const PRINT_FUSION_RULES = RewriteRule[
    @rewrite Intrinsics.print_tko(Intrinsics.format_string(~parts...), ~rest...) =>
             Intrinsics.print_tko(~parts..., ~rest...)
]

#=============================================================================
 FMA Fusion (rewrite)
=============================================================================#

# mul+add/sub → fma to reduce register pressure.
# Mirrors cuTile Python's fuse_mul_addsub in rewrite_patterns.py.
# RM/FTZ are applied by the ambient @fpmode scope during codegen, not per-op.

const FMA_RULES = RewriteRule[
    @rewrite Intrinsics.addf(one_use(Intrinsics.mulf(~x, ~y)), ~z) =>
            Intrinsics.fma(~x, ~y, ~z)
    @rewrite Intrinsics.addf(~z, one_use(Intrinsics.mulf(~x, ~y))) =>
            Intrinsics.fma(~x, ~y, ~z)
    @rewrite Intrinsics.subf(one_use(Intrinsics.mulf(~x, ~y)), ~z) =>
            Intrinsics.fma(~x, ~y, Intrinsics.negf(~z))
    @rewrite Intrinsics.subf(~z, one_use(Intrinsics.mulf(~x, ~y))) =>
            Intrinsics.fma(Intrinsics.negf(~x), ~y, ~z)
]

fma_fusion_pass!(sci::StructuredIRCode) = rewrite_patterns!(sci, FMA_RULES)

#=============================================================================
 Algebraic Simplification (rewrite)
=============================================================================#

# Cancel inverse addi/subi pairs: x+c-c → x, x-c+c → x.
# Repeated ~c binds enforce that both operands are the same value.

# Guard factory: check that the given bindings all resolve to the same constant
# value. Like MLIR's ConstantLikeMatcher: matches on attribute value, not SSA
# identity. Returns a guard function for use with @rewrite.
function same_const(keys::Symbol...)
    (match, driver) -> begin
        vals = map(keys) do k
            const_value(driver.constants, match.bindings[k])
        end
        all(!isnothing, vals) && allequal(vals)
    end
end

"""Commute addi/subi past a transparent op (reshape or broadcast) by recreating
the constant at the pre-transparent shape. The transparent op is determined from
the matched intermediate instruction, so one function handles both reshape and
broadcast patterns."""
function commute_arith_transparent(sci, block, inst, match, driver)
    x = match.bindings[:x]
    c = match.bindings[:c]

    scalar = const_value(driver.constants, c)
    scalar === nothing && return false

    # Don't commute when x is also a constant (would loop).
    const_value(driver.constants, x) !== nothing && return false

    x_type = value_type(block, x)
    x_type === nothing && return false
    xT = CC.widenconst(x_type)
    xT <: Tile || return false

    val = SSAValue(inst)
    root_func = driver.defs[val].func  # Intrinsics.subi or Intrinsics.addi

    # Determine transparent op (reshape or broadcast) from first operand
    inner_val = def_operands(driver.defs[val])[1]
    transparent_func = driver.defs[inner_val].func

    # Don't commute through identity ops (same shape in/out) — that's a
    # pointless restructuring that breaks pattern matching for other rules.
    # Identity transparent ops are handled by IDENTITY_RULES instead.
    inst_type = value_type(block, val)
    inst_type === nothing && return false
    size(CC.widenconst(inst_type)) == size(xT) && return false

    # Insert broadcast of the scalar to x's shape and register as constant
    x_shape = size(xT)
    bc_type = Tile{eltype(xT), Tuple{x_shape...}}
    bc = insert_before!(block, val, Expr(:call, Intrinsics.broadcast, scalar, x_shape), bc_type)
    notify_insert!(driver, block, bc)
    # Side-inject the freshly synthesized constant into the dataflow result so
    # downstream pattern matches see it. Bypasses tmerge (this is a brand-new
    # SSA value, not a merge).
    driver.constants[SSAValue(bc)] = convert(eltype(xT), scalar)

    # Insert op(x, broadcast) with x's type
    op = insert_before!(block, val, Expr(:call, root_func, x, SSAValue(bc)), xT)
    notify_insert!(driver, block, op)

    # Replace root with transparent_op(op_result, s)
    pos = findfirst(==(val.id), block.body.ssa_idxes)
    block.body.stmts[pos] = Expr(:call, transparent_func, SSAValue(op), match.bindings[:s])
    driver.defs[val] = DefEntry(block, val, transparent_func)
    push!(driver.worklist, val)
    add_users_to_worklist!(driver, val)
    return true
end

const ALGEBRA_RULES = RewriteRule[
    # SSA-identity cancellation: subi(addi(x, c), c) where c is the same SSA value
    @rewrite Intrinsics.subi(Intrinsics.addi(~x, ~c), ~c) => ~x
    @rewrite Intrinsics.addi(Intrinsics.subi(~x, ~c), ~c) => ~x

    # Constant-value cancellation: subi(addi(x, c0), c1) where c0 == c1 as values
    # (different SSA defs, same constant). Catches 1-based indexing patterns where
    # arange(N)+1 produces one broadcast(1) and gather's -1 produces another.
    # Generalizes MLIR's arith.addi/subi canonicalization for matching constants.
    @rewrite(Intrinsics.subi(Intrinsics.addi(~x, ~c0), ~c1) => ~x, same_const(:c0, :c1))
    @rewrite(Intrinsics.addi(Intrinsics.subi(~x, ~c0), ~c1) => ~x, same_const(:c0, :c1))

    # Nested cancellation: (a + (b + c)) - c → a + b
    # Catches arange pattern where iota+1 is added to an offset, then gather/scatter
    # subtracts 1: subi(addi(offset, addi(iota, 1)), 1) → addi(offset, iota).
    @rewrite(Intrinsics.subi(Intrinsics.addi(~a, Intrinsics.addi(~b, ~c0)), ~c1) =>
             Intrinsics.addi(~a, ~b), same_const(:c0, :c1))
    @rewrite(Intrinsics.addi(Intrinsics.subi(~a, Intrinsics.subi(~b, ~c0)), ~c1) =>
             Intrinsics.subi(~a, ~b), same_const(:c0, :c1))

    # Commute addi/subi through transparent ops (reshape, broadcast).
    # Moves arithmetic past the transparent op so cancellation rules above can fire.
    @rewriter Intrinsics.subi(Intrinsics.reshape(~x, ~s), ~c) => commute_arith_transparent
    @rewriter Intrinsics.addi(Intrinsics.reshape(~x, ~s), ~c) => commute_arith_transparent
    @rewriter Intrinsics.subi(Intrinsics.broadcast(~x, ~s), ~c) => commute_arith_transparent
    @rewriter Intrinsics.addi(Intrinsics.broadcast(~x, ~s), ~c) => commute_arith_transparent
]

algebra_pass!(sci::StructuredIRCode) = rewrite_patterns!(sci, ALGEBRA_RULES)

#=============================================================================
 Identity Fold (rewrite)
=============================================================================#

# Eliminate identity broadcasts and reshapes (same shape in/out). These are
# no-ops left behind by the broadcast system after scalar elimination.

function is_identity_op(match, driver)
    x = match.bindings[:x]
    val = first(match.matched_ssas)
    entry = driver.defs[val]
    in_t = value_type(entry.block, x)
    out_t = value_type(entry.block, val)
    in_t === nothing && return false
    out_t === nothing && return false
    in_T = CC.widenconst(in_t)
    out_T = CC.widenconst(out_t)
    in_T <: Tile && out_T <: Tile || return false
    return size(in_T) == size(out_T)
end

const IDENTITY_RULES = RewriteRule[
    @rewrite(Intrinsics.broadcast(~x, ~shape) => ~x, is_identity_op)
    @rewrite(Intrinsics.reshape(~x, ~shape) => ~x, is_identity_op)
]

#=============================================================================
 Comparison Strength Reduction (rewrite)
=============================================================================#

# (x + 1) <= y  →  x < y  for signed integers.
# Canonicalizes Julia's 1-based `arange(N) .+ 1 .<= limit` mask pattern
# into 0-based `arange(N) .< limit`, eliminating the tile-wide addi(iota, 1).

const COMPARISON_RULES = RewriteRule[
    # Direct: cmpi(addi(x, 1), y, <=, signed) → cmpi(x, y, <, signed)
    @rewrite Intrinsics.cmpi(Intrinsics.addi(~x, $(1)), ~y,
                              $(ComparisonPredicate.LessThanOrEqual), $(Signedness.Signed)) =>
             Intrinsics.cmpi(~x, ~y, $(ComparisonPredicate.LessThan), $(Signedness.Signed))

    # Nested: cmpi(addi(a, addi(b, 1)), y, <=, signed) → cmpi(addi(a, b), y, <, signed)
    # Uses inplace=true to modify the existing addi and cmpi ops' operands rather
    # than creating new ones (which would cascade the worklist).
    @rewrite(inplace=true,
             Intrinsics.cmpi(Intrinsics.addi(~a, Intrinsics.addi(~b, $(1))), ~y,
                              $(ComparisonPredicate.LessThanOrEqual), $(Signedness.Signed)) =>
             Intrinsics.cmpi(Intrinsics.addi(~a, ~b), ~y,
                              $(ComparisonPredicate.LessThan), $(Signedness.Signed)))
]

#=============================================================================
 Power Strength Reduction (rewrite)
=============================================================================#

# pow(x, 2) → mulf(x, x): replaces an expensive transcendental with a multiply.
# The MLIR Tile IR backend has no canonicalization for pow, so this is purely
# a Julia-level optimization. Applies to the variance computation in layernorm
# (centered_tx .^ 2.0f0). Uses a guard with == so it matches any float type
# (Float16, BFloat16, Float32, Float64, TFloat32). Integer-literal exponents
# (x .^ 2) are already handled by Julia's literal_pow → x*x → mulf(x, x).

function is_pow_two(match, driver)
    c = const_value(driver.constants, match.bindings[:exp])
    c !== nothing && c == 2
end

const POWER_RULES = RewriteRule[
    @rewrite(Intrinsics.pow(~x, ~exp) => Intrinsics.mulf(~x, ~x), is_pow_two)
]

#=============================================================================
 Combined Rule Set
=============================================================================#

const OPTIMIZATION_RULES = RewriteRule[
    IDENTITY_RULES...,
    ALGEBRA_RULES...,
    FMA_RULES...,
    COMPARISON_RULES...,
    POWER_RULES...,
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

    rewrite_patterns!(sci, PRINT_FUSION_RULES)

    constants = analyze_constants(sci)
    rewrite_patterns!(sci, OPTIMIZATION_RULES; constants)

    alias_info = analyze_aliases(sci)

    token_order_pass!(sci, alias_info)

    licm_pass!(sci)

    dce_pass!(sci)
end
