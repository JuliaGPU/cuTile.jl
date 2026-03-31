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
 Combined Rule Set
=============================================================================#

const OPTIMIZATION_RULES = RewriteRule[
    IDENTITY_RULES...,
    ALGEBRA_RULES...,
    FMA_RULES...,
]

#=============================================================================
 Constant Propagation (analysis)
=============================================================================#

# Tracks which SSA values have known constant values. Constants are represented
# as Julia Arrays matching the tile's element type and shape. This enables O(1)
# constant lookups in the rewrite pattern matcher (PLiteral).

"""
    propagate_constants(sci) -> Dict{SSAValue, Any}

Build a map from SSA values to their known constant values. Walks all blocks
in program order so transitive constants (e.g. reshape of a broadcast of a
literal) resolve correctly.
"""
function propagate_constants(sci::StructuredIRCode)
    constants = Dict{SSAValue, Any}()
    propagate_constants!(constants, sci.entry)
    return constants
end

function propagate_constants!(constants::Dict{SSAValue, Any}, block::Block)
    # Recurse into nested control flow first
    for inst in instructions(block)
        s = stmt(inst)
        if s isa ForOp
            propagate_constants!(constants, s.body)
        elseif s isa IfOp
            propagate_constants!(constants, s.then_region)
            propagate_constants!(constants, s.else_region)
        elseif s isa WhileOp
            propagate_constants!(constants, s.before)
            propagate_constants!(constants, s.after)
        elseif s isa LoopOp
            propagate_constants!(constants, s.body)
        end
    end

    for inst in instructions(block)
        call = resolve_call(block, inst)
        call === nothing && continue
        func, ops = call

        # Transparent ops (broadcast, reshape) propagate constants from operand
        if (func === Intrinsics.broadcast || func === Intrinsics.reshape) &&
                length(ops) >= 1
            scalar = const_value(constants, ops[1])
            scalar === nothing && continue
            vt = value_type(block, SSAValue(inst))
            vt === nothing && continue
            T = CC.widenconst(vt)
            T <: Tile || continue
            S = size(T)
            constants[SSAValue(inst)] = fill(convert(eltype(T), scalar), S)
        end
    end
end

"""Resolve an operand to its scalar constant value, or `nothing`."""
function const_value(constants::Dict{SSAValue, Any}, @nospecialize(op))
    if op isa Number
        return op
    elseif op isa QuoteNode && op.value isa Number
        return op.value
    elseif op isa SSAValue
        c = get(constants, op, nothing)
        c isa AbstractArray || return nothing
        isempty(c) && return nothing
        v = first(c)
        all(==(v), c) && return v
        return nothing
    end
    return nothing
end

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

    constants = propagate_constants(sci)
    rewrite_patterns!(sci, OPTIMIZATION_RULES; constants)

    alias_result = alias_analysis_pass!(sci)
    token_order_pass!(sci, alias_result)

    dce_pass!(sci)
end
