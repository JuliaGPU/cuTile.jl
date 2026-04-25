# Constant Analysis
#
# Tracks which SSA values have known constant scalar values. "Constant" here
# means the Tile is a splat of a single scalar — shape is recoverable from
# the SSA's type, so the analysis stores only the scalar (or ⊤).
#
# Implemented as a `ConstantAnalysis <: ForwardAnalysis{ConstantElement}` on
# the generic dataflow framework. The driver handles block walking, fixpoint
# iteration, and control-flow merges (IfOp / ForOp / WhileOp / LoopOp) via
# the shared protocol.
#
# Consumed by `rewrite_patterns!` via `const_value`. Consumers go through
# the public query API (`const_value`); the dataflow framework is an
# implementation detail.

"""⊤ of the constant-analysis lattice: "known non-constant / conflicting merges"."""
struct ConstantTop end
const CONSTANT_TOP = ConstantTop()

"""
    ConstantElement

3-state lattice for constant analysis:

    nothing        — ⊥: not yet analysed (dict-absent return value)
    Number         — known splat-constant with this scalar value
    CONSTANT_TOP   — ⊤: known non-constant (or conflicting merges)
"""
const ConstantElement = Union{Nothing, Number, ConstantTop}

"""
    ConstantAnalysis

Forward analysis propagating "this Tile is a splat of a known scalar" facts
through straight-line code and structured control flow. Recognises:

- `Intrinsics.constant(shape, scalar, T)` — seeds from the scalar operand
- `Intrinsics.broadcast(x, _)`, `Intrinsics.reshape(x, _)` — pass-through

Everything else → `CONSTANT_TOP`.
"""
struct ConstantAnalysis <: ForwardAnalysis{ConstantElement} end

bottom(::ConstantAnalysis) = nothing
top(::ConstantAnalysis) = CONSTANT_TOP

function tmerge(::ConstantAnalysis, a, b)
    a === nothing && return b
    b === nothing && return a
    a === CONSTANT_TOP && return CONSTANT_TOP
    b === CONSTANT_TOP && return CONSTANT_TOP
    a == b ? a : CONSTANT_TOP
end

function operand_value(::ConstantAnalysis, r::DataflowResult, @nospecialize(op))
    op isa Number && return op
    op isa QuoteNode && op.value isa Number && return op.value
    op isa LatticeAnchor && return r[op]
    CONSTANT_TOP
end

function transfer(a::ConstantAnalysis, r::DataflowResult, @nospecialize(func),
                  ops, block::Block, ::Any)
    if func === Intrinsics.constant && length(ops) >= 2
        v = operand_value(a, r, ops[2])
        return v isa Number ? v : CONSTANT_TOP
    end
    if (func === Intrinsics.broadcast || func === Intrinsics.reshape) &&
       length(ops) >= 1
        return operand_value(a, r, ops[1])
    end
    CONSTANT_TOP
end

#=============================================================================
 Public query API
=============================================================================#

"""
    ConstantInfo

Result of running constant analysis. Consumers query it via `const_value`;
the underlying lattice representation is an implementation detail.
"""
const ConstantInfo = DataflowResult{ConstantAnalysis, ConstantElement}

"""
    analyze_constants(sci::StructuredIRCode) -> ConstantInfo

Run forward constant analysis on `sci`.
"""
analyze_constants(sci::StructuredIRCode) = analyze(ConstantAnalysis(), sci)::ConstantInfo

"""
    const_value(info, op) -> Number | nothing

Resolve an operand to its scalar constant value, or `nothing` if unknown /
non-constant. Accepts raw numeric operands, QuoteNodes, and SSAValues
(resolved through the `ConstantInfo` result). When called with `nothing`
(no analysis result supplied), returns the scalar only for raw literal
operands — SSAValues are treated as non-constant.
"""
function const_value(info::ConstantInfo, @nospecialize(op))
    op isa Number && return op
    op isa QuoteNode && op.value isa Number && return op.value
    if op isa SSAValue
        v = info[op]
        return v isa Number ? v : nothing
    end
    nothing
end

function const_value(::Nothing, @nospecialize(op))
    op isa Number && return op
    op isa QuoteNode && op.value isa Number && return op.value
    nothing
end
