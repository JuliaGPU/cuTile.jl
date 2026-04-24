# Generic forward sparse dataflow framework.
#
# One reusable driver for monotonic forward analyses over StructuredIRCode.
# Each concrete analysis subtypes `ForwardAnalysis{T}` and implements a small
# protocol (`bottom`, `top`, `tmerge`, `transfer`). The driver handles block
# walking, fixpoint iteration, and the structured-control-flow merges at
# IfOp / ForOp / WhileOp / LoopOp.
#
# Inspired by MLIR's `SparseForwardDataFlowAnalysis` (per-Value lattice,
# `ChangeResult` worklist, queryable state after convergence) and Julia's
# `AbstractLattice` (lattice-as-interface, fallthrough composition).

"""
    LatticeAnchor

The keys that dataflow results are indexed by: an SSA value defined in some
block, a block argument, or a kernel parameter reference (`Argument` /
`SlotNumber`). `BlockArgument` is used for loop-carried values and IfOp
outputs.
"""
const LatticeAnchor = Union{SSAValue, BlockArgument, Argument, SlotNumber}


#=============================================================================
 Analysis protocol
=============================================================================#

"""
    ForwardAnalysis{T}

Abstract supertype for monotonic forward sparse dataflow analyses whose
lattice element type is `T`. Concrete analyses subtype this and implement:

- `bottom(a)::T`  — the "no info yet" element. Returned for any anchor absent
  from the result dict; every anchor starts at `bottom` before propagation.
- `top(a)::T`     — the "maximally loose" element. Returned by the default
  `init_arg` and used for unknown-op transfer fallback. `tmerge(x, top) == top`.
- `tmerge(a, x::T, y::T)::T` — least upper bound. Must be monotone and commute.
- `transfer(a, result, func, ops, block, inst)::T` — lattice element for the
  result of a resolved call. `result` is the live `DataflowResult` being built
  (read-only; use `result[op]` to query operand anchors).

Optional methods, with defaults:

- `operand_value(a, result, op)::T` — lattice value for an operand in a
  transfer-function position. Default routes anchors to `result[anchor]`
  and returns `bottom(a)` for anything else. Override to recognise raw
  literal operands (e.g. an integer literal is its own constant for a
  constant-propagation analysis).
- `init_arg(a, i, argtype)::T` — seed for kernel parameter `Argument(i)` with
  inferred type `argtype`. Default returns `top(a)`.
- `max_iters(a)::Int` — convergence cap. Default 32. The driver errors if the
  dirty flag is still set after this many outer iterations.

The default control-flow transfer handles `IfOp`, `ForOp`, `WhileOp`, and
`LoopOp` by recursing into sub-blocks and joining yielded / loop-carried
values via `tmerge`. Override `transfer_cf` per analysis if a specialized rule
is needed.
"""
abstract type ForwardAnalysis{T} end

function bottom end
function top end
function tmerge end
function transfer end

init_arg(a::ForwardAnalysis{T}, ::Int, @nospecialize(_)) where {T} = top(a)::T
max_iters(::ForwardAnalysis) = 32

# `operand_value` is defined after `DataflowResult` below (its method
# signature takes a `DataflowResult`).


#=============================================================================
 Result
=============================================================================#

"""
    DataflowResult{A, T}

Queryable result of a forward analysis `A` with lattice element type `T`.
Read via `r[anchor]` — absent keys collapse to `bottom(analysis)`, so the
query is total.

The underlying `values::Dict` is exposed for callers that need to synthesize
new entries after analysis has converged (the rewriter does this when it
inserts a fresh constant mid-rewrite). Direct writes bypass the analysis'
`tmerge`, which is intentional for synthesis of *new* SSA values. For updating
existing keys, use `record!`.
"""
struct DataflowResult{A <: ForwardAnalysis, T}
    analysis::A
    values::Dict{LatticeAnchor, T}
end

DataflowResult(a::ForwardAnalysis{T}) where {T} =
    DataflowResult{typeof(a), T}(a, Dict{LatticeAnchor, T}())

Base.getindex(r::DataflowResult, @nospecialize(key)) =
    key isa LatticeAnchor ? get(r.values, key, bottom(r.analysis)) : bottom(r.analysis)

"""
    r[key] = val

Record a lattice value for `key`, bypassing `tmerge`. Use this only to
side-inject facts for *newly synthesised* SSA values (e.g. when a rewriter
creates a broadcast op whose constant value is known by construction).
For re-analysis of an existing anchor, call the analysis again instead.
"""
function Base.setindex!(r::DataflowResult{A, T}, val::T,
                        key::LatticeAnchor) where {A, T}
    r.values[key] = val
    return r
end

"""
    has_value(r, key) -> Bool

True iff `key` has a non-bottom entry in the result. Useful for telling
"genuinely unknown" from "happens to equal bottom".
"""
has_value(r::DataflowResult, @nospecialize(key)) =
    key isa LatticeAnchor && haskey(r.values, key)

"""
    operand_value(analysis, result, op) -> T

Lattice value for an operand as used *inside a transfer function*. The
default handles `LatticeAnchor`s (via dict lookup) and treats everything
else (integer/float literals, QuoteNodes, GlobalRefs, …) as `bottom`.
Analyses override this to recognise raw literals when those are
meaningful lattice inputs — e.g. a constant-propagation analysis returns
the integer itself for a raw `Int` operand.
"""
operand_value(a::ForwardAnalysis, r::DataflowResult, @nospecialize(op)) =
    op isa LatticeAnchor ? r[op] : bottom(a)


#=============================================================================
 Driver
=============================================================================#

"""
    analyze(analysis, sci) -> DataflowResult

Run `analysis` over `sci` to fixpoint. Kernel parameter anchors
(`Argument(i)`) are seeded from `init_arg(analysis, i, T)`; everything
else is derived from the IR by the transfer rules.
"""
function analyze(analysis::A, sci::StructuredIRCode) where {A <: ForwardAnalysis}
    result = DataflowResult(analysis)

    for (i, argtype) in enumerate(sci.argtypes)
        v = init_arg(analysis, i, argtype)
        if !is_bottom(analysis, v)
            result.values[Argument(i)] = v
        end
    end

    dirty = Ref(true)
    cap = max_iters(analysis)
    for iter in 1:cap
        dirty[] || return result
        dirty[] = false
        walk!(analysis, result, sci.entry, dirty)
        iter == cap && dirty[] &&
            error("dataflow analysis $(nameof(A)) did not converge in $cap iterations")
    end
    return result
end

"""Predicate: is `v` the lattice bottom for `analysis`?"""
is_bottom(analysis::ForwardAnalysis, @nospecialize(v)) = v === bottom(analysis)


#=============================================================================
 Block walk & transfer dispatch
=============================================================================#

function walk!(analysis::ForwardAnalysis, result::DataflowResult, block::Block, dirty::Ref{Bool})
    for inst in instructions(block)
        s = stmt(inst)
        if s isa ControlFlowOp
            transfer_cf!(analysis, result, s, SSAValue(inst.ssa_idx), block, dirty)
        elseif s isa Expr
            transfer_call!(analysis, result, block, inst, dirty)
        end
    end
end

function transfer_call!(analysis::ForwardAnalysis, result::DataflowResult,
                        block::Block, inst::Instruction, dirty::Ref{Bool})
    call = resolve_call(block, inst)
    call === nothing && return
    func, ops = call
    new_val = transfer(analysis, result, func, ops, block, inst)
    record!(analysis, result, SSAValue(inst.ssa_idx), new_val, dirty)
end

"""
    record!(analysis, result, key, new_val, dirty)

Write `new_val` into `result` at `key`. On insert, the new value lands as-is;
on conflict, it is merged with the existing entry via `tmerge`. Sets `dirty` when
the stored element actually changes.
"""
function record!(analysis::ForwardAnalysis{T}, result::DataflowResult,
                 key::LatticeAnchor, new_val::T, dirty::Ref{Bool}) where {T}
    old = get(result.values, key, nothing)
    if old === nothing
        is_bottom(analysis, new_val) && return  # don't pollute the dict with ⊥
        result.values[key] = new_val
        dirty[] = true
    else
        merged = tmerge(analysis, old, new_val)
        if merged != old
            result.values[key] = merged
            dirty[] = true
        end
    end
    return
end


#=============================================================================
 Default structured-control-flow transfer
=============================================================================#

"""
    transfer_cf!(analysis, result, op, ssa, block, dirty)

Default control-flow transfer. Concrete analyses rarely need to override this
— add a method for a particular `op` type if a specialized rule is required
(e.g., an analysis that sharpens based on a branch condition).
"""
function transfer_cf! end

# IfOp — a single aggregate lattice element covering all yielded positions
# from both regions. Our IR uses one SSA per IfOp and downstream access via
# `getfield`, so a per-position lattice adds complexity without a consumer.
# Same aggregate model as the existing divisibility analysis.
function transfer_cf!(analysis::ForwardAnalysis, result::DataflowResult,
                      op::IfOp, ssa::SSAValue, ::Block, dirty::Ref{Bool})
    walk!(analysis, result, op.then_region, dirty)
    walk!(analysis, result, op.else_region, dirty)

    tt = op.then_region.terminator
    et = op.else_region.terminator
    (tt isa YieldOp && et isa YieldOp) || return
    (isempty(tt.values) || isempty(et.values)) && return

    merged = bottom(analysis)
    n = min(length(tt.values), length(et.values))
    for i in 1:n
        merged = tmerge(analysis, merged, operand_value(analysis, result, tt.values[i]))
        merged = tmerge(analysis, merged, operand_value(analysis, result, et.values[i]))
    end
    record!(analysis, result, ssa, merged, dirty)
end

# ForOp — body args receive loop-carried values: join(init_values, ContinueOp yields).
# The IV argument is seeded to `top` (unknown at compile time).
function transfer_cf!(analysis::ForwardAnalysis, result::DataflowResult,
                      op::ForOp, ::SSAValue, ::Block, dirty::Ref{Bool})
    record!(analysis, result, op.iv_arg, top(analysis), dirty)
    propagate_loop_carried!(analysis, result, op.body, op.init_values, dirty)
    walk!(analysis, result, op.body, dirty)
    term = op.body.terminator
    if term isa ContinueOp
        propagate_loop_carried!(analysis, result, op.body, term.values, dirty)
    end
end

# WhileOp — before.args ← init ⊔ after yields; after.args ← before's ConditionOp args.
function transfer_cf!(analysis::ForwardAnalysis, result::DataflowResult,
                      op::WhileOp, ::SSAValue, ::Block, dirty::Ref{Bool})
    propagate_loop_carried!(analysis, result, op.before, op.init_values, dirty)
    walk!(analysis, result, op.before, dirty)
    before_term = op.before.terminator
    if before_term isa ConditionOp
        propagate_loop_carried!(analysis, result, op.after, before_term.args, dirty)
    end
    walk!(analysis, result, op.after, dirty)
    after_term = op.after.terminator
    if after_term isa YieldOp
        propagate_loop_carried!(analysis, result, op.before, after_term.values, dirty)
    end
end

# LoopOp — body.args ← init ⊔ ContinueOp values ⊔ BreakOp values (walked via
# nested blocks, which the terminator walker handles).
function transfer_cf!(analysis::ForwardAnalysis, result::DataflowResult,
                      op::LoopOp, ::SSAValue, ::Block, dirty::Ref{Bool})
    propagate_loop_carried!(analysis, result, op.body, op.init_values, dirty)
    walk!(analysis, result, op.body, dirty)
    walk_terminators(op.body) do t
        if t isa ContinueOp || t isa BreakOp
            propagate_loop_carried!(analysis, result, op.body, t.values, dirty)
        end
    end
end

function propagate_loop_carried!(analysis::ForwardAnalysis, result::DataflowResult,
                                 body::Block, values::Vector, dirty::Ref{Bool})
    for (i, arg) in enumerate(body.args)
        i <= length(values) || break
        record!(analysis, result, arg, operand_value(analysis, result, values[i]), dirty)
    end
end

"""Recursively visit every (nested) block's terminator, invoking `f(terminator)`."""
function walk_terminators(f, block::Block)
    t = block.terminator
    t === nothing || f(t)
    for inst in instructions(block)
        s = stmt(inst)
        if s isa ControlFlowOp
            for sub in blocks(s)
                walk_terminators(f, sub)
            end
        end
    end
end
