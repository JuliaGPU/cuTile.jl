# Generic forward sparse dataflow framework.
#
# One reusable driver for monotonic forward analyses over StructuredIRCode.
# Each concrete analysis subtypes `ForwardAnalysis{T}` and implements a small
# protocol (`bottom`, `top`, `tmerge`, `transfer`). The driver handles block
# walking, fixpoint iteration, and the structured-control-flow merges at
# IfOp / ForOp / WhileOp / LoopOp.
#
# Driver shape: per-anchor lattice in a Dict, whole-CFG re-walk until no
# element changes for a full pass (capped at `max_iters`). Deliberately no
# worklist: cuTile kernels are small and our lattices are short-height, so
# the `ops × height` bound already dominates any per-op worklist overhead.
# Upgrade path if this ever becomes a bottleneck: reuse the rewrite engine's
# `Worklist` (SSAValue-keyed), generalise the key to `LatticeAnchor`, and
# add an inverse-of-`transfer_cf!` index for indirect flow through
# IfOp/ForOp/WhileOp/LoopOp (so a moved `ContinueOp.values[i]` re-enqueues
# the matching `body.args[i]`, etc.).
#
# API shape borrowed from MLIR's `SparseForwardDataFlowAnalysis` (per-Value
# lattice, queryable result post-convergence) and Julia's `AbstractLattice`
# (lattice-as-interface, fallthrough composition).

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
- `T` must support `==` — the driver uses equality on `T` to detect when a
  merge made no progress (short-circuits the `dirty` flag).
- `transfer(a, result, func, ops, block, inst)::T` — lattice element for the
  result of a resolved call. `result` is the live `DataflowResult` being built
  (read-only; use `result[op]` to query operand anchors).

Optional methods, with defaults:

- `operand_value(a, result, op)::T` — lattice value for an operand in a
  transfer-function position. Default routes anchors to `result[anchor]`
  and returns `bottom(a)` for anything else. Override to recognise raw
  literal operands (e.g. an integer literal is its own constant for a
  constant analysis).
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

Public API:
- `r[anchor]`        — read lattice value; absent keys collapse to
  `bottom(analysis)` so the query is total.
- `r[anchor] = val`  — synthesise a fresh entry (bypasses `tmerge`). Use
  only for *newly created* SSA values (e.g. a rewriter inserting a
  broadcast whose constant value is known by construction). Not for
  updating anchors that the analysis already visited — call the analysis
  again instead.
- `has_value(r, key)` — distinguish "known bottom" from "absent".
- `pairs(r)`         — iterate non-bottom (anchor, value) entries.

`values` is an internal dict backing the above; consumers must go through
this API rather than poking the field directly.
"""
struct DataflowResult{A <: ForwardAnalysis, T}
    analysis::A
    values::Dict{LatticeAnchor, T}
end

DataflowResult(a::ForwardAnalysis{T}) where {T} =
    DataflowResult{typeof(a), T}(a, Dict{LatticeAnchor, T}())

Base.getindex(r::DataflowResult, @nospecialize(key)) =
    key isa LatticeAnchor ? get(r.values, key, bottom(r.analysis)) : bottom(r.analysis)

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

"""Iterate (anchor, value) entries — only non-bottom entries are stored."""
Base.pairs(r::DataflowResult) = pairs(r.values)

"""Iterate the lattice values — only non-bottom entries are stored."""
Base.values(r::DataflowResult) = values(r.values)

"""
    operand_value(analysis, result, op) -> T

Lattice value for an operand as used *inside a transfer function*. The
default handles `LatticeAnchor`s (via dict lookup) and treats everything
else (integer/float literals, QuoteNodes, GlobalRefs, …) as `bottom`.
Analyses override this to recognise raw literals when those are
meaningful lattice inputs — e.g. a constant analysis returns
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
            result[Argument(i)] = v
        end
    end

    tracker = ChangeTracker(true, nothing)
    cap = max_iters(analysis)
    for _ in 1:cap
        tracker.dirty || return result
        tracker.dirty = false
        walk!(analysis, result, sci.entry, tracker)
    end
    tracker.dirty && error(
        "dataflow analysis $(nameof(A)) did not converge in $cap iterations " *
        "(last changed anchor: $(tracker.last_changed))")
    return result
end

"""
    ChangeTracker

Mutable iteration state. `dirty` is the outer-loop continuation signal;
`last_changed` carries the most recent anchor whose lattice value moved
(`nothing` when no change has been recorded yet this run) and feeds the
non-convergence diagnostic.
"""
mutable struct ChangeTracker
    dirty::Bool
    last_changed::Union{Nothing, LatticeAnchor}
end

"""Predicate: is `v` the lattice bottom for `analysis`?"""
is_bottom(analysis::ForwardAnalysis, @nospecialize(v)) = v === bottom(analysis)


#=============================================================================
 Block walk & transfer dispatch
=============================================================================#

function walk!(analysis::ForwardAnalysis, result::DataflowResult, block::Block, tracker::ChangeTracker)
    for inst in instructions(block)
        s = inst[:stmt]
        if s isa ControlFlowOp
            transfer_cf!(analysis, result, s, SSAValue(inst.ssa_idx), block, tracker)
        elseif s isa Expr
            transfer_call!(analysis, result, block, inst, tracker)
        end
    end
end

function transfer_call!(analysis::ForwardAnalysis, result::DataflowResult,
                        block::Block, inst::Instruction, tracker::ChangeTracker)
    call = resolve_call(block, inst)
    call === nothing && return
    func, ops = call
    new_val = transfer(analysis, result, func, ops, block, inst)
    record!(analysis, result, SSAValue(inst.ssa_idx), new_val, tracker)
end

"""
    record!(analysis, result, key, new_val, tracker)

Write `new_val` into `result` at `key`. On insert, the new value lands as-is;
on conflict, it is merged with the existing entry via `tmerge`. On any
actual change, flags `tracker.dirty` and records `key` as the last-changed
anchor (feeds the non-convergence diagnostic).
"""
function record!(analysis::ForwardAnalysis{T}, result::DataflowResult,
                 key::LatticeAnchor, new_val::T, tracker::ChangeTracker) where {T}
    if has_value(result, key)
        old = result[key]
        merged = tmerge(analysis, old, new_val)
        merged == old && return
        result[key] = merged
    else
        is_bottom(analysis, new_val) && return  # don't pollute the dict with ⊥
        result[key] = new_val
    end
    tracker.dirty = true
    tracker.last_changed = key
    return
end


#=============================================================================
 Default structured-control-flow transfer
=============================================================================#

"""
    transfer_cf!(analysis, result, op, ssa, block, tracker)

Dispatched per `ControlFlowOp` subtype. Dedicated methods below cover the
four shapes IRStructurizer currently produces (`IfOp`, `ForOp`, `WhileOp`,
`LoopOp`). Analyses rarely need to override these — add a method for a
particular `op` type if a specialized rule is required (e.g., an analysis
that sharpens based on a branch condition).

The generic fallback below walks every sub-block (so SSAs defined inside
are still visited) and conservatively records the op's own SSA as `top`.
It fires for any new `ControlFlowOp` subtype added upstream, keeping the
framework forward-compatible instead of failing with `MethodError`.
"""
function transfer_cf!(analysis::ForwardAnalysis, result::DataflowResult,
                      op::ControlFlowOp, ssa::SSAValue, ::Block, tracker::ChangeTracker)
    for sub in blocks(op)
        walk!(analysis, result, sub, tracker)
    end
    record!(analysis, result, ssa, top(analysis), tracker)
end

# IfOp — a single aggregate lattice element covering all yielded positions
# from both regions. Our IR uses one SSA per IfOp and downstream access via
# `getfield`, so a per-position lattice adds complexity without a consumer.
# Same aggregate model as the existing divisibility analysis.
function transfer_cf!(analysis::ForwardAnalysis, result::DataflowResult,
                      op::IfOp, ssa::SSAValue, ::Block, tracker::ChangeTracker)
    walk!(analysis, result, op.then_region, tracker)
    walk!(analysis, result, op.else_region, tracker)

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
    record!(analysis, result, ssa, merged, tracker)
end

# ForOp — body.args receives loop-carried values: join(init_values, ContinueOp
# yields). The IV lives in `op.iv_arg` (not in `body.args`), so init/continue
# values line up positionally with `body.args` without skipping slot 1.
# The IV is seeded to `top` (unknown at compile time).
function transfer_cf!(analysis::ForwardAnalysis, result::DataflowResult,
                      op::ForOp, ::SSAValue, ::Block, tracker::ChangeTracker)
    record!(analysis, result, op.iv_arg, top(analysis), tracker)
    for (arg, v) in zip(op.body.args, op.init_values)
        record!(analysis, result, arg, operand_value(analysis, result, v), tracker)
    end
    walk!(analysis, result, op.body, tracker)
    term = op.body.terminator
    if term isa ContinueOp
        for (arg, v) in zip(op.body.args, term.values)
            record!(analysis, result, arg, operand_value(analysis, result, v), tracker)
        end
    end
end

# WhileOp — before.args ← init ⊔ after yields; after.args ← before's ConditionOp args.
function transfer_cf!(analysis::ForwardAnalysis, result::DataflowResult,
                      op::WhileOp, ::SSAValue, ::Block, tracker::ChangeTracker)
    for (arg, v) in zip(op.before.args, op.init_values)
        record!(analysis, result, arg, operand_value(analysis, result, v), tracker)
    end
    walk!(analysis, result, op.before, tracker)
    before_term = op.before.terminator
    if before_term isa ConditionOp
        for (arg, v) in zip(op.after.args, before_term.args)
            record!(analysis, result, arg, operand_value(analysis, result, v), tracker)
        end
    end
    walk!(analysis, result, op.after, tracker)
    after_term = op.after.terminator
    if after_term isa YieldOp
        for (arg, v) in zip(op.before.args, after_term.values)
            record!(analysis, result, arg, operand_value(analysis, result, v), tracker)
        end
    end
end

# LoopOp — body.args ← init ⊔ ContinueOp values ⊔ BreakOp values. ContinueOp /
# BreakOp can appear directly as `body.terminator` or as the terminator of a
# nested IfOp branch (transparent); nested LoopOp/ForOp/WhileOp introduce their
# own scope and their terminators target *themselves*, not this outer LoopOp.
# `reachable_terminators` encodes that scoping rule.
function transfer_cf!(analysis::ForwardAnalysis, result::DataflowResult,
                      op::LoopOp, ::SSAValue, ::Block, tracker::ChangeTracker)
    for (arg, v) in zip(op.body.args, op.init_values)
        record!(analysis, result, arg, operand_value(analysis, result, v), tracker)
    end
    walk!(analysis, result, op.body, tracker)
    for t in reachable_terminators(op.body)
        if t isa ContinueOp || t isa BreakOp
            for (arg, v) in zip(op.body.args, t.values)
                record!(analysis, result, arg, operand_value(analysis, result, v), tracker)
            end
        end
    end
end
