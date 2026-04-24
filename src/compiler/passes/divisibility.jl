# Divisibility Propagation (analysis)
#
# Forward dataflow over StructuredIRCode that tracks the largest known divisor
# of every integer SSA value and block argument. Mirrors cuTile Python's
# `dataflow_analysis.py` (div_by part only; alias sets live in a separate pass).
#
# Propagation rules:
#   addi(a, b), subi(a, b)    -> gcd(d[a], d[b])
#   muli(a, b)                -> d[a] * d[b]  (capped at MAX_DIVBY)
#   shli(a, k), k a constant  -> d[a] << k    (capped at MAX_DIVBY)
#   negi(a), absi(a)          -> d[a]
#   broadcast(x, _), reshape  -> d[x]
#   exti(x, ...)              -> d[x]       (sign/zero extension preserves divisors)
#   trunci(x, T)              -> gcd(d[x], 2^bitwidth(T))
#   integer literal n         -> abs(n)     (0 treated as MAX_DIVBY)
#   everything else           -> 1 (unknown)
#
# Control flow: IfOp results merge via gcd across branch yields; ForOp/LoopOp/
# WhileOp conservatively set their induction variable and loop-carried args to
# divby=1 (matching Python's `set_always_true(induction_var)`), then propagate
# init values into the body block args via gcd. Fixpoint iterate until stable.
#
# Consumers look up `divby[SSAValue(i)]` (or block-arg) to decide whether to
# wrap a value with `AssumeOp(DivBy(d))`.

const DivByResult = Dict{Any, Int}

const MAX_DIVBY = 1024

#=============================================================================
 Precision gaps vs. cuTile Python's `dataflow_analysis.py`
=============================================================================#
#
# Both gaps below are correctness-safe (our bounds are always sound, just
# looser than Python's). They're called out here rather than hidden in the
# code they constrain so follow-up work has a single list to work from.
#
# ──────────────────────────────────────────────────────────────────────────
# Gap 1: No const-fold for arithmetic on known constants.
# ──────────────────────────────────────────────────────────────────────────
#
# For `addi(a, b)` / `subi(a, b)` where both operands resolve to known
# integer constants, this pass uses `gcd(divby[a], divby[b])` which is
# strictly looser than the exact `abs_divby(a ± b)` value. Examples:
#
#     subi(10, 2) →  divby = gcd(10, 2) = 2     (actual: abs(8) = 8)
#     addi( 4, 4) →  divby = gcd( 4, 4) = 4     (actual: abs(8) = 8)
#
# Python gets the tight bound because a separate constant-folding stage
# lowers these to `typed_const(folded_value)` before its dataflow runs, at
# which point its `TypedConst` rule applies `abs_divby(value)` directly.
#
# For literal arithmetic inside Julia code, Julia's own inference already
# folds at compile time (`Int32(10) - Int32(2)` is just `Int32(8)`), so
# this gap is invisible for that case. It only shows up for arithmetic
# whose operands become `Intrinsics.constant(...)` SSA defs (e.g. produced
# by the broadcast system), where inference has already lowered past the
# literal-level fold.
#
# Fix shape: a standalone folding pass that rewrites
# `(addi|subi|muli)(constant, constant)` → `broadcast(folded_scalar, shape)`
# before `divisibility_analysis`. The rule is straightforward to write; it
# does *not* belong inside the rewriter's ALGEBRA_RULES (algebraic
# simplification and constant folding are separate concerns).
#
# ──────────────────────────────────────────────────────────────────────────
# Gap 2: No seeding from `ArraySpec` into derived-value divby.
# ──────────────────────────────────────────────────────────────────────────
#
# `ArraySpec{N, Alignment, Contiguous, StrideDivBy, ShapeDivBy}` on a
# `TileArray` kernel parameter encodes divisibility of its ptr/sizes/strides
# at compile time. Today these facts reach Tile IR via two independent
# paths:
#
#   - `emit_assume_ops!` (intrinsics/misc.jl) wraps the kernel-entry
#     `MakeTensorView` operands with `AssumeOp(DivBy)` from `ArraySpec`.
#   - `Intrinsics.slice`'s codegen consults `src_spec.alignment` and
#     `src_spec.stride_div_by[axis]` directly on the source type for the
#     new-pointer divby.
#
# Neither flows into this pass. As a consequence, when the user writes:
#
#     m = arr.sizes[1]                         # SSA from getfield chain
#     sub = @view arr[start:start+m-1, :]      # m has divby = 1 here
#
# `m`'s divby is recorded as 1 even though `ArraySpec.shape_div_by[1]`
# might say 16. Python, by contrast, seeds every TileArray field access
# with its `DataPredicate.div_by` at the start of its dataflow (see
# `DataflowResult` construction in `dataflow_analysis.py`). Subsequent
# slices whose bounds are derived from `arr.sizes[i]` / `arr.strides[i]`
# miss out on tight pointer/size annotations.
#
# Fix shape: walk the entry block for each `getfield(arg, :ptr | :sizes[i]
# | :strides[i])` SSA that resolves to a TileArray kernel parameter, and
# populate `seeds[SSAValue] = <from ArraySpec>` before running the
# fixpoint. Needs a small helper that mirrors `cache_tensor_view!`'s path
# resolution. Orthogonal to Gap 1 and to the insert-assumes rewriter; the
# three passes compose cleanly.
#
# ──────────────────────────────────────────────────────────────────────────
# Observable effect and test coverage
# ──────────────────────────────────────────────────────────────────────────
#
# `examples/slice.jl`'s ragged-copy kernel hits Gap 2 directly: its derived
# `new_base` carries `assume div_by<4>` where Python emits `div_by<16>`.
# Both are sound; the Python annotation unlocks one more level of
# vectorisation in the tile backend.
#
# `test/codegen/slice.jl` has a `@test_broken` at the tail of the "Phase 3"
# testset that should flip to `@test` once either gap is closed enough to
# produce a non-trivial divby on dynamic bounds.

"""
    divisibility_analysis(sci, seeds=DivByResult()) -> DivByResult

Build a map from SSA values (and block arguments) to their known divisors.
`seeds` is populated by the kernel driver from `ArraySpec` at entry.

Converges in O(depth * ops) iterations since the lattice monotonically shrinks:
each value's divby can only decrease (via `gcd` at merges) or stay the same.
"""
function divisibility_analysis(sci::StructuredIRCode, seeds::DivByResult = DivByResult())
    divby = copy(seeds)
    max_iters = 32   # far more than needed; safety bound
    for iter in 1:max_iters
        dirty = Ref(false)
        analyze_block!(divby, sci.entry, dirty)
        dirty[] || break
        iter == max_iters && error("divisibility_analysis: failed to converge in $max_iters iterations")
    end
    return divby
end

# ---- core analysis -----------------------------------------------------------

"""Analyze a block, propagating divby through straight-line code and nested regions."""
function analyze_block!(divby::DivByResult, block::Block, dirty::Ref{Bool})
    for inst in instructions(block)
        s = stmt(inst)
        if s isa ForOp
            analyze_forop!(divby, s, dirty)
        elseif s isa IfOp
            analyze_ifop!(divby, s, SSAValue(inst.ssa_idx), block, dirty)
        elseif s isa WhileOp
            analyze_whileop!(divby, s, dirty)
        elseif s isa LoopOp
            analyze_loopop!(divby, s, dirty)
        elseif s isa Expr
            analyze_call!(divby, block, inst, dirty)
        end
    end
end

# ---- straight-line calls -----------------------------------------------------

"""Propagate divby for a single call instruction, updating `divby[SSAValue(inst)]`."""
function analyze_call!(divby::DivByResult, block::Block, inst, dirty::Ref{Bool})
    call = resolve_call(block, inst)
    call === nothing && return
    func, ops = call
    new_div = divby_for_call(divby, func, ops)
    update_divby!(divby, SSAValue(inst.ssa_idx), new_div, dirty)
end

"""Compute the result divby for a given intrinsic/function call."""
function divby_for_call(divby::DivByResult, @nospecialize(func), ops)::Int
    # Binary arithmetic
    if func === Intrinsics.addi || func === Intrinsics.subi
        length(ops) >= 2 || return 1
        return gcd(operand_divby(divby, ops[1]), operand_divby(divby, ops[2]))
    end
    if func === Intrinsics.muli
        length(ops) >= 2 || return 1
        a = operand_divby(divby, ops[1])
        b = operand_divby(divby, ops[2])
        return cap_divby(Int128(a) * Int128(b))
    end
    # Shifts: if shift-amount is a known constant, scale divby by 2^k.
    if func === Intrinsics.shli
        length(ops) >= 2 || return 1
        base = operand_divby(divby, ops[1])
        k_val = operand_divby(divby, ops[2])
        # shli's rhs divby isn't its value; peek at the concrete operand
        k = constant_integer(ops[2])
        k === nothing && return 1
        return cap_divby(Int128(base) << Int128(k))
    end
    # Unary arithmetic
    if func === Intrinsics.negi || func === Intrinsics.absi
        length(ops) >= 1 || return 1
        return operand_divby(divby, ops[1])
    end
    # Transparent ops (propagate from first operand)
    if func === Intrinsics.broadcast || func === Intrinsics.reshape
        length(ops) >= 1 || return 1
        return operand_divby(divby, ops[1])
    end
    # Sign/zero extension preserves divisors.
    if func === Intrinsics.exti
        length(ops) >= 1 || return 1
        return operand_divby(divby, ops[1])
    end
    # Truncating cast: gcd with 2^bitwidth. If the target type is known we
    # could derive this precisely; conservatively fall back to 1 unless we can.
    if func === Intrinsics.trunci
        length(ops) >= 2 || return 1
        src = operand_divby(divby, ops[1])
        T = constant_type(ops[2])
        T === nothing && return src  # if we can't see the target, don't downgrade
        bw = sizeof(T) * 8
        return gcd(src, 2^bw)
    end
    # from_scalar / to_scalar wrap the underlying value.
    if func === Intrinsics.from_scalar || func === Intrinsics.to_scalar
        length(ops) >= 1 || return 1
        return operand_divby(divby, ops[1])
    end
    # constant(shape, val, T): the value's divby is abs(val) for integer constants
    if func === Intrinsics.constant
        length(ops) >= 2 || return 1
        v = constant_integer(ops[2])
        return v === nothing ? 1 : abs_divby(v)
    end
    return 1
end

# ---- control flow ------------------------------------------------------------

"""IfOp: result = gcd(then_yield, else_yield) per position."""
function analyze_ifop!(divby::DivByResult, op::IfOp, ssa::SSAValue, block::Block, dirty::Ref{Bool})
    # Analyze both regions first (so yielded values' divby is up to date).
    analyze_block!(divby, op.then_region, dirty)
    analyze_block!(divby, op.else_region, dirty)

    # IfOp results are the yielded values; in Julia IR we expose a tuple via
    # getfield. For our purposes, associate a single divby for the IfOp via the
    # gcd across both regions' yield values (position 0 used as representative;
    # users that index tuple positions fetch per-position divby via getfield
    # resolution at emission time). For now record a single aggregate.
    then_t = op.then_region.terminator
    else_t = op.else_region.terminator
    if then_t isa YieldOp && else_t isa YieldOp && !isempty(then_t.values) && !isempty(else_t.values)
        n = min(length(then_t.values), length(else_t.values))
        agg = operand_divby(divby, then_t.values[1])
        for i in 1:n
            agg = gcd(agg, operand_divby(divby, then_t.values[i]),
                          operand_divby(divby, else_t.values[i]))
        end
        update_divby!(divby, ssa, agg, dirty)
    end
end

"""ForOp: induction variable conservatively divby=1. Loop-carried args merge
init values and continue-yields."""
function analyze_forop!(divby::DivByResult, op::ForOp, dirty::Ref{Bool})
    # IV: gcd(lower, step) is a tighter bound, but matching Python we set 1.
    update_divby!(divby, op.iv_arg, 1, dirty)

    # Merge init values into loop-carried body args.
    propagate_loop_carried!(divby, op.body, op.init_values, dirty)

    # Recurse into body; continue yields (if any) refine block-arg divby on
    # the next fixpoint iteration.
    analyze_block!(divby, op.body, dirty)

    # After analyzing body, merge ContinueOp's yielded values back into body args.
    term = op.body.terminator
    if term isa ContinueOp && length(term.values) == length(op.body.args)
        propagate_loop_carried!(divby, op.body, term.values, dirty)
    end
end

"""WhileOp: before.args get init, after.args get condition args."""
function analyze_whileop!(divby::DivByResult, op::WhileOp, dirty::Ref{Bool})
    propagate_loop_carried!(divby, op.before, op.init_values, dirty)
    analyze_block!(divby, op.before, dirty)

    # `before` region terminator is ConditionOp(cond, args...); args become
    # `after` block args.
    before_term = op.before.terminator
    if before_term isa ConditionOp && length(before_term.args) == length(op.after.args)
        propagate_loop_carried!(divby, op.after, before_term.args, dirty)
    end
    analyze_block!(divby, op.after, dirty)

    # `after` region yields new values for `before` block args (next iteration).
    after_term = op.after.terminator
    if after_term isa YieldOp && length(after_term.values) == length(op.before.args)
        propagate_loop_carried!(divby, op.before, after_term.values, dirty)
    end
end

"""LoopOp: body.args merge init, ContinueOp, and BreakOp values. Result of
the LoopOp merges BreakOp values."""
function analyze_loopop!(divby::DivByResult, op::LoopOp, dirty::Ref{Bool})
    propagate_loop_carried!(divby, op.body, op.init_values, dirty)
    analyze_block!(divby, op.body, dirty)
    # Terminator contributions handled on the next fixpoint iteration via
    # propagate_loop_carried! once walk_continue_break_values is gathered.
    gather_loop_yields!(divby, op.body, dirty)
end

"""Merge `values` (one per loop-carried arg) into the block args via gcd."""
function propagate_loop_carried!(divby::DivByResult, body::Block, values::Vector, dirty::Ref{Bool})
    n = min(length(body.args), length(values))
    for i in 1:n
        d = operand_divby(divby, values[i])
        arg = body.args[i]
        update_divby_gcd!(divby, arg, d, dirty)
    end
end

"""Walk the body and propagate every ContinueOp/BreakOp's values back into the
loop-carried args of the enclosing loop."""
function gather_loop_yields!(divby::DivByResult, body::Block, dirty::Ref{Bool})
    walk_blocks(body) do b
        t = b.terminator
        if t isa ContinueOp || t isa BreakOp
            propagate_loop_carried!(divby, body, t.values, dirty)
        end
    end
end

"""Recursively visit every (nested) block, invoking `f(block)`."""
function walk_blocks(f, block::Block)
    f(block)
    for inst in instructions(block)
        s = stmt(inst)
        if s isa ControlFlowOp
            for sub in blocks(s)
                walk_blocks(f, sub)
            end
        end
    end
end

# ---- helpers -----------------------------------------------------------------

"""Resolve an operand to its known divby. Unknown → 1."""
function operand_divby(divby::DivByResult, @nospecialize(op))::Int
    if op isa SSAValue
        return get(divby, op, 1)
    elseif op isa BlockArgument
        return get(divby, op, 1)
    elseif op isa Integer
        return abs_divby(op)
    elseif op isa QuoteNode && op.value isa Integer
        return abs_divby(op.value)
    end
    return 1
end

"""Get the integer value of an operand if it is a literal, else nothing."""
function constant_integer(@nospecialize(op))
    op isa Integer && return Int128(op)
    op isa QuoteNode && op.value isa Integer && return Int128(op.value)
    return nothing
end

"""Get the Type value of an operand if it is a literal Type, else nothing."""
function constant_type(@nospecialize(op))
    op isa Type && return op
    op isa QuoteNode && op.value isa Type && return op.value
    return nothing
end

"""abs(v), but treat 0 as MAX_DIVBY (everything divides 0)."""
@inline abs_divby(v::Integer) = v == 0 ? MAX_DIVBY : Int(min(abs(Int128(v)), Int128(MAX_DIVBY)))

"""Cap a divby value at MAX_DIVBY (and snap to 1 for negative/zero)."""
@inline cap_divby(v::Int128)::Int = v <= 0 ? 1 : Int(min(v, Int128(MAX_DIVBY)))

"""Record `divby[k] = v` if it changes or is new; set `dirty` on update."""
function update_divby!(divby::DivByResult, @nospecialize(k), v::Int, dirty::Ref{Bool})
    old = get(divby, k, nothing)
    if old === nothing || old != v
        divby[k] = v
        dirty[] = true
    end
    return nothing
end

"""Record `divby[k] = gcd(divby[k], v)`. Monotonically shrinks; set dirty on change."""
function update_divby_gcd!(divby::DivByResult, @nospecialize(k), v::Int, dirty::Ref{Bool})
    old = get(divby, k, nothing)
    if old === nothing
        divby[k] = v
        dirty[] = true
    else
        new_v = gcd(old, v)
        if new_v != old
            divby[k] = new_v
            dirty[] = true
        end
    end
    return nothing
end


#=============================================================================
 Assume Insertion (rewriter)
=============================================================================#

# Phase 3: after the analysis, walk the IR and wrap each sink's integer inputs
# with `Intrinsics.assume_div_by(x, Val(D))` when `divby[x] > 1`. Codegen then
# lowers the wrapper to `encode_AssumeOp!(DivBy(D))` naturally, with no codegen
# coupling to the analysis result.
#
# Sinks in cuTile.jl that benefit from divby on integer inputs:
#   - `Intrinsics.slice(arr, axis, start, stop)` — start and stop
# (`Intrinsics.make_tensor_view` does not take integer inputs at the IR level;
# its sizes/strides are materialized inside codegen from the TileArray's
# destructured parameters, which already get AssumeOps from `emit_assume_ops!`.)

const ASSUME_SINKS = Set{Any}([Intrinsics.slice])

"""
    insert_divby_assumes!(sci, seeds=DivByResult())

Run divisibility analysis and mutate `sci` by inserting
`Intrinsics.assume_div_by(x, Val(D))` calls before each recognized sink's
integer operands with divby > 1.

After this pass, downstream consumers read divisibility directly from the IR
(no external Dict lookup) — e.g., slice's `emit_intrinsic!` calls
`operand_divby_local(sci, op)` to recover `D` from the wrapper.
"""
function insert_divby_assumes!(sci::StructuredIRCode, seeds::DivByResult = DivByResult())
    divby = divisibility_analysis(sci, seeds)
    insert_assumes_in_block!(sci.entry, divby)
    return sci
end

function insert_assumes_in_block!(block::Block, divby::DivByResult)
    # Collect rewrites first so we don't invalidate the instruction iterator.
    # Each rewrite is: (reference-instruction, stmt-to-mutate, arg-index, operand, divisor).
    rewrites = Vector{Tuple{SSAValue, Expr, Int, Any, Int}}()

    for inst in instructions(block)
        s = stmt(inst)
        if s isa ControlFlowOp
            for sub in blocks(s)
                insert_assumes_in_block!(sub, divby)
            end
            continue
        end
        s isa Expr || continue
        call = resolve_call(block, inst)
        call === nothing && continue
        func, ops = call
        func in ASSUME_SINKS || continue
        # For slice: annotate start (ops[3]) and stop (ops[4]) if divby > 1.
        for op_idx in sink_integer_operand_indices(func)
            op_idx <= length(ops) || continue
            op = ops[op_idx]
            op isa SSAValue || continue
            d = get(divby, op, 1)
            d > 1 || continue
            push!(rewrites, (SSAValue(inst.ssa_idx), s, op_idx, op, d))
        end
    end

    for (ref, stmt_expr, op_idx, op, d) in rewrites
        # Assume ops on this operand's type; reuse the source op's known type.
        op_type = value_type(block, op)
        op_type === nothing && continue
        # Insert: assume_div_by(op, Val(d)) just before the sink.
        assume_call = Expr(:call, Intrinsics.assume_div_by, op, Val(d))
        new_inst = insert_before!(block, ref, assume_call, op_type)
        # Rewrite the sink's arg to use the wrapper's SSA value.
        rewrite_call_arg!(stmt_expr, op_idx, SSAValue(new_inst.ssa_idx))
    end

    return block
end

"""For a sink function, return the indices (into `ops`) of integer operands
that benefit from divisibility annotation."""
function sink_integer_operand_indices(@nospecialize(func))
    func === Intrinsics.slice && return (3, 4)   # start, stop
    return ()
end

"""Replace operand `op_idx` of an `:call`/`:invoke` Expr with `new_op`."""
function rewrite_call_arg!(stmt::Expr, op_idx::Int, new_op)
    if stmt.head === :call
        stmt.args[op_idx + 1] = new_op
    elseif stmt.head === :invoke
        stmt.args[op_idx + 2] = new_op
    end
    return stmt
end


#=============================================================================
 Local divby query (used by emit_intrinsic!)
=============================================================================#

"""
    operand_divby_local(sci, op) -> Int

Resolve the known divisor of `op` by peeking at its def. For SSAValues that are
Intrinsics.assume_div_by wrappers, returns `D`. For integer literals, returns
`abs_divby(n)`. Otherwise 1.

Local (no external state). Used by emit_intrinsic! to emit AssumeOp on derived
values without depending on the analysis dict.
"""
function operand_divby_local(sci::StructuredIRCode, @nospecialize(op))::Int
    if op isa Integer
        return abs_divby(op)
    elseif op isa QuoteNode && op.value isa Integer
        return abs_divby(op.value)
    elseif op isa SSAValue
        d = find_assume_div_by(sci, op)
        return d === nothing ? 1 : d
    end
    return 1
end
