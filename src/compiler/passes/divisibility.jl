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
