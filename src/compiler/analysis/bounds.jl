# Bounds Analysis
#
# Tracks integer-valued SSA values to a closed interval `[lo, hi]` (with
# optionally-open endpoints). Mirrors LLVM's `ConstantRange` /
# `ValueTracking::computeConstantRange` adapted to `StructuredIRCode`.
# Sibling to `DivByAnalysis` (analysis/divisibility.jl), built on the
# same `ForwardAnalysis` framework.
#
# Convergence: aggressive widening — `tmerge(a, b)` keeps each endpoint
# only when both inputs agree on it, otherwise widens to `nothing`
# (±∞). This caps each anchor's lattice walk at *two* state changes
# (bottom → some IntRange → wider IntRange / top), so loop back-edges
# converge in one or two iterations regardless of body shape. Loses
# precision on loop-carried values that vary per iteration; gains
# precision on literal-constant flow and ForOp induction variables.
#
# Consumed by `analyze_assume_info` (analysis/assume.jl) to emit
# sharper `Bounded(...)` predicates on `make_tensor_view` operands at
# codegen time, and by `no_wrap_pass!` (transform/no_wrap.jl) to
# attach `nsw`/`nuw` flags on integer arithmetic that provably fits in
# its destination width.

"""
    IntRange(lo, hi)

Closed integer interval `[lo, hi]`. Either endpoint may be `nothing`
to indicate an open bound (`-∞` for `lo`, `+∞` for `hi`).
`IntRange(nothing, nothing)` is the top element of the lattice;
`IntRange(n, n)` is an exact known value.
"""
struct IntRange
    lo::Union{Int, Nothing}
    hi::Union{Int, Nothing}
end

const TOP_RANGE = IntRange(nothing, nothing)
nonneg_range() = IntRange(0, nothing)

"""
    BoundsAnalysis

Forward interval analysis. Lattice element `Union{Nothing, IntRange}`:
`nothing` is bottom (not yet seen), `IntRange` carries a closed
interval, `TOP_RANGE` is the top element.
"""
struct BoundsAnalysis <: ForwardAnalysis{Union{Nothing, IntRange}} end

bottom(::BoundsAnalysis) = nothing
top(::BoundsAnalysis) = TOP_RANGE

# Aggressive widening LUB. Each endpoint is preserved iff both inputs
# agree on it; on disagreement the endpoint widens to `nothing` (open).
# Two `IntRange`s with different endpoints meet at the open form, then
# stay there — bounded ascent in O(1) per anchor.
function tmerge(::BoundsAnalysis, a::Union{Nothing, IntRange},
                b::Union{Nothing, IntRange})
    a === nothing && return b
    b === nothing && return a
    lo = a.lo == b.lo ? a.lo : nothing
    hi = a.hi == b.hi ? a.hi : nothing
    return IntRange(lo, hi)
end

init_arg(::BoundsAnalysis, ::Int, @nospecialize(_)) = TOP_RANGE

function operand_value(::BoundsAnalysis, r::DataflowResult, @nospecialize(op))
    if op isa Integer
        v = Int(op)
        return IntRange(v, v)
    end
    if op isa QuoteNode && op.value isa Integer
        v = Int(op.value)
        return IntRange(v, v)
    end
    op isa LatticeAnchor && return r[op]
    return TOP_RANGE
end

#=============================================================================
 Transfer rules
=============================================================================#

function transfer(a::BoundsAnalysis, r::DataflowResult, @nospecialize(func),
                  ops, block::Block, ::Any)
    if func === Intrinsics.addi
        length(ops) >= 2 || return TOP_RANGE
        return range_add(operand_value(a, r, ops[1]), operand_value(a, r, ops[2]))
    end
    if func === Intrinsics.subi
        length(ops) >= 2 || return TOP_RANGE
        return range_sub(operand_value(a, r, ops[1]), operand_value(a, r, ops[2]))
    end
    if func === Intrinsics.muli
        length(ops) >= 2 || return TOP_RANGE
        return range_mul(operand_value(a, r, ops[1]), operand_value(a, r, ops[2]))
    end
    if func === Intrinsics.negi
        length(ops) >= 1 || return TOP_RANGE
        return range_neg(operand_value(a, r, ops[1]))
    end

    if func === Intrinsics.constant
        length(ops) >= 2 || return TOP_RANGE
        sv = ops[2]
        sv isa Integer && return IntRange(Int(sv), Int(sv))
        sv isa QuoteNode && sv.value isa Integer && return IntRange(Int(sv.value), Int(sv.value))
        return TOP_RANGE
    end

    # Pure shape ops — element-wise range passes through.
    if func === Intrinsics.broadcast || func === Intrinsics.reshape
        length(ops) >= 1 || return TOP_RANGE
        return operand_value(a, r, ops[1])
    end

    # `Intrinsics.assume(x, predicate)` — refines via `Bounded` (interval
    # intersection), passes through for any other predicate kind.
    if func === Intrinsics.assume
        length(ops) >= 2 || return TOP_RANGE
        x_range = operand_value(a, r, ops[1])
        pred = ops[2]
        if pred isa Bounded
            return range_intersect(x_range,
                                   IntRange(pred.lb === nothing ? nothing : Int(pred.lb),
                                            pred.ub === nothing ? nothing : Int(pred.ub)))
        end
        return x_range
    end

    # Casts: `bitcast`/`exti` preserve the integer value (exti is sign-
    # or zero-extend; both widen the type without changing the
    # mathematical value). `trunci` clamps to the destination width's
    # signed range — conservatively bail to top when truncating.
    if func === Intrinsics.bitcast || func === Intrinsics.exti
        length(ops) >= 1 || return TOP_RANGE
        return operand_value(a, r, ops[1])
    end
    if func === Intrinsics.trunci
        return TOP_RANGE
    end

    # Field access on a TileArray-typed Argument — `arr.sizes[i]` and
    # `arr.strides[i]` are non-negative by construction (Int32 widths
    # carry `[0, ∞)` even before launch-value specialisation).
    if func === Base.getfield
        return getfield_bounds(r, ops, block)
    end

    return TOP_RANGE
end

# Override the framework's default ForOp transfer to seed the IV with
# its static bounds. ForOp iterates `iv ∈ [lower, upper)` with stride
# `step`; when `lower`/`upper`/`step` are literal-resolvable, the IV
# range is exactly `[lower, last_iv]` where `last_iv = upper - step`
# (assuming `step > 0`). Without literal bounds, the IV defaults to
# top (the framework's default).
function transfer_cf!(analysis::BoundsAnalysis, result::DataflowResult,
                      op::ForOp, ::SSAValue, ::Block, tracker::ChangeTracker)
    iv = compute_iv_range(analysis, result, op)
    record!(analysis, result, op.iv_arg, iv, tracker)

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

function compute_iv_range(a::BoundsAnalysis, r::DataflowResult, op::ForOp)
    lower = operand_value(a, r, op.lower)
    upper = operand_value(a, r, op.upper)
    step  = operand_value(a, r, op.step)
    iv_lo = (lower === nothing) ? nothing : lower.lo
    # Last IV is `upper - step` (loop iterates while iv < upper). Need
    # both an upper bound on `upper` and a lower bound on `step` to
    # compute it.
    iv_hi = (upper === nothing || step === nothing ||
             upper.hi === nothing || step.lo === nothing || step.lo < 1) ?
            nothing : upper.hi - step.lo
    return IntRange(iv_lo, iv_hi)
end

#=============================================================================
 Interval arithmetic
=============================================================================#

# `nothing` is ±∞; arithmetic with it produces `nothing`. Integer
# arithmetic is otherwise saturated at `Int` limits — overflow at the
# lattice level widens to `nothing` rather than wrapping.

@inline function add_or_nothing(a::Union{Int,Nothing}, b::Union{Int,Nothing})
    (a === nothing || b === nothing) && return nothing
    r = a + b
    # Detect overflow (signed): result has different sign from inputs.
    (a > 0 && b > 0 && r < 0) && return nothing
    (a < 0 && b < 0 && r > 0) && return nothing
    return r
end

@inline function sub_or_nothing(a::Union{Int,Nothing}, b::Union{Int,Nothing})
    (a === nothing || b === nothing) && return nothing
    r = a - b
    (a >= 0 && b < 0 && r < 0) && return nothing
    (a < 0 && b > 0 && r > 0) && return nothing
    return r
end

@inline function mul_or_nothing(a::Union{Int,Nothing}, b::Union{Int,Nothing})
    (a === nothing || b === nothing) && return nothing
    a == 0 && return 0
    b == 0 && return 0
    abs(a) > typemax(Int) ÷ abs(b) && return nothing
    return a * b
end

range_add(a::Union{Nothing,IntRange}, b::Union{Nothing,IntRange}) = begin
    (a === nothing || b === nothing) && return nothing
    IntRange(add_or_nothing(a.lo, b.lo), add_or_nothing(a.hi, b.hi))
end

range_sub(a::Union{Nothing,IntRange}, b::Union{Nothing,IntRange}) = begin
    (a === nothing || b === nothing) && return nothing
    IntRange(sub_or_nothing(a.lo, b.hi), sub_or_nothing(a.hi, b.lo))
end

range_neg(a::Union{Nothing,IntRange}) = begin
    a === nothing && return nothing
    lo = a.hi === nothing ? nothing : -a.hi
    hi = a.lo === nothing ? nothing : -a.lo
    IntRange(lo, hi)
end

# General multiplication — pick min/max over all four corner products.
function range_mul(a::Union{Nothing,IntRange}, b::Union{Nothing,IntRange})
    (a === nothing || b === nothing) && return nothing
    # Common fast path: both ranges are non-negative (sizes/strides ×
    # offsets etc.). Result is `[a.lo*b.lo, a.hi*b.hi]`.
    if a.lo !== nothing && a.lo >= 0 && b.lo !== nothing && b.lo >= 0
        return IntRange(mul_or_nothing(a.lo, b.lo), mul_or_nothing(a.hi, b.hi))
    end
    # General case bails to top: signed multiplication's range depends
    # on all four corners, and the open-endpoint logic gets fiddly.
    # Worth elaborating only if a workload demands it.
    return TOP_RANGE
end

function range_intersect(a::Union{Nothing,IntRange}, b::Union{Nothing,IntRange})
    a === nothing && return b
    b === nothing && return a
    lo = a.lo === nothing ? b.lo :
         b.lo === nothing ? a.lo : max(a.lo, b.lo)
    hi = a.hi === nothing ? b.hi :
         b.hi === nothing ? a.hi : min(a.hi, b.hi)
    return IntRange(lo, hi)
end

#=============================================================================
 getfield bounds
=============================================================================#

# Recognise `getfield(arg::TileArray, :ptr|:sizes|:strides)` (top — they
# return tuples or pointers, no integer range) and the two-step chain
# `getfield(getfield(arg, :sizes|:strides), i)` (non-negative integer).
function getfield_bounds(r::DataflowResult, ops, block::Block)
    length(ops) >= 2 || return TOP_RANGE
    obj = ops[1]
    field = ops[2] isa QuoteNode ? ops[2].value : ops[2]

    obj_T = value_type(block, obj)
    obj_T = obj_T === nothing ? Any : CC.widenconst(obj_T)
    obj_T <: TileArray && return TOP_RANGE  # ptr or whole tuple — no scalar range

    obj isa SSAValue || return TOP_RANGE
    obj_def = lookup_def_call(block, obj)
    obj_def === nothing && return TOP_RANGE
    obj_func, obj_ops = obj_def
    obj_func === Base.getfield || return TOP_RANGE
    length(obj_ops) >= 2 || return TOP_RANGE

    inner_field = obj_ops[2] isa QuoteNode ? obj_ops[2].value : obj_ops[2]
    (inner_field === :sizes || inner_field === :strides) || return TOP_RANGE

    inner_T = value_type(block, obj_ops[1])
    inner_T = inner_T === nothing ? Any : CC.widenconst(inner_T)
    inner_T <: TileArray || return TOP_RANGE

    return nonneg_range()
end

#=============================================================================
 Public query API
=============================================================================#

const BoundsInfo = DataflowResult{BoundsAnalysis, Union{Nothing, IntRange}}

"""
    analyze_bounds(sci::StructuredIRCode) -> BoundsInfo

Run forward bounds analysis on `sci`.
"""
analyze_bounds(sci::StructuredIRCode) = analyze(BoundsAnalysis(), sci)::BoundsInfo

"""
    bounds(info, op) -> IntRange

Resolve an operand to its known interval. Returns `TOP_RANGE` for
unknown / non-integer operands. Returns an exact `IntRange(n, n)` for
integer literals.
"""
function bounds(info::BoundsInfo, @nospecialize(op))
    if op isa Integer
        v = Int(op)
        return IntRange(v, v)
    end
    if op isa QuoteNode && op.value isa Integer
        v = Int(op.value)
        return IntRange(v, v)
    end
    if op isa LatticeAnchor
        v = info[op]
        return v === nothing ? TOP_RANGE : v
    end
    return TOP_RANGE
end

bounds(::Nothing, @nospecialize(_)) = TOP_RANGE
