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
# Consumed at codegen time by `op_predicates` (analysis/assume.jl) to
# emit sharper `Bounded(...)` predicates on `make_tensor_view` size /
# stride operands, and by `no_wrap_pass!` (transform/no_wrap.jl) to
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
signed_max_value(T::DataType) = T === Bool ? 0 : Int(typemax(signed(T)))

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
                  ops, block::Block, @nospecialize(inst))
    # Arithmetic transfers compute the exact mathematical interval, but
    # the hardware op wraps modulo the result's element width — clamp
    # each result through `clamp_to_width` so a range the runtime value
    # can escape never reaches downstream consumers.
    if func === Intrinsics.addi
        length(ops) >= 2 || return TOP_RANGE
        return clamp_to_width(
            range_add(operand_value(a, r, ops[1]), operand_value(a, r, ops[2])), inst)
    end
    if func === Intrinsics.subi
        length(ops) >= 2 || return TOP_RANGE
        return clamp_to_width(
            range_sub(operand_value(a, r, ops[1]), operand_value(a, r, ops[2])), inst)
    end
    if func === Intrinsics.muli
        length(ops) >= 2 || return TOP_RANGE
        return clamp_to_width(
            range_mul(operand_value(a, r, ops[1]), operand_value(a, r, ops[2])), inst)
    end
    if func === Intrinsics.negi
        length(ops) >= 1 || return TOP_RANGE
        return clamp_to_width(range_neg(operand_value(a, r, ops[1])), inst)
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
    # intersection), passes through for any other predicate kind. The
    # predicate is resolved through `constant_operand`: pass-constructed
    # assumes embed it raw, user-written ones arrive as a `QuoteNode` or
    # const-inferred SSAValue.
    if func === Intrinsics.assume
        length(ops) >= 2 || return TOP_RANGE
        x_range = operand_value(a, r, ops[1])
        pred = constant_operand(block, ops[2])
        if pred isa Bounded
            return range_intersect(x_range,
                                   IntRange(pred.lb === nothing ? nothing : Int(pred.lb),
                                            pred.ub === nothing ? nothing : Int(pred.ub)))
        end
        return x_range
    end

    # Casts: `bitcast` preserves the integer value. `exti` preserves it
    # for sign-extension; zero-extension reinterprets the source bits as
    # unsigned, so it is value-preserving only when the source provably
    # can't be negative. `trunci` clamps to the destination width's
    # signed range — conservatively bail to top when truncating.
    if func === Intrinsics.bitcast
        length(ops) >= 1 || return TOP_RANGE
        return operand_value(a, r, ops[1])
    end
    if func === Intrinsics.exti
        length(ops) >= 3 || return TOP_RANGE
        v = operand_value(a, r, ops[1])
        s = constant_operand(block, ops[3])
        src_T = bitinteger_eltype(value_type(block, ops[1]))
        dst_T = bitinteger_eltype(constant_operand(block, ops[2]))
        (v isa IntRange && src_T !== nothing && dst_T !== nothing) || return TOP_RANGE
        nonnegative = v.lo !== nothing && v.lo >= 0
        if s === Signedness.Signed
            source_ok = src_T <: Signed ||
                        (v.hi !== nothing && v.hi <= signed_max_value(src_T))
            target_ok = dst_T <: Signed || nonnegative
        elseif s === Signedness.Unsigned
            source_ok = !(src_T <: Signed) || nonnegative
            target_ok = !(dst_T <: Signed) ||
                        (v.hi !== nothing && v.hi <= signed_max_value(dst_T))
        else
            return TOP_RANGE
        end
        return source_ok && target_ok ? v : TOP_RANGE
    end
    if func === Intrinsics.trunci
        return TOP_RANGE
    end

    # Field access on a TileArray-typed Argument — `arr.sizes[i]` and
    # `arr.strides[i]` are non-negative by construction (Int32 widths
    # carry `[0, ∞)` even before launch-value specialisation).
    if func === Base.getfield
        ref = decode_tilearray_field(block, ops)
        ref === nothing && return TOP_RANGE
        return tilearray_field_bounds(ref)
    end

    return TOP_RANGE
end

# Override the framework's default ForOp transfer to seed the IV with
# its static bounds (see `compute_iv_range`). Back-edges are collected
# via `reachable_terminators`, mirroring the framework's ForOp handler.
function transfer_cf!(analysis::BoundsAnalysis, result::DataflowResult,
                      op::ForOp, ssa::SSAValue, ::Block, tracker::ChangeTracker)
    iv = compute_iv_range(analysis, result, op)
    record!(analysis, result, op.iv_arg, iv, tracker)

    for (arg, v) in zip(op.body.args, op.init_values)
        record!(analysis, result, arg, operand_value(analysis, result, v), tracker)
    end
    walk!(analysis, result, op.body, tracker)
    for t in reachable_terminators(op.body)
        t isa ContinueOp || continue
        for (arg, v) in zip(op.body.args, t.values)
            record!(analysis, result, arg, operand_value(analysis, result, v), tracker)
        end
    end
    record!(analysis, result, ssa, top(analysis), tracker)
end

# ForOp iterates `iv ∈ [lower, upper)` with stride `step > 0`, so
# `upper.hi - 1` is always a sound upper bound on the IV. The sharper
# `upper - step` (the last IV taken) requires the trip length to divide
# the step — slt/sle-promoted while loops with `step > 1` overshoot it
# otherwise (0:4:10 visits 8, not 10 - 4 = 6) — so it is used only when
# `lower`, `upper`, and `step` are exact and the divisibility holds.
function compute_iv_range(a::BoundsAnalysis, r::DataflowResult, op::ForOp)
    lower = operand_value(a, r, op.lower)
    upper = operand_value(a, r, op.upper)
    step  = operand_value(a, r, op.step)
    iv_lo = (lower === nothing) ? nothing : lower.lo
    iv_hi = (upper === nothing || upper.hi === nothing) ? nothing :
            sub_or_nothing(upper.hi, 1)
    if iv_hi !== nothing &&
       step !== nothing && step.lo !== nothing && step.lo == step.hi && step.lo > 1 &&
       lower !== nothing && lower.lo !== nothing && lower.lo == lower.hi &&
       upper.lo == upper.hi && mod(upper.hi - lower.lo, step.lo) == 0
        iv_hi = sub_or_nothing(upper.hi, step.lo)
    end
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
    r, overflow = Base.add_with_overflow(a, b)
    return overflow ? nothing : r
end

@inline function sub_or_nothing(a::Union{Int,Nothing}, b::Union{Int,Nothing})
    (a === nothing || b === nothing) && return nothing
    r, overflow = Base.sub_with_overflow(a, b)
    return overflow ? nothing : r
end

@inline function mul_or_nothing(a::Union{Int,Nothing}, b::Union{Int,Nothing})
    (a === nothing || b === nothing) && return nothing
    r, overflow = Base.mul_with_overflow(a, b)
    return overflow ? nothing : r
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
    IntRange(sub_or_nothing(0, a.hi), sub_or_nothing(0, a.lo))
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

# Arithmetic above is over ℤ. An open, inverted, or out-of-width interval
# may wrap at runtime, so it degrades to top.
function clamp_to_width(rng::Union{Nothing, IntRange}, @nospecialize(inst))
    rng === nothing && return rng               # ⊥ stays ⊥
    T = inst isa Instruction ? bitinteger_eltype(inst[:type]) : nothing
    T === nothing && return rng
    sizeof(T) <= 8 || return TOP_RANGE
    (rng.lo === nothing || rng.hi === nothing) && return TOP_RANGE
    lo, hi = Int128(rng.lo), Int128(rng.hi)
    return Int128(typemin(T)) <= lo <= hi <= Int128(typemax(T)) ? rng : TOP_RANGE
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

# Project a `TileArrayFieldRef` to its bound: per-axis `sizes[i]` /
# `strides[i]` reads are non-negative (Int32 fields carry `[0, ∞)` even
# before launch-value specialisation); pointer and whole-tuple reads have
# no scalar range.
function tilearray_field_bounds(ref::TileArrayFieldRef)
    ref.index === nothing && return TOP_RANGE
    (ref.field === :sizes || ref.field === :strides) || return TOP_RANGE
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
