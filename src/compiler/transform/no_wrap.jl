# No-Wrap Inference
#
# Attaches `no_signed_wrap` / `no_unsigned_wrap` flags to integer
# arithmetic ops (`addi`, `subi`, `muli`) when the bounds analysis
# proves the result fits in the destination width without
# wraparound. Mirrors LLVM `IndVarSimplify` / `ScalarEvolution`'s
# `proveNoWrapViaConstantRanges` (in spirit), simpler in shape: a
# single forward walk consuming the existing `BoundsAnalysis` result.
#
# Tile IR encodes the flag as the `overflow` attribute on the
# arithmetic op (`IntegerOverflow.{None, NSW, NUW, NW}`). The pass
# stores its decision as an extra positional operand on the SCI
# `Expr(:call|:invoke, addi, x, y)` — the addi/subi/muli emitters
# read this trailing operand at codegen and forward it to
# `encode_{Add,Sub,Mul}IOp!` as the overflow kwarg. Adding it after
# inference is done is safe: type inference already accepted the
# 2-arg form, and the codegen path resolves Exprs by raw operand
# extraction, not Julia dispatch.
#
# An op already carrying an overflow operand is left alone — the
# pass is idempotent.

"""
    no_wrap_pass!(sci::StructuredIRCode, bounds_info::BoundsInfo)

Walk every `addi`/`subi`/`muli` and attach the strongest
`IntegerOverflow` flag the bounds analysis proves. Mutates the SCI in
place by appending an `IntegerOverflow.T` value to each eligible
instruction's operand list; the corresponding emitters read it as the
overflow kwarg.
"""
function no_wrap_pass!(sci::StructuredIRCode, bounds_info::BoundsInfo)
    walk_no_wrap!(sci.entry, bounds_info)
    return nothing
end

function walk_no_wrap!(block::Block, bounds_info::BoundsInfo)
    for inst in instructions(block)
        s = inst[:stmt]
        if s isa ControlFlowOp
            for sub in blocks(s)
                walk_no_wrap!(sub, bounds_info)
            end
            continue
        end
        s isa Expr || continue
        try_attach_no_wrap!(block, inst, bounds_info)
    end
end

function try_attach_no_wrap!(block::Block, inst::Instruction, bounds_info::BoundsInfo)
    call = resolve_call(block, inst)
    call === nothing && return
    func, ops = call
    is_no_wrap_eligible(func) || return
    length(ops) == 2 || return  # already annotated, leave alone

    width = result_int_width(inst[:type])
    width === nothing && return

    a = bounds(bounds_info, ops[1])
    b = bounds(bounds_info, ops[2])

    flag = prove_no_wrap(func, width, a, b)
    flag === IntegerOverflow.None && return

    push!(inst[:stmt].args, flag)
    return
end

is_no_wrap_eligible(@nospecialize(func)) =
    func === Intrinsics.addi || func === Intrinsics.subi || func === Intrinsics.muli

# Width (in bits) of an integer instruction's element type. Returns
# `nothing` for non-integer or non-Tile-of-integer results.
function result_int_width(@nospecialize(typ))
    T = CC.widenconst(typ)
    elem = T <: Tile ? eltype(T) : T
    elem <: Integer || return nothing
    return sizeof(elem) * 8
end

#=============================================================================
 Proving no-wrap from interval ranges
=============================================================================#

# Compute the *ideal* (overflow-unaware) result range for the op,
# then check whether it fits in the destination width as a signed and/
# or unsigned integer. Returns the corresponding `IntegerOverflow`
# flag — `NW` if both, `NSW`/`NUW` if one, `None` otherwise.
function prove_no_wrap(@nospecialize(func), width::Int, a::IntRange, b::IntRange)
    # Exact-zero shortcut for multiplication: `0 × anything = 0` is
    # never wrapping, regardless of the other operand's range.
    if func === Intrinsics.muli && (is_exact_zero(a) || is_exact_zero(b))
        return IntegerOverflow.NW
    end

    finite(a) || return IntegerOverflow.None
    finite(b) || return IntegerOverflow.None

    lo, hi = if func === Intrinsics.addi
        ideal_add(a, b)
    elseif func === Intrinsics.subi
        ideal_sub(a, b)
    elseif func === Intrinsics.muli
        ideal_mul(a, b)
    else
        return IntegerOverflow.None
    end

    (lo === nothing || hi === nothing) && return IntegerOverflow.None

    fits_signed = lo >= signed_min(width) && hi <= signed_max(width)
    fits_unsigned = lo >= 0 && hi <= unsigned_max(width)

    return fits_signed && fits_unsigned ? IntegerOverflow.NW :
           fits_signed                  ? IntegerOverflow.NSW :
           fits_unsigned                ? IntegerOverflow.NUW :
                                          IntegerOverflow.None
end

# Both endpoints of the interval are concrete integers (no open ±∞).
finite(r::IntRange) = r.lo !== nothing && r.hi !== nothing

# Range pinned to the single value `0`.
is_exact_zero(r::IntRange) = r.lo === 0 && r.hi === 0

# Ideal arithmetic — Int (typically Int64) is wide enough that the
# result of any int32 × int32 product won't overflow the working
# precision. For Int64 inputs we fall back to `nothing` if the
# computation itself would overflow Int.
function ideal_add(a::IntRange, b::IntRange)
    return (saturated_add(a.lo, b.lo), saturated_add(a.hi, b.hi))
end

function ideal_sub(a::IntRange, b::IntRange)
    return (saturated_sub(a.lo, b.hi), saturated_sub(a.hi, b.lo))
end

# Multiplication: pick min/max over the four corner products. Returns
# `(nothing, nothing)` if any working-precision product overflows.
function ideal_mul(a::IntRange, b::IntRange)
    p1 = saturated_mul(a.lo, b.lo)
    p2 = saturated_mul(a.lo, b.hi)
    p3 = saturated_mul(a.hi, b.lo)
    p4 = saturated_mul(a.hi, b.hi)
    (p1 === nothing || p2 === nothing || p3 === nothing || p4 === nothing) &&
        return (nothing, nothing)
    return (min(p1, p2, p3, p4), max(p1, p2, p3, p4))
end

@inline function saturated_add(a::Int, b::Int)
    r, overflow = Base.add_with_overflow(a, b)
    overflow && return nothing
    return r
end

@inline function saturated_sub(a::Int, b::Int)
    r, overflow = Base.sub_with_overflow(a, b)
    overflow && return nothing
    return r
end

@inline function saturated_mul(a::Int, b::Int)
    r, overflow = Base.mul_with_overflow(a, b)
    overflow && return nothing
    return r
end

# Integer width helpers. Caller guarantees `1 ≤ width ≤ 64`.
signed_min(width::Int)    = -(Int128(1) << (width - 1))
signed_max(width::Int)    =  (Int128(1) << (width - 1)) - 1
unsigned_max(width::Int)  =  (Int128(1) << width) - 1
