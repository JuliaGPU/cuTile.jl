# Divisibility Analysis
#
# Tracks which integer (and pointer) SSA values are known to be divisible
# by some integer N > 1. Mirrors cuTile Python's `dataflow_analysis.py`
# divby tracker: per-anchor `Int` lattice element, gcd-based merge,
# transfer rules for arithmetic, pointer offset, and getfield chains
# rooted at TileArray arguments.
#
# Consumed by `analyze_assume_info` (analysis/assume.jl), which combines
# this with the bounds analysis and the operand TileArray's `ArraySpec`
# into per-`make_tensor_view` predicate bundles. Codegen reads those
# bundles and wraps each operand `Value` with `encode_AssumeOp!` —
# the analysis itself does *not* mutate the SCI.

"""
    DivByAnalysis

3-state-ish lattice for divisibility:

    0  — ⊥: not analysed yet (`gcd(0, x) == x`, so this acts as identity
         under merge — newly-seeded facts replace it cleanly).
    1  — ⊤: nothing useful (divisible by 1 is trivially true).
    n  — known divisor; the SSA value is a multiple of `n`.

Recognises:

- `Intrinsics.addi` / `Intrinsics.subi` — `gcd(lhs, rhs)`
- `Intrinsics.muli`                     — `lhs * rhs`
- `Intrinsics.negi`                     — pass-through
- `Intrinsics.offset(ptr, off)`         — `gcd(ptr, off * sizeof(elem))`
- `Intrinsics.constant(_, scalar, _)`   — `|scalar|` if integer
- `Intrinsics.assume(x, DivBy(d))`      — `lcm(divby(x), d)` (refines)
- `Intrinsics.assume(x, Bounded|...)`   — pass-through (other predicates)
- `Intrinsics.broadcast` / `Intrinsics.reshape` — pass-through
- `Base.getfield(arg::TileArray, :ptr)` — `spec.alignment`
- `Base.getfield(getfield(arg::TileArray, :sizes), i)`   — `spec.shape_div_by[i]`
- `Base.getfield(getfield(arg::TileArray, :strides), i)` — `spec.stride_div_by[i]`
- Integer literal operands resolve to `|value|` (≥ 1).
"""
struct DivByAnalysis <: ForwardAnalysis{Int} end

bottom(::DivByAnalysis) = 0
top(::DivByAnalysis) = 1
tmerge(::DivByAnalysis, a::Int, b::Int) = gcd(a, b)

# Initial seed for kernel arguments. The TileArray case is handled via the
# `getfield(arg, :ptr|:sizes|:strides)` transfer rules (richer per-field
# facts can't be encoded in a single Int); this default just contributes
# nothing for the bare argument.
init_arg(::DivByAnalysis, ::Int, @nospecialize(_)) = 1

# Lattice convention (matches cuTile Python's `dataflow_analysis.py`):
#  - 0 is dual-purpose: bottom (not yet analysed) and "infinitely divisible"
#    (e.g. literal `0`). Both behave correctly under `gcd` because
#    `gcd(0, x) == x` makes 0 the identity element of the merge.
#  - Concrete divisors are positive integers; `1` means "no useful info".
#  - Public-query consumers (`div_by` in this file, `divby_query` in the
#    assume pass) collapse 0 to 1 — there's no concrete divisor to assert.
function operand_value(::DivByAnalysis, r::DataflowResult, @nospecialize(op))
    op isa Integer && return abs(Int(op))
    op isa QuoteNode && op.value isa Integer && return abs(Int(op.value))
    op isa LatticeAnchor && return r[op]
    return 1
end

function transfer(a::DivByAnalysis, r::DataflowResult, @nospecialize(func),
                  ops, block::Block, ::Any)
    # Arithmetic on integer tiles
    if func === Intrinsics.addi || func === Intrinsics.subi
        length(ops) >= 2 || return 1
        return gcd(operand_value(a, r, ops[1]), operand_value(a, r, ops[2]))
    end
    if func === Intrinsics.muli
        length(ops) >= 2 || return 1
        x = operand_value(a, r, ops[1])
        y = operand_value(a, r, ops[2])
        # 0 (∞ divisibility) absorbs in the product — `0 * x` is divisible
        # by anything, so the result stays at ∞.
        (x == 0 || y == 0) && return 0
        # Saturate to avoid runaway products on lattice movement.
        return x <= typemax(Int) ÷ y ? x * y : 1
    end
    if func === Intrinsics.negi
        length(ops) >= 1 || return 1
        return operand_value(a, r, ops[1])
    end

    # Pointer offset: result is divisible by gcd(ptr_div, off_div * elem_bytes).
    # The base may be either `Tile{Ptr{T}, ()}` (post-canonicalize tile form)
    # or raw `Ptr{T}` (the SCI annotation on `getfield(arg, :ptr)`); both
    # lower to a 0-D pointer Value at codegen.
    if func === Intrinsics.offset
        length(ops) >= 2 || return 1
        base_T = value_type(block, ops[1])
        base_T = base_T === nothing ? Any : CC.widenconst(base_T)
        elem_T = ptr_pointee(base_T)
        elem_T === nothing && return 1
        elem_bytes = sizeof(elem_T)
        ptr_div = operand_value(a, r, ops[1])
        off_div = operand_value(a, r, ops[2])
        # off_div == 0 (∞) means the offset is a multiple of arbitrarily
        # large powers (e.g. literal `0`); the result inherits ptr_div.
        off_div == 0 && return ptr_div
        off_bytes = elem_bytes > 0 && off_div <= typemax(Int) ÷ elem_bytes ?
                    off_div * elem_bytes : 1
        return gcd(ptr_div, off_bytes)
    end

    # Constant op: derive |scalar| if integer (0 stays 0, the ∞-divisor).
    if func === Intrinsics.constant
        length(ops) >= 2 || return 1
        sv = ops[2]
        sv isa Integer && return abs(Int(sv))
        sv isa QuoteNode && sv.value isa Integer && return abs(Int(sv.value))
        return 1
    end

    # `Intrinsics.assume(x, predicate)` — refines when the predicate is
    # `DivBy`, otherwise passes through. The predicate is an embedded
    # `AssumePredicate` value (constructed at pass time), so it lives in
    # the operand list directly rather than being wrapped in a QuoteNode.
    if func === Intrinsics.assume
        length(ops) >= 2 || return 1
        x_div = operand_value(a, r, ops[1])
        pred = ops[2]
        if pred isa DivBy
            d = pred.divisor
            return d > 0 ? lcm(max(x_div, 1), d) : x_div
        end
        return x_div
    end

    # Pure shape ops — divisor passes through.
    if func === Intrinsics.broadcast || func === Intrinsics.reshape
        length(ops) >= 1 || return 1
        return operand_value(a, r, ops[1])
    end

    # Integer casts: `bitcast`/`exti` preserve the value (and so the
    # divisor); `trunci` reduces to mod 2^bitwidth (matches Python's
    # `TileAsType` rule on integer→integer).
    if func === Intrinsics.bitcast || func === Intrinsics.exti
        length(ops) >= 1 || return 1
        return operand_value(a, r, ops[1])
    end
    if func === Intrinsics.trunci
        length(ops) >= 2 || return 1
        x_div = operand_value(a, r, ops[1])
        T = ops[2] isa Type ? ops[2] : (ops[2] isa QuoteNode && ops[2].value isa Type ? ops[2].value : nothing)
        T isa Type && T <: Integer || return x_div
        bw = sizeof(T) * 8
        return gcd(x_div, 1 << bw)
    end

    # Field access on a TileArray-typed Argument: derive from the ArraySpec.
    if func === Base.getfield
        return getfield_divby(r, ops, block)
    end

    return 1
end

# `getfield(obj, field)` — derive divisibility when `obj` traces back to a
# TileArray-typed `Argument`. Handles the two-step chain
# `getfield(getfield(arg, :sizes), i)` and `getfield(getfield(arg, :strides), i)`.
function getfield_divby(r::DataflowResult, ops, block::Block)
    length(ops) >= 2 || return 1
    obj = ops[1]
    field = ops[2] isa QuoteNode ? ops[2].value : ops[2]

    obj_T = value_type(block, obj)
    obj_T = obj_T === nothing ? Any : CC.widenconst(obj_T)

    # First-level: getfield(arg, :ptr | :sizes | :strides)
    if obj_T <: TileArray
        spec = array_spec(obj_T)
        spec === nothing && return 1
        if field === :ptr
            return spec.alignment > 0 ? spec.alignment : 1
        end
        # :sizes / :strides return a tuple — element-level facts come from
        # the second-level getfield handler below.
        return 1
    end

    # Second-level: getfield(getfield(arg, :sizes|:strides), i)
    # Only meaningful when `obj` is itself a getfield SSA defined in this
    # block (or a parent), with `obj.field ∈ {:sizes, :strides}` and its
    # source object is a TileArray-typed Argument.
    obj isa SSAValue || return 1
    obj_def = lookup_def_call(block, obj)
    obj_def === nothing && return 1
    obj_func, obj_ops = obj_def
    obj_func === Base.getfield || return 1
    length(obj_ops) >= 2 || return 1

    inner_field = obj_ops[2] isa QuoteNode ? obj_ops[2].value : obj_ops[2]
    inner_field === :sizes || inner_field === :strides || return 1

    inner_obj = obj_ops[1]
    inner_T = value_type(block, inner_obj)
    inner_T = inner_T === nothing ? Any : CC.widenconst(inner_T)
    inner_T <: TileArray || return 1
    spec = array_spec(inner_T)
    spec === nothing && return 1

    idx = field isa Integer ? Int(field) : nothing
    idx === nothing && return 1
    idx >= 1 || return 1
    if inner_field === :sizes
        idx <= length(spec.shape_div_by) || return 1
        d = spec.shape_div_by[idx]
        return d > 0 ? d : 1
    else  # :strides
        idx <= length(spec.stride_div_by) || return 1
        d = spec.stride_div_by[idx]
        return d > 0 ? d : 1
    end
end

# Pointee element type of a 0-D pointer base, accepting both `Ptr{T}` and
# `Tile{Ptr{T}, ()}`. Returns `nothing` for shapes that don't fit (tile-of-
# pointers, non-pointer types, …).
function ptr_pointee(@nospecialize(T))
    if T <: Tile
        size(T) == () || return nothing
        ptr_T = eltype(T)
        ptr_T <: Ptr || return nothing
        return eltype(ptr_T)
    elseif T <: Ptr
        return eltype(T)
    end
    return nothing
end

# Walk parent blocks searching for the def of an SSAValue, returning the
# resolved (func, operands) tuple if it's a call. Returns nothing otherwise.
function lookup_def_call(block::Block, val::SSAValue)
    p = block
    while p isa Block
        entry = get(p.body, val.id, nothing)
        if entry !== nothing
            return resolve_call(p, entry.stmt)
        end
        p = p.parent
    end
    return nothing
end

#=============================================================================
 Public query API
=============================================================================#

const DivByInfo = DataflowResult{DivByAnalysis, Int}

"""
    analyze_divisibility(sci::StructuredIRCode) -> DivByInfo

Run forward divisibility analysis on `sci`.
"""
analyze_divisibility(sci::StructuredIRCode) = analyze(DivByAnalysis(), sci)::DivByInfo

"""
    div_by(info, op) -> Int

Resolve an operand to its known divisor (≥ 1). Returns `1` for unknown
or non-integer operands.
"""
function div_by(info::DivByInfo, @nospecialize(op))
    if op isa Integer
        v = abs(Int(op))
        return v == 0 ? 1 : v
    end
    op isa QuoteNode && op.value isa Integer && return begin
        v = abs(Int(op.value))
        v == 0 ? 1 : v
    end
    if op isa LatticeAnchor
        v = info[op]
        return v == 0 ? 1 : v
    end
    return 1
end

div_by(::Nothing, @nospecialize(op)) = 1
