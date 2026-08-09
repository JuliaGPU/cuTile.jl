# Assume Helpers
#
# `DivByAnalysis` and `BoundsAnalysis` propagate per-anchor facts; this
# file is the projection layer that turns those facts (plus the
# operand's `ArraySpec`) into the `AssumePredicate` chains that codegen
# wraps `Value`s with. There's no precomputed sidecar — the analyses'
# results live on the `CGCtx` and consumer-op codegen calls
# `op_predicates` / `arg_chain` on demand for each operand it cares
# about.
#
# Mirrors cuTile Python's `_passes/propagate_divby.py`: same
# `_OPS_NEED_ASSUME = (MakeTensorView, LoadPointer, StorePointer)`
# consumer set, same per-consumer derivation. Where Python mutates the
# IR (inserts `AssumeDivBy` ops with a `var_map` to dedup), we wrap at
# bytecode emission time and the per-`Value` cache on `CGCtx`
# (`assume_wrapped`) plays the role of `var_map`, ensuring a `Value`
# reused across consumers — e.g. a kernel-arg pointer threaded through
# both an MTV and a gather — is wrapped exactly once.
#
# Pure analysis: does not mutate the SCI.

const EMPTY_PREDS = AssumePredicate[]

#=============================================================================
 Per-operand chain derivation
=============================================================================#

"""
    op_predicates(divby, bounds, op, kind, spec_div=1) -> Vector{AssumePredicate}

Derive the `AssumePredicate` chain for a consumer-op operand. `kind`
selects the structural prior:
- `:ptr` — pointer operand, no `Bounded` (a pointer's range is
  meaningless to tileiras's vectorizer); chain is `[DivBy(d)]` when
  `d > 1`, else empty.
- `:size` / `:stride` — integer operand; always `Bounded(0, ?)` since
  sizes / strides are non-negative, plus `DivBy(d)` when `d > 1`.

`spec_div` is the consumer-side type-level divisor hint
(`spec.alignment`, `spec.shape_div_by[i]`, `spec.stride_div_by[i]`)
combined with the dataflow result via `lcm` — both inputs are
guarantees, so the value is divisible by their lcm.

Returns `EMPTY_PREDS` for literal operands; the Tile IR translator
already sees the literal directly.
"""
function op_predicates(divby_info::Union{DivByInfo, Nothing},
                        bounds_info::Union{BoundsInfo, Nothing},
                        @nospecialize(op),
                        kind::Symbol,
                        spec_div::Int=1)
    is_literal_op(op) && return EMPTY_PREDS

    df_div = op === nothing ? 0 : divby_query(divby_info, op)
    d = pow2_divisor(combine_divisor(spec_div, df_div))

    if kind === :ptr
        return d > 1 ? AssumePredicate[DivBy(d)] : EMPTY_PREDS
    end

    # :size / :stride — always assert non-negativity, refine with
    # dataflow's tighter range when available.
    df_bound = op === nothing ? TOP_RANGE : bounds_query(bounds_info, op)
    bound = combine_bound(nonneg_range(), df_bound)
    chain = AssumePredicate[as_bounded(bound)]
    d > 1 && push!(chain, DivBy(d))
    return chain
end

#=============================================================================
 Kernel-arg flat-slot chain derivation (spec-only)
=============================================================================#

"""
    arg_chain(T::Type{<:TileArray}, path) -> Vector{AssumePredicate}

Per-flat-slot chain for a `TileArray` kernel argument. Thin
dispatcher over `op_predicates` keyed on the flat slot path
produced by `flatten_struct_params!`:

- `[1]` → `:ptr`     (with `spec.alignment`)
- `[2, i]` → `:size` (with `spec.shape_div_by[i]`)
- `[3, i]` → `:stride` (with `spec.stride_div_by[i]`)

Dataflow inputs are `nothing` because the kernel-arg slot is the
analysis anchor — there's no upstream IR for the dataflow to refine
against. Consumer-site queries against an SSA derived from the slot
*do* carry dataflow refinement (and combine with the same spec hints
via `lcm`), so the entry-time chain is an upper bound on what any
consumer would derive — important for the `wrap_for` cache invariant
(see its docstring).

Used by `apply_arg_assume_predicates!` (codegen/kernel.jl) at kernel
entry to wrap each flat kernel-arg `Value` *before* any consumer
reads it. Important for raw `offset` / `load_ptr_tko` /
`store_ptr_tko` access paths (gather/scatter): the assume must attach
to the base pointer, not just to the post-offset operand, for
tileiras's vectorizer to prove the wide-vector address alignment its
STG.E.128 / LDG.E.128 lowering requires.

Returns `EMPTY_PREDS` when no useful fact exists (no spec on `T`,
unrecognised path, or contiguous-axis stride which is a static `1`).
"""
function arg_chain(::Type{T}, path::Vector{Int}) where {T <: TileArray}
    spec = array_spec(T)
    spec === nothing && return EMPTY_PREDS

    if length(path) == 1 && path[1] == 1
        return op_predicates(nothing, nothing, nothing, :ptr, Int(spec.alignment))
    end

    if length(path) == 2
        i = path[2]
        1 <= i <= ndims(T) || return EMPTY_PREDS
        if path[1] == 2  # sizes[i]
            spec.singleton[i] && return EMPTY_PREDS
            return op_predicates(nothing, nothing, nothing, :size, Int(spec.shape_div_by[i]))
        elseif path[1] == 3  # strides[i]
            # Static contiguous/singleton strides need no runtime assumption.
            ((spec.contiguous && i == 1) || spec.singleton[i]) && return EMPTY_PREDS
            return op_predicates(nothing, nothing, nothing, :stride, Int(spec.stride_div_by[i]))
        end
    end
    return EMPTY_PREDS
end

#=============================================================================
 Tuple element source resolution
=============================================================================#

"""
    tuple_element_source(block, tuple_op, i) -> SSAValue / literal / nothing

Resolve a tuple-typed operand to its i-th element's SCI handle.
Recognises:
- Literal `Tuple` values (`(64, 64)`): returns the i-th literal.
- `Core.tuple(s1, ..., sN)` SSA: returns the i-th operand.
- Anything else (e.g. `getfield(arg, :sizes)`): returns `nothing`,
  leaving the caller to use spec-only facts.

The walk-up parent chain mirrors `value_type` / `lookup_def_call`.
"""
function tuple_element_source(block::Block, @nospecialize(tuple_op), i::Int)
    if tuple_op isa Tuple
        return length(tuple_op) >= i ? tuple_op[i] : nothing
    end
    tuple_op isa SSAValue || return nothing
    p = block
    while p isa Block
        entry = get(p.body, tuple_op.id, nothing)
        if entry !== nothing
            call = resolve_call(p, entry.stmt)
            call === nothing && return nothing
            func, ops = call
            func === Core.tuple || return nothing
            return length(ops) >= i ? ops[i] : nothing
        end
        p = p.parent
    end
    return nothing
end

@inline is_literal_op(::Nothing) = false
@inline is_literal_op(@nospecialize(op)) = op isa Number || op isa QuoteNode

#=============================================================================
 Operand-type extraction
=============================================================================#

"""
    resolve_tilearray_type(block, op) -> Union{Type, Nothing}

Extract a `Type{TileArray{...}}` value from an SCI operand. Recognises
a constant `Type` literal, a `QuoteNode(::Type)`, and an SSA whose
inferred type is `Const(T)` / `Type{T}`. Returns the unwrapped `T` or
`nothing`.
"""
function resolve_tilearray_type(block::Block, @nospecialize(op))
    if op isa Type
        op <: TileArray && return op
        return nothing
    end
    if op isa QuoteNode && op.value isa Type
        op.value <: TileArray && return op.value
        return nothing
    end
    T_lat = value_type(block, op)
    T_lat === nothing && return nothing
    if T_lat isa CC.Const
        v = T_lat.val
        v isa Type && v <: TileArray && return v
    end
    Tw = CC.widenconst(T_lat)
    if Tw isa DataType && Tw <: Type && length(Tw.parameters) == 1
        v = Tw.parameters[1]
        v isa Type && v <: TileArray && return v
    end
    return nothing
end

#=============================================================================
 Fact combination & projection
=============================================================================#

# Combine type-level (ArraySpec) and dataflow-level facts into one divisor.
# Inputs use the lattice convention from `DivByAnalysis`: 0 = "no info"
# (treated as 1), positive = "divisible by N". Type-level facts are
# always powers of 2 by construction; dataflow facts may not be (a
# constant `12` enters the lattice as 12). The output is normalised to a
# power of 2 by `pow2_divisor` before being baked into a `DivBy(...)`
# predicate — see that helper for why.
#
# Combine semantics is `lcm` (the value is divisible by both inputs, so
# also by their lcm). For power-of-2 inputs `lcm == max`, but the more
# general form is harmless and keeps things honest if one input ever
# carries non-power-of-2 information.
@inline function combine_divisor(spec_div::Int, df_div::Int)
    s = spec_div > 0 ? spec_div : 1
    d = df_div  > 0 ? df_div  : 1
    return lcm(s, d)
end

# Project a divisibility fact onto the largest power-of-2 dividing it,
# capped at `MAX_POW2_DIVBY`. Mirrors cuTile Python's
# `power_of_2_d = min(divisor & -divisor, MAX_DIVBY)` in
# `_passes/propagate_divby.py`.
#
# The downstream consumer of `DivBy` (vectorised memory ops in
# cuda_tile_translate) picks a vector width based on the largest
# *power-of-2* alignment it can prove — `div_by<12>` is unusable for
# any vectorised load even though `12 = 4·3` implies `div_by<4>`,
# which *is* usable. Project before emission so the annotation is in
# the form the consumer can act on directly.
const MAX_POW2_DIVBY = 1024
@inline function pow2_divisor(d::Int)
    d <= 1 && return 1
    return min(d & -d, MAX_POW2_DIVBY)
end

# Tighten one range with another (interval intersection). Used to mix
# the structural type-level bound (sizes ≥ 0) with the dataflow result.
function combine_bound(spec::IntRange, df::IntRange)
    lo = spec.lo === nothing ? df.lo :
         df.lo === nothing ? spec.lo : max(spec.lo, df.lo)
    hi = spec.hi === nothing ? df.hi :
         df.hi === nothing ? spec.hi : min(spec.hi, df.hi)
    return IntRange(lo, hi)
end

@inline as_bounded(r::IntRange) = Bounded(r.lo, r.hi)

#=============================================================================
 Operand → lattice value queries
=============================================================================#

# Only `LatticeAnchor` operands have entries in the dataflow result;
# literals collapse via `literal_divisor` / `literal_range`.

@inline divby_query(::Nothing, @nospecialize(op)) = literal_divisor(op)
@inline divby_query(info::DivByInfo, @nospecialize(op)) =
    op isa LatticeAnchor ? div_by(info, op) : literal_divisor(op)

@inline bounds_query(::Nothing, @nospecialize(op)) = literal_range(op)
@inline bounds_query(info::BoundsInfo, @nospecialize(op)) =
    op isa LatticeAnchor ? bounds(info, op) : literal_range(op)

# Largest divisor a literal operand contributes (≥ 1). `0` is the
# DivByAnalysis ∞-divisible element; we collapse to `1` here because
# the `combine_divisor`/`pow2_divisor` pipeline expects a positive
# concrete divisor.
function literal_divisor(@nospecialize(op))
    if op isa Integer
        v = abs(Int(op))
        return v == 0 ? 1 : v
    end
    if op isa QuoteNode && op.value isa Integer
        v = abs(Int(op.value))
        return v == 0 ? 1 : v
    end
    return 1
end

# Literal operands collapse to a singleton range `[v, v]`.
function literal_range(@nospecialize(op))
    op isa Integer && return IntRange(Int(op), Int(op))
    op isa QuoteNode && op.value isa Integer && return IntRange(Int(op.value), Int(op.value))
    return TOP_RANGE
end
