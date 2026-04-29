# Assume Aggregation
#
# Read-only aggregator that bundles `DivByAnalysis` and `BoundsAnalysis`
# results with `ArraySpec` lookup, projection (pow2 + bound clamping),
# and tuple-element-source resolution into a per-`make_tensor_view`
# predicate bundle. Codegen consumes one bundle per call:
# `predicates_for(ctx.assume_info, mtv_ssa)` returns an `MTVPredicates`
# struct with `ptr` / `sizes[i]` / `strides[i]` chains ready to wrap
# the corresponding bytecode `Value`s.
#
# This replaces the prior `assume_pass!` (transform/assume.jl). The
# difference is that the aggregator does *not* mutate the SCI: no
# `Intrinsics.assume` ops are inserted, no `Core.tuple` SSAs are rebuilt,
# no `getfield` SSAs are synthesised. The "where do I attach this fact"
# step happens at bytecode emission, where per-element `Value`s already
# exist as a natural product of `resolve_tuple` (which the
# `make_tensor_view` codegen had to do anyway to feed
# `encode_MakeTensorViewOp!` its flat operands).
#
# Tuple-element-source navigation (the recovery of per-axis SCI handles
# from a tuple-typed operand) lives in `tuple_element_source` and is
# entirely an analysis-internal concern — codegen requests "facts for
# this make_tensor_view" and gets back tuple-shaped chains it can
# consume positionally.
#
# Mirrors cuTile Python's `add_divby_pass` + inline `assume_bounded(0,
# None)` emission, but as a sidecar query rather than an IR-mutation pass.
# Their MakeTensorView has variadic per-axis operands so attaching is a
# per-slot operand swap; ours has tuple-typed operands so the bytecode
# emission is the natural attachment point.

#=============================================================================
 AssumeInfo
=============================================================================#

"""
    MTVPredicates

Per-operand `AssumePredicate` chains for one `make_tensor_view` call.
- `ptr`: chain to wrap the base-pointer operand.
- `sizes[i]`: chain to wrap the i-th size operand (Julia/column-major order).
- `strides[i]`: chain to wrap the i-th stride operand.

`sizes` and `strides` always have length `N` (the TileArray rank);
slots that produce no useful fact (literal element, contiguous-axis
static stride) carry an empty chain. Empty chains mean "emit no
`AssumeOp`" — same observable result as omitting the entry.
"""
struct MTVPredicates
    ptr::Vector{AssumePredicate}
    sizes::Vector{Vector{AssumePredicate}}
    strides::Vector{Vector{AssumePredicate}}
end

const EMPTY_PREDS = AssumePredicate[]

"""
    AssumeInfo

Sidecar carrying per-`make_tensor_view` predicate bundles. Built by
`analyze_assume_info` from `DivByInfo` + `BoundsInfo` + the operand
TileArray's `ArraySpec`; queried by codegen via `predicates_for`.

Each entry collapses ptr / per-axis sizes / per-axis strides for one
make_tensor_view into a single `MTVPredicates` struct, so codegen sees
the tuple-shaped result directly rather than reconstructing it from
flat indexed lookups. The walk from a tuple operand to its per-element
sources lives in `tuple_element_source` — the cost of recovering
"per-field facts on a tuple-valued operand" stays inside the analysis.
"""
struct AssumeInfo
    predicates::Dict{Int, MTVPredicates}
end

AssumeInfo() = AssumeInfo(Dict{Int, MTVPredicates}())

"""
    predicates_for(info, mtv_ssa) -> Union{MTVPredicates, Nothing}

Return the predicate bundle for the `make_tensor_view` at SSA index
`mtv_ssa`, or `nothing` if no entry exists (e.g. the analysis didn't
run, or the make_tensor_view's TileArray type was unresolvable).
Codegen treats `nothing` as "no assumes" — same as all-empty chains.
"""
@inline predicates_for(info::AssumeInfo, mtv_ssa::Int) =
    get(info.predicates, mtv_ssa, nothing)

predicates_for(::Nothing, ::Int) = nothing

#=============================================================================
 Analysis driver
=============================================================================#

"""
    analyze_assume_info(sci, divby_info, bounds_info) -> AssumeInfo

Walk every `Intrinsics.make_tensor_view` in `sci`, derive
divisibility / bound facts from the operand TileArray's `ArraySpec`
combined with the optional dataflow analyses, and store the resulting
`AssumePredicate` chains keyed by `(mtv_ssa, kind, slot)`. Pure
analysis: does not mutate `sci`.
"""
function analyze_assume_info(sci::StructuredIRCode,
                              divby_info::Union{DivByInfo, Nothing}=nothing,
                              bounds_info::Union{BoundsInfo, Nothing}=nothing)
    info = AssumeInfo()
    walk_collect!(info, sci.entry, divby_info, bounds_info)
    return info
end

function walk_collect!(info::AssumeInfo, block::Block,
                        divby_info::Union{DivByInfo, Nothing},
                        bounds_info::Union{BoundsInfo, Nothing})
    for inst in instructions(block)
        s = inst[:stmt]
        if s isa ControlFlowOp
            for sub in blocks(s)
                walk_collect!(info, sub, divby_info, bounds_info)
            end
            continue
        end
        call = resolve_call(block, inst)
        call === nothing && continue
        func, ops = call
        if func === Intrinsics.make_tensor_view
            collect_make_tensor_view!(info, block, inst, ops, divby_info, bounds_info)
        end
    end
end

function collect_make_tensor_view!(info::AssumeInfo, block::Block,
                                     inst::Instruction, ops,
                                     divby_info::Union{DivByInfo, Nothing},
                                     bounds_info::Union{BoundsInfo, Nothing})
    length(ops) >= 4 || return
    T_arg      = ops[1]
    ptr_op     = ops[2]
    sizes_op   = ops[3]
    strides_op = ops[4]

    T = resolve_tilearray_type(block, T_arg)
    T === nothing && return
    spec = array_spec(T)
    spec === nothing && return

    N = ndims(T)
    mtv_ssa = inst.ssa_idx

    # ---- Pointer ---------------------------------------------------------
    ptr_div = pow2_divisor(combine_divisor(Int(spec.alignment),
                                            divby_query(divby_info, ptr_op)))
    ptr_chain = ptr_div > 1 ? AssumePredicate[DivBy(ptr_div)] : EMPTY_PREDS

    # ---- Sizes -----------------------------------------------------------
    # Lower bound is structurally `0` (sizes are non-negative). Combine
    # with the dataflow result to refine: an exact known size collapses
    # to `Bounded(N, N)`; a ForOp-IV-derived size to `Bounded(0, max)`,
    # etc.
    sizes_chains = Vector{Vector{AssumePredicate}}(undef, N)
    for i in 1:N
        sizes_chains[i] = element_chain(block, sizes_op, i,
                                         Int(spec.shape_div_by[i]),
                                         divby_info, bounds_info)
    end

    # ---- Strides ---------------------------------------------------------
    strides_chains = Vector{Vector{AssumePredicate}}(undef, N)
    for i in 1:N
        # Skip the contiguous axis: its stride is statically `1` and never
        # enters the bytecode kernel signature (filter_dynamic_strides).
        if spec.contiguous && i == 1
            strides_chains[i] = EMPTY_PREDS
            continue
        end
        strides_chains[i] = element_chain(block, strides_op, i,
                                           Int(spec.stride_div_by[i]),
                                           divby_info, bounds_info)
    end

    info.predicates[mtv_ssa] = MTVPredicates(ptr_chain, sizes_chains, strides_chains)
    return
end

# Build the predicate chain for a single tuple element (size or stride).
# Walks back through `tuple_element_source` to recover a per-element SCI
# handle when one exists (`Core.tuple(...)` constructor); falls through
# to spec-only facts (`spec_div`, structural `[0, ∞)`) when the source is
# wholesale (`getfield(arg, :sizes)`) or otherwise opaque. Returns
# `EMPTY_PREDS` for literal elements and for the all-trivial case.
function element_chain(block::Block, tuple_op, i::Int, spec_div::Int,
                        divby_info::Union{DivByInfo, Nothing},
                        bounds_info::Union{BoundsInfo, Nothing})
    elem_op = tuple_element_source(block, tuple_op, i)
    # Literals — `assume bounded<N, N>` on `<i32: 64>` adds no info
    # the Tile IR translator can't see directly.
    is_literal_op(elem_op) && return EMPTY_PREDS

    df_div   = elem_op === nothing ? 0 : divby_query(divby_info, elem_op)
    df_bound = elem_op === nothing ? TOP_RANGE : bounds_query(bounds_info, elem_op)

    d     = pow2_divisor(combine_divisor(spec_div, df_div))
    bound = combine_bound(nonneg_range(), df_bound)

    preds = AssumePredicate[as_bounded(bound)]
    d > 1 && push!(preds, DivBy(d))
    return preds
end

#=============================================================================
 Tuple element source resolution
=============================================================================#

# Resolve a tuple-typed operand to its i-th element's SCI handle.
# Recognises:
#   - Literal `Tuple` values (`(64, 64)`): returns the i-th literal.
#   - `Core.tuple(s1, ..., sN)` SSA: returns the i-th operand.
#   - Anything else (e.g. `getfield(arg, :sizes)`): returns `nothing`,
#     leaving the caller to use spec-only facts.
#
# The walk-up parent chain mirrors `value_type` / `lookup_def_call`.
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

# Extract a `Type{TileArray{...}}` value from an SCI operand. Recognises
# a constant `Type` literal, a `QuoteNode(::Type)`, and an SSA whose
# inferred type is `Const(T)` / `Type{T}`. Returns the unwrapped `T` or
# `nothing`.
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
