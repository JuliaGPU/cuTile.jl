# Alias Analysis Pass
#
# Forward dataflow over StructuredIRCode that determines which SSA values may
# point into the same allocation. Each pointer-carrying kernel argument starts
# in its own alias set; the analysis propagates those sets through getfield
# (for TileArray.ptr access), pointer arithmetic, view constructors, and
# pointer passthroughs.
#
# Unknown operations conservatively produce ALIAS_UNIVERSE (may alias anything).
#
# Consumers go through the public query API (`alias_class`, `alias_classes`,
# `aliases`) which hides the underlying lattice; in particular, the
# "unvisited → may alias anything" policy is encoded once, in `alias_class`.
# The dataflow framework is an implementation detail.

#=============================================================================
 Lattice
=============================================================================#

"""
    AliasElement

3-state lattice for alias analysis (internal):

    nothing         — ⊥: not yet analysed
    Set{Any}        — concrete root tags this anchor may point into
    ALIAS_UNIVERSE  — ⊤: may alias anything
"""
const AliasElement = Union{Nothing, AliasSet}

"""
    AliasAnalysis

Forward sparse dataflow analysis whose lattice element is `AliasElement`.
Concrete `Set{Any}`s carry root alias tags (`Argument(i)`); `ALIAS_UNIVERSE`
is the top. Join is set union; `ALIAS_UNIVERSE ∪ x = ALIAS_UNIVERSE`.

The framework handles block walking, fixpoint iteration, and structured-
control-flow merges — this file only supplies the per-op transfer rules.
"""
struct AliasAnalysis <: ForwardAnalysis{AliasElement} end

bottom(::AliasAnalysis) = nothing
top(::AliasAnalysis) = ALIAS_UNIVERSE

tmerge(::AliasAnalysis, ::Nothing, ::Nothing) = nothing
tmerge(::AliasAnalysis, ::Nothing, b::AliasSet) = b
tmerge(::AliasAnalysis, a::AliasSet, ::Nothing) = a
tmerge(::AliasAnalysis, a::AliasSet, b::AliasSet) = union(a, b)

function init_arg(::AliasAnalysis, i::Int, @nospecialize(argtype))
    T = CC.widenconst(argtype)
    contains_pointers(T) || return nothing
    arg = Argument(i)
    Set{Any}([arg])
end

function transfer(a::AliasAnalysis, r::DataflowResult, @nospecialize(func),
                  ops, ::Block, ::Any)
    # getfield: propagate from parent on :ptr, UNIVERSE elsewhere.
    if func === getfield && length(ops) >= 1
        field = length(ops) >= 2 ? ops[2] : nothing
        if field isa QuoteNode && field.value === :ptr
            return operand_value(a, r, ops[1])
        end
        return ALIAS_UNIVERSE
    end

    # Pointer arithmetic: propagate from the pointer operand (first operand
    # whose alias set is concrete).
    if func === Base.:+ || func === Base.:-
        for arg in ops
            av = operand_value(a, r, arg)
            av isa Set && return av
        end
        return ALIAS_UNIVERSE
    end

    # View constructors and pointer passthroughs: propagate from first operand.
    if is_view_constructor(func) || is_pointer_passthrough(func)
        length(ops) >= 1 && return operand_value(a, r, ops[1])
        return ALIAS_UNIVERSE
    end

    ALIAS_UNIVERSE
end


# Helper functions

contains_pointers(T) = T <: Ptr || T <: TileArray || (T <: Tile && eltype(T) <: Ptr)

"""
    is_view_constructor(func) -> Bool

Check if a resolved function is a tensor/partition view constructor.
These propagate alias identity from their first operand.
"""
function is_view_constructor(func)
    return func === Intrinsics.make_tensor_view ||
        func === Intrinsics.make_partition_view
end

function is_pointer_passthrough(func)
    return func === Intrinsics.offset ||
        func === Core.Intrinsics.bitcast
end


#=============================================================================
 Public query API
=============================================================================#

"""
    AliasInfo

Result of running alias analysis. Consumers query it via `alias_class`,
`alias_classes`, and `aliases`; the underlying lattice representation is an
implementation detail.
"""
const AliasInfo = DataflowResult{AliasAnalysis, AliasElement}

"""
    analyze_aliases(sci::StructuredIRCode) -> AliasInfo

Run forward alias analysis on `sci`.
"""
analyze_aliases(sci::StructuredIRCode) = analyze(AliasAnalysis(), sci)::AliasInfo

"""
    alias_class(info::AliasInfo, op) -> AliasSet

Alias set associated with `op`. Operands the analysis didn't reach (and
non-anchor operands like literals) collapse to `ALIAS_UNIVERSE` — the
"may alias anything" default policy is encoded here, once.
"""
function alias_class(info::AliasInfo, @nospecialize(op))
    op isa SSAValue || op isa Argument || op isa SlotNumber || return ALIAS_UNIVERSE
    return something(info[op], ALIAS_UNIVERSE)
end

"""
    alias_classes(info::AliasInfo)

Iterator over the distinct alias sets the analysis recorded. Used by
consumers that need to enumerate alias equivalence classes (e.g. to seed
per-class state).
"""
alias_classes(info::AliasInfo) = values(info)

"""
    AliasResult

Result of a binary alias query (`aliases`). `MustAlias` / `PartialAlias` are
not produced — the analysis isn't flow-precise enough — so the lattice is
just `NoAlias` / `MayAlias`.
"""
@enum AliasResult NoAlias MayAlias

"""
    aliases(a::AliasSet, b::AliasSet) -> AliasResult

Binary alias query. `MayAlias` if either set is `ALIAS_UNIVERSE` or the two
sets share a root tag; `NoAlias` otherwise.
"""
function aliases(a::AliasSet, b::AliasSet)
    (a isa AliasUniverse || b isa AliasUniverse) && return MayAlias
    isempty(intersect(a, b)) ? NoAlias : MayAlias
end
