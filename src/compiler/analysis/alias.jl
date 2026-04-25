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
# Consumed by token_order_pass! to partition memory operations into
# independent token chains, enabling parallelism across independent regions.

"""
    AliasElement

3-state lattice for alias analysis:

    nothing         — ⊥: not yet analysed (dict-absent return value)
    Set{Any}        — concrete root tags this anchor may point into
    ALIAS_UNIVERSE  — ⊤: may alias anything

Consumers must map a missing entry (or `nothing`) to `ALIAS_UNIVERSE`
themselves: an SSA the analysis hasn't reached is conservatively
may-alias-anything. `something(result[anchor], ALIAS_UNIVERSE)` is the
canonical idiom.
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
