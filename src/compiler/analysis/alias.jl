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
    AliasAnalysis

Forward sparse dataflow analysis whose lattice element is `AliasSet`:
either a `Set{Any}` of root alias tags (`Argument(i)`) or the sentinel
`ALIAS_UNIVERSE` (top, "may alias anything").

Join is set union; `ALIAS_UNIVERSE ∪ x = ALIAS_UNIVERSE`. The framework
handles block walking, fixpoint iteration, and structured-control-flow
merges — this file only supplies the per-op transfer rules.
"""
struct AliasAnalysis <: ForwardAnalysis{AliasSet} end

# Lattice flattening: bottom and top are both `ALIAS_UNIVERSE`. A proper
# three-level lattice (⊥ / concrete Set / ⊤=UNIVERSE) would distinguish
# "unvisited" from "may alias anything"; here we collapse them because
# every consumer (token_order_pass!) treats absent keys as UNIVERSE
# anyway. The `record!` "skip bottom on insert" optimisation then elides
# the universe-on-unknown-op dict writes, without affecting semantics.
bottom(::AliasAnalysis) = ALIAS_UNIVERSE
top(::AliasAnalysis) = ALIAS_UNIVERSE

tmerge(::AliasAnalysis, a::AliasSet, b::AliasSet) = union(a, b)

function init_arg(::AliasAnalysis, i::Int, @nospecialize(argtype))
    T = CC.widenconst(argtype)
    if contains_pointers(T)
        arg = Argument(i)
        Set{Any}([arg])
    else
        ALIAS_UNIVERSE
    end
end

# `operand_value` default (LatticeAnchor → r[op], else bottom) is already
# correct here since bottom == ALIAS_UNIVERSE — no override needed.

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
            if av isa Set
                return av
            end
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


"""
    alias_analysis_pass!(sci::StructuredIRCode) -> Dict{Any, AliasSet}

Run forward sparse alias analysis and return the result in the legacy
`Dict{Any, AliasSet}` form that `token_order_pass!` consumes.
"""
function alias_analysis_pass!(sci::StructuredIRCode)
    result = analyze(AliasAnalysis(), sci)
    # token_order still consumes the legacy dict shape; expose that.
    legacy = Dict{Any, AliasSet}()
    for (k, v) in pairs(result)
        legacy[k] = v
    end
    return legacy
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
