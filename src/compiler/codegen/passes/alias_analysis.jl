# Alias Analysis Pass
#
# Fixed-point alias analysis over StructuredIRCode. Determines which memory
# operations may access the same underlying data (i.e., which SSA values
# point into the same allocation).
#
# WHY: The token ordering pass needs alias information to decide which memory
# operations require token dependencies between them. Without alias analysis,
# all memory ops would be serialized through a single token chain — correct,
# but overly conservative. With per-alias-set information, independent memory
# regions (e.g., separate kernel arguments) get independent token chains,
# enabling more parallelism in the generated Tile IR.
#
# HOW: Each pointer-containing kernel argument starts in its own alias set.
# Alias sets propagate forward through:
#   - getfield (for TileArray.ptr field access)
#   - pointer arithmetic (+, -)
#   - view constructors (make_tensor_view, make_partition_view)
#   - pointer passthroughs (bitcast, assume_aligned, etc.)
# Unknown operations conservatively produce ALIAS_UNIVERSE (may alias anything).
# Fixed-point iteration handles back-edges from loops.
#
# OUTPUT: Dict{Any, AliasSet} mapping SSA values and Arguments to their alias
# sets, consumed by token_order_pass!.

"""
    AliasTracker

Tracks alias sets for each SSA value during fixed-point analysis.
"""
mutable struct AliasTracker
    dirty::Bool
    aliases::Dict{Any, AliasSet}  # SSAValue/Argument/SlotNumber -> AliasSet
end

AliasTracker() = AliasTracker(false, Dict{Any, AliasSet}())

function Base.getindex(tracker::AliasTracker, key)
    return get(tracker.aliases, key, ALIAS_UNIVERSE)
end

function Base.setindex!(tracker::AliasTracker, value::AliasSet, key)
    current = get(tracker.aliases, key, nothing)
    if current !== value
        tracker.dirty = true
        tracker.aliases[key] = value
    end
    return
end

"""
    alias_analysis_pass!(sci::StructuredIRCode) -> Dict{Any, AliasSet}

Perform fixed-point alias analysis on structured IR.
Returns mapping from SSA values to alias sets.
"""
function alias_analysis_pass!(sci::StructuredIRCode)
    tracker = AliasTracker()

    # Initialize: each argument gets its own alias set
    for (idx, argtype) in enumerate(sci.argtypes)
        argtype_unwrapped = CC.widenconst(argtype)
        if contains_pointers(argtype_unwrapped)
            arg_ref = Argument(idx)
            tracker[arg_ref] = Set{Any}([arg_ref])
        end
    end

    # Fixed-point iteration
    iteration = 0
    max_iterations = 100

    tracker.dirty = true
    while tracker.dirty && iteration < max_iterations
        tracker.dirty = false
        iteration += 1

        analyze_block!(tracker, sci.entry)
    end

    @debug "Alias analysis converged in $iteration iterations"

    return tracker.aliases
end

"""
    propagate!(tracker::AliasTracker, from, to)

Propagate alias set from `from` to `to`.
Uses direct assignment when `to` is uninitialized, union otherwise.
"""
function propagate!(tracker::AliasTracker, from, to)
    from_aliases = tracker[from]

    if from_aliases === ALIAS_UNIVERSE
        # Propagating UNIVERSE is always conservative
        tracker[to] = ALIAS_UNIVERSE
        return
    end

    if haskey(tracker.aliases, to)
        # Target already has an alias set union with it
        to_aliases = tracker.aliases[to]
        new_aliases = union(from_aliases, to_aliases)
        if new_aliases != to_aliases
            tracker[to] = new_aliases
        end
    else
        # Target not yet analyzed assign directly
        tracker[to] = from_aliases
    end
    return
end

"""
    analyze_block!(tracker::AliasTracker, block)

Analyze all statements in a block, recursing into nested control flow.
"""
function analyze_block!(tracker::AliasTracker, block)
    for (ssa_idx, entry) in block.body
        if entry.stmt isa ControlFlowOp
            analyze_control_flow!(tracker, entry.stmt)
        else
            analyze_statement!(tracker, SSAValue(ssa_idx), entry.stmt)
        end
    end
    return
end

# Recurse into nested control flow regions
function analyze_control_flow!(tracker::AliasTracker, op::IfOp)
    analyze_block!(tracker, op.then_region)
    return analyze_block!(tracker, op.else_region)
end

function analyze_control_flow!(tracker::AliasTracker, op::ForOp)
    return analyze_block!(tracker, op.body)
end

function analyze_control_flow!(tracker::AliasTracker, op::WhileOp)
    analyze_block!(tracker, op.before)
    return analyze_block!(tracker, op.after)
end

function analyze_control_flow!(tracker::AliasTracker, op::LoopOp)
    return analyze_block!(tracker, op.body)
end

# Fallback for unknown control flow ops
function analyze_control_flow!(::AliasTracker, ::ControlFlowOp)
    return
end

"""
    analyze_statement!(tracker::AliasTracker, ssa::SSAValue, stmt)

Analyze a single statement and propagate aliases.
Handles both `:call` and `:invoke` expression forms.
"""
function analyze_statement!(tracker::AliasTracker, ssa::SSAValue, stmt)
    call = resolve_call(stmt)
    if call !== nothing
        resolved_func, operands = call

        # Also need the raw func ref for GlobalRef comparisons
        func = stmt.head === :call ? stmt.args[1] : stmt.args[2]

        # getfield: propagate from parent
        if func === GlobalRef(Core, :getfield) && length(operands) >= 1
            field = length(operands) >= 2 ? operands[2] : nothing

            # For TileArray.ptr field access, propagate pointer alias
            if field isa QuoteNode && field.value === :ptr
                propagate!(tracker, operands[1], ssa)
            else
                # Conservatively mark as UNIVERSE for non-pointer fields
                tracker[ssa] = ALIAS_UNIVERSE
            end

        # Pointer arithmetic: propagate from pointer operand
        elseif func === GlobalRef(Base, :+) || func === GlobalRef(Base, :-)
            for arg in operands
                # Find the pointer argument and propagate
                arg_aliases = tracker[arg]
                if arg_aliases !== ALIAS_UNIVERSE && arg_aliases isa Set
                    propagate!(tracker, arg, ssa)
                    break
                end
            end

        # View construction: propagate alias from first operand
        elseif is_view_constructor(resolved_func) || is_pointer_passthrough(resolved_func)
            if length(operands) >= 1
                propagate!(tracker, operands[1], ssa)
            end

        # Default: unknown operation -> UNIVERSE
        else
            tracker[ssa] = ALIAS_UNIVERSE
        end

    elseif stmt isa ReturnNode
        # No alias propagation needed

    else
        # Unknown statement type -> conservative
        tracker[ssa] = ALIAS_UNIVERSE
    end
    return
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
    func === GlobalRef(Core.Intrinsics, :bitcast) && return true

    # Safely check by name to avoid UndefVarError if intrinsics aren't exposed
    if func isa Core.IntrinsicFunction || func isa Function
        n = nameof(func)
        return n === :bitcast || n === :assume_div_by || n === :assume_bounded || n === :assume_aligned
    end
    return false
end
