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
    return if current !== value
        tracker.dirty = true
        tracker.aliases[key] = value
    end
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

Propagate alias set from `from` to `to` (union operation).
"""
function propagate!(tracker::AliasTracker, from, to)
    from_aliases = tracker[from]
    to_aliases = tracker[to]

    # Union the alias sets
    new_aliases = union(from_aliases, to_aliases)

    return if new_aliases != to_aliases
        tracker[to] = new_aliases
    end
end

"""
    analyze_block!(tracker::AliasTracker, block)

Analyze all statements in a block.
"""
function analyze_block!(tracker::AliasTracker, block)
    # Block has args, body, terminator
    # body is an iterator that yields (ssa_idx, entry) where entry has .stmt and .typ
    for (ssa_idx, entry) in block.body
        analyze_statement!(tracker, SSAValue(ssa_idx), entry.stmt)
    end
    return
end

"""
    analyze_statement!(tracker::AliasTracker, ssa::SSAValue, stmt)

Analyze a single statement and propagate aliases.
"""
function analyze_statement!(tracker::AliasTracker, ssa::SSAValue, stmt)
    return if stmt isa Expr && stmt.head === :call
        func = stmt.args[1]

        # getfield: propagate from parent
        if func === GlobalRef(Core, :getfield) && length(stmt.args) >= 2
            parent = stmt.args[2]
            field = length(stmt.args) >= 3 ? stmt.args[3] : nothing

            # For TileArray.ptr field access, propagate pointer alias
            if field isa QuoteNode && field.value === :ptr
                propagate!(tracker, parent, ssa)
            else
                # Conservatively mark as UNIVERSE for non-pointer fields
                tracker[ssa] = ALIAS_UNIVERSE
            end

            # Pointer arithmetic: propagate from pointer operand
        elseif func === GlobalRef(Base, :+) || func === GlobalRef(Base, :-)
            for arg in stmt.args[2:end]
                # Find the pointer argument and propagate
                arg_aliases = tracker[arg]
                if arg_aliases !== ALIAS_UNIVERSE || arg_aliases isa Set
                    propagate!(tracker, arg, ssa)
                    break
                end
            end

            # TileArray construction: propagate from pointer argument
        elseif is_tile_array_constructor(func)
            # First argument is typically the pointer
            if length(stmt.args) >= 2
                propagate!(tracker, stmt.args[2], ssa)
            end

            # Default: unknown operation -> UNIVERSE
        else
            tracker[ssa] = ALIAS_UNIVERSE
        end

        # Control flow operations need special handling
    elseif stmt isa ReturnNode
        # No alias propagation needed

    else
        # Unknown statement type -> conservative
        tracker[ssa] = ALIAS_UNIVERSE
    end
end

# Helper functions
contains_pointers(T) = T <: Ptr || T <: TileArray || (T <: Tile && eltype(T) <: Ptr)

function is_tile_array_constructor(func)
    # Check if this is a TileArray constructor call
    # You'll need to detect the specific GlobalRef for TileArray
    return false  # TODO: implement
end
