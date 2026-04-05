# Loop-Invariant Code Motion (LICM)
#
# Hoists loop-invariant loads (and their dependency chains) out of loops.
#
# Pure operations (arithmetic, broadcasts, view constructors) are NOT hoisted
# here — MLIR's built-in LICM handles those at optLevel >= 2.
#
# This pass targets what MLIR cannot hoist: memory loads. After token ordering,
# loads have token dependencies that anchor them inside loops. By hoisting
# before token insertion, we avoid creating unnecessary token carries.
#
# Safety: a load is hoistable only when (1) all its operands are loop-invariant,
# and (2) no store in the loop body aliases with the load's memory region.
# Alias information comes from alias_analysis_pass!, which must run first.

"""
    licm_pass!(sci::StructuredIRCode, alias_result::Dict{Any, AliasSet})

Hoist loop-invariant loads out of loops. Must run after alias_analysis_pass!
and before token_order_pass!.

A load is hoistable when:
- All operands are defined outside the loop
- No store in the loop body writes to an aliasing memory region
"""
function licm_pass!(sci::StructuredIRCode, alias_result::Dict{Any, AliasSet})
    def_depth = Dict{Any, Int}()
    for i in 1:length(sci.argtypes)
        def_depth[Argument(i)] = 0
    end
    _hoist_loads!(sci.entry, Vector{Vector{Tuple{Int,Any,Any}}}(), def_depth,
                  alias_result, false)
    return
end

# Collect alias sets of all stores in a block (recursively through nested CFs).
function _collect_store_aliases(block::Block, alias_result::Dict{Any, AliasSet})
    store_aliases = AliasSet[]
    for inst in instructions(block)
        s = stmt(inst)
        if s isa ControlFlowOp
            for b in blocks(s)
                append!(store_aliases, _collect_store_aliases(b, alias_result))
            end
        else
            call = resolve_call(block, s)
            call === nothing && continue
            resolved_func, operands = call
            if classify_memory_op(resolved_func) == MEM_STORE
                aset = get_alias_set_for_operand(alias_result, first(operands))
                push!(store_aliases, aset)
            end
        end
    end
    return store_aliases
end

# Check if a load's alias set conflicts with any store alias set in the loop.
function _aliases_with_store(load_alias::AliasSet, store_aliases::Vector{AliasSet})
    for sa in store_aliases
        if load_alias isa AliasUniverse || sa isa AliasUniverse
            return true
        end
        if !isempty(intersect(load_alias, sa))
            return true
        end
    end
    return false
end

function _hoist_loads!(block::Block, stack::Vector{Vector{Tuple{Int,Any,Any}}},
                       def_depth::Dict{Any,Int}, alias_result::Dict{Any, AliasSet},
                       is_loop_body::Bool)
    depth = length(stack)
    push!(stack, Tuple{Int,Any,Any}[])

    # Register block args at current depth
    for ba in block.args
        def_depth[ba] = depth
    end

    # If this is a loop body, collect store alias sets for the load safety check
    store_aliases = is_loop_body ? _collect_store_aliases(block, alias_result) : AliasSet[]

    for inst in instructions(block)
        s = stmt(inst)
        hoisted = false

        if s isa ForOp || s isa LoopOp
            body = s.body
            if s isa ForOp
                def_depth[s.iv_arg] = depth + 1
            end
            for ba in body.args
                def_depth[ba] = depth + 1
            end
            _hoist_loads!(body, stack, def_depth, alias_result, true)

        elseif s isa WhileOp
            for ba in s.before.args
                def_depth[ba] = depth + 1
            end
            for ba in s.after.args
                def_depth[ba] = depth + 1
            end
            _hoist_loads!(s.before, stack, def_depth, alias_result, true)
            _hoist_loads!(s.after, stack, def_depth, alias_result, true)

        elseif s isa IfOp
            _hoist_loads!(s.then_region, stack, def_depth, alias_result, false)
            _hoist_loads!(s.else_region, stack, def_depth, alias_result, false)

        elseif is_loop_body && _is_hoistable_load(block, s, def_depth, depth,
                                                   alias_result, store_aliases)
            # Hoist this load to the enclosing scope
            target_depth = depth - 1
            while target_depth > 0 && _can_hoist_to(stack, target_depth)
                target_depth -= 1
            end
            push!(stack[target_depth + 1], (inst.ssa_idx, s, inst.typ))
            def_depth[SSAValue(inst.ssa_idx)] = target_depth
            hoisted = true
        end

        if !hoisted
            # Keep at current depth
            push!(stack[depth + 1], (inst.ssa_idx, s, inst.typ))
            def_depth[SSAValue(inst.ssa_idx)] = depth
        end
    end

    # Rebuild block body from collected entries
    entries = pop!(stack)
    empty!(block)
    for (idx, s, typ) in entries
        push!(block, idx, s, typ)
    end
end

# Check if a stack entry is a loop body (for multi-level hoisting)
function _can_hoist_to(stack::Vector{Vector{Tuple{Int,Any,Any}}}, target_depth::Int)
    # We'd need to track is_loop_body per stack entry to do multi-level hoisting.
    # For now, only hoist one level out.
    return false
end

# Check if a statement is a load that can be safely hoisted.
function _is_hoistable_load(block::Block, @nospecialize(s), def_depth::Dict{Any,Int},
                            cur_depth::Int, alias_result::Dict{Any, AliasSet},
                            store_aliases::Vector{AliasSet})
    s isa Expr || return false
    call = resolve_call(block, s)
    call === nothing && return false
    resolved_func, operands = call

    # Must be a load operation
    classify_memory_op(resolved_func) == MEM_LOAD || return false

    # All operands must be defined outside this loop
    _all_operands_outside(s, def_depth, cur_depth) || return false

    # Load must not alias with any store in the loop
    load_alias = get_alias_set_for_operand(alias_result, first(operands))
    return !_aliases_with_store(load_alias, store_aliases)
end

# Check that all operands of a statement are defined at depth < cur_depth.
function _all_operands_outside(@nospecialize(s), def_depth::Dict{Any,Int}, cur_depth::Int)
    s isa Expr || return true
    start = s.head === :invoke ? 3 : 2
    for i in start:length(s.args)
        d = get(def_depth, s.args[i], nothing)
        d === nothing && continue  # constants/literals are always available
        d >= cur_depth && return false
    end
    return true
end
