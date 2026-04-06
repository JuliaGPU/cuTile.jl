# Loop-Invariant Code Motion (LICM)
#
# Hoists loop-invariant operations out of loops. Runs AFTER token_order_pass!
# so that token dependencies correctly prevent unsafe hoisting of aliasing loads.
#
# Operations classified as stores (store_partition_view, store_ptr_tko, atomics,
# print_tko) and control flow exits (return) are never hoisted. All other
# operations — including loads, arithmetic, partition views, token nodes — are
# hoisted when all their data dependencies are defined outside the loop.
#
# This mirrors cuTile Python's code_motion.py:hoist_loop_invariants.

# Indicates whether a block could in theory be moved, based on the operations
# it contains (side effects, jumps). Does not consider data dependencies.
@enum _BlockMobility::Int8 begin
    # The block (or any ancestor) cannot be moved due to side effects.
    _IMMOVABLE = 0
    # The block itself can't be hoisted alone, but its containing loop can.
    # Happens when the block contains Continue or Break.
    _CAN_MOVE_WITH_LOOP = 1
    # The block can move (subject to data dependencies).
    _CAN_MOVE = 2
end

struct _BlockResult
    mobility::_BlockMobility
    min_depth::Int  # deepest outside dependency of any hoisted-out op
end

# Helper for accumulating data dependency information per operation.
mutable struct _DepInfo
    must_stay::Bool
    max_outside_depth::Int
end

function _update_dep!(di::_DepInfo, dep_depth::Int, cur_depth::Int)
    if dep_depth >= cur_depth
        di.must_stay = true
    else
        di.max_outside_depth = max(di.max_outside_depth, dep_depth)
    end
end

# Update dependency info from an SSA value or literal.
function _check_val!(di::_DepInfo, val, def_depth::Dict{Any,Int}, cur_depth::Int)
    d = get(def_depth, val, nothing)
    d === nothing && return  # constants/literals always available
    _update_dep!(di, d, cur_depth)
end

# Extract all SSA dependencies from a statement.
function _check_stmt_deps!(di::_DepInfo, @nospecialize(s), def_depth::Dict{Any,Int},
                           cur_depth::Int)
    if s isa Expr
        start = s.head === :invoke ? 3 : 2
        for i in start:length(s.args)
            _check_val!(di, s.args[i], def_depth, cur_depth)
        end
    elseif s isa JoinTokensNode
        for tok in s.tokens
            _check_val!(di, tok, def_depth, cur_depth)
        end
    elseif s isa TokenResultNode
        _check_val!(di, SSAValue(s.mem_op_ssa), def_depth, cur_depth)
    end
    # MakeTokenNode, PiNode, GlobalRef, literals: no SSA deps
end

struct _StackItem
    entries::Vector{Tuple{Int,Any,Any}}  # (ssa_idx, stmt, type)
    is_loop_body::Bool
end

"""
    licm_pass!(sci::StructuredIRCode)

Hoist loop-invariant operations out of loops. Must run after token_order_pass!.
"""
function licm_pass!(sci::StructuredIRCode)
    def_depth = Dict{Any,Int}()
    for i in 1:length(sci.argtypes)
        def_depth[Argument(i)] = 0
    end
    _hoist!(sci.entry, _StackItem[], def_depth, false)
    return
end

function _hoist!(block::Block, stack::Vector{_StackItem}, def_depth::Dict{Any,Int},
                 is_loop_body::Bool)
    depth = length(stack)
    push!(stack, _StackItem(Tuple{Int,Any,Any}[], is_loop_body))

    for ba in block.args
        def_depth[ba] = depth
    end

    mobility = _CAN_MOVE
    min_depth = 0

    for inst in instructions(block)
        s = stmt(inst)
        di = _DepInfo(!is_loop_body, 0)

        if s isa ForOp
            def_depth[s.iv_arg] = depth + 1
            for ba in s.body.args
                def_depth[ba] = depth + 1
            end
            body_res = _hoist!(s.body, stack, def_depth, true)
            if body_res.mobility == _IMMOVABLE
                mobility = _IMMOVABLE
                di.must_stay = true
            end
            for v in s.init_values
                _check_val!(di, v, def_depth, depth)
            end
            _update_dep!(di, body_res.min_depth, depth)

        elseif s isa LoopOp
            for ba in s.body.args
                def_depth[ba] = depth + 1
            end
            body_res = _hoist!(s.body, stack, def_depth, true)
            if body_res.mobility == _IMMOVABLE
                mobility = _IMMOVABLE
                di.must_stay = true
            end
            for v in s.init_values
                _check_val!(di, v, def_depth, depth)
            end
            _update_dep!(di, body_res.min_depth, depth)

        elseif s isa WhileOp
            for ba in s.before.args
                def_depth[ba] = depth + 1
            end
            for ba in s.after.args
                def_depth[ba] = depth + 1
            end
            before_res = _hoist!(s.before, stack, def_depth, true)
            after_res = _hoist!(s.after, stack, def_depth, true)
            if min(before_res.mobility, after_res.mobility) == _IMMOVABLE
                mobility = _IMMOVABLE
                di.must_stay = true
            end
            for v in s.init_values
                _check_val!(di, v, def_depth, depth)
            end
            _update_dep!(di, before_res.min_depth, depth)
            _update_dep!(di, after_res.min_depth, depth)

        elseif s isa IfOp
            _check_val!(di, s.condition, def_depth, depth)
            for region in (s.then_region, s.else_region)
                branch_res = _hoist!(region, stack, def_depth, false)
                _update_dep!(di, branch_res.min_depth, depth)
                if branch_res.mobility != _CAN_MOVE
                    mobility = min(mobility, branch_res.mobility)
                    di.must_stay = true
                end
            end

        elseif _is_store(block, s)
            mobility = _IMMOVABLE
            di.must_stay = true

        elseif s isa ContinueOp || s isa BreakOp
            mobility = min(mobility, _CAN_MOVE_WITH_LOOP)
            di.must_stay = true

        elseif s isa YieldOp || s isa ConditionOp || s isa ReturnNode
            di.must_stay = true
            # Track deps for YieldOp/ConditionOp so min_depth is correct
            if s isa YieldOp
                for v in s.values
                    _check_val!(di, v, def_depth, depth)
                end
            elseif s isa ConditionOp
                _check_val!(di, s.condition, def_depth, depth)
                for v in s.args
                    _check_val!(di, v, def_depth, depth)
                end
            end

        else
            # Movable operation: loads, arithmetic, make_partition_view, etc.
            _check_stmt_deps!(di, s, def_depth, depth)
        end

        # Determine target depth
        target_depth = depth
        if di.must_stay
            min_depth = max(min_depth, di.max_outside_depth)
        else
            while target_depth > di.max_outside_depth && stack[target_depth].is_loop_body
                target_depth -= 1
            end
        end

        push!(stack[target_depth + 1].entries, (inst.ssa_idx, s, inst.typ))

        # Record definition depth AFTER hoisting so subsequent ops see the new depth
        def_depth[SSAValue(inst.ssa_idx)] = target_depth
    end

    # Rebuild block body from collected entries
    entries = pop!(stack).entries
    empty!(block)
    for (idx, s, typ) in entries
        push!(block, idx, s, typ)
    end

    return _BlockResult(mobility, min_depth)
end

# Check if a statement is a store/atomic (side-effecting memory write).
function _is_store(block::Block, @nospecialize(s))
    call = resolve_call(block, s)
    call === nothing && return false
    resolved_func, _ = call
    return classify_memory_op(resolved_func) == MEM_STORE
end
