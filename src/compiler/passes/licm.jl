# Loop-Invariant Code Motion (LICM)
#
# Single-pass depth-tracking algorithm that hoists loop-invariant operations
# out of loops. Port of cuTile Python's `hoist_loop_invariants` (code_motion.py).
#
# The algorithm walks the IR recursively while tracking the definition depth
# of each value. An operation whose data dependencies all resolve to depths
# *less than* its containing loop can be hoisted above that loop. A stack of
# SSAMaps collects operations at their target depth; at the end of each block,
# the original body is replaced with the (filtered) rebuilt map.

# Whether a block can be moved, based on the operations it contains.
@enum BlockMobility begin
    IMMOVABLE          # contains stores, returns, or nested IMMOVABLE blocks
    CAN_MOVE_WITH_LOOP # contains continue/break
    CAN_MOVE           # pure operations only
end

struct BlockResult
    mobility::BlockMobility
    min_depth::Int   # minimum depth any op in this block needs
end

mutable struct DependencyInfo
    must_stay::Bool
    max_outside_depth::Int
end

function update!(di::DependencyInfo, dep_depth::Int, cur_depth::Int)
    if dep_depth >= cur_depth
        di.must_stay = true
    else
        di.max_outside_depth = max(di.max_outside_depth, dep_depth)
    end
end

struct StackItem
    new_body::SSAMap
    is_loop_body::Bool
end

"""
    licm_pass!(sci::StructuredIRCode)

Hoist loop-invariant operations out of loops. Must run after rewrite_patterns!
and before token_order_pass! (which inserts tokens that should not be moved).
"""
function licm_pass!(sci::StructuredIRCode)
    def_depth = Dict{Any, Int}()
    for i in 1:length(sci.argtypes)
        def_depth[Argument(i)] = 0
    end
    _hoist!(sci.entry, StackItem[], def_depth, false)
    return
end

function _hoist!(block::Block, stack::Vector{StackItem}, def_depth::Dict{Any,Int},
                 is_loop_body::Bool)
    depth = length(stack)
    new_body = SSAMap()
    push!(stack, StackItem(new_body, is_loop_body))

    mobility = CAN_MOVE
    min_depth = 0

    # Register block args at current depth
    for ba in block.args
        def_depth[ba] = depth
    end

    for inst in instructions(block)
        s = stmt(inst)
        depinfo = DependencyInfo(!is_loop_body, 0)

        if s isa ForOp || s isa LoopOp
            body = s.body
            # ForOp's iv_arg is separate from body.args (which holds only carries)
            if s isa ForOp
                def_depth[s.iv_arg] = depth + 1
            end
            for ba in body.args
                def_depth[ba] = depth + 1
            end
            body_result = _hoist!(body, stack, def_depth, true)
            if body_result.mobility == IMMOVABLE
                mobility = IMMOVABLE
                depinfo.must_stay = true
            end
            for v in s.init_values
                _update_from_value!(depinfo, def_depth, v, depth)
            end
            if s isa ForOp
                for v in (s.lower, s.upper, s.step)
                    _update_from_value!(depinfo, def_depth, v, depth)
                end
            end
            update!(depinfo, body_result.min_depth, depth)

        elseif s isa WhileOp
            for ba in s.before.args
                def_depth[ba] = depth + 1
            end
            for ba in s.after.args
                def_depth[ba] = depth + 1
            end
            before_result = _hoist!(s.before, stack, def_depth, true)
            after_result = _hoist!(s.after, stack, def_depth, true)
            worst = min(before_result.mobility, after_result.mobility)
            if worst == IMMOVABLE
                mobility = IMMOVABLE
                depinfo.must_stay = true
            end
            for v in s.init_values
                _update_from_value!(depinfo, def_depth, v, depth)
            end
            update!(depinfo, before_result.min_depth, depth)
            update!(depinfo, after_result.min_depth, depth)

        elseif s isa IfOp
            _update_from_value!(depinfo, def_depth, s.condition, depth)
            then_result = _hoist!(s.then_region, stack, def_depth, false)
            else_result = _hoist!(s.else_region, stack, def_depth, false)
            update!(depinfo, then_result.min_depth, depth)
            update!(depinfo, else_result.min_depth, depth)
            for r in (then_result, else_result)
                if r.mobility != CAN_MOVE
                    mobility = min(mobility, r.mobility)
                    depinfo.must_stay = true
                end
            end

        elseif _is_memory_store(block, s)
            mobility = IMMOVABLE
            depinfo.must_stay = true
        else
            # Pure operation: check operand depths
            _update_operand_depths!(depinfo, def_depth, s, depth)
        end

        # Determine target depth
        target_depth = depth
        if depinfo.must_stay
            min_depth = max(min_depth, depinfo.max_outside_depth)
        else
            while target_depth > depinfo.max_outside_depth && stack[target_depth + 1].is_loop_body
                target_depth -= 1
            end
        end

        # Place at target depth
        push!(stack[target_depth + 1].new_body, (inst.ssa_idx, s, inst.typ))

        # Record definition depth AFTER hoisting (enables cascading hoists)
        def_depth[SSAValue(inst.ssa_idx)] = target_depth
    end

    # Handle terminator operands for min_depth computation
    term = block.terminator
    if term isa ContinueOp || term isa BreakOp
        mobility = min(mobility, CAN_MOVE_WITH_LOOP)
    end

    pop!(stack)
    block.body = new_body
    return BlockResult(mobility, min_depth)
end

# Check if a statement is a memory store (IMMOVABLE for LICM purposes).
# Loads are hoistable (they're pure if operands are invariant).
function _is_memory_store(block::Block, @nospecialize(s))
    s isa Expr || return false
    call = resolve_call(block, s)
    call === nothing && return false
    resolved_func, _ = call
    effect = classify_memory_op(resolved_func)
    return effect == MEM_STORE
end

# Update DependencyInfo from a single IR value
function _update_from_value!(di::DependencyInfo, def_depth::Dict{Any,Int}, @nospecialize(val), cur_depth::Int)
    d = get(def_depth, val, nothing)
    d !== nothing && update!(di, d, cur_depth)
end

# Update DependencyInfo from all operands of a statement
function _update_operand_depths!(di::DependencyInfo, def_depth::Dict{Any,Int}, @nospecialize(s), cur_depth::Int)
    if s isa Expr
        start = s.head === :invoke ? 3 : 2
        for i in start:length(s.args)
            _update_from_value!(di, def_depth, s.args[i], cur_depth)
        end
    elseif s isa Core.PiNode
        _update_from_value!(di, def_depth, s.val, cur_depth)
    end
end
