# Phase 2b: Pattern matching and loop upgrades
#
# This file contains functions for upgrading LoopOp to ForOp/WhileOp.
# All pattern matching operates on the structured IR after substitutions.

#=============================================================================
 Helper Functions (Pattern matching on structured IR)
=============================================================================#

"""
    find_ifop(block::Block) -> Union{Tuple{Int, IfOp}, Nothing}

Find the first IfOp in a block's ops.
Returns (position, IfOp) or nothing.
"""
function find_ifop(block::Block)
    for (i, op) in enumerate(block.ops)
        if op.expr isa IfOp
            return (i, op.expr)
        end
    end
    return nothing
end

"""
    find_op_by_local_ssa(block::Block, ssa::LocalSSA) -> Union{Operation, Nothing}

Find an Operation in the block by its local SSA position.
"""
function find_op_by_local_ssa(block::Block, ssa::LocalSSA)
    if 1 <= ssa.id <= length(block.ops)
        return block.ops[ssa.id]
    end
    return nothing
end

"""
    find_add_int_for_iv(block::Block, iv_arg::BlockArg) -> Union{Tuple{Int, Operation}, Nothing}

Find an Operation containing `add_int(iv_arg, step)` in the block.
Searches inside IfOp blocks (since condition creates IfOp structure),
but NOT into nested LoopOps (those have their own IVs).
Returns (position, Operation) or nothing.
"""
function find_add_int_for_iv(block::Block, iv_arg::BlockArg)
    for (i, op) in enumerate(block.ops)
        if op.expr isa ControlFlowOp
            if op.expr isa IfOp
                # Search in IfOp blocks (condition structure)
                result = find_add_int_for_iv(op.expr.then_block, iv_arg)
                result !== nothing && return result
                result = find_add_int_for_iv(op.expr.else_block, iv_arg)
                result !== nothing && return result
            end
            # Don't recurse into LoopOp - nested loops have their own IVs
        else
            expr = op.expr
            if expr isa Expr && expr.head === :call && length(expr.args) >= 3
                func = expr.args[1]
                if func isa GlobalRef && func.name === :add_int
                    if expr.args[2] == iv_arg
                        return (i, op)
                    end
                end
            end
        end
    end
    return nothing
end

"""
    is_loop_invariant(val, block::Block) -> Bool

Check if a value is loop-invariant (not defined inside the loop body).
- BlockArgs are loop-variant (they're loop-carried values)
- LocalSSA values that point to ops in this block are loop-variant
- Constants and Arguments are always loop-invariant
"""
function is_loop_invariant(val, block::Block)
    # BlockArgs are loop-carried values - not invariant
    val isa BlockArg && return false

    # LocalSSA: check if defined in this block (by position)
    if val isa LocalSSA
        return !(1 <= val.id <= length(block.ops))
    end

    # SSAValues shouldn't appear after structurization, but handle gracefully
    if val isa SSAValue
        return !defines(block, val)
    end

    # Constants, Arguments, etc. are invariant
    return true
end

"""
    is_for_condition(expr) -> Bool

Check if an expression is a for-loop condition pattern: slt_int or ult_int.
"""
function is_for_condition(expr)
    expr isa Expr || return false
    expr.head === :call || return false
    length(expr.args) >= 3 || return false
    func = expr.args[1]
    return func isa GlobalRef && func.name in (:slt_int, :ult_int)
end

#=============================================================================
 Loop Pattern Matching (upgrade LoopOp → ForOp/WhileOp)
=============================================================================#

"""
    apply_loop_patterns!(block::Block)

Upgrade LoopOps to ForOp/WhileOp where patterns match.
Assumes substitutions have already been applied.
Pattern matches entirely on the structured IR.
"""
function apply_loop_patterns!(block::Block)
    for (i, op) in enumerate(block.ops)
        cfop = op.expr
        cfop isa ControlFlowOp || continue

        if cfop isa LoopOp
            upgraded = try_upgrade_loop(cfop)
            if upgraded !== nothing
                block.ops[i] = Operation(upgraded, get_result_type(upgraded))
                # Recursively apply to the upgraded op's blocks
                if upgraded isa ForOp
                    apply_loop_patterns!(upgraded.body)
                elseif upgraded isa WhileOp
                    apply_loop_patterns!(upgraded.before)
                    apply_loop_patterns!(upgraded.after)
                end
            else
                apply_loop_patterns!(cfop.body)
            end
        elseif cfop isa IfOp
            apply_loop_patterns!(cfop.then_block)
            apply_loop_patterns!(cfop.else_block)
        elseif cfop isa WhileOp
            apply_loop_patterns!(cfop.before)
            apply_loop_patterns!(cfop.after)
        elseif cfop isa ForOp
            apply_loop_patterns!(cfop.body)
        end
    end
end

"""
    try_upgrade_loop(loop::LoopOp) -> Union{ForOp, WhileOp, Nothing}

Try to upgrade a LoopOp to a more specific loop type (ForOp or WhileOp).
Returns the upgraded op, or nothing if no pattern matches.
Pattern matches entirely on the structured IR (after substitutions).
"""
function try_upgrade_loop(loop::LoopOp)
    # Try ForOp pattern first
    for_op = try_upgrade_to_for(loop)
    for_op !== nothing && return for_op

    # Try WhileOp pattern
    while_op = try_upgrade_to_while(loop)
    while_op !== nothing && return while_op

    return nothing
end

"""
    try_upgrade_to_for(loop::LoopOp) -> Union{ForOp, Nothing}

Try to upgrade a LoopOp to a ForOp by detecting the for-loop pattern.
Pattern matches entirely on the structured IR (after substitutions).
"""
function try_upgrade_to_for(loop::LoopOp)
    # Find the IfOp in the loop body - this contains the condition check
    ifop_result = find_ifop(loop.body)
    ifop_result === nothing && return nothing
    ifop_pos, condition_ifop = ifop_result

    # The condition should be a LocalSSA pointing to a comparison Operation
    condition_ifop.condition isa LocalSSA || return nothing
    cond_op = find_op_by_local_ssa(loop.body, condition_ifop.condition)
    cond_op === nothing && return nothing
    cond_pos = condition_ifop.condition.id

    # Check it's a for-loop condition: slt_int(iv_arg, upper_bound)
    is_for_condition(cond_op.expr) || return nothing

    # After substitution, the IV should be a BlockArg
    iv_arg = cond_op.expr.args[2]
    iv_arg isa BlockArg || return nothing
    upper_bound = cond_op.expr.args[3]

    # Find which index this BlockArg corresponds to
    iv_idx = findfirst(==(iv_arg), loop.body.args)
    iv_idx === nothing && return nothing

    # Get lower bound from init_values
    iv_idx > length(loop.init_values) && return nothing
    lower_bound = loop.init_values[iv_idx]

    # Find the step: add_int(iv_arg, step)
    step_result = find_add_int_for_iv(loop.body, iv_arg)
    step_result === nothing && return nothing
    step_pos, step_op = step_result
    step = step_op.expr.args[3]

    # Verify upper_bound and step are loop-invariant
    is_loop_invariant(upper_bound, loop.body) || return nothing
    is_loop_invariant(step, loop.body) || return nothing

    # Separate non-IV carried values and init values
    other_init_values = IRValue[]
    other_block_args = BlockArg[]
    for (j, arg) in enumerate(loop.body.args)
        if j != iv_idx && j <= length(loop.init_values)
            push!(other_init_values, loop.init_values[j])
            # Renumber block args sequentially
            push!(other_block_args, BlockArg(length(other_block_args) + 1, arg.type))
        end
    end

    # Rebuild body block without condition structure
    # LoopOp body: [header_ops..., IfOp(cond, continue_block, break_block)]
    # ForOp body: [body_ops...] with ContinueOp terminator
    new_body = Block(loop.body.id)
    # Only include carried values, not IV (IV is stored separately in ForOp.iv_arg)
    new_body.args = other_block_args

    # Extract body operations, filtering out iv-related ones
    for (i, op) in enumerate(loop.body.ops)
        # Skip iv increment and condition comparison
        i == step_pos && continue
        i == cond_pos && continue

        if op.expr isa IfOp
            # Extract the continue path's body (skip condition check structure)
            for (j, sub_op) in enumerate(op.expr.then_block.ops)
                # Skip step increment if it's in here
                if sub_op.expr isa Expr && sub_op.expr.head === :call
                    func = sub_op.expr.args[1]
                    if func isa GlobalRef && func.name === :add_int && length(sub_op.expr.args) >= 3
                        if sub_op.expr.args[2] == iv_arg
                            continue  # Skip IV increment
                        end
                    end
                end
                push!(new_body.ops, sub_op)
            end
        else
            push!(new_body.ops, op)
        end
    end

    # Get yield values from continue terminator, excluding the IV
    yield_values = IRValue[]
    if !isempty(loop.body.ops)
        last_op = loop.body.ops[end]
        if last_op.expr isa IfOp && last_op.expr.then_block.terminator isa ContinueOp
            for (j, v) in enumerate(last_op.expr.then_block.terminator.values)
                j != iv_idx && push!(yield_values, v)
            end
        end
    end
    new_body.terminator = ContinueOp(yield_values)

    # Compute result type from non-IV block args
    result_type = compute_result_type(other_block_args)

    return ForOp(lower_bound, upper_bound, step, iv_arg,
                 other_init_values, new_body, result_type)
end

"""
    try_upgrade_to_while(loop::LoopOp) -> Union{WhileOp, Nothing}

Try to upgrade a LoopOp to a WhileOp by detecting the while-loop pattern.
Pattern matches entirely on the structured IR (after substitutions).

Creates MLIR-style scf.while with before/after regions:
- before: condition computation, ends with ConditionOp
- after: loop body, ends with YieldOp
"""
function try_upgrade_to_while(loop::LoopOp)
    # The body already has substitutions applied from Phase 2a

    # Find the IfOp in the loop body - its condition is the while condition
    ifop_result = find_ifop(loop.body)
    ifop_result === nothing && return nothing
    ifop_pos, condition_ifop = ifop_result

    # Build "before" region: operations before the IfOp + ConditionOp
    before = Block(loop.body.id)
    before.args = copy(loop.body.args)

    for (i, op) in enumerate(loop.body.ops)
        if op.expr isa IfOp
            # Stop before IfOp - the condition becomes ConditionOp
            break
        else
            push!(before.ops, op)
        end
    end

    # ConditionOp args must include ALL block args (MLIR scf.while semantics).
    # - When condition is true: all args are passed to after region
    # - When condition is false: all args become the loop results
    # We always pass all block args, not just the BreakOp values which may be a subset.
    condition_args = IRValue[arg for arg in before.args]

    before.terminator = ConditionOp(condition_ifop.condition, condition_args)

    # Build "after" region: operations from the then_block + YieldOp
    after = Block(loop.body.id + 1000)  # Different block ID
    # After region receives args from ConditionOp
    for (i, arg) in enumerate(before.args)
        push!(after.args, BlockArg(i, arg.type))
    end

    # Copy body operations from the continue path
    for op in condition_ifop.then_block.ops
        push!(after.ops, op)
    end

    # Get yield values from the continue terminator.
    # The ContinueOp values are the "updated" carried values.
    # For loop-invariant values (captured from outer scope), we pass them through unchanged.
    yield_values = IRValue[]
    if condition_ifop.then_block.terminator isa ContinueOp
        yield_values = copy(condition_ifop.then_block.terminator.values)
    end
    # For any remaining block args (loop-invariant), use the after block args directly
    for i in (length(yield_values) + 1):length(after.args)
        push!(yield_values, after.args[i])
    end
    after.terminator = YieldOp(yield_values)

    # Use the same result type as the original loop
    return WhileOp(before, after, loop.init_values, loop.result_type)
end
