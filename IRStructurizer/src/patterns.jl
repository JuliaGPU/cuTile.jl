# Phase 2b: Pattern matching and loop upgrades
#
# This file contains functions for upgrading LoopOp to ForOp/WhileOp.
# All pattern matching operates on the structured IR after substitutions.

#=============================================================================
 Helper Functions (Pattern matching on structured IR)
=============================================================================#

"""
    find_ifop(block::Block) -> Union{IfOp, Nothing}

Find the first IfOp in a block's body.
"""
function find_ifop(block::Block)
    for item in block.body
        if item isa IfOp
            return item
        end
    end
    return nothing
end

"""
    find_statement_by_ssa(block::Block, ssa::SSAValue) -> Union{Statement, Nothing}

Find a Statement in the block whose idx matches the SSAValue's id.
"""
function find_statement_by_ssa(block::Block, ssa::SSAValue)
    for item in block.body
        if item isa Statement && item.idx == ssa.id
            return item
        end
    end
    return nothing
end

"""
    find_add_int_for_iv(block::Block, iv_arg::BlockArg) -> Union{Statement, Nothing}

Find a Statement containing `add_int(iv_arg, step)` in the block.
Searches inside IfOp blocks (since condition creates IfOp structure),
but NOT into nested LoopOps (those have their own IVs).
"""
function find_add_int_for_iv(block::Block, iv_arg::BlockArg)
    for item in block.body
        if item isa Statement
            expr = item.expr
            if expr isa Expr && expr.head === :call && length(expr.args) >= 3
                func = expr.args[1]
                if func isa GlobalRef && func.name === :add_int
                    if expr.args[2] == iv_arg
                        return item
                    end
                end
            end
        elseif item isa IfOp
            # Search in IfOp blocks (condition structure)
            result = find_add_int_for_iv(item.then_block, iv_arg)
            result !== nothing && return result
            result = find_add_int_for_iv(item.else_block, iv_arg)
            result !== nothing && return result
        end
        # Don't recurse into LoopOp - nested loops have their own IVs
    end
    return nothing
end

"""
    is_loop_invariant(val, block::Block) -> Bool

Check if a value is loop-invariant (not defined inside the loop body).
- BlockArgs are loop-variant (they're loop-carried values)
- SSAValues are loop-invariant if no Statement in the body defines them
- Constants and Arguments are always loop-invariant
"""
function is_loop_invariant(val, block::Block)
    # BlockArgs are loop-carried values - not invariant
    val isa BlockArg && return false

    # SSAValues: check if defined in the loop body (including nested blocks)
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
    for (i, item) in enumerate(block.body)
        if item isa LoopOp
            upgraded = try_upgrade_loop(item)
            if upgraded !== nothing
                block.body[i] = upgraded
                # Recursively apply to the upgraded op's blocks
                if upgraded isa ForOp
                    apply_loop_patterns!(upgraded.body)
                elseif upgraded isa WhileOp
                    apply_loop_patterns!(upgraded.before)
                    apply_loop_patterns!(upgraded.after)
                end
            else
                apply_loop_patterns!(item.body)
            end
        elseif item isa IfOp
            apply_loop_patterns!(item.then_block)
            apply_loop_patterns!(item.else_block)
        elseif item isa WhileOp
            apply_loop_patterns!(item.before)
            apply_loop_patterns!(item.after)
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
    condition_ifop = find_ifop(loop.body)
    condition_ifop === nothing && return nothing

    # The condition should be an SSAValue pointing to a comparison Statement
    condition_ifop.condition isa SSAValue || return nothing
    cond_stmt = find_statement_by_ssa(loop.body, condition_ifop.condition)
    cond_stmt === nothing && return nothing

    # Check it's a for-loop condition: slt_int(iv_arg, upper_bound)
    is_for_condition(cond_stmt.expr) || return nothing

    # After substitution, the IV should be a BlockArg
    iv_arg = cond_stmt.expr.args[2]
    iv_arg isa BlockArg || return nothing
    upper_bound = cond_stmt.expr.args[3]

    # Find which index this BlockArg corresponds to
    iv_idx = findfirst(==(iv_arg), loop.body.args)
    iv_idx === nothing && return nothing

    # Get lower bound from init_values
    iv_idx > length(loop.init_values) && return nothing
    lower_bound = loop.init_values[iv_idx]

    # Find the step: add_int(iv_arg, step)
    step_stmt = find_add_int_for_iv(loop.body, iv_arg)
    step_stmt === nothing && return nothing
    step = step_stmt.expr.args[3]

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
            push!(other_block_args, BlockArg(length(other_block_args) + 1, arg.type, arg.ssa_origin))
        end
    end

    # Rebuild body block without condition structure
    # LoopOp body: [header_stmts..., IfOp(cond, continue_block, break_block)]
    # ForOp body: [body_stmts...] with ContinueOp terminator
    new_body = Block(loop.body.id)
    # Only include carried values, not IV (IV is stored separately in ForOp.iv_arg)
    new_body.args = other_block_args

    # Extract body statements, filtering out iv-related ones
    for item in loop.body.body
        if item isa Statement
            # Skip iv increment and condition comparison
            item === step_stmt && continue
            item === cond_stmt && continue
            push!(new_body.body, item)
        elseif item isa IfOp
            # Extract the continue path's body (skip condition check structure)
            for sub_item in item.then_block.body
                if sub_item isa Statement
                    sub_item === step_stmt && continue
                    push!(new_body.body, sub_item)
                else
                    push!(new_body.body, sub_item)
                end
            end
        else
            push!(new_body.body, item)
        end
    end

    # Get yield values from continue terminator, excluding the IV
    yield_values = IRValue[]
    if !isempty(loop.body.body)
        last_item = loop.body.body[end]
        if last_item isa IfOp && last_item.then_block.terminator isa ContinueOp
            for (j, v) in enumerate(last_item.then_block.terminator.values)
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
    condition_ifop = find_ifop(loop.body)
    condition_ifop === nothing && return nothing

    # Build "before" region: statements before the IfOp + ConditionOp
    before = Block(loop.body.id)
    before.args = copy(loop.body.args)

    for item in loop.body.body
        if item isa Statement
            push!(before.body, item)
        elseif item isa IfOp
            # Stop before IfOp - the condition becomes ConditionOp
            break
        else
            push!(before.body, item)
        end
    end

    # Get break values (become results when condition is false)
    # Also used as args passed to after region when condition is true
    condition_args = IRValue[]
    if condition_ifop.else_block.terminator isa BreakOp
        condition_args = copy(condition_ifop.else_block.terminator.values)
    elseif !isempty(before.args)
        # Default: forward the block args
        condition_args = [arg for arg in before.args]
    end

    before.terminator = ConditionOp(condition_ifop.condition, condition_args)

    # Build "after" region: statements from the then_block + YieldOp
    after = Block(loop.body.id + 1000)  # Different block ID
    # After region receives args from ConditionOp - create new BlockArgs (no ssa_origin needed)
    for (i, arg) in enumerate(before.args)
        push!(after.args, BlockArg(i, arg.type, 0))
    end

    # Copy body statements from the continue path
    for item in condition_ifop.then_block.body
        push!(after.body, item)
    end

    # Get yield values from the continue terminator
    yield_values = IRValue[]
    if condition_ifop.then_block.terminator isa ContinueOp
        yield_values = copy(condition_ifop.then_block.terminator.values)
    end
    after.terminator = YieldOp(yield_values)

    # Use the same result type as the original loop
    return WhileOp(before, after, loop.init_values, loop.result_type)
end
