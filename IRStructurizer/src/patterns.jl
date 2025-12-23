# Phase 3: Pattern matching and loop upgrades
#
# This file contains functions for upgrading :loop to :for/:while.
# All pattern matching operates on the structured IR after substitutions.

#=============================================================================
 Helper Functions (Pattern matching on structured IR)
=============================================================================#

"""
    find_ifop(block::Block) -> Union{PartialControlFlowOp, Nothing}

Find the first :if op in a block's body.
"""
function find_ifop(block::Block)
    for (idx, item) in block.body
        if item isa PartialControlFlowOp && item.head == :if
            return item
        end
    end
    return nothing
end

"""
    find_expr_by_ssa(block::Block, ssa::SSAValue) -> Union{Tuple{Int, Any, Any}, Nothing}

Find an expression in the block whose SSA index matches the SSAValue's id.
Returns (idx, expr, type) tuple or nothing.
"""
function find_expr_by_ssa(block::Block, ssa::SSAValue)
    for (idx, item) in block.body
        if !(item isa PartialControlFlowOp) && idx == ssa.id
            return (idx, item, block.types[idx])
        end
    end
    return nothing
end

"""
    find_add_int_for_iv(block::Block, iv_arg::BlockArg) -> Union{Tuple{Int, Any, Any}, Nothing}

Find an expression containing `add_int(iv_arg, step)` in the block.
Searches inside :if ops (since condition creates :if structure),
but NOT into nested :loop ops (those have their own IVs).
Returns (idx, expr, type) tuple or nothing.
"""
function find_add_int_for_iv(block::Block, iv_arg::BlockArg)
    for (idx, item) in block.body
        if item isa PartialControlFlowOp
            if item.head == :if
                # Search in :if blocks (condition structure)
                then_blk = item.regions[:then]::Block
                else_blk = item.regions[:else]::Block
                result = find_add_int_for_iv(then_blk, iv_arg)
                result !== nothing && return result
                result = find_add_int_for_iv(else_blk, iv_arg)
                result !== nothing && return result
            end
            # Don't recurse into :loop - nested loops have their own IVs
        else  # Statement
            expr = item
            if expr isa Expr && expr.head === :call && length(expr.args) >= 3
                func = expr.args[1]
                if func isa GlobalRef && func.name === :add_int
                    if expr.args[2] == iv_arg
                        return (idx, item, block.types[idx])
                    end
                end
            end
        end
    end
    return nothing
end

"""
    is_loop_invariant(val, block::Block, n_result_vars::Int) -> Bool

Check if a value is loop-invariant (not defined inside the loop body).
- BlockArgs for phi values (indices 1..n_result_vars) are loop-variant
- BlockArgs for outer captures (indices > n_result_vars) are loop-invariant
- SSAValues are loop-invariant if no statement in the body defines them
- Constants and Arguments are always loop-invariant
"""
function is_loop_invariant(val, block::Block, n_result_vars::Int)
    # BlockArgs: phi values are variant, outer captures are invariant
    if val isa BlockArg
        # BlockArgs beyond result_vars count are outer captures (invariant)
        return val.id > n_result_vars
    end

    # SSAValues: check if defined in the loop body (including nested blocks)
    if val isa SSAValue
        return !defines(block, val)
    end

    # Constants, Arguments, etc. are invariant
    return true
end

"""
    defines(block::Block, ssa::SSAValue) -> Bool

Check if a block defines an SSA value (i.e., contains an expression that produces it).
Searches nested blocks recursively.
"""
function defines(block::Block, ssa::SSAValue)
    for (idx, item) in block.body
        if item isa PartialControlFlowOp
            for (_, region) in item.regions
                defines(region, ssa) && return true
            end
        else  # Statement
            if idx == ssa.id
                return true
            end
        end
    end
    return false
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
 Loop Pattern Matching (upgrade :loop → :for/:while)
=============================================================================#

"""
    apply_loop_patterns!(block::Block, ctx::StructurizationContext)

Upgrade :loop ops to :for/:while where patterns match.
Modifies ops in-place by changing their head and operands.
"""
function apply_loop_patterns!(block::Block, ctx::StructurizationContext)
    for (idx, item) in block.body
        if item isa PartialControlFlowOp
            if item.head == :loop
                if try_upgrade_loop!(item, ctx)
                    # Successfully upgraded, recurse into the modified op's regions
                    for (_, region) in item.regions
                        apply_loop_patterns!(region, ctx)
                    end
                else
                    # Not upgraded, recurse into body
                    apply_loop_patterns!(item.regions[:body], ctx)
                end
            elseif item.head == :if
                apply_loop_patterns!(item.regions[:then], ctx)
                apply_loop_patterns!(item.regions[:else], ctx)
            elseif item.head == :while
                apply_loop_patterns!(item.regions[:before], ctx)
                apply_loop_patterns!(item.regions[:after], ctx)
            elseif item.head == :for
                apply_loop_patterns!(item.regions[:body], ctx)
            end
        end
    end
end

"""
    try_upgrade_loop!(loop::PartialControlFlowOp, ctx::StructurizationContext) -> Bool

Try to upgrade a :loop op to :for or :while by modifying it in-place.
Returns true if upgraded, false otherwise.
"""
function try_upgrade_loop!(loop::PartialControlFlowOp, ctx::StructurizationContext)
    @assert loop.head == :loop

    # Try ForOp pattern first
    if try_upgrade_to_for!(loop, ctx)
        return true
    end

    # Try WhileOp pattern
    if try_upgrade_to_while!(loop, ctx)
        return true
    end

    return false
end

"""
    try_upgrade_to_for!(loop::PartialControlFlowOp, ctx::StructurizationContext) -> Bool

Try to upgrade a :loop op to :for by detecting the for-loop pattern.
Modifies the op in-place. Returns true if upgraded.
"""
function try_upgrade_to_for!(loop::PartialControlFlowOp, ctx::StructurizationContext)
    @assert loop.head == :loop
    body = loop.regions[:body]::Block
    result_vars = get_result_vars(ctx, loop)

    # Find the :if op in the loop body - this contains the condition check
    condition_ifop = find_ifop(body)
    condition_ifop === nothing && return false

    # The condition should be an SSAValue pointing to a comparison expression
    cond_val = condition_ifop.operands.condition
    cond_val isa SSAValue || return false
    cond_result = find_expr_by_ssa(body, cond_val)
    cond_result === nothing && return false
    cond_idx, cond_expr, _ = cond_result

    # Check it's a for-loop condition: slt_int(iv_arg, upper_bound)
    is_for_condition(cond_expr) || return false

    # After substitution, the IV should be a BlockArg
    iv_arg = cond_expr.args[2]
    iv_arg isa BlockArg || return false
    upper_bound_raw = cond_expr.args[3]

    # If upper_bound is a BlockArg (from outer capture), resolve to original SSAValue
    upper_bound = if upper_bound_raw isa BlockArg && upper_bound_raw.id <= length(loop.init_values)
        loop.init_values[upper_bound_raw.id]
    else
        upper_bound_raw
    end

    # Find which index this BlockArg corresponds to
    iv_idx = findfirst(==(iv_arg), body.args)
    iv_idx === nothing && return false

    # Get lower bound from init_values
    iv_idx > length(loop.init_values) && return false
    iv_idx > length(result_vars) && return false
    lower_bound = loop.init_values[iv_idx]

    # Find the step: add_int(iv_arg, step)
    step_result = find_add_int_for_iv(body, iv_arg)
    step_result === nothing && return false
    step_idx, step_expr, _ = step_result
    step_raw = step_expr.args[3]

    # If step is a BlockArg (from outer capture), resolve to original SSAValue
    step = if step_raw isa BlockArg && step_raw.id <= length(loop.init_values)
        loop.init_values[step_raw.id]
    else
        step_raw
    end

    # Verify upper_bound and step are loop-invariant
    is_loop_invariant(upper_bound, body, length(result_vars)) || return false
    is_loop_invariant(step, body, length(result_vars)) || return false

    # Separate non-IV carried values (from result_vars)
    other_result_vars = SSAValue[]
    other_init_values = IRValue[]
    for (j, rv) in enumerate(result_vars)
        if j != iv_idx && j <= length(loop.init_values)
            push!(other_result_vars, rv)
            push!(other_init_values, loop.init_values[j])
        end
    end

    # Add outer captures (init_values beyond result_vars)
    for j in (length(result_vars)+1):length(loop.init_values)
        push!(other_init_values, loop.init_values[j])
    end

    # Rebuild body block without condition structure
    then_blk = condition_ifop.regions[:then]::Block
    new_body = Block()
    # Only include carried values, not IV
    new_body.args = [arg for arg in body.args if arg !== iv_arg]

    # Extract body items, filtering out iv-related ones
    for (idx, item) in body.body
        if item isa PartialControlFlowOp
            if item.head == :if && item === condition_ifop
                # Extract the continue path's body (skip condition check structure)
                for (sub_idx, sub_item) in then_blk.body
                    if sub_item isa PartialControlFlowOp
                        push_op!(new_body, sub_idx, sub_item)
                    else  # Statement
                        sub_idx == step_idx && continue
                        push_stmt!(new_body, sub_idx, sub_item, then_blk.types[sub_idx])
                    end
                end
            else
                push_op!(new_body, idx, item)
            end
        else  # Statement
            idx == step_idx && continue
            idx == cond_idx && continue
            push_stmt!(new_body, idx, item, body.types[idx])
        end
    end

    # Get yield values from continue terminator, excluding the IV
    yield_values = IRValue[]
    if then_blk.terminator isa ContinueOp
        for (j, v) in enumerate(then_blk.terminator.values)
            j != iv_idx && push!(yield_values, v)
        end
    end

    # Add outer captures unchanged
    for j in (length(other_result_vars)+1):length(new_body.args)
        push!(yield_values, new_body.args[j])
    end

    new_body.terminator = ContinueOp(yield_values)

    # Modify the loop in-place to become :for
    loop.head = :for
    loop.regions = Dict{Symbol,Any}(:body => new_body)
    loop.init_values = other_init_values
    loop.operands = (lower=lower_bound, upper=upper_bound, step=step, iv_arg=iv_arg)
    # Update result_vars in context
    set_result_vars!(ctx, loop, other_result_vars)

    return true
end

"""
    try_upgrade_to_while!(loop::PartialControlFlowOp, ctx::StructurizationContext) -> Bool

Try to upgrade a :loop op to :while by detecting the while-loop pattern.
Modifies the op in-place. Returns true if upgraded.

Creates MLIR-style scf.while with before/after regions:
- before: condition computation, ends with ConditionOp
- after: loop body, ends with YieldOp
"""
function try_upgrade_to_while!(loop::PartialControlFlowOp, ctx::StructurizationContext)
    @assert loop.head == :loop
    body = loop.regions[:body]::Block
    result_vars = get_result_vars(ctx, loop)

    # Find the :if op in the loop body - its condition is the while condition
    condition_ifop = find_ifop(body)
    condition_ifop === nothing && return false

    then_blk = condition_ifop.regions[:then]::Block
    else_blk = condition_ifop.regions[:else]::Block

    # Build "before" region: statements before the :if + ConditionOp
    before = Block()
    before.args = copy(body.args)

    for (idx, item) in body.body
        if item isa PartialControlFlowOp
            if item === condition_ifop
                # Stop before :if - the condition becomes ConditionOp
                break
            else
                push_op!(before, idx, item)
            end
        else  # Statement
            push_stmt!(before, idx, item, body.types[idx])
        end
    end

    # Get break values (become results when condition is false)
    condition_args = IRValue[]
    if else_blk.terminator isa BreakOp
        condition_args = copy(else_blk.terminator.values)
        # Add outer captures unchanged
        for j in (length(result_vars)+1):length(before.args)
            push!(condition_args, before.args[j])
        end
    elseif !isempty(before.args)
        condition_args = [arg for arg in before.args]
    end

    cond_val = condition_ifop.operands.condition
    before.terminator = ConditionOp(cond_val, condition_args)

    # Build "after" region: statements from the then_block + YieldOp
    after = Block()
    for (i, arg) in enumerate(before.args)
        push!(after.args, BlockArg(i, arg.type))
    end

    for (idx, item) in then_blk.body
        if item isa PartialControlFlowOp
            push_op!(after, idx, item)
        else  # Statement
            push_stmt!(after, idx, item, then_blk.types[idx])
        end
    end

    # Get yield values from the continue terminator
    yield_values = IRValue[]
    if then_blk.terminator isa ContinueOp
        yield_values = copy(then_blk.terminator.values)
    end

    # Add outer captures unchanged
    for j in (length(result_vars)+1):length(after.args)
        push!(yield_values, after.args[j])
    end

    after.terminator = YieldOp(yield_values)

    # Modify the loop in-place to become :while
    loop.head = :while
    loop.regions = Dict{Symbol,Any}(:before => before, :after => after)
    # init_values and result_vars stay the same (result_vars already in context)

    return true
end
