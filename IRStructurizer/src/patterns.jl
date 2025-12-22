# Phase 3: Pattern matching and loop upgrades
#
# This file contains functions for upgrading :loop to :for/:while.
# All pattern matching operates on the structured IR after substitutions.

#=============================================================================
 Helper Functions (Pattern matching on structured IR)
=============================================================================#

"""
    find_ifop(block::PartialBlock) -> Union{PartialControlFlowOp, Nothing}

Find the first :if op in a block's body.
"""
function find_ifop(block::PartialBlock)
    for item in block.body
        if item isa PartialControlFlowOp && item.head == :if
            return item
        end
    end
    return nothing
end

"""
    find_statement_by_ssa(block::PartialBlock, ssa::SSAValue) -> Union{Statement, Nothing}

Find a Statement in the block whose idx matches the SSAValue's id.
"""
function find_statement_by_ssa(block::PartialBlock, ssa::SSAValue)
    for item in block.body
        if item isa Statement && item.idx == ssa.id
            return item
        end
    end
    return nothing
end

"""
    find_add_int_for_iv(block::PartialBlock, iv_arg::BlockArg) -> Union{Statement, Nothing}

Find a Statement containing `add_int(iv_arg, step)` in the block.
Searches inside :if ops (since condition creates :if structure),
but NOT into nested :loop ops (those have their own IVs).
"""
function find_add_int_for_iv(block::PartialBlock, iv_arg::BlockArg)
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
        elseif item isa PartialControlFlowOp && item.head == :if
            # Search in :if blocks (condition structure)
            then_blk = item.regions[:then]::PartialBlock
            else_blk = item.regions[:else]::PartialBlock
            result = find_add_int_for_iv(then_blk, iv_arg)
            result !== nothing && return result
            result = find_add_int_for_iv(else_blk, iv_arg)
            result !== nothing && return result
        end
        # Don't recurse into :loop - nested loops have their own IVs
    end
    return nothing
end

"""
    is_loop_invariant(val, block::PartialBlock; result_vars::Vector{SSAValue}=SSAValue[]) -> Bool

Check if a value is loop-invariant (not defined inside the loop body).
- BlockArgs for phi values (indices 1..len(result_vars)) are loop-variant
- BlockArgs for outer captures (indices > len(result_vars)) are loop-invariant
- SSAValues are loop-invariant if no Statement in the body defines them
- Constants and Arguments are always loop-invariant
"""
function is_loop_invariant(val, block::PartialBlock; result_vars::Vector{SSAValue}=SSAValue[])
    # BlockArgs: phi values are variant, outer captures are invariant
    if val isa BlockArg
        # If no result_vars provided, conservatively treat all BlockArgs as variant
        isempty(result_vars) && return false
        # BlockArgs beyond result_vars count are outer captures (invariant)
        return val.id > length(result_vars)
    end

    # SSAValues: check if defined in the loop body (including nested blocks)
    if val isa SSAValue
        return !defines(block, val)
    end

    # Constants, Arguments, etc. are invariant
    return true
end

"""
    defines(block::PartialBlock, ssa::SSAValue) -> Bool

Check if a block defines an SSA value (i.e., contains a Statement that produces it).
Searches nested blocks recursively.
"""
function defines(block::PartialBlock, ssa::SSAValue)
    for item in block.body
        if item isa Statement && item.idx == ssa.id
            return true
        elseif item isa PartialControlFlowOp
            for (_, region) in item.regions
                defines(region, ssa) && return true
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
    apply_loop_patterns!(block::PartialBlock)

Upgrade :loop ops to :for/:while where patterns match.
Modifies ops in-place by changing their head and operands.
"""
function apply_loop_patterns!(block::PartialBlock)
    for (i, item) in enumerate(block.body)
        if item isa PartialControlFlowOp
            if item.head == :loop
                if try_upgrade_loop!(item)
                    # Successfully upgraded, recurse into the modified op's regions
                    for (_, region) in item.regions
                        apply_loop_patterns!(region)
                    end
                else
                    # Not upgraded, recurse into body
                    apply_loop_patterns!(item.regions[:body])
                end
            elseif item.head == :if
                apply_loop_patterns!(item.regions[:then])
                apply_loop_patterns!(item.regions[:else])
            elseif item.head == :while
                apply_loop_patterns!(item.regions[:before])
                apply_loop_patterns!(item.regions[:after])
            elseif item.head == :for
                apply_loop_patterns!(item.regions[:body])
            end
        end
    end
end

"""
    try_upgrade_loop!(loop::PartialControlFlowOp) -> Bool

Try to upgrade a :loop op to :for or :while by modifying it in-place.
Returns true if upgraded, false otherwise.
"""
function try_upgrade_loop!(loop::PartialControlFlowOp)
    @assert loop.head == :loop

    # Try ForOp pattern first
    if try_upgrade_to_for!(loop)
        return true
    end

    # Try WhileOp pattern
    if try_upgrade_to_while!(loop)
        return true
    end

    return false
end

"""
    try_upgrade_to_for!(loop::PartialControlFlowOp) -> Bool

Try to upgrade a :loop op to :for by detecting the for-loop pattern.
Modifies the op in-place. Returns true if upgraded.
"""
function try_upgrade_to_for!(loop::PartialControlFlowOp)
    @assert loop.head == :loop
    body = loop.regions[:body]::PartialBlock

    # Find the :if op in the loop body - this contains the condition check
    condition_ifop = find_ifop(body)
    condition_ifop === nothing && return false

    # The condition should be an SSAValue pointing to a comparison Statement
    cond_val = condition_ifop.operands.condition
    cond_val isa SSAValue || return false
    cond_stmt = find_statement_by_ssa(body, cond_val)
    cond_stmt === nothing && return false

    # Check it's a for-loop condition: slt_int(iv_arg, upper_bound)
    is_for_condition(cond_stmt.expr) || return false

    # After substitution, the IV should be a BlockArg
    iv_arg = cond_stmt.expr.args[2]
    iv_arg isa BlockArg || return false
    upper_bound_raw = cond_stmt.expr.args[3]

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
    iv_idx > length(loop.result_vars) && return false
    lower_bound = loop.init_values[iv_idx]

    # Find the step: add_int(iv_arg, step)
    step_stmt = find_add_int_for_iv(body, iv_arg)
    step_stmt === nothing && return false
    step_raw = step_stmt.expr.args[3]

    # If step is a BlockArg (from outer capture), resolve to original SSAValue
    step = if step_raw isa BlockArg && step_raw.id <= length(loop.init_values)
        loop.init_values[step_raw.id]
    else
        step_raw
    end

    # Verify upper_bound and step are loop-invariant
    is_loop_invariant(upper_bound, body; result_vars=loop.result_vars) || return false
    is_loop_invariant(step, body; result_vars=loop.result_vars) || return false

    # Separate non-IV carried values (from result_vars)
    other_result_vars = SSAValue[]
    other_init_values = IRValue[]
    for (j, rv) in enumerate(loop.result_vars)
        if j != iv_idx && j <= length(loop.init_values)
            push!(other_result_vars, rv)
            push!(other_init_values, loop.init_values[j])
        end
    end

    # Add outer captures (init_values beyond result_vars)
    for j in (length(loop.result_vars)+1):length(loop.init_values)
        push!(other_init_values, loop.init_values[j])
    end

    # Rebuild body block without condition structure
    then_blk = condition_ifop.regions[:then]::PartialBlock
    new_body = PartialBlock()
    # Only include carried values, not IV
    new_body.args = [arg for arg in body.args if arg !== iv_arg]

    # Extract body statements, filtering out iv-related ones
    for item in body.body
        if item isa Statement
            item === step_stmt && continue
            item === cond_stmt && continue
            push!(new_body.body, item)
        elseif item isa PartialControlFlowOp && item.head == :if && item === condition_ifop
            # Extract the continue path's body (skip condition check structure)
            for sub_item in then_blk.body
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
    loop.result_vars = other_result_vars

    return true
end

"""
    try_upgrade_to_while!(loop::PartialControlFlowOp) -> Bool

Try to upgrade a :loop op to :while by detecting the while-loop pattern.
Modifies the op in-place. Returns true if upgraded.

Creates MLIR-style scf.while with before/after regions:
- before: condition computation, ends with ConditionOp
- after: loop body, ends with YieldOp
"""
function try_upgrade_to_while!(loop::PartialControlFlowOp)
    @assert loop.head == :loop
    body = loop.regions[:body]::PartialBlock

    # Find the :if op in the loop body - its condition is the while condition
    condition_ifop = find_ifop(body)
    condition_ifop === nothing && return false

    then_blk = condition_ifop.regions[:then]::PartialBlock
    else_blk = condition_ifop.regions[:else]::PartialBlock

    # Build "before" region: statements before the :if + ConditionOp
    before = PartialBlock()
    before.args = copy(body.args)

    for item in body.body
        if item isa Statement
            push!(before.body, item)
        elseif item isa PartialControlFlowOp && item === condition_ifop
            # Stop before :if - the condition becomes ConditionOp
            break
        else
            push!(before.body, item)
        end
    end

    # Get break values (become results when condition is false)
    condition_args = IRValue[]
    if else_blk.terminator isa BreakOp
        condition_args = copy(else_blk.terminator.values)
        # Add outer captures unchanged
        for j in (length(loop.result_vars)+1):length(before.args)
            push!(condition_args, before.args[j])
        end
    elseif !isempty(before.args)
        condition_args = [arg for arg in before.args]
    end

    cond_val = condition_ifop.operands.condition
    before.terminator = ConditionOp(cond_val, condition_args)

    # Build "after" region: statements from the then_block + YieldOp
    after = PartialBlock()
    for (i, arg) in enumerate(before.args)
        push!(after.args, BlockArg(i, arg.type))
    end

    for item in then_blk.body
        push!(after.body, item)
    end

    # Get yield values from the continue terminator
    yield_values = IRValue[]
    if then_blk.terminator isa ContinueOp
        yield_values = copy(then_blk.terminator.values)
    end

    # Add outer captures unchanged
    for j in (length(loop.result_vars)+1):length(after.args)
        push!(yield_values, after.args[j])
    end

    after.terminator = YieldOp(yield_values)

    # Modify the loop in-place to become :while
    loop.head = :while
    loop.regions = Dict{Symbol,Any}(:before => before, :after => after)
    # init_values and result_vars stay the same

    return true
end
