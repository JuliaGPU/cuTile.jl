# Phase 3: Pattern matching and loop upgrades
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
    for stmt in statements(block.body)
        if stmt isa IfOp
            return stmt
        end
    end
    return nothing
end

"""
    find_expr_by_ssa(block::Block, ssa::SSAValue) -> Union{Tuple{Int, SSAEntry}, Nothing}

Find an expression in the block whose SSA index matches the SSAValue's id.
Returns (idx, entry) tuple or nothing.
"""
function find_expr_by_ssa(block::Block, ssa::SSAValue)
    for (idx, entry) in block.body
        if !(entry.stmt isa ControlFlowOp) && idx == ssa.id
            return (idx, entry)
        end
    end
    return nothing
end

"""
    find_add_int_for_iv(block::Block, iv_arg::BlockArg) -> Union{Tuple{Int, SSAEntry}, Nothing}

Find an expression containing `add_int(iv_arg, step)` in the block.
Searches inside IfOp (since condition creates if structure),
but NOT into nested LoopOp (those have their own IVs).
Returns (idx, entry) tuple or nothing.
"""
function find_add_int_for_iv(block::Block, iv_arg::BlockArg)
    for (idx, entry) in block.body
        if entry.stmt isa IfOp
            # Search in IfOp blocks (condition structure)
            result = find_add_int_for_iv(entry.stmt.then_region, iv_arg)
            result !== nothing && return result
            result = find_add_int_for_iv(entry.stmt.else_region, iv_arg)
            result !== nothing && return result
            # Don't recurse into LoopOp - nested loops have their own IVs
        elseif !(entry.stmt isa ControlFlowOp)  # Statement
            expr = entry.stmt
            if expr isa Expr && expr.head === :call && length(expr.args) >= 3
                func = expr.args[1]
                if func isa GlobalRef && func.name === :add_int
                    if expr.args[2] == iv_arg
                        return (idx, entry)
                    end
                end
            end
        end
    end
    return nothing
end

"""
    is_loop_invariant(val, block::Block, n_iter_args::Int) -> Bool

Check if a value is loop-invariant (not defined inside the loop body).
- BlockArgs (all of which are iter_args) are loop-variant (carries)
- SSAValues are loop-invariant (outer scope references)
- Constants and Arguments are always loop-invariant
"""
function is_loop_invariant(val, block::Block, n_iter_args::Int)
    # BlockArgs are all iter_args (carries) - loop-variant
    if val isa BlockArg
        return false  # All BlockArgs are carries now
    end

    # SSAValues reference outer scope (loop-invariant) or local body defs (loop-variant)
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
    for (idx, entry) in block.body
        if entry.stmt isa ControlFlowOp
            defines_in_op(entry.stmt, ssa) && return true
        else  # Statement
            if idx == ssa.id
                return true
            end
        end
    end
    return false
end

# Helper to check if an SSA is defined in a control flow op's regions
defines_in_op(op::IfOp, ssa::SSAValue) = defines(op.then_region, ssa) || defines(op.else_region, ssa)
defines_in_op(op::LoopOp, ssa::SSAValue) = defines(op.body, ssa)
defines_in_op(op::ForOp, ssa::SSAValue) = defines(op.body, ssa)
defines_in_op(op::WhileOp, ssa::SSAValue) = defines(op.before, ssa) || defines(op.after, ssa)

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
    apply_loop_patterns!(block::Block, ctx::StructurizationContext)

Upgrade LoopOp to ForOp/WhileOp where patterns match.
Creates new ops and replaces them in the block body.
When upgrading to ForOp, re-keys the op if the IV was the first result.
"""
function apply_loop_patterns!(block::Block, ctx::StructurizationContext)
    # Collect replacements: (old_idx => (new_op, new_key))
    replacements = Dict{Int, Tuple{ControlFlowOp, Int}}()

    for (idx, entry) in block.body
        if entry.stmt isa LoopOp
            result = try_upgrade_loop(entry.stmt, ctx, idx)
            if result !== nothing
                new_op, new_key = result
                replacements[idx] = (new_op, new_key)
            end
        end
    end

    # Apply replacements and recurse
    if !isempty(replacements)
        new_body = SSAVector()
        for (old_key, entry) in block.body
            if haskey(replacements, old_key)
                new_op, new_key = replacements[old_key]
                push!(new_body, (new_key, new_op, entry.typ))
            else
                push!(new_body, (old_key, entry.stmt, entry.typ))
            end
        end
        block.body = new_body
    end

    # Recurse into all control flow ops (including newly created ones)
    for stmt in statements(block.body)
        if stmt isa LoopOp
            apply_loop_patterns!(stmt.body, ctx)
        elseif stmt isa IfOp
            apply_loop_patterns!(stmt.then_region, ctx)
            apply_loop_patterns!(stmt.else_region, ctx)
        elseif stmt isa WhileOp
            apply_loop_patterns!(stmt.before, ctx)
            apply_loop_patterns!(stmt.after, ctx)
        elseif stmt isa ForOp
            apply_loop_patterns!(stmt.body, ctx)
        end
    end
end

"""
    try_upgrade_loop(loop::LoopOp, ctx::StructurizationContext, current_key::Int) -> Union{Tuple{ControlFlowOp, Int}, Nothing}

Try to upgrade a LoopOp to ForOp or WhileOp.
Returns (new_op, new_key) if upgraded, or nothing if not upgraded.
"""
function try_upgrade_loop(loop::LoopOp, ctx::StructurizationContext, current_key::Int)
    # Try ForOp pattern first
    result = try_upgrade_to_for(loop, ctx, current_key)
    if result !== nothing
        return result
    end

    # Try WhileOp pattern
    while_op = try_upgrade_to_while(loop, ctx)
    if while_op !== nothing
        return (while_op, current_key)  # WhileOp doesn't change keying
    end

    return nothing
end

"""
    try_upgrade_to_for(loop::LoopOp, ctx::StructurizationContext, current_key::Int) -> Union{Tuple{ForOp, Int}, Nothing}

Try to upgrade a LoopOp to ForOp by detecting the for-loop pattern.
Returns (ForOp, new_key) if upgraded, or nothing if not upgraded.
The new key is the first non-IV result's SSA index (needed for correct result storage in codegen).
"""
function try_upgrade_to_for(loop::LoopOp, ctx::StructurizationContext, current_key::Int)
    body = loop.body::Block
    n_iter_args = length(loop.iter_args)

    # Get original result SSA indices before modifying (needed for re-keying)
    original_result_indices = derive_result_vars(loop)

    # Find the IfOp in the loop body - this contains the condition check
    condition_ifop = find_ifop(body)
    condition_ifop === nothing && return nothing

    # The condition should be an SSAValue pointing to a comparison expression
    cond_val = condition_ifop.condition
    cond_val isa SSAValue || return nothing
    cond_result = find_expr_by_ssa(body, cond_val)
    cond_result === nothing && return nothing
    cond_idx, cond_entry = cond_result
    cond_expr = cond_entry.stmt

    # Check it's a for-loop condition: slt_int(iv_arg, upper_bound)
    is_for_condition(cond_expr) || return nothing

    # After substitution, the IV should be a BlockArg
    iv_arg = cond_expr.args[2]
    iv_arg isa BlockArg || return nothing
    upper_bound_raw = cond_expr.args[3]

    # Helper to resolve BlockArg to original value from iter_args
    function resolve_blockarg(arg)
        if arg isa BlockArg && arg.id <= n_iter_args
            return loop.iter_args[arg.id]
        end
        return arg
    end

    upper_bound = resolve_blockarg(upper_bound_raw)

    # Find which index this BlockArg corresponds to
    iv_idx = findfirst(==(iv_arg), body.args)
    iv_idx === nothing && return nothing

    # IV must be an iter_arg (in the iter_args range)
    iv_idx > n_iter_args && return nothing
    lower_bound = loop.iter_args[iv_idx]

    # Find the step: add_int(iv_arg, step)
    step_result = find_add_int_for_iv(body, iv_arg)
    step_result === nothing && return nothing
    step_idx, step_entry = step_result
    step_expr = step_entry.stmt
    step_raw = step_expr.args[3]
    step = resolve_blockarg(step_raw)

    # Verify upper_bound and step are loop-invariant
    is_loop_invariant(upper_bound, body, n_iter_args) || return nothing
    is_loop_invariant(step, body, n_iter_args) || return nothing

    # Separate non-IV iter_args (the new iter_args for ForOp)
    other_iter_args = IRValue[]
    for (j, v) in enumerate(loop.iter_args)
        j != iv_idx && push!(other_iter_args, v)
    end

    # Compute the new key: first non-IV result's SSA index
    # If no non-IV results, keep the current key
    new_key = current_key
    for (j, rv) in enumerate(original_result_indices)
        if j != iv_idx
            new_key = rv.id
            break
        end
    end

    # Rebuild body block without condition structure
    then_blk = condition_ifop.then_region::Block
    new_body = Block()
    # Only include carried values, not IV
    new_body.args = [arg for arg in body.args if arg !== iv_arg]

    # Extract body items, filtering out iv-related ones
    for (idx, entry) in body.body
        if entry.stmt isa IfOp && entry.stmt === condition_ifop
            # Extract the continue path's body (skip condition check structure)
            for (sub_idx, sub_entry) in then_blk.body
                sub_idx == step_idx && continue
                push!(new_body, sub_idx, sub_entry.stmt, sub_entry.typ)
            end
        elseif entry.stmt isa ControlFlowOp
            push!(new_body, idx, entry.stmt, entry.typ)
        else  # Statement
            idx == step_idx && continue
            idx == cond_idx && continue
            push!(new_body, idx, entry.stmt, entry.typ)
        end
    end

    # Get yield values from continue terminator, excluding the IV
    yield_values = IRValue[]
    if then_blk.terminator isa ContinueOp
        for (j, v) in enumerate(then_blk.terminator.values)
            # Only include non-IV values
            if j != iv_idx && j <= n_iter_args
                push!(yield_values, v)
            end
        end
    end

    new_body.terminator = ContinueOp(yield_values)

    # Create ForOp
    for_op = ForOp(lower_bound, upper_bound, step, iv_arg, new_body, other_iter_args)

    return (for_op, new_key)
end

"""
    try_upgrade_to_while(loop::LoopOp, ctx::StructurizationContext) -> Union{WhileOp, Nothing}

Try to upgrade a LoopOp to WhileOp by detecting the while-loop pattern.
Returns WhileOp if upgraded, or nothing if not upgraded.

Creates MLIR-style scf.while with before/after regions:
- before: condition computation, ends with ConditionOp (only passes iter_args)
- after: loop body, ends with YieldOp (only yields iter_args)
"""
function try_upgrade_to_while(loop::LoopOp, ctx::StructurizationContext)
    body = loop.body::Block
    n_iter_args = length(loop.iter_args)

    # Find the IfOp in the loop body - its condition is the while condition
    condition_ifop = find_ifop(body)
    condition_ifop === nothing && return nothing

    then_blk = condition_ifop.then_region::Block
    else_blk = condition_ifop.else_region::Block

    # Build "before" region: statements before the IfOp + ConditionOp
    before = Block()
    before.args = copy(body.args)

    for (idx, entry) in body.body
        if entry.stmt isa IfOp && entry.stmt === condition_ifop
            # Stop before IfOp - the condition becomes ConditionOp
            break
        elseif entry.stmt isa ControlFlowOp
            push!(before, idx, entry.stmt, entry.typ)
        else  # Statement
            push!(before, idx, entry.stmt, entry.typ)
        end
    end

    # ConditionOp args: iter_args (carries)
    condition_args = IRValue[before.args[i] for i in 1:n_iter_args]

    cond_val = condition_ifop.condition
    before.terminator = ConditionOp(cond_val, condition_args)

    # Build "after" region: statements from the then_block + YieldOp
    # After region has the same block args as before region
    after = Block()
    for (i, arg) in enumerate(before.args)
        push!(after.args, BlockArg(i, arg.type))
    end

    for (idx, entry) in then_blk.body
        push!(after, idx, entry.stmt, entry.typ)
    end

    # Get yield values from the continue terminator (iter_args / carries)
    yield_values = IRValue[]
    if then_blk.terminator isa ContinueOp
        for (j, v) in enumerate(then_blk.terminator.values)
            if j <= n_iter_args
                push!(yield_values, v)
            end
        end
    end

    after.terminator = YieldOp(yield_values)

    # Create WhileOp
    return WhileOp(before, after, loop.iter_args)
end
