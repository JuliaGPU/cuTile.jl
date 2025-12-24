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
    for stmt in statements(block.body)
        if stmt isa PartialControlFlowOp && stmt.head == :if
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
        if !(entry.stmt isa PartialControlFlowOp) && idx == ssa.id
            return (idx, entry)
        end
    end
    return nothing
end

"""
    find_add_int_for_iv(block::Block, iv_arg::BlockArg) -> Union{Tuple{Int, SSAEntry}, Nothing}

Find an expression containing `add_int(iv_arg, step)` in the block.
Searches inside :if ops (since condition creates :if structure),
but NOT into nested :loop ops (those have their own IVs).
Returns (idx, entry) tuple or nothing.
"""
function find_add_int_for_iv(block::Block, iv_arg::BlockArg)
    for (idx, entry) in block.body
        if entry.stmt isa PartialControlFlowOp
            if entry.stmt.head == :if
                # Search in :if blocks (condition structure)
                then_blk = entry.stmt.regions[:then]::Block
                else_blk = entry.stmt.regions[:else]::Block
                result = find_add_int_for_iv(then_blk, iv_arg)
                result !== nothing && return result
                result = find_add_int_for_iv(else_blk, iv_arg)
                result !== nothing && return result
            end
            # Don't recurse into :loop - nested loops have their own IVs
        else  # Statement
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
        if entry.stmt isa PartialControlFlowOp
            for (_, region) in entry.stmt.regions
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
When upgrading to :for, re-keys the op if the IV was the first result.
"""
function apply_loop_patterns!(block::Block, ctx::StructurizationContext)
    # Collect all re-keying needed (old_key => new_key)
    rekey_map = Dict{Int,Int}()

    for (idx, entry) in block.body
        if entry.stmt isa PartialControlFlowOp
            if entry.stmt.head == :loop
                new_key = try_upgrade_loop!(entry.stmt, ctx, idx)
                if new_key !== nothing
                    # Successfully upgraded - record re-keying if needed
                    if new_key != idx
                        rekey_map[idx] = new_key
                    end
                    # Recurse into the modified op's regions
                    for (_, region) in entry.stmt.regions
                        apply_loop_patterns!(region, ctx)
                    end
                else
                    # Not upgraded, recurse into body
                    apply_loop_patterns!(entry.stmt.regions[:body], ctx)
                end
            elseif entry.stmt.head == :if
                apply_loop_patterns!(entry.stmt.regions[:then], ctx)
                apply_loop_patterns!(entry.stmt.regions[:else], ctx)
            elseif entry.stmt.head == :while
                apply_loop_patterns!(entry.stmt.regions[:before], ctx)
                apply_loop_patterns!(entry.stmt.regions[:after], ctx)
            elseif entry.stmt.head == :for
                apply_loop_patterns!(entry.stmt.regions[:body], ctx)
            end
        end
    end

    # Apply re-keying by rebuilding SSAVector
    if !isempty(rekey_map)
        new_body = SSAVector()
        for (old_key, entry) in block.body
            new_key = get(rekey_map, old_key, old_key)
            push!(new_body, (new_key, entry.stmt, entry.typ))
        end
        block.body = new_body
    end
end

"""
    try_upgrade_loop!(loop::PartialControlFlowOp, ctx::StructurizationContext, current_key::Int) -> Union{Int, Nothing}

Try to upgrade a :loop op to :for or :while by modifying it in-place.
Returns the new key if upgraded (may be same as current_key), or nothing if not upgraded.
"""
function try_upgrade_loop!(loop::PartialControlFlowOp, ctx::StructurizationContext, current_key::Int)
    @assert loop.head == :loop

    # Try ForOp pattern first
    new_key = try_upgrade_to_for!(loop, ctx, current_key)
    if new_key !== nothing
        return new_key
    end

    # Try WhileOp pattern
    if try_upgrade_to_while!(loop, ctx)
        return current_key  # WhileOp doesn't change keying
    end

    return nothing
end

"""
    try_upgrade_to_for!(loop::PartialControlFlowOp, ctx::StructurizationContext, current_key::Int) -> Union{Int, Nothing}

Try to upgrade a :loop op to :for by detecting the for-loop pattern.
Modifies the op in-place. Returns the new key if upgraded, or nothing if not upgraded.
The new key is the first non-IV result's SSA index (needed for correct result storage in codegen).
"""
function try_upgrade_to_for!(loop::PartialControlFlowOp, ctx::StructurizationContext, current_key::Int)
    @assert loop.head == :loop
    body = loop.regions[:body]::Block
    n_iter_args = length(loop.iter_args)

    # Get original result SSA indices before modifying (needed for re-keying)
    original_result_indices = derive_result_vars(loop)

    # Find the :if op in the loop body - this contains the condition check
    condition_ifop = find_ifop(body)
    condition_ifop === nothing && return nothing

    # The condition should be an SSAValue pointing to a comparison expression
    cond_val = condition_ifop.operands.condition
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

    # Separate non-IV iter_args (the new iter_args for :for)
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
    then_blk = condition_ifop.regions[:then]::Block
    new_body = Block()
    # Only include carried values, not IV
    new_body.args = [arg for arg in body.args if arg !== iv_arg]

    # Extract body items, filtering out iv-related ones
    for (idx, entry) in body.body
        if entry.stmt isa PartialControlFlowOp
            if entry.stmt.head == :if && entry.stmt === condition_ifop
                # Extract the continue path's body (skip condition check structure)
                for (sub_idx, sub_entry) in then_blk.body
                    sub_idx == step_idx && continue
                    push!(new_body, sub_idx, sub_entry.stmt, sub_entry.typ)
                end
            else
                push!(new_body, idx, entry.stmt, entry.typ)
            end
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

    # Modify the loop in-place to become :for
    loop.head = :for
    loop.regions = Dict{Symbol,Any}(:body => new_body)
    loop.iter_args = other_iter_args
    loop.operands = (lower=lower_bound, upper=upper_bound, step=step, iv_arg=iv_arg)

    return new_key
end

"""
    try_upgrade_to_while!(loop::PartialControlFlowOp, ctx::StructurizationContext) -> Bool

Try to upgrade a :loop op to :while by detecting the while-loop pattern.
Modifies the op in-place. Returns true if upgraded.

Creates MLIR-style scf.while with before/after regions:
- before: condition computation, ends with ConditionOp (only passes iter_args)
- after: loop body, ends with YieldOp (only yields iter_args)
"""
function try_upgrade_to_while!(loop::PartialControlFlowOp, ctx::StructurizationContext)
    @assert loop.head == :loop
    body = loop.regions[:body]::Block
    n_iter_args = length(loop.iter_args)

    # Find the :if op in the loop body - its condition is the while condition
    condition_ifop = find_ifop(body)
    condition_ifop === nothing && return false

    then_blk = condition_ifop.regions[:then]::Block
    else_blk = condition_ifop.regions[:else]::Block

    # Build "before" region: statements before the :if + ConditionOp
    before = Block()
    before.args = copy(body.args)

    for (idx, entry) in body.body
        if entry.stmt isa PartialControlFlowOp
            if entry.stmt === condition_ifop
                # Stop before :if - the condition becomes ConditionOp
                break
            else
                push!(before, idx, entry.stmt, entry.typ)
            end
        else  # Statement
            push!(before, idx, entry.stmt, entry.typ)
        end
    end

    # ConditionOp args: iter_args (carries)
    condition_args = IRValue[before.args[i] for i in 1:n_iter_args]

    cond_val = condition_ifop.operands.condition
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

    # Modify the loop in-place to become :while
    loop.head = :while
    loop.regions = Dict{Symbol,Any}(:before => before, :after => after)

    return true
end
