# Pattern matching and loop upgrades (LoopOp → ForOp/WhileOp)

#=============================================================================
 Getfield Helper Functions (for tuple+getfield loop result handling)
=============================================================================#

"""
    find_getfields_for(block::Block, loop_ssa::SSAValue) -> Vector{NamedTuple}

Find all getfield expressions in the block that reference the given loop SSA value.
Returns a vector of (ssa_idx, field_idx) named tuples.
"""
function find_getfields_for(block::Block, loop_ssa::SSAValue)
    getfields = NamedTuple{(:ssa_idx, :field_idx), Tuple{Int, Int}}[]
    for (idx, entry) in block.body
        expr = entry.stmt
        if expr isa Expr && expr.head === :call &&
           length(expr.args) >= 3 && expr.args[1] === Core.getfield &&
           expr.args[2] == loop_ssa
            push!(getfields, (ssa_idx=idx, field_idx=expr.args[3]))
        end
    end
    return getfields
end

"""
    remove_stmt!(block::Block, ssa_idx::Int)

Remove a statement from the block by its SSA index.
"""
function remove_stmt!(block::Block, ssa_idx::Int)
    new_body = SSAVector()
    for (idx, entry) in block.body
        idx == ssa_idx || push!(new_body, (idx, entry.stmt, entry.typ))
    end
    block.body = new_body
end

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
            result = find_add_int_for_iv(entry.stmt.then_region, iv_arg)
            result !== nothing && return result
            result = find_add_int_for_iv(entry.stmt.else_region, iv_arg)
            result !== nothing && return result
        elseif !(entry.stmt isa ControlFlowOp)
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
    is_loop_invariant(val, block::Block, n_init_values::Int) -> Bool

Check if a value is loop-invariant (not defined inside the loop body).
- BlockArgs (all of which are init_values) are loop-variant (carries)
- SSAValues are loop-invariant (outer scope references)
- Constants and Arguments are always loop-invariant
"""
function is_loop_invariant(val, block::Block, n_init_values::Int)
    if val isa BlockArg
        return false
    end

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
        elseif idx == ssa.id
            return true
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

For ForOp upgrades:
- The loop's key remains unchanged (no re-keying)
- The IV's getfield statement is removed from the parent block
- Remaining getfield indices are adjusted to account for the removed IV
"""
function apply_loop_patterns!(block::Block, ctx::StructurizationContext)
    # Collect upgrades: (loop_key => (new_op, iv_getfield_idx, iv_field_idx))
    # iv_getfield_idx: SSA index of the IV's getfield (to remove), or nothing for WhileOp
    # iv_field_idx: field index of the IV in the tuple (to adjust other getfields)
    upgrades = Dict{Int, Tuple{ControlFlowOp, Union{Nothing, Int}, Int}}()

    for (idx, entry) in block.body
        if entry.stmt isa LoopOp
            result = try_upgrade_loop(entry.stmt, ctx, idx, block)
            if result !== nothing
                upgrades[idx] = result
            end
        end
    end

    # Apply upgrades: replace LoopOp with ForOp/WhileOp (same key!)
    if !isempty(upgrades)
        new_body = SSAVector()
        for (old_key, entry) in block.body
            if haskey(upgrades, old_key)
                new_op, _, _ = upgrades[old_key]
                # Update result type: ForOp has one fewer result than LoopOp
                new_typ = compute_upgraded_type(entry.typ, upgrades[old_key])
                push!(new_body, (old_key, new_op, new_typ))
            else
                push!(new_body, (old_key, entry.stmt, entry.typ))
            end
        end
        block.body = new_body

        # Remove IV getfields and adjust remaining getfield indices
        for (loop_key, (new_op, iv_getfield_idx, iv_field_idx)) in upgrades
            if new_op isa ForOp && iv_getfield_idx !== nothing
                remove_stmt!(block, iv_getfield_idx)
                # Adjust remaining getfield indices for this loop
                adjust_getfield_indices!(block, loop_key, iv_field_idx)
            end
        end
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
    compute_upgraded_type(old_typ, upgrade_info) -> Type

Compute the new result type after ForOp upgrade (one fewer result due to IV removal).
Always returns a Tuple type (uniform handling in codegen).
"""
function compute_upgraded_type(@nospecialize(old_typ), upgrade_info)
    new_op, _, iv_field_idx = upgrade_info

    # WhileOp keeps all results
    !(new_op isa ForOp) && return old_typ

    # Must be a Tuple type (uniform loop result handling)
    @assert old_typ <: Tuple "Expected Tuple type for loop results, got $old_typ"

    # Remove the IV's type from the tuple
    params = collect(old_typ.parameters)
    deleteat!(params, iv_field_idx)

    # Always return Tuple (may be empty Tuple{})
    return Tuple{params...}
end

"""
    adjust_getfield_indices!(block::Block, loop_key::Int, removed_field_idx::Int)

Adjust getfield field indices after removing the IV's getfield.
All getfields referencing the loop with field index > removed_field_idx
have their field index decremented.
"""
function adjust_getfield_indices!(block::Block, loop_key::Int, removed_field_idx::Int)
    loop_ssa = SSAValue(loop_key)
    for (idx, entry) in block.body
        expr = entry.stmt
        if expr isa Expr && expr.head === :call &&
           length(expr.args) >= 3 && expr.args[1] === Core.getfield &&
           expr.args[2] == loop_ssa
            old_field = expr.args[3]
            if old_field > removed_field_idx
                expr.args[3] = old_field - 1
            end
        end
    end
end

"""
    try_upgrade_loop(loop::LoopOp, ctx::StructurizationContext, current_key::Int, parent_block::Block)
        -> Union{Tuple{ControlFlowOp, Union{Nothing, Int}, Int}, Nothing}

Try to upgrade a LoopOp to ForOp or WhileOp.
Returns (new_op, iv_getfield_idx, iv_field_idx) if upgraded, or nothing if not upgraded.
- new_op: The ForOp or WhileOp
- iv_getfield_idx: SSA index of the IV's getfield to remove (ForOp only, nothing for WhileOp)
- iv_field_idx: Field index of the IV in the tuple (for adjusting other getfields)
"""
function try_upgrade_loop(loop::LoopOp, ctx::StructurizationContext, current_key::Int, parent_block::Block)
    # Try ForOp pattern first
    result = try_upgrade_to_for(loop, ctx, current_key, parent_block)
    if result !== nothing
        return result
    end

    # Try WhileOp pattern
    while_op = try_upgrade_to_while(loop, ctx)
    if while_op !== nothing
        return (while_op, nothing, 0)  # WhileOp doesn't remove any getfields
    end

    return nothing
end

"""
    try_upgrade_to_for(loop::LoopOp, ctx::StructurizationContext, current_key::Int, parent_block::Block)
        -> Union{Tuple{ForOp, Union{Nothing, Int}, Int}, Nothing}

Try to upgrade a LoopOp to ForOp by detecting the for-loop pattern.
Returns (ForOp, iv_getfield_idx, iv_field_idx) if upgraded, or nothing if not upgraded.
- ForOp: The created ForOp
- iv_getfield_idx: SSA index of the IV's getfield statement to remove (or nothing if not found)
- iv_field_idx: Field index of the IV in the tuple (for adjusting other getfields)
"""
function try_upgrade_to_for(loop::LoopOp, ctx::StructurizationContext, current_key::Int, parent_block::Block)
    body = loop.body::Block
    n_init_values = length(loop.init_values)

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

    # Helper to resolve BlockArg to original value from init_values
    function resolve_blockarg(arg)
        if arg isa BlockArg && arg.id <= n_init_values
            return loop.init_values[arg.id]
        end
        return arg
    end

    upper_bound = resolve_blockarg(upper_bound_raw)

    # Find which index this BlockArg corresponds to
    iv_idx = findfirst(==(iv_arg), body.args)
    iv_idx === nothing && return nothing

    # IV must be an init_value (in the init_values range)
    iv_idx > n_init_values && return nothing
    lower_bound = loop.init_values[iv_idx]

    # Find the step: add_int(iv_arg, step)
    step_result = find_add_int_for_iv(body, iv_arg)
    step_result === nothing && return nothing
    step_idx, step_entry = step_result
    step_expr = step_entry.stmt
    step_raw = step_expr.args[3]
    step = resolve_blockarg(step_raw)

    # Verify upper_bound and step are loop-invariant
    is_loop_invariant(upper_bound, body, n_init_values) || return nothing
    is_loop_invariant(step, body, n_init_values) || return nothing

    # Separate non-IV init_values (the new init_values for ForOp)
    other_init_values = IRValue[]
    for (j, v) in enumerate(loop.init_values)
        j != iv_idx && push!(other_init_values, v)
    end

    # Find the IV's getfield in the parent block
    loop_ssa = SSAValue(current_key)
    getfields = find_getfields_for(parent_block, loop_ssa)
    iv_getfield_idx = nothing
    for gf in getfields
        if gf.field_idx == iv_idx
            iv_getfield_idx = gf.ssa_idx
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
        else
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
            if j != iv_idx && j <= n_init_values
                push!(yield_values, v)
            end
        end
    end

    new_body.terminator = ContinueOp(yield_values)

    # Create ForOp
    for_op = ForOp(lower_bound, upper_bound, step, iv_arg, new_body, other_init_values)

    return (for_op, iv_getfield_idx, iv_idx)
end

"""
    try_upgrade_to_while(loop::LoopOp, ctx::StructurizationContext) -> Union{WhileOp, Nothing}

Try to upgrade a LoopOp to WhileOp by detecting the while-loop pattern.
Returns WhileOp if upgraded, or nothing if not upgraded.

Creates MLIR-style scf.while with before/after regions:
- before: condition computation, ends with ConditionOp (only passes init_values)
- after: loop body, ends with YieldOp (only yields init_values)
"""
function try_upgrade_to_while(loop::LoopOp, ctx::StructurizationContext)
    body = loop.body::Block
    n_init_values = length(loop.init_values)

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
            break
        elseif entry.stmt isa ControlFlowOp
            push!(before, idx, entry.stmt, entry.typ)
        else
            push!(before, idx, entry.stmt, entry.typ)
        end
    end

    condition_args = IRValue[before.args[i] for i in 1:n_init_values]

    cond_val = condition_ifop.condition
    before.terminator = ConditionOp(cond_val, condition_args)

    after = Block()
    for (i, arg) in enumerate(before.args)
        push!(after.args, BlockArg(i, arg.type))
    end

    for (idx, entry) in then_blk.body
        push!(after, idx, entry.stmt, entry.typ)
    end

    yield_values = IRValue[]
    if then_blk.terminator isa ContinueOp
        for (j, v) in enumerate(then_blk.terminator.values)
            if j <= n_init_values
                push!(yield_values, v)
            end
        end
    end

    after.terminator = YieldOp(yield_values)

    return WhileOp(before, after, loop.init_values)
end
