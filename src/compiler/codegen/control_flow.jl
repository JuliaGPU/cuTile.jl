# Structured IR Emission

"""
    result_count(T) -> Int

Compute the number of results from a Block.types entry.
Block.types contains Julia types:
- For Statement: Julia type → 1 result
- For ControlFlowOp with 0 results: Nothing → 0 results
- For ControlFlowOp with 1 result: Julia type → 1 result
- For ControlFlowOp with N results: Tuple{T1, T2, ...} → N results
"""
function result_count(@nospecialize(T))
    T === Nothing && return 0
    T <: Tuple && return length(T.parameters)
    return 1
end

"""
    emit_block!(ctx, block::Block)

Emit bytecode for a structured IR block.
All SSA values use original Julia SSA indices (no local renumbering).
Values are stored in ctx.values by their original index.
"""
function emit_block!(ctx::CGCtx, block::Block; skip_terminator::Bool=false)
    # Emit body items (interleaved expressions and control flow ops)
    # SSAVector iteration yields (ssa_idx, entry) where entry has .stmt and .typ
    for (ssa_idx, entry) in block.body
        if entry.stmt isa ControlFlowOp
            n_results = result_count(entry.typ)
            emit_control_flow_op!(ctx, entry.stmt, entry.typ, n_results, ssa_idx)
        else
            emit_statement!(ctx, entry.stmt, ssa_idx, entry.typ)
        end
    end

    # Emit terminator (unless skipped)
    if !skip_terminator && block.terminator !== nothing
        emit_terminator!(ctx, block.terminator)
    end
end

"""
    emit_control_flow_op!(ctx, op::ControlFlowOp, result_type, n_results, original_idx)

Emit bytecode for a structured control flow operation.
Uses multiple dispatch on the concrete ControlFlowOp type.
Results are stored at indices assigned AFTER nested regions (DFS order).
original_idx is the original Julia SSA index for cross-block references.
"""
emit_control_flow_op!(ctx::CGCtx, op::IfOp, @nospecialize(result_type), n_results::Int, original_idx::Int) =
    emit_if_op!(ctx, op, result_type, n_results, original_idx)
emit_control_flow_op!(ctx::CGCtx, op::ForOp, @nospecialize(result_type), n_results::Int, original_idx::Int) =
    emit_for_op!(ctx, op, result_type, n_results, original_idx)
emit_control_flow_op!(ctx::CGCtx, op::WhileOp, @nospecialize(result_type), n_results::Int, original_idx::Int) =
    emit_while_op!(ctx, op, result_type, n_results, original_idx)
emit_control_flow_op!(ctx::CGCtx, op::LoopOp, @nospecialize(result_type), n_results::Int, original_idx::Int) =
    emit_loop_op!(ctx, op, result_type, n_results, original_idx)

function emit_if_op!(ctx::CGCtx, op::IfOp, @nospecialize(parent_result_type), n_results::Int, ssa_idx::Int)
    cb = ctx.cb
    then_blk = op.then_region
    else_blk = op.else_region

    # Get condition value
    cond_tv = emit_value!(ctx, op.condition)
    cond_tv === nothing && throw(IRError("Cannot resolve condition for IfOp"))

    # Determine result types from parent_result_type
    result_types = TypeId[]
    julia_result_types = Type[]
    if parent_result_type === Nothing
        # No results
    elseif parent_result_type <: Tuple
        for T in parent_result_type.parameters
            push!(result_types, tile_type_for_julia!(ctx, T))
            push!(julia_result_types, T)
        end
    else
        push!(result_types, tile_type_for_julia!(ctx, parent_result_type))
        push!(julia_result_types, parent_result_type)
    end
    n_user_results = length(result_types)
    # Add token type as additional result (for memory ordering)
    push!(result_types, ctx.token_type)

    # Save token before branches
    token_before = ctx.token

    # Emit IfOp with callback-based region building
    then_body = function(_)
        saved_block_args = copy(ctx.block_args)
        ctx.token = token_before  # Reset to pre-branch token
        emit_block!(ctx, then_blk)
        if then_blk.terminator === nothing
            encode_YieldOp!(ctx.cb, [ctx.token])
        end
        empty!(ctx.block_args)
        merge!(ctx.block_args, saved_block_args)
    end
    else_body = function(_)
        saved_block_args = copy(ctx.block_args)
        ctx.token = token_before  # Reset to pre-branch token
        emit_block!(ctx, else_blk)
        if else_blk.terminator === nothing
            encode_YieldOp!(ctx.cb, [ctx.token])
        end
        empty!(ctx.block_args)
        merge!(ctx.block_args, saved_block_args)
    end
    results = encode_IfOp!(then_body, else_body, cb, result_types, cond_tv.v)

    # Last result is the merged token from both branches
    ctx.token = results[end]

    # Store results at IfOp's SSA index (may be empty for void-returning ifs)
    ctx.values[ssa_idx] = CGVal(results[1:n_user_results], parent_result_type)
end

function emit_for_op!(ctx::CGCtx, op::ForOp, @nospecialize(parent_result_type), n_results::Int, ssa_idx::Int)
    cb = ctx.cb
    tt = ctx.tt
    body_blk = op.body

    # Get bounds values
    lower_tv = emit_value!(ctx, op.lower)
    upper_tv = emit_value!(ctx, op.upper)
    step_tv = emit_value!(ctx, op.step)
    iv_arg = op.iv_arg

    (lower_tv === nothing || upper_tv === nothing || step_tv === nothing) &&
        throw(IRError("Cannot resolve ForOp bounds"))

    # Assert all bounds have the same type
    lower_tv.jltype === upper_tv.jltype === step_tv.jltype ||
        throw(IRError("ForOp bounds must all have the same type: lower=$(lower_tv.jltype), upper=$(upper_tv.jltype), step=$(step_tv.jltype)"))
        iv_jl_type = lower_tv.jltype
        iv_type = tile_type_for_julia!(ctx, iv_jl_type)

    # Get init values (init_values are loop-carried values)
    init_values = Value[]
    for init_val in op.init_values
        tv = emit_value!(ctx, init_val)
        (tv === nothing || tv.v === nothing) && throw(IRError("Cannot resolve ForOp init value"))
        push!(init_values, tv.v)
    end
    # Add token as additional init value (for memory ordering)
    push!(init_values, ctx.token)

    # Number of carries (init_values) - these are the loop results
    n_carries = length(op.init_values)

    # Determine result types from carries (body.args)
    result_types = TypeId[]
    for i in 1:n_carries
        body_arg = body_blk.args[i]
        type_id = tile_type_for_julia!(ctx, body_arg.type)
        push!(result_types, type_id)
    end
    # Add token type as additional result (for memory ordering)
    push!(result_types, ctx.token_type)

    # Number of user result types (excluding token)
    n_user_results = n_carries

    # Emit ForOp with callback-based region building
    body_builder = function(block_args)
        saved_block_args = copy(ctx.block_args)

        # Tile IR block args layout: [iv, carries..., token]
        # Julia IR body.args layout: [carries...]

        # Map the induction variable
        iv_tv = CGVal(block_args[1], iv_type, iv_jl_type)
        ctx[iv_arg] = iv_tv

        # Map carried values (body.args)
        for i in 1:n_carries
            body_arg = body_blk.args[i]
            shape = extract_tile_shape(body_arg.type)
            tv = CGVal(block_args[i + 1], result_types[i], body_arg.type, shape)
            ctx[body_arg] = tv
        end

        # Set token from last block arg
        ctx.token = block_args[end]

        emit_block!(ctx, body_blk)

        empty!(ctx.block_args)
        merge!(ctx.block_args, saved_block_args)
    end
    results = encode_ForOp!(body_builder, cb, result_types, iv_type, lower_tv.v, upper_tv.v, step_tv.v, init_values)

    # Last result is the token
    ctx.token = results[end]

    # Store results at the loop's SSA index (may be empty for void-returning loops)
    ctx.values[ssa_idx] = CGVal(results[1:n_user_results], parent_result_type)
end

function emit_loop_op!(ctx::CGCtx, op::LoopOp, @nospecialize(parent_result_type), n_results::Int, ssa_idx::Int)
    cb = ctx.cb
    body_blk = op.body

    # Get init values (init_values are loop-carried values)
    init_values = Value[]
    for init_val in op.init_values
        tv = emit_value!(ctx, init_val)
        (tv === nothing || tv.v === nothing) && throw(IRError("Cannot resolve LoopOp init value"))
        push!(init_values, tv.v)
    end
    # Add token as additional init value (for memory ordering)
    push!(init_values, ctx.token)

    # Number of carries (init_values) - these are the loop results
    n_carries = length(op.init_values)

    # Determine result types from carries (body.args)
    result_types = TypeId[]
    for i in 1:n_carries
        body_arg = body_blk.args[i]
        type_id = tile_type_for_julia!(ctx, body_arg.type)
        push!(result_types, type_id)
    end
    # Add token type as additional result (for memory ordering)
    push!(result_types, ctx.token_type)

    # Number of user result types (excluding token)
    n_user_results = n_carries

    # Emit LoopOp with callback-based region building
    body_builder = function(block_args)
        saved_block_args = copy(ctx.block_args)

        # Tile IR block args layout: [carries..., token]
        # Julia IR body.args layout: [carries...]

        # Map carried values (body.args)
        for i in 1:n_carries
            body_arg = body_blk.args[i]
            shape = extract_tile_shape(body_arg.type)
            ctx[body_arg] = CGVal(block_args[i], result_types[i], body_arg.type, shape)
        end

        # Set token from last block arg
        ctx.token = block_args[end]

        emit_block!(ctx, body_blk)

        # In Tile IR, if the loop body ends with an IfOp (even one with continue/break
        # in all branches), the if is NOT a terminator. We need an explicit terminator
        # after the if. Add an unreachable ContinueOp as fallback terminator.
        if body_blk.terminator === nothing
            fallback_operands = copy(block_args)
            fallback_operands[end] = ctx.token
            encode_ContinueOp!(ctx.cb, fallback_operands)
        end

        empty!(ctx.block_args)
        merge!(ctx.block_args, saved_block_args)
    end
    results = encode_LoopOp!(body_builder, cb, result_types, init_values)

    # Last result is the token
    ctx.token = results[end]

    # Store results at the loop's SSA index (may be empty for void-returning loops)
    ctx.values[ssa_idx] = CGVal(results[1:n_user_results], parent_result_type)
end

function emit_while_op!(ctx::CGCtx, op::WhileOp, @nospecialize(parent_result_type), n_results::Int, ssa_idx::Int)
    cb = ctx.cb
    before_blk = op.before
    after_blk = op.after

    # Get init values (init_values are loop-carried values)
    init_values = Value[]
    for init_val in op.init_values
        tv = emit_value!(ctx, init_val)
        (tv === nothing || tv.v === nothing) && throw(IRError("Cannot resolve WhileOp init value: $init_val"))
        push!(init_values, tv.v)
    end
    # Add token as additional init value (for memory ordering)
    push!(init_values, ctx.token)

    # Number of carries (init_values) - these are the loop results
    n_carries = length(op.init_values)

    # Determine result types from carries (before.args)
    result_types = TypeId[]
    for i in 1:n_carries
        before_arg = before_blk.args[i]
        type_id = tile_type_for_julia!(ctx, before_arg.type)
        push!(result_types, type_id)
    end
    # Add token type as additional result (for memory ordering)
    push!(result_types, ctx.token_type)

    # Number of user result types (excluding token)
    n_user_results = n_carries

    # Emit WhileOp as cuda_tile.loop with conditional break pattern
    # MLIR structure: before { stmts; condition(cond) args } do { stmts; yield vals }
    # Emitted as: loop { before_stmts; if(!cond) { break } else { yield }; after_stmts; continue }
    # This structure keeps the "after" statements at LoopOp body level, avoiding
    # nested region issues when "after" contains loops.
    body_builder = function(block_args)
        saved_block_args = copy(ctx.block_args)

        # Tile IR block args layout: [carries..., token]
        # Julia IR before.args layout: [carries...]

        # Map carried values (before.args)
        for i in 1:n_carries
            before_arg = before_blk.args[i]
            shape = extract_tile_shape(before_arg.type)
            ctx[before_arg] = CGVal(block_args[i], result_types[i], before_arg.type, shape)
        end

        # Set token from last block arg
        ctx.token = block_args[end]

        # Emit "before" region
        emit_block!(ctx, before_blk)

        # Get condition from ConditionOp terminator
        cond_op = before_blk.terminator
        cond_op isa ConditionOp || throw(IRError("WhileOp before region must end with ConditionOp"))

        cond_tv = emit_value!(ctx, cond_op.condition)
        (cond_tv === nothing || cond_tv.v === nothing) && throw(IRError("Cannot resolve WhileOp condition: $(cond_op.condition)"))

        # Emit conditional break: if (cond) { yield } else { break }
        # This keeps nested loops in "after" at LoopOp body level
        then_body = function(_)
            # Just yield (empty) - control continues to after_stmts
            encode_YieldOp!(ctx.cb, Value[])
        end

        else_body = function(_)
            # Break with ConditionOp args (become loop results)
            break_operands = Value[]
            for arg in cond_op.args
                tv = emit_value!(ctx, arg)
                tv !== nothing && tv.v !== nothing && push!(break_operands, tv.v)
            end
            if isempty(break_operands)
                for i in 1:n_carries
                    push!(break_operands, block_args[i])
                end
            end
            push!(break_operands, ctx.token)
            encode_BreakOp!(ctx.cb, break_operands)
        end

        # Emit IfOp with NO results: if (cond) continue flow, else break
        if_result_types = TypeId[]
        encode_IfOp!(then_body, else_body, cb, if_result_types, cond_tv.v)

        # Now emit "after" region at LoopOp body level (not inside IfOp!)
        # Map "after" region block args - carries from ConditionOp.args
        for i in 1:n_carries
            after_arg = after_blk.args[i]
            if i <= length(cond_op.args)
                tv = emit_value!(ctx, cond_op.args[i])
                if tv !== nothing
                    ctx[after_arg] = tv
                else
                    shape = extract_tile_shape(after_arg.type)
                    ctx[after_arg] = CGVal(block_args[i], result_types[i], after_arg.type, shape)
                end
            end
        end

        # Emit "after" region body (skip terminator - we emit ContinueOp instead)
        emit_block!(ctx, after_blk; skip_terminator=true)

        # Emit ContinueOp with yield values from after region's YieldOp
        continue_operands = Value[]
        if after_blk.terminator isa YieldOp
            for val in after_blk.terminator.values
                tv = emit_value!(ctx, val)
                tv !== nothing && tv.v !== nothing && push!(continue_operands, tv.v)
            end
        end
        push!(continue_operands, ctx.token)
        encode_ContinueOp!(ctx.cb, continue_operands)

        empty!(ctx.block_args)
        merge!(ctx.block_args, saved_block_args)
    end
    results = encode_LoopOp!(body_builder, cb, result_types, init_values)

    # Last result is the token
    ctx.token = results[end]

    # Store results at the loop's SSA index (may be empty for void-returning loops)
    ctx.values[ssa_idx] = CGVal(results[1:n_user_results], parent_result_type)
end

"""
    emit_terminator!(ctx, terminator)

Emit bytecode for a block terminator.
"""
function emit_terminator!(ctx::CGCtx, node::ReturnNode)
    emit_return!(ctx, node)
end

function emit_terminator!(ctx::CGCtx, op::YieldOp)
    # Collect yield operands
    operands = Value[]
    for val in op.values
        tv = emit_value!(ctx, val)
        tv !== nothing && tv.v !== nothing && push!(operands, tv.v)
    end
    # Append current token for memory ordering
    push!(operands, ctx.token)
    encode_YieldOp!(ctx.cb, operands)
end

function emit_terminator!(ctx::CGCtx, op::ContinueOp)
    # Collect continue operands (updated carried values)
    operands = Value[]
    for val in op.values
        tv = emit_value!(ctx, val)
        tv !== nothing && tv.v !== nothing && push!(operands, tv.v)
    end
    # Append current token for memory ordering
    push!(operands, ctx.token)
    encode_ContinueOp!(ctx.cb, operands)
end

function emit_terminator!(ctx::CGCtx, op::BreakOp)
    # Collect break operands (final values)
    operands = Value[]
    for val in op.values
        tv = emit_value!(ctx, val)
        tv !== nothing && tv.v !== nothing && push!(operands, tv.v)
    end
    # Append current token for memory ordering
    push!(operands, ctx.token)
    encode_BreakOp!(ctx.cb, operands)
end

function emit_terminator!(ctx::CGCtx, ::Nothing)
    # No terminator, nothing to emit
end

function emit_terminator!(ctx::CGCtx, ::ConditionOp)
    # ConditionOp is handled specially by emit_while_op!, not emitted as a terminator
end

#=============================================================================
 Early Return Hoisting

 tileiras rejects ReturnNode (cuda_tile.return) inside IfOp (cuda_tile.if)
 regions. This pre-pass rewrites the structured IR so that ReturnNode only
 appears at the top level, replacing nested returns with YieldOp.
=============================================================================#

"""
    hoist_returns!(block::Block)

Rewrite `ReturnNode` terminators inside `IfOp` regions into `YieldOp`,
hoisting the return to the parent block. Operates recursively so that
nested early returns (multiple successive `if ... return end` patterns)
are handled automatically.

Only handles the case where BOTH branches of an IfOp terminate with
ReturnNode (REGION_TERMINATION with 3 children). The 2-child case
(early return inside a loop) is not handled.
"""
function hoist_returns!(block::Block)
    # First, recurse into all nested control flow ops
    for (_, entry) in block.body
        stmt = entry.stmt
        if stmt isa IfOp
            hoist_returns!(stmt.then_region)
            hoist_returns!(stmt.else_region)
        elseif stmt isa ForOp
            hoist_returns!(stmt.body)
        elseif stmt isa WhileOp
            hoist_returns!(stmt.before)
            hoist_returns!(stmt.after)
        elseif stmt isa LoopOp
            hoist_returns!(stmt.body)
        end
    end

    # Now check: does this block contain an IfOp where both branches return?
    # If so, replace branch ReturnNodes with YieldOp and set block terminator.
    for (_, entry) in block.body
        entry.stmt isa IfOp || continue
        op = entry.stmt::IfOp
        op.then_region.terminator isa ReturnNode || continue
        op.else_region.terminator isa ReturnNode || continue

        # Both branches return — hoist to parent block.
        # Replace branch terminators with YieldOp (void — no values to yield).
        op.then_region.terminator = YieldOp()
        op.else_region.terminator = YieldOp()
        block.terminator = ReturnNode(nothing)
    end
end

"""
    emit_getfield!(ctx, args) -> Union{CGVal, Nothing}

Handle getfield on multi-value results (loops, ifs). Returns CGVal if handled,
nothing if this is not a multi-value extraction and normal handling should proceed.
This is a compile-time lookup - no Tile IR is emitted.
"""
function emit_loop_getfield!(ctx::CGCtx, args::Vector{Any})
    length(args) >= 2 || return nothing
    args[1] isa SSAValue || return nothing

    ref_cgval = get(ctx.values, args[1].id, nothing)
    ref_cgval === nothing && return nothing
    ref_cgval.v isa Vector{Value} || return nothing

    field_idx = args[2]::Int
    v = ref_cgval.v[field_idx]
    elem_type = ref_cgval.jltype.parameters[field_idx]
    type_id = tile_type_for_julia!(ctx, elem_type)
    shape = extract_tile_shape(elem_type)
    CGVal(v, type_id, elem_type, shape)
end
