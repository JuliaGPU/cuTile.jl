# Structured IR Emission

"""
    result_count(T) -> Int

Compute the number of results from a Block.types entry.
"""
function result_count(@nospecialize(T))
    T === Nothing && return 0
    T <: Tuple && return length(T.parameters)
    return 1
end

"""
    emit_block!(ctx, block::Block)

Emit bytecode for a structured IR block.
"""
function emit_block!(ctx::CGCtx, block::Block; skip_terminator::Bool=false)
    for (ssa_idx, entry) in block.body
        if entry.stmt isa ControlFlowOp
            n_results = result_count(entry.typ)
            emit_control_flow_op!(ctx, entry.stmt, entry.typ, n_results, ssa_idx)
        else
            emit_statement!(ctx, entry.stmt, ssa_idx, entry.typ)
        end
    end

    if !skip_terminator && block.terminator !== nothing
        emit_terminator!(ctx, block.terminator)
    end
end

emit_control_flow_op!(ctx::CGCtx, op::IfOp, @nospecialize(result_type), n_results::Int, original_idx::Int) =
    emit_if_op!(ctx, op, result_type, n_results, original_idx)
emit_control_flow_op!(ctx::CGCtx, op::ForOp, @nospecialize(result_type), n_results::Int, original_idx::Int) =
    emit_for_op!(ctx, op, result_type, n_results, original_idx)
emit_control_flow_op!(ctx::CGCtx, op::WhileOp, @nospecialize(result_type), n_results::Int, original_idx::Int) =
    emit_while_op!(ctx, op, result_type, n_results, original_idx)
emit_control_flow_op!(ctx::CGCtx, op::LoopOp, @nospecialize(result_type), n_results::Int, original_idx::Int) =
    emit_loop_op!(ctx, op, result_type, n_results, original_idx)

#=============================================================================
 Control flow emitters
 Token threading through control flow is still manual (conservative approach).
 The token_order_pass handles straight-line code; control flow uses ctx.token.
=============================================================================#

function emit_if_op!(ctx::CGCtx, op::IfOp, @nospecialize(parent_result_type), n_results::Int, ssa_idx::Int)
    cb = ctx.cb
    then_blk = op.then_region
    else_blk = op.else_region

    cond_tv = emit_value!(ctx, op.condition)
    cond_tv === nothing && throw(IRError("Cannot resolve condition for IfOp"))

    # User result types
    result_types = TypeId[]
    if parent_result_type === Nothing
        # No results
    elseif parent_result_type <: Tuple
        for T in parent_result_type.parameters
            push!(result_types, tile_type_for_julia!(ctx, T))
        end
    else
        push!(result_types, tile_type_for_julia!(ctx, parent_result_type))
    end
    n_user_results = length(result_types)
    # Add token as additional result
    push!(result_types, ctx.token_type)

    token_before = ctx.token

    then_body = function(_)
        saved_block_args = copy(ctx.block_args)
        ctx.token = token_before
        emit_block!(ctx, then_blk)
        if then_blk.terminator === nothing
            encode_YieldOp!(ctx.cb, [ctx.token])
        end
        empty!(ctx.block_args)
        merge!(ctx.block_args, saved_block_args)
    end
    else_body = function(_)
        saved_block_args = copy(ctx.block_args)
        ctx.token = token_before
        emit_block!(ctx, else_blk)
        if else_blk.terminator === nothing
            encode_YieldOp!(ctx.cb, [ctx.token])
        end
        empty!(ctx.block_args)
        merge!(ctx.block_args, saved_block_args)
    end
    results = encode_IfOp!(then_body, else_body, cb, result_types, cond_tv.v)

    ctx.token = results[end]
    ctx.values[ssa_idx] = CGVal(results[1:n_user_results], parent_result_type)
end

function emit_for_op!(ctx::CGCtx, op::ForOp, @nospecialize(parent_result_type), n_results::Int, ssa_idx::Int)
    cb = ctx.cb
    body_blk = op.body

    lower_tv = emit_value!(ctx, op.lower)
    upper_tv = emit_value!(ctx, op.upper)
    step_tv = emit_value!(ctx, op.step)
    iv_arg = op.iv_arg

    (lower_tv === nothing || upper_tv === nothing || step_tv === nothing) &&
        throw(IRError("Cannot resolve ForOp bounds"))
    lower_tv.jltype === upper_tv.jltype === step_tv.jltype ||
        throw(IRError("ForOp bounds must all have the same type"))
        iv_jl_type = lower_tv.jltype
        iv_type = tile_type_for_julia!(ctx, iv_jl_type)

    # Init values + token
    init_values = Value[]
    for init_val in op.init_values
        tv = emit_value!(ctx, init_val)
        (tv === nothing || tv.v === nothing) && throw(IRError("Cannot resolve ForOp init value"))
        push!(init_values, tv.v)
    end
    push!(init_values, ctx.token)

    n_carries = length(op.init_values)

    result_types = TypeId[]
    for i in 1:n_carries
        body_arg = body_blk.args[i]
        push!(result_types, tile_type_for_julia!(ctx, body_arg.type))
    end
    push!(result_types, ctx.token_type)

    body_builder = function(block_args)
        saved_block_args = copy(ctx.block_args)

        iv_tv = CGVal(block_args[1], iv_type, iv_jl_type)
        ctx[iv_arg] = iv_tv

        for i in 1:n_carries
            body_arg = body_blk.args[i]
            shape = RowMajorShape(extract_tile_shape(body_arg.type))
            ctx[body_arg] = CGVal(block_args[i + 1], result_types[i], body_arg.type, shape)
        end
        ctx.token = block_args[end]

        emit_block!(ctx, body_blk)

        empty!(ctx.block_args)
        merge!(ctx.block_args, saved_block_args)
    end
    results = encode_ForOp!(body_builder, cb, result_types, iv_type, lower_tv.v, upper_tv.v, step_tv.v, init_values)

    ctx.token = results[end]
    ctx.values[ssa_idx] = CGVal(results[1:n_carries], parent_result_type)
end

function emit_loop_op!(ctx::CGCtx, op::LoopOp, @nospecialize(parent_result_type), n_results::Int, ssa_idx::Int)
    cb = ctx.cb
    body_blk = op.body

    init_values = Value[]
    for init_val in op.init_values
        tv = emit_value!(ctx, init_val)
        (tv === nothing || tv.v === nothing) && throw(IRError("Cannot resolve LoopOp init value"))
        push!(init_values, tv.v)
    end
    push!(init_values, ctx.token)

    n_carries = length(op.init_values)

    result_types = TypeId[]
    for i in 1:n_carries
        body_arg = body_blk.args[i]
        push!(result_types, tile_type_for_julia!(ctx, body_arg.type))
    end
    push!(result_types, ctx.token_type)

    body_builder = function(block_args)
        saved_block_args = copy(ctx.block_args)

        for i in 1:n_carries
            body_arg = body_blk.args[i]
            shape = RowMajorShape(extract_tile_shape(body_arg.type))
            ctx[body_arg] = CGVal(block_args[i], result_types[i], body_arg.type, shape)
        end
        ctx.token = block_args[end]

        emit_block!(ctx, body_blk)

        if body_blk.terminator === nothing
            fallback_operands = copy(block_args)
            fallback_operands[end] = ctx.token
            encode_ContinueOp!(ctx.cb, fallback_operands)
        end

        empty!(ctx.block_args)
        merge!(ctx.block_args, saved_block_args)
    end
    results = encode_LoopOp!(body_builder, cb, result_types, init_values)

    ctx.token = results[end]
    ctx.values[ssa_idx] = CGVal(results[1:n_carries], parent_result_type)
end

function emit_while_op!(ctx::CGCtx, op::WhileOp, @nospecialize(parent_result_type), n_results::Int, ssa_idx::Int)
    cb = ctx.cb
    before_blk = op.before
    after_blk = op.after

    init_values = Value[]
    for init_val in op.init_values
        tv = emit_value!(ctx, init_val)
        (tv === nothing || tv.v === nothing) && throw(IRError("Cannot resolve WhileOp init value: $init_val"))
        push!(init_values, tv.v)
    end
    push!(init_values, ctx.token)

    n_carries = length(op.init_values)

    result_types = TypeId[]
    for i in 1:n_carries
        before_arg = before_blk.args[i]
        push!(result_types, tile_type_for_julia!(ctx, before_arg.type))
    end
    push!(result_types, ctx.token_type)

    body_builder = function(block_args)
        saved_block_args = copy(ctx.block_args)

        for i in 1:n_carries
            before_arg = before_blk.args[i]
            shape = RowMajorShape(extract_tile_shape(before_arg.type))
            ctx[before_arg] = CGVal(block_args[i], result_types[i], before_arg.type, shape)
        end
        ctx.token = block_args[end]

        emit_block!(ctx, before_blk)

        cond_op = before_blk.terminator
        cond_op isa ConditionOp || throw(IRError("WhileOp before region must end with ConditionOp"))

        cond_tv = emit_value!(ctx, cond_op.condition)
        (cond_tv === nothing || cond_tv.v === nothing) && throw(IRError("Cannot resolve WhileOp condition"))

        then_body = function(_)
            encode_YieldOp!(ctx.cb, Value[])
        end

        else_body = function(_)
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

        encode_IfOp!(then_body, else_body, cb, TypeId[], cond_tv.v)

        for i in 1:length(after_blk.args)
            after_arg = after_blk.args[i]
            if i <= length(cond_op.args)
                tv = emit_value!(ctx, cond_op.args[i])
                if tv !== nothing
                    ctx[after_arg] = tv
                else
                    shape = RowMajorShape(extract_tile_shape(after_arg.type))
                    ctx[after_arg] = CGVal(block_args[i], result_types[i], after_arg.type, shape)
                end
            end
        end

        emit_block!(ctx, after_blk; skip_terminator=true)

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

    ctx.token = results[end]
    ctx.values[ssa_idx] = CGVal(results[1:n_carries], parent_result_type)
end

#=============================================================================
 Terminators
 Token is appended manually for control flow threading (conservative approach).
=============================================================================#

function emit_terminator!(ctx::CGCtx, node::ReturnNode)
    emit_return!(ctx, node)
end

function emit_terminator!(ctx::CGCtx, op::YieldOp)
    operands = Value[]
    for val in op.values
        tv = emit_value!(ctx, val)
        tv !== nothing && tv.v !== nothing && push!(operands, tv.v)
    end
    push!(operands, ctx.token)
    encode_YieldOp!(ctx.cb, operands)
end

function emit_terminator!(ctx::CGCtx, op::ContinueOp)
    operands = Value[]
    for val in op.values
        tv = emit_value!(ctx, val)
        tv !== nothing && tv.v !== nothing && push!(operands, tv.v)
    end
    push!(operands, ctx.token)
    encode_ContinueOp!(ctx.cb, operands)
end

function emit_terminator!(ctx::CGCtx, op::BreakOp)
    operands = Value[]
    for val in op.values
        tv = emit_value!(ctx, val)
        tv !== nothing && tv.v !== nothing && push!(operands, tv.v)
    end
    push!(operands, ctx.token)
    encode_BreakOp!(ctx.cb, operands)
end

function emit_terminator!(ctx::CGCtx, ::Nothing) end
function emit_terminator!(ctx::CGCtx, ::ConditionOp) end

#=============================================================================
 Early Return Hoisting
=============================================================================#

function hoist_returns!(block::Block)
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

    for (_, entry) in block.body
        entry.stmt isa IfOp || continue
        op = entry.stmt::IfOp
        op.then_region.terminator isa ReturnNode || continue
        op.else_region.terminator isa ReturnNode || continue

        op.then_region.terminator = YieldOp()
        op.else_region.terminator = YieldOp()
        block.terminator = ReturnNode(nothing)
    end
end

#=============================================================================
 Loop getfield extraction
=============================================================================#

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
    shape = RowMajorShape(extract_tile_shape(elem_type))
    CGVal(v, type_id, elem_type, shape)
end
