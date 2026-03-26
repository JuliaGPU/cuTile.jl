# Structured IR Emission

function result_count(@nospecialize(T))
    T === Nothing && return 0
    T <: Tuple && return length(T.parameters)
    return 1
end

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

emit_control_flow_op!(ctx::CGCtx, op::IfOp, @nospecialize(rt), n::Int, idx::Int) = emit_if_op!(ctx, op, rt, n, idx)
emit_control_flow_op!(ctx::CGCtx, op::ForOp, @nospecialize(rt), n::Int, idx::Int) = emit_for_op!(ctx, op, rt, n, idx)
emit_control_flow_op!(ctx::CGCtx, op::WhileOp, @nospecialize(rt), n::Int, idx::Int) = emit_while_op!(ctx, op, rt, n, idx)
emit_control_flow_op!(ctx::CGCtx, op::LoopOp, @nospecialize(rt), n::Int, idx::Int) = emit_loop_op!(ctx, op, rt, n, idx)

#=============================================================================
 IfOp
=============================================================================#

function emit_if_op!(ctx::CGCtx, op::IfOp, @nospecialize(parent_result_type), n_results::Int, ssa_idx::Int)
    cb = ctx.cb

    cond_tv = emit_value!(ctx, op.condition)
    cond_tv === nothing && throw(IRError("Cannot resolve condition for IfOp"))

    # Build result types uniformly from parent_result_type (pass set it correctly)
    result_types = TypeId[]
    if parent_result_type !== Nothing
        if parent_result_type <: Tuple
            for T in parent_result_type.parameters
                push!(result_types, tile_type_for_julia!(ctx, T))
            end
        else
            push!(result_types, tile_type_for_julia!(ctx, parent_result_type))
        end
    end

    then_body = function(_)
        saved = copy(ctx.block_args)
        emit_block!(ctx, op.then_region)
        op.then_region.terminator === nothing && encode_YieldOp!(ctx.cb, Value[])
        empty!(ctx.block_args); merge!(ctx.block_args, saved)
    end
    else_body = function(_)
        saved = copy(ctx.block_args)
        emit_block!(ctx, op.else_region)
        op.else_region.terminator === nothing && encode_YieldOp!(ctx.cb, Value[])
        empty!(ctx.block_args); merge!(ctx.block_args, saved)
    end
    results = encode_IfOp!(then_body, else_body, cb, result_types, cond_tv.v)

    ctx.values[ssa_idx] = CGVal(results, parent_result_type)
end

#=============================================================================
 ForOp
=============================================================================#

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

    # Emit ALL init values (user + token carries from pass)
    init_values = Value[]
    for init_val in op.init_values
        tv = emit_value!(ctx, init_val)
        (tv === nothing || tv.v === nothing) && throw(IRError("Cannot resolve ForOp init value"))
        push!(init_values, tv.v)
    end

    # Build result types uniformly from block args
    n_carries = length(body_blk.args)
    result_types = TypeId[tile_type_for_julia!(ctx, arg.type) for arg in body_blk.args]

    body_builder = function(block_args)
        saved = copy(ctx.block_args)
        # Tile IR layout: [iv, carries...]
        ctx[iv_arg] = CGVal(block_args[1], iv_type, iv_jl_type)
        for i in 1:n_carries
            arg = body_blk.args[i]
            shape = RowMajorShape(extract_tile_shape(arg.type))
            ctx[arg] = CGVal(block_args[i + 1], result_types[i], arg.type, shape)
        end
        emit_block!(ctx, body_blk)
        # If body has no terminator, emit a ContinueOp with all block args
        if body_blk.terminator === nothing
            # block_args[1] is iv, block_args[2:end] are carries
            encode_ContinueOp!(ctx.cb, block_args[2:end])
        end
        empty!(ctx.block_args); merge!(ctx.block_args, saved)
    end
    results = encode_ForOp!(body_builder, cb, result_types, iv_type,
                             lower_tv.v, upper_tv.v, step_tv.v, init_values)

    # Trust parent_result_type (pass set it via update_type!)
    ctx.values[ssa_idx] = CGVal(results, parent_result_type)
end

#=============================================================================
 LoopOp
=============================================================================#

function emit_loop_op!(ctx::CGCtx, op::LoopOp, @nospecialize(parent_result_type), n_results::Int, ssa_idx::Int)
    cb = ctx.cb
    body_blk = op.body

    init_values = Value[]
    for init_val in op.init_values
        tv = emit_value!(ctx, init_val)
        (tv === nothing || tv.v === nothing) && throw(IRError("Cannot resolve LoopOp init value"))
        push!(init_values, tv.v)
    end

    n_carries = length(body_blk.args)
    result_types = TypeId[tile_type_for_julia!(ctx, arg.type) for arg in body_blk.args]

    body_builder = function(block_args)
        saved = copy(ctx.block_args)
        for i in 1:n_carries
            arg = body_blk.args[i]
            shape = RowMajorShape(extract_tile_shape(arg.type))
            ctx[arg] = CGVal(block_args[i], result_types[i], arg.type, shape)
        end
        emit_block!(ctx, body_blk)
        if body_blk.terminator === nothing
            encode_ContinueOp!(ctx.cb, copy(block_args))
        end
        empty!(ctx.block_args); merge!(ctx.block_args, saved)
    end
    results = encode_LoopOp!(body_builder, cb, result_types, init_values)

    ctx.values[ssa_idx] = CGVal(results, parent_result_type)
end

#=============================================================================
 WhileOp — lowered to LoopOp pattern in codegen
=============================================================================#

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

    n_carries = length(before_blk.args)
    result_types = TypeId[tile_type_for_julia!(ctx, arg.type) for arg in before_blk.args]

    body_builder = function(block_args)
        saved = copy(ctx.block_args)

        for i in 1:n_carries
            arg = before_blk.args[i]
            shape = RowMajorShape(extract_tile_shape(arg.type))
            ctx[arg] = CGVal(block_args[i], result_types[i], arg.type, shape)
        end

        emit_block!(ctx, before_blk)

        cond_op = before_blk.terminator
        cond_op isa ConditionOp || throw(IRError("WhileOp before region must end with ConditionOp"))

        cond_tv = emit_value!(ctx, cond_op.condition)
        (cond_tv === nothing || cond_tv.v === nothing) && throw(IRError("Cannot resolve WhileOp condition"))

        then_body = (_) -> encode_YieldOp!(ctx.cb, Value[])
        else_body = function(_)
            break_operands = Value[]
            for arg in cond_op.args
                tv = emit_value!(ctx, arg)
                tv !== nothing && tv.v !== nothing && push!(break_operands, tv.v)
            end
            if isempty(break_operands)
                append!(break_operands, block_args[1:n_carries])
            else
                # Append token carries (block_args beyond user carries from ConditionOp)
                n_user = length(break_operands)
                for i in (n_user + 1):n_carries
                    push!(break_operands, block_args[i])
                end
            end
            encode_BreakOp!(ctx.cb, break_operands)
        end
        encode_IfOp!(then_body, else_body, cb, TypeId[], cond_tv.v)

        for i in 1:length(after_blk.args)
            arg = after_blk.args[i]
            if i <= length(cond_op.args)
                tv = emit_value!(ctx, cond_op.args[i])
                if tv !== nothing
                    ctx[arg] = tv
                else
                    shape = RowMajorShape(extract_tile_shape(arg.type))
                    ctx[arg] = CGVal(block_args[i], result_types[i], arg.type, shape)
                end
            else
                # Token carries beyond ConditionOp.args: use block_args directly
                ctx[arg] = CGVal(block_args[i], result_types[i], arg.type,
                                  RowMajorShape(extract_tile_shape(arg.type)))
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
        # Ensure all carries (including tokens from pass) are in the ContinueOp
        while length(continue_operands) < n_carries
            push!(continue_operands, block_args[length(continue_operands) + 1])
        end
        encode_ContinueOp!(ctx.cb, continue_operands)

        empty!(ctx.block_args); merge!(ctx.block_args, saved)
    end
    results = encode_LoopOp!(body_builder, cb, result_types, init_values)

    ctx.values[ssa_idx] = CGVal(results, parent_result_type)
end

#=============================================================================
 Terminators — tokens already in op.values from token_order_pass!
=============================================================================#

emit_terminator!(ctx::CGCtx, node::ReturnNode) = emit_return!(ctx, node)

function emit_terminator!(ctx::CGCtx, op::YieldOp)
    operands = Value[]
    for val in op.values
        tv = emit_value!(ctx, val)
        tv !== nothing && tv.v !== nothing && push!(operands, tv.v)
    end
    encode_YieldOp!(ctx.cb, operands)
end

function emit_terminator!(ctx::CGCtx, op::ContinueOp)
    operands = Value[]
    for val in op.values
        tv = emit_value!(ctx, val)
        tv !== nothing && tv.v !== nothing && push!(operands, tv.v)
    end
    encode_ContinueOp!(ctx.cb, operands)
end

function emit_terminator!(ctx::CGCtx, op::BreakOp)
    operands = Value[]
    for val in op.values
        tv = emit_value!(ctx, val)
        tv !== nothing && tv.v !== nothing && push!(operands, tv.v)
    end
    encode_BreakOp!(ctx.cb, operands)
end

emit_terminator!(ctx::CGCtx, ::Nothing) = nothing
emit_terminator!(ctx::CGCtx, ::ConditionOp) = nothing

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
 Loop getfield extraction — uniform, no token special cases
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
