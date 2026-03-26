# statement emission

"""
    emit_statement!(ctx, stmt, ssa_idx, result_type)

Emit bytecode for a single SSA statement.
The ssa_idx is the original Julia SSA index to store the result at.
"""
function emit_statement!(ctx::CGCtx, @nospecialize(stmt), ssa_idx::Int, @nospecialize(result_type))
    ctx.current_ssa_idx = ssa_idx
    tv = nothing
    if stmt isa MakeTokenNode
        tv = emit_make_token!(ctx)
    elseif stmt isa JoinTokensNode
        tv = emit_join_tokens!(ctx, stmt)
    elseif stmt isa TokenResultNode
        tv = emit_token_result!(ctx, stmt)
    elseif stmt isa ReturnNode
        emit_return!(ctx, stmt)
    elseif stmt isa Expr
        tv = emit_expr!(ctx, stmt, result_type)
        if tv === nothing
            # If emit_expr! returns nothing, try emit_value! for ghost values like tuples
            tv = emit_value!(ctx, stmt)
        end
    elseif stmt isa GlobalRef
        tv = emit_value!(ctx, stmt)
    elseif stmt isa QuoteNode
        tv = emit_constant!(ctx, stmt.value, result_type)
    elseif stmt isa SlotNumber
        tv = ctx[stmt]
    elseif stmt isa Argument
        tv = emit_value!(ctx, stmt)
    elseif stmt isa SSAValue
        tv = emit_value!(ctx, stmt)
    elseif stmt isa PiNode
        # PiNode is a type narrowing assertion - store the resolved value
        tv = emit_value!(ctx, stmt)
    elseif stmt isa Undef
        tv = emit_value!(ctx, stmt)
    elseif stmt === nothing
        # Dead code elimination artifact — no value to register
    else
        # Literal values from constant folding or concrete eval.
        # Try emit_constant! first (numbers/ghost types), fall back to emit_value!.
        tv = emit_constant!(ctx, stmt, result_type)
        if tv === nothing
            tv = emit_value!(ctx, stmt)
        end
    end

    # Store result by original Julia SSA index
    if tv !== nothing
        ctx.values[ssa_idx] = tv
    end
end

"""
    emit_return!(ctx, node::ReturnNode)

Emit a return operation.
"""
function emit_return!(ctx::CGCtx, node::ReturnNode)
    if !isdefined(node, :val) || node.val === nothing || (node.val isa GlobalRef && node.val.name === :nothing)
        encode_ReturnOp!(ctx.cb, Value[])
    else
        tv = emit_value!(ctx, node.val)
        if tv !== nothing && !is_ghost(tv)
            encode_ReturnOp!(ctx.cb, [tv.v])
        else
            encode_ReturnOp!(ctx.cb, Value[])
        end
    end
end

#=============================================================================
 Token IR node emission
=============================================================================#

function emit_make_token!(ctx::CGCtx)
    token_type = ctx.token_type
    if token_type === nothing
        token_type = Token(ctx.tt)
        ctx.token_type = token_type
    end
    v = encode_MakeTokenOp!(ctx.cb, token_type)
    return CGVal(v, token_type, TokenType)
end

function emit_join_tokens!(ctx::CGCtx, node::JoinTokensNode)
    tokens = Value[]
    for tok_ref in node.tokens
        tv = emit_value!(ctx, tok_ref)
        tv === nothing && throw(IRError("JoinTokensNode: cannot resolve token operand $tok_ref"))
        push!(tokens, tv.v)
    end
    # Deduplicate by identity (avoid join_tokens(%x, %x))
    unique_tokens = Value[]
    for t in tokens
        any(u -> u === t, unique_tokens) || push!(unique_tokens, t)
    end
    if length(unique_tokens) == 1
        return CGVal(unique_tokens[1], ctx.token_type, TokenType)
    end
    v = encode_JoinTokensOp!(ctx.cb, ctx.token_type, unique_tokens)
    return CGVal(v, ctx.token_type, TokenType)
end

function emit_token_result!(ctx::CGCtx, node::TokenResultNode)
    # The memory op at node.mem_op_ssa should have stored its result token
    v = get(ctx.result_tokens, node.mem_op_ssa, nothing)
    v === nothing && throw(IRError("TokenResultNode: no result token for memory op at SSA %$(node.mem_op_ssa)"))
    return CGVal(v, ctx.token_type, TokenType)
end

