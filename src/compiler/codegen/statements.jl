# statement emission

"""
    emit_statement!(ctx, stmt, ssa_idx, result_type)

Emit bytecode for a single SSA statement.
The ssa_idx is the original Julia SSA index to store the result at.
"""
function emit_statement!(ctx::CGCtx, @nospecialize(stmt), ssa_idx::Int, @nospecialize(result_type))
    tv = nothing
    if stmt isa ReturnNode
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
    elseif stmt isa PiNode
        # PiNode is a type narrowing assertion - store the resolved value
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
