# expression emission

"""
    emit_expr!(ctx, expr::Expr, result_type) -> Union{CGVal, Nothing}

Emit bytecode for an expression.
"""
function emit_expr!(ctx::CGCtx, expr::Expr, @nospecialize(result_type))
    if expr.head === :call
        return emit_call!(ctx, expr, result_type)
    elseif expr.head === :invoke
        return emit_invoke!(ctx, expr, result_type)
    elseif expr.head === :(=)
        return emit_assignment!(ctx, expr, result_type)
    elseif expr.head === :new
        return nothing  # Struct construction handled elsewhere
    elseif expr.head === :foreigncall
        error("Foreign calls not supported in Tile IR")
    elseif expr.head === :boundscheck
        return nothing
    else
        @warn "Unhandled expression head" expr.head expr
        return nothing
    end
end

function emit_assignment!(ctx::CGCtx, expr::Expr, @nospecialize(result_type))
    lhs = expr.args[1]
    rhs = expr.args[2]

    tv = emit_rhs!(ctx, rhs, result_type)

    if lhs isa SlotNumber && tv !== nothing
        ctx[lhs] = tv
    end

    return tv
end

function emit_rhs!(ctx::CGCtx, @nospecialize(rhs), @nospecialize(result_type))
    if rhs isa Expr
        return emit_expr!(ctx, rhs, result_type)
    elseif rhs isa SSAValue || rhs isa SlotNumber || rhs isa Argument
        return emit_value!(ctx, rhs)
    elseif rhs isa QuoteNode
        return emit_constant!(ctx, rhs.value, result_type)
    elseif rhs isa GlobalRef
        return nothing
    else
        return emit_constant!(ctx, rhs, result_type)
    end
end

#=============================================================================
 Call Emission
=============================================================================#

"""
    emit_call!(ctx, expr::Expr, result_type) -> Union{CGVal, Nothing}

Emit bytecode for a function call.
"""
function emit_call!(ctx::CGCtx, expr::Expr, @nospecialize(result_type))
    args = expr.args
    func = resolve_function(ctx, args[1])
    call_args = args[2:end]

    # TODO: This is normally dynamic dispatch, which we should allow.
    #       However, we currently trigger this when emitting Julia intrinsics.
    #       We should switch to our own intrinsics entirely, which are only invoked.

    if func === Core.getfield
        tv = emit_getfield!(ctx, call_args, result_type)
        tv !== nothing && return tv
    elseif func === Base.getindex
        tv = emit_getindex!(ctx, call_args, result_type)
        tv !== nothing && return tv
    elseif func === Core.apply_type
        # Type construction is compile-time only - result_type tells us the constructed type
        return ghost_value(result_type, result_type)
    end

    result = emit_intrinsic!(ctx, func, call_args)
    result === missing && error("Unknown function call: $func")
    validate_result_type(result, result_type, func)
    return result
end

"""
    emit_invoke!(ctx, expr::Expr, result_type) -> Union{CGVal, Nothing}

Emit bytecode for a method invocation.
"""
function emit_invoke!(ctx::CGCtx, expr::Expr, @nospecialize(result_type))
    # invoke has: (MethodInstance, func, args...)
    func = resolve_function(ctx, expr.args[2])
    call_args = expr.args[3:end]

    result = emit_intrinsic!(ctx, func, call_args)
    result === missing && error("Unknown function call: $func")
    validate_result_type(result, result_type, func)
    return result
end

"""
    validate_result_type(result, expected_type, func)

Assert that the intrinsic returned a type compatible with what the IR expects.
"""
function validate_result_type(@nospecialize(result), @nospecialize(expected_type), @nospecialize(func))
    result === nothing && return  # void return
    result isa CGVal || return

    actual = unwrap_type(result.jltype)
    expected = unwrap_type(expected_type)

    # Check subtype relationship (actual should be at least as specific as expected)
    actual <: expected && return

    error("Type mismatch in $func: expected $expected, got $actual")
end

"""
    resolve_function(ctx, ref) -> Any

Resolve a function reference to its actual value.
"""
function resolve_function(ctx::CGCtx, @nospecialize(ref))
    if ref isa GlobalRef
        return getfield(ref.mod, ref.name)
    elseif ref isa QuoteNode
        return ref.value
    elseif ref isa SSAValue
        tv = ctx[ref]
        tv !== nothing && tv.constant !== nothing && return something(tv.constant)
    end
    return ref
end
