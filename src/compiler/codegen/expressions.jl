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
        throw(IRError("Foreign calls not supported in Tile IR"))
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
    func = get_constant(ctx, args[1])
    call_args = args[2:end]

    # TODO: This is normally dynamic dispatch, which we should allow.
    #       However, we currently trigger this when emitting Julia intrinsics.
    #       We should switch to our own intrinsics entirely, which are only invoked.

    @static if isdefined(Core, :throw_methoderror)
        if func === Core.throw_methoderror
            _throw_method_error(ctx, call_args)
        end
    end
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
    result === missing && _unsupported_call(ctx, func, call_args)
    validate_result_type(result, result_type, func)
    return result
end

"""
    emit_invoke!(ctx, expr::Expr, result_type) -> Union{CGVal, Nothing}

Emit bytecode for a method invocation.
"""
function emit_invoke!(ctx::CGCtx, expr::Expr, @nospecialize(result_type))
    # invoke has: (MethodInstance, func, args...)
    func = get_constant(ctx, expr.args[2])
    call_args = expr.args[3:end]

    @static if isdefined(Core, :throw_methoderror)
        if func === Core.throw_methoderror
            _throw_method_error(ctx, call_args)
        end
    end

    result = emit_intrinsic!(ctx, func, call_args)
    result === missing && _unsupported_call(ctx, func, call_args)
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

    actual = CC.widenconst(result.jltype)
    expected = CC.widenconst(expected_type)

    # Check subtype relationship (actual should be at least as specific as expected)
    actual <: expected && return

    throw(IRError("Type mismatch in $func: expected $expected, got $actual"))
end

"""
    _throw_method_error(ctx, call_args)

Provide a clear error message when Julia inserts a `throw_methoderror` call,
indicating that type inference found no matching method for a function call.
"""
function _throw_method_error(ctx::CGCtx, call_args)
    # call_args typically contains: (function, arg1, arg2, ...)
    if isempty(call_args)
        throw(IRError("MethodError during Tile IR compilation"))
    end

    func_val = try
        get_constant(ctx, call_args[1])
    catch
        call_args[1]
    end

    argtypes = argextype.(Ref(ctx), call_args[2:end])
    typestr = isempty(argtypes) ? "" : " with argument types ($(join(argtypes, ", ")))"
    throw(IRError("MethodError during Tile IR compilation: no matching method for $func_val$typestr"))
end

"""
    _unsupported_call(ctx, func, call_args)

Provide a clear error message when a function has no Tile IR intrinsic mapping.
"""
function _unsupported_call(ctx::CGCtx, @nospecialize(func), call_args)
    argtypes = argextype.(Ref(ctx), call_args)
    typestr = isempty(argtypes) ? "" : " with argument types ($(join(argtypes, ", ")))"
    throw(IRError("Unsupported function call during Tile IR compilation: $func$typestr has no Tile IR equivalent"))
end

"""
    argextype(ctx, x) -> Type

Get the Julia type of an IR value.
"""
function argextype(ctx::CGCtx, @nospecialize(x))
    tv = emit_value!(ctx, x)
    tv === nothing ? Any : CC.widenconst(tv.jltype)
end
