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
        return emit_new!(ctx, expr, result_type)
    elseif expr.head === :foreigncall
        throw(IRError("Foreign calls not supported in Tile IR"))
    elseif expr.head === :boundscheck
        # Bounds checking is always disabled in Tile IR kernels.
        # Emit false so IfOps referencing this SSA can resolve the condition.
        return emit_constant!(ctx, false, Bool)
    elseif expr.head === :static_parameter
        # Static type parameter reference (e.g., V in `f(::T{V}) where {V}`).
        # Look up the concrete value from the method's sptypes.
        idx = expr.args[1]::Int
        sp = ctx.sci.sptypes[idx]
        sptyp = sp isa CC.VarState ? sp.typ : sp
        val = sptyp isa CC.Const ? sptyp.val : CC.widenconst(sptyp)
        return emit_value!(ctx, val)
    elseif expr.head === :code_coverage_effect
        return nothing
    else
        @warn "Unhandled expression head" expr.head expr
        return nothing
    end
end

"""
    emit_new!(ctx, expr::Expr, result_type) -> Union{CGVal, Nothing}

Emit bytecode for a `:new` expression (struct construction).

In Tile IR codegen, only ghost types (zero-size immutables like `Val{V}`,
`Constant{T,V}`) are supported — these have no runtime representation.
"""
function emit_new!(ctx::CGCtx, expr::Expr, @nospecialize(result_type))
    T = CC.widenconst(result_type)
    is_ghost_type(T) && return ghost_value(T)
    # On older Julia versions, method errors are emitted as
    # %new(MethodError, func, args_tuple, world) instead of Core.throw_methoderror
    if T === MethodError
        # expr.args: (MethodError, func, args_tuple, world)
        _throw_method_error(ctx, expr.args[2:end-1])
    end
    throw(IRError("Struct construction not supported in Tile IR: $T"))
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
        return emit_value!(ctx, rhs)
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
    func = @something get_constant(ctx, args[1]) throw(IRError("Cannot resolve called function"))
    call_args = args[2:end]

    @static if isdefined(Core, :throw_methoderror)
        if func === Core.throw_methoderror
            _throw_method_error(ctx, call_args)
        end
    end
    if func === Core.kwcall
        throw(IRError("Keyword argument call (Core.kwcall) was not inlined during compilation. " *
                      "Ensure all argument types are fully concrete."))
    elseif func === Core.getfield
        tv = emit_getfield!(ctx, call_args, result_type)
        tv !== nothing && return tv
    elseif func === Base.getindex
        tv = emit_getindex!(ctx, call_args, result_type)
        tv !== nothing && return tv
    elseif func === Core.apply_type
        # Type construction is compile-time only - result_type tells us the constructed type
        return ghost_value(result_type, result_type)
    elseif func === Core.getglobal
        return emit_getglobal!(ctx, call_args)
    end

    result = emit_intrinsic!(ctx, func, call_args)
    result === missing && _throw_method_error(ctx, [func; call_args])
    validate_result_type(result, result_type, func)
    return result
end

"""
    emit_invoke!(ctx, expr::Expr, result_type) -> Union{CGVal, Nothing}

Emit bytecode for a method invocation.
"""
function emit_invoke!(ctx::CGCtx, expr::Expr, @nospecialize(result_type))
    # invoke has: (MethodInstance, func, args...)
    func = @something get_constant(ctx, expr.args[2]) throw(IRError("Cannot resolve invoked function"))
    call_args = expr.args[3:end]

    @static if isdefined(Core, :throw_methoderror)
        if func === Core.throw_methoderror
            _throw_method_error(ctx, call_args)
        end
    end
    if func === Core.getglobal
        return emit_getglobal!(ctx, call_args)
    end

    result = emit_intrinsic!(ctx, func, call_args)
    result === missing && _throw_method_error(ctx, [func; call_args])
    validate_result_type(result, result_type, func)
    return result
end

# Resolve `Core.getglobal(mod::Module, name::Symbol)` at compile time. Both
# arguments must be constants — the result is the value bound in `mod`.
function emit_getglobal!(ctx::CGCtx, call_args)
    length(call_args) >= 2 || throw(IRError("getglobal: expected (Module, Symbol) arguments"))
    mod = @something get_constant(ctx, call_args[1]) throw(IRError("getglobal: module argument must be a compile-time constant"))
    name = @something get_constant(ctx, call_args[2]) throw(IRError("getglobal: name argument must be a compile-time constant"))
    mod isa Module || throw(IRError("getglobal: expected a Module, got $(typeof(mod))"))
    name isa Symbol || throw(IRError("getglobal: expected a Symbol, got $(typeof(name))"))
    return emit_value!(ctx, getglobal(mod, name))
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

Provide a clear error message when an unsupported function is encountered during
Tile IR compilation, either from an explicit `throw_methoderror` / `MethodError`
construction, or from a function call with no Tile IR intrinsic mapping.

`call_args` contains `(function, arg1, arg2, ...)`.
"""
function _throw_method_error(ctx::CGCtx, call_args)
    if isempty(call_args)
        throw(IRError("Unsupported function call during Tile IR compilation"))
    end

    func_val = try
        @something get_constant(ctx, call_args[1]) call_args[1]
    catch
        call_args[1]
    end

    argtypes = argextype.(Ref(ctx), call_args[2:end])
    typestr = isempty(argtypes) ? "" : " with argument types ($(join(argtypes, ", ")))"
    throw(IRError("Unsupported function call during Tile IR compilation: $func_val$typestr has no Tile IR equivalent"))
end

"""
    argextype(ctx, x) -> Type

Get the Julia type of an IR value.
"""
function argextype(ctx::CGCtx, @nospecialize(x))
    tv = emit_value!(ctx, x)
    tv === nothing ? Any : CC.widenconst(tv.jltype)
end
