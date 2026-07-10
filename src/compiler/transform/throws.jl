# Julia throw lowering

"""Return `Some(x)` when `x` can be reconstructed without runtime values."""
function throw_constant(sci::StructuredIRCode, def_index, @nospecialize(x))
    if x isa QuoteNode
        return Some(x.value)
    elseif x isa GlobalRef
        return IRStructurizer.const_value(sci, x)
    elseif x isa SSAValue
        inst = def(def_index, x)
        inst === nothing && return nothing
        call = resolve_call(inst.block, inst[:stmt])
        call === nothing && return nothing
        func, args = call
        if func === Core.tuple
            vals = Any[]
            for arg in args
                val = throw_constant(sci, def_index, arg)
                val === nothing && return nothing
                push!(vals, something(val))
            end
            return Some(Tuple(vals))
        elseif func === Intrinsics.format_string
            vals = Any[]
            for arg in args
                val = throw_constant(sci, def_index, arg)
                val === nothing && return nothing
                push!(vals, something(val))
            end
            return Some(string(vals...))
        end
        return nothing
    elseif x isa Union{Argument, SlotNumber, BlockArgument, Expr}
        return nothing
    else
        return Some(x)
    end
end

function thrown_type(block::Block, @nospecialize(exception))
    if exception isa QuoteNode
        return typeof(exception.value)
    elseif !(exception isa Union{SSAValue, Argument, SlotNumber, BlockArgument, Expr, GlobalRef})
        return typeof(exception)
    end
    T = value_type(block, exception)
    T === nothing && return Exception
    return CC.widenconst(T)
end

"""
Best-effort reconstruction of an exception from its optimized constructor.
Only constructors Julia marked removable are evaluated at compile time.
"""
function thrown_exception(sci::StructuredIRCode, def_index, block::Block,
                          @nospecialize(exception))
    direct = throw_constant(sci, def_index, exception)
    direct !== nothing && something(direct) isa Exception && return something(direct)
    exception isa SSAValue || return nothing

    inst = def(def_index, exception)
    inst === nothing && return nothing
    flag = inst[:flag]
    (flag & CC.IR_FLAGS_REMOVABLE) == CC.IR_FLAGS_REMOVABLE || return nothing

    stmt = inst[:stmt]
    T = thrown_type(block, exception)
    T isa Type && T <: Exception || return nothing

    args = if stmt isa Expr && stmt.head === :new
        stmt.args[2:end]
    else
        call = resolve_call(inst.block, stmt)
        call === nothing && return nothing
        func, call_args = call
        func === T || return nothing
        call_args
    end

    vals = Any[]
    for arg in args
        val = throw_constant(sci, def_index, arg)
        val === nothing && return nothing
        push!(vals, something(val))
    end
    try
        return T(vals...)
    catch
        return nothing
    end
end

function exception_message(sci::StructuredIRCode, def_index, block::Block,
                           @nospecialize(exception))
    if thrown_type(block, exception) === MethodError && exception isa SSAValue
        inst = def(def_index, exception)
        if inst !== nothing
            stmt = inst[:stmt]
            args = if stmt isa Expr && stmt.head === :new
                stmt.args[2:end]
            else
                call = resolve_call(inst.block, stmt)
                call === nothing ? Any[] : last(call)
            end
            if length(args) >= 2 && args[2] isa SSAValue
                tuple_inst = def(def_index, args[2])
                tuple_call = tuple_inst === nothing ? nothing :
                             resolve_call(tuple_inst.block, tuple_inst[:stmt])
                if tuple_call !== nothing && first(tuple_call) === Core.tuple
                    return method_error_message(sci, def_index, block,
                                                Any[args[1]; last(tuple_call)])
                end
            end
        end
    end

    ex = thrown_exception(sci, def_index, block, exception)
    if ex !== nothing
        if hasfield(typeof(ex), :msg) && isdefined(ex, :msg)
            msg = getfield(ex, :msg)
            msg isa AbstractString && return String(msg)
        end
        return sprint(showerror, ex)
    end

    T = thrown_type(block, exception)
    return T isa Type && T <: Exception ? "$(T) was thrown" : "exception was thrown"
end

function method_error_message(sci::StructuredIRCode, def_index, block::Block, args)
    isempty(args) && return "Unsupported function call during Tile IR compilation"
    func = throw_constant(sci, def_index, args[1])
    funcstr = func === nothing ? string(args[1]) : string(something(func))
    argtypes = Any[value_type(block, arg) for arg in args[2:end]]
    typestr = isempty(argtypes) ? "" : " with argument types ($(join(argtypes, ", ")))"
    return "Unsupported function call during Tile IR compilation: $funcstr$typestr has no Tile IR equivalent"
end

"""
    lower_throws!(sci)

Canonicalize Julia throws before normal optimization and codegen. Throws nested
in structured control flow remain runtime failures. A throw in the entry block
is unavoidable after Julia's CFG simplification and becomes a collected
compile-time diagnostic.
"""
function lower_throws!(sci::StructuredIRCode)
    def_index = defs(sci)
    for block in eachblock(sci)
        runtime = block !== sci.entry
        for inst in instructions(block)
            call = resolve_call(block, inst[:stmt])
            call === nothing && continue
            func, args = call
            message = if func === throw
                length(args) == 1 || continue
                exception_message(sci, def_index, block, args[1])
            elseif @static isdefined(Core, :throw_methoderror) && func === Core.throw_methoderror
                method_error_message(sci, def_index, block, args)
            else
                continue
            end
            inst[:stmt] = ThrowNode(message, runtime)
            inst[:flag] = CC.flags_for_effects(CC.EFFECTS_THROWS)
        end
    end
    return sci
end
