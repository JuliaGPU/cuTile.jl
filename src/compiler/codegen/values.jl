# value emission and tracking

"""
    emit_value!(ctx, ref) -> Union{CGVal, Nothing}

Emit/resolve a value reference to a CGVal using multiple dispatch.
"""
function emit_value!(ctx::CGCtx, ssa::SSAValue)
    tv = ctx[ssa]
    tv !== nothing && return tv
    error("SSAValue %$(ssa.id) not found in context")
end
emit_value!(ctx::CGCtx, arg::Argument) = ctx[arg]
emit_value!(ctx::CGCtx, slot::SlotNumber) = ctx[slot]
emit_value!(ctx::CGCtx, block_arg::BlockArg) = ctx[block_arg]

function emit_value!(ctx::CGCtx, val::Integer)
    jltype = typeof(val)
    type_id = tile_type_for_julia!(ctx, jltype)
    bytes = reinterpret(UInt8, [jltype(val)])
    v = encode_ConstantOp!(ctx.cb, type_id, collect(bytes))
    CGVal(v, type_id, jltype, Int[], nothing, Some(val), nothing)
end

function emit_value!(ctx::CGCtx, val::AbstractFloat)
    jltype = typeof(val)
    type_id = tile_type_for_julia!(ctx, jltype)
    bytes = reinterpret(UInt8, [jltype(val)])
    v = encode_ConstantOp!(ctx.cb, type_id, collect(bytes))
    CGVal(v, type_id, jltype, Int[], nothing, Some(val), nothing)
end

function emit_value!(ctx::CGCtx, node::QuoteNode)
    emit_value!(ctx, node.value)
end

function emit_value!(ctx::CGCtx, expr::Expr)
    # Try to extract constant from Expr (e.g., call to tuple or type assert)
    if expr.head === :call
        callee = expr.args[1]
        # Handle Core.tuple - store component refs and extract constants if all available
        if callee isa GlobalRef && callee.mod === Core && callee.name === :tuple
            component_refs = collect(Any, expr.args[2:end])  # [SSAValue(3), SSAValue(4), ...]

            # Emit each component and collect types/constants
            component_types = Type[]
            component_constants = Any[]
            for ref in component_refs
                tv = emit_value!(ctx, ref)
                tv === nothing && return nothing
                push!(component_types, tv.jltype)
                push!(component_constants, tv.constant === nothing ? nothing : something(tv.constant))
            end

            jltype = Tuple{component_types...}
            return tuple_value(jltype, component_refs, component_constants)
        end
    end
    nothing
end

function emit_value!(ctx::CGCtx, ref::GlobalRef)
    val = getfield(ref.mod, ref.name)
    ghost_value(typeof(val), val)
end

function emit_value!(ctx::CGCtx, node::PiNode)
    # PiNode narrows the type. If the narrowed type is Type{T}, extract T
    if node.typ isa Type{<:Type}
        return ghost_value(node.typ, node.typ.parameters[1])
    end
    # Otherwise emit the underlying value
    emit_value!(ctx, node.val)
end

emit_value!(ctx::CGCtx, ::Nothing) = nothing

"""
    get_constant(ctx, ref) -> Union{Any, Nothing}

Get the compile-time constant from an IR reference or direct value.
Returns the value directly if it's not an IR reference.
"""
function get_constant(ctx::CGCtx, @nospecialize(ref))
    # Direct values (not IR references) - return as-is
    if !(ref isa SSAValue || ref isa Argument || ref isa SlotNumber ||
         ref isa Expr || ref isa GlobalRef || ref isa QuoteNode)
        return ref
    end
    # IR references - extract constant through emit_value!
    tv = emit_value!(ctx, ref)
    tv === nothing ? nothing : (tv.constant === nothing ? nothing : something(tv.constant))
end

# Symbols are compile-time only values
emit_value!(ctx::CGCtx, val::Symbol) = ghost_value(Symbol, val)

# Tuples are compile-time only values
emit_value!(ctx::CGCtx, val::Tuple) = ghost_value(typeof(val), val)

# Types are compile-time only values
emit_value!(ctx::CGCtx, @nospecialize(val::Type)) = ghost_value(Type{val}, val)

# Fallback for other types (constants embedded in IR)
function emit_value!(ctx::CGCtx, @nospecialize(val))
    T = typeof(val)
    # Handle Val{V} instances
    if T <: Val && length(T.parameters) == 1
        return ghost_value(T, T.parameters[1])
    end
    # Handle Constant{T, V} instances
    if T <: Constant && length(T.parameters) >= 2
        return ghost_value(T, T.parameters[2])
    end
    @warn "Unhandled value type in emit_value!" typeof(val)
    nothing
end


## constants

function emit_constant!(ctx::CGCtx, @nospecialize(value), @nospecialize(result_type))
    result_type_unwrapped = unwrap_type(result_type)

    # Ghost types have no runtime representation
    if is_ghost_type(result_type_unwrapped)
        return ghost_value(result_type_unwrapped)
    end

    # Skip non-primitive types
    if !(result_type_unwrapped <: Number || result_type_unwrapped === Bool)
        return nothing
    end

    type_id = tile_type_for_julia!(ctx, result_type_unwrapped)
    bytes = constant_to_bytes(value, result_type_unwrapped)
    v = encode_ConstantOp!(ctx.cb, type_id, bytes)

    CGVal(v, type_id, result_type_unwrapped)
end

function constant_to_bytes(@nospecialize(value), @nospecialize(T::Type))
    if T === Bool
        return UInt8[value ? 0xff : 0x00]
    elseif T === Int32 || T === UInt32
        return collect(reinterpret(UInt8, [Int32(value)]))
    elseif T === Int64 || T === UInt64
        return collect(reinterpret(UInt8, [Int64(value)]))
    elseif T === Float16
        return collect(reinterpret(UInt8, [Float16(value)]))
    elseif T === Float32
        return collect(reinterpret(UInt8, [Float32(value)]))
    elseif T === Float64
        return collect(reinterpret(UInt8, [Float64(value)]))
    else
        error("Cannot convert $T to constant bytes")
    end
end
