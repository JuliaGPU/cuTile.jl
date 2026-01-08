#-----------------------------------------------------------------------------
# Argument helpers
#-----------------------------------------------------------------------------

"""
    extract_argument_index(arg) -> Union{Int, Nothing}

Extract the raw argument index from a SlotNumber or Argument.
Returns the index that corresponds directly to `ir.argtypes[idx]`.
Note: index 1 is the function itself; user args start at index 2.
"""
function extract_argument_index(@nospecialize(arg))
    if arg isa SlotNumber
        return arg.id
    elseif arg isa Argument
        return arg.n
    end
    nothing
end

function resolve_or_constant(ctx::CGCtx, @nospecialize(arg), type_id::TypeId)
    tv = emit_value!(ctx, arg)
    # If we have a runtime value, use it
    tv.v !== nothing && return tv.v
    # Otherwise emit a constant from the compile-time value
    tv.constant === nothing && error("Cannot resolve argument")
    val = something(tv.constant)
    bytes = reinterpret(UInt8, [Int32(val)])
    encode_ConstantOp!(ctx.cb, type_id, collect(bytes))
end

#-----------------------------------------------------------------------------
# Tile helpers
#-----------------------------------------------------------------------------

"""
    extract_tile_shape(T) -> Vector{Int}

Extract shape from a Tile{T, Shape} type, returning Int[] if not a Tile type.
"""
function extract_tile_shape(@nospecialize(T))
    T = unwrap_type(T)
    if T <: Tile && length(T.parameters) >= 2
        shape = T.parameters[2]
        if shape isa Tuple
            return collect(Int, shape)
        end
    end
    Int[]
end
