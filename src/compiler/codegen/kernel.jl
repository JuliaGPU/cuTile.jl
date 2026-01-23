# kernel and argument handling

"""
    emit_kernel!(writer, func_buf, sci, rettype; name, sm_arch=nothing, is_entry=true, num_ctas=nothing, occupancy=nothing)

Compile a StructuredIRCode to Tile IR bytecode.
"""
function emit_kernel!(writer::BytecodeWriter, func_buf::Vector{UInt8},
                      sci::StructuredIRCode, rettype::Type;
                      name::String,
                      sm_arch::Union{String, Nothing} = nothing,
                      is_entry::Bool = true,
                      num_ctas::Union{Int, Nothing} = nothing,
                      occupancy::Union{Int, Nothing} = nothing)
    ctx = CGCtx(writer, sci, sm_arch)
    tt = ctx.tt

    # Validate non-ghost argument types are concrete
    for (i, argtype) in enumerate(sci.argtypes)
        is_ghost_type(unwrap_type(argtype)) && continue
        require_concrete_type(argtype, "kernel argument $i")
    end

    # Build parameter list, handling ghost types and struct destructuring
    param_types = TypeId[]
    param_mapping = Tuple{Int, Union{Nothing, Symbol}}[]

    for (i, argtype) in enumerate(sci.argtypes)
        argtype_unwrapped = unwrap_type(argtype)
        if is_ghost_type(argtype_unwrapped)
            continue
        elseif should_destructure(argtype_unwrapped)
            # Destructure TileArray into flat parameters
            params = argtype_unwrapped.parameters
            ndims = params[2]::Integer
            for fi in 1:fieldcount(argtype_unwrapped)
                fname = fieldname(argtype_unwrapped, fi)
                ftype = fieldtype(argtype_unwrapped, fi)
                if fname === :sizes || fname === :strides
                    fcount = ndims
                    elem_type = Int32
                else
                    fcount = flat_field_count(ftype)
                    elem_type = ftype <: Ptr ? Ptr{params[1]} : (ftype <: Tuple ? eltype(ftype) : ftype)
                end
                for _ in 1:fcount
                    push!(param_types, tile_type_for_julia!(ctx, elem_type))
                    push!(param_mapping, (i, fname))
                end
            end
            ctx.arg_types[i] = argtype_unwrapped
        else
            push!(param_types, tile_type_for_julia!(ctx, argtype_unwrapped))
            push!(param_mapping, (i, nothing))
        end
    end

    # Return types
    result_types = TypeId[]
    if rettype !== Nothing && rettype !== Union{}
        push!(result_types, tile_type_for_julia!(ctx, rettype))
    end

    # Create entry hints if provided
    entry_hints = encode_entry_hints(writer, sm_arch, EntryHints(; num_ctas, occupancy))

    # Create function
    cb = add_function!(writer, func_buf, name, param_types, result_types;
                       is_entry, entry_hints)
    ctx.cb = cb

    # Set up argument values
    arg_values = make_block_args!(cb, length(param_types))

    # Build arg_flat_values map
    field_values = Dict{Tuple{Int, Union{Nothing, Symbol}}, Vector{Value}}()
    for (param_idx, val) in enumerate(arg_values)
        key = param_mapping[param_idx]
        if !haskey(field_values, key)
            field_values[key] = Value[]
        end
        push!(field_values[key], val)
    end

    # Store in context and set up slot/argument mappings
    # arg_idx is the direct index into argtypes (2, 3, ...) which matches SlotNumber/Argument
    for (key, values) in field_values
        arg_idx, field = key
        ctx.arg_flat_values[key] = values

        if field === nothing
            # Regular argument - create concrete CGVal
            @assert length(values) == 1
            val = values[1]
            type_id = tile_type_for_julia!(ctx, sci.argtypes[arg_idx])
            tv = CGVal(val, type_id, sci.argtypes[arg_idx])
            ctx[SlotNumber(arg_idx)] = tv
            ctx[Argument(arg_idx)] = tv
        end
    end

    # For destructured args, create lazy CGVals that track the argument index
    for (arg_idx, argtype) in ctx.arg_types
        tv = arg_ref_value(arg_idx, Union{Symbol, Int}[], argtype)
        ctx[SlotNumber(arg_idx)] = tv
        ctx[Argument(arg_idx)] = tv
    end

    # Create TensorViews for all TileArray arguments at kernel entry
    for (arg_idx, _) in ctx.arg_types
        cache_tensor_view!(ctx, arg_idx)
    end

    # Create memory ordering token
    token_type = Token(tt)
    ctx.token_type = token_type
    ctx.token = encode_MakeTokenOp!(cb, token_type)

    # Emit the structured IR (uses original Julia SSA indices everywhere)
    emit_block!(ctx, ctx.sci.entry)

    finalize_function!(func_buf, cb, writer.debug_info)
end

# getfield for destructured arguments (lazy chain extension)
function emit_getfield!(ctx::CGCtx, args, @nospecialize(result_type))
    length(args) >= 2 || return nothing

    # special case: multi-valued loops rely on getfield to extract values
    tv = emit_loop_getfield!(ctx, args)
    tv !== nothing && return tv

    obj_arg = args[1]
    field_arg = args[2]

    # Extract field name or index
    field = get_constant(ctx, field_arg)

    # Try to get the object as a CGVal
    obj_tv = emit_value!(ctx, obj_arg)

    # If obj is a lazy arg_ref, extend the chain
    if obj_tv !== nothing && is_arg_ref(obj_tv)
        arg_idx, chain = obj_tv.arg_ref

        if field isa Symbol
            # Field access: extend chain with symbol
            new_chain = Union{Symbol, Int}[chain..., field]
            # Check if this resolves to a scalar field (auto-materialize leaf)
            # Don't auto-materialize tuple types - they need indexing first
            rt = unwrap_type(result_type)
            if !(rt <: Tuple)
                values = get_arg_flat_values(ctx, arg_idx, field)
                if values !== nothing && length(values) == 1
                    # Scalar field - materialize immediately
                    type_id = tile_type_for_julia!(ctx, rt)
                    return CGVal(values[1], type_id, rt)
                end
            end
            return arg_ref_value(arg_idx, new_chain, rt)
        elseif field isa Integer && !isempty(chain) && chain[end] isa Symbol
            # Tuple indexing: chain ends with field name, now indexing into it
            # This is a leaf - materialize immediately
            field_name = chain[end]
            values = get_arg_flat_values(ctx, arg_idx, field_name)
            if values !== nothing && 1 <= field <= length(values)
                type_id = tile_type_for_julia!(ctx, unwrap_type(result_type))
                return CGVal(values[field], type_id, unwrap_type(result_type))
            end
        end
    end

    nothing
end

# getindex for tuple field access (lazy chain extension)
function emit_getindex!(ctx::CGCtx, args, @nospecialize(result_type))
    length(args) >= 2 || return nothing

    obj_arg = args[1]
    index_arg = args[2]

    # Extract constant index
    index = get_constant(ctx, index_arg)
    index isa Integer || return nothing

    # Try to get the object as a CGVal
    obj_tv = emit_value!(ctx, obj_arg)
    obj_tv === nothing && return nothing

    # If obj is a lazy arg_ref, try to materialize or extend the chain
    if is_arg_ref(obj_tv)
        arg_idx, chain = obj_tv.arg_ref

        # If chain ends with a symbol (field name), we're indexing into a tuple field
        # Try to materialize immediately
        if !isempty(chain) && chain[end] isa Symbol
            field_name = chain[end]
            values = get_arg_flat_values(ctx, arg_idx, field_name)
            if values !== nothing && 1 <= index <= length(values)
                type_id = tile_type_for_julia!(ctx, unwrap_type(result_type))
                return CGVal(values[index], type_id, unwrap_type(result_type))
            end
        end

        # Otherwise extend the chain
        new_chain = Union{Symbol, Int}[chain..., Int(index)]
        return arg_ref_value(arg_idx, new_chain, unwrap_type(result_type))
    end

    # Not an arg_ref - not handled here
    nothing
end
