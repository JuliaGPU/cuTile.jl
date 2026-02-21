# kernel and argument handling

"""
    emit_kernel!(writer, func_buf, sci, rettype; name, sm_arch=nothing, is_entry=true, num_ctas=nothing, occupancy=nothing, const_argtypes=nothing)

Compile a StructuredIRCode to Tile IR bytecode.

When `const_argtypes` is provided, arguments with `CC.Const` entries are treated
as compile-time constants: no kernel parameter is generated and a ConstantOp is
emitted instead. The `const_argtypes` vector is 1-indexed matching `sci.argtypes`
(index 1 = function itself, user args from index 2).
"""
function emit_kernel!(writer::BytecodeWriter, func_buf::Vector{UInt8},
                      sci::StructuredIRCode, rettype::Type;
                      name::String,
                      sm_arch::Union{String, Nothing} = nothing,
                      is_entry::Bool = true,
                      num_ctas::Union{Int, Nothing} = nothing,
                      occupancy::Union{Int, Nothing} = nothing,
                      cache::CacheView,
                      const_argtypes::Union{Vector{Any}, Nothing} = nothing)
    tt = writer.type_table
    cb = CodeBuilder(writer.string_table, writer.constant_table, tt)
    ctx = CGCtx(; cb, tt, sci, sm_arch, cache)

    # Determine which argument positions are const-seeded
    # const_argtypes is 1-indexed: [Const(f), arg2, arg3, ...]
    # sci.argtypes is also 1-indexed: [f_type, arg2_type, arg3_type, ...]
    is_const_arg(i) = const_argtypes !== nothing && i <= length(const_argtypes) &&
                      const_argtypes[i] isa CC.Const

    # Validate non-ghost, non-const argument types are concrete
    for (i, argtype) in enumerate(sci.argtypes)
        is_ghost_type(CC.widenconst(argtype)) && continue
        is_const_arg(i) && continue
        require_concrete_type(argtype, "kernel argument $i")
    end

    # Build parameter list, handling ghost types, const args, and struct destructuring
    param_types = TypeId[]
    param_mapping = Tuple{Int, Union{Nothing, Symbol}}[]

    for (i, argtype) in enumerate(sci.argtypes)
        argtype_unwrapped = CC.widenconst(argtype)
        if is_ghost_type(argtype_unwrapped)
            continue
        elseif is_const_arg(i)
            continue  # const arg: no kernel parameter
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
            if length(values) != 1
                throw(IRError("Expected exactly one value for argument $arg_idx, got $(length(values))"))
            end
            val = values[1]
            type_id = tile_type_for_julia!(ctx, sci.argtypes[arg_idx])
            tv = CGVal(val, type_id, sci.argtypes[arg_idx])
            ctx[SlotNumber(arg_idx)] = tv
            ctx[Argument(arg_idx)] = tv
        end
    end

    # Emit ConstantOps for const-seeded arguments (no kernel parameter)
    if const_argtypes !== nothing
        for (i, cat) in enumerate(const_argtypes)
            cat isa CC.Const || continue
            i > length(sci.argtypes) && continue
            val = cat.val
            T = typeof(val)
            type_id = tile_type_for_julia!(ctx, T; throw_error=false)
            if type_id !== nothing
                # Scalar: emit ConstantOp
                bytes = constant_to_bytes(val, T)
                v = encode_ConstantOp!(ctx.cb, type_id, bytes)
                tv = CGVal(v, type_id, T, Int[], nothing, Some(val), nothing)
            else
                # Non-primitive (tuple etc.): ghost with constant
                tv = ghost_value(T, val)
            end
            ctx[SlotNumber(i)] = tv
            ctx[Argument(i)] = tv
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

    # Hoist early returns out of IfOp regions (tileiras rejects ReturnOp inside IfOp)
    hoist_returns!(ctx.sci.entry)

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

    # Tuple indexing: extract component by integer index
    if obj_tv !== nothing && obj_tv.tuple !== nothing && field isa Integer
        return emit_value!(ctx, obj_tv.tuple[field])
    end

    # If obj is a lazy arg_ref, extend the chain
    if obj_tv !== nothing && is_arg_ref(obj_tv)
        arg_idx, chain = obj_tv.arg_ref

        if field isa Symbol
            # Field access: extend chain with symbol
            new_chain = Union{Symbol, Int}[chain..., field]
            # Check if this resolves to a scalar field (auto-materialize leaf)
            # Don't auto-materialize tuple types - they need indexing first
            rt = CC.widenconst(result_type)
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
                type_id = tile_type_for_julia!(ctx, CC.widenconst(result_type))
                return CGVal(values[field], type_id, CC.widenconst(result_type))
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
                type_id = tile_type_for_julia!(ctx, CC.widenconst(result_type))
                return CGVal(values[index], type_id, CC.widenconst(result_type))
            end
        end

        # Otherwise extend the chain
        new_chain = Union{Symbol, Int}[chain..., Int(index)]
        return arg_ref_value(arg_idx, new_chain, CC.widenconst(result_type))
    end

    # Not an arg_ref - not handled here
    nothing
end


#=============================================================================
 Subprogram compilation
=============================================================================#

"""
    emit_subprogram!(ctx, func, arg_types, block_args, block_type_ids) -> Vector{Value}

Compile a Julia function into the current region body. Resolves `func` via the cuTile
pipeline (method_instance → code_ircode → StructuredIRCode), creates a sub-context,
maps `block_args` to the function's positional arguments, emits the body, and returns
the yielded result values.

- `func`: the Julia function to compile (e.g., `+`, `max`, a lambda)
- `arg_types`: Julia types for each block arg (e.g., `[Tile{Float32,()}]` repeated)
- `block_args`: IR `Value`s from the enclosing region (e.g., `[acc, elem]`)
- `block_type_ids`: `TypeId`s corresponding to each block arg

A `YieldOp` is emitted with the return value(s).
"""
function emit_subprogram!(ctx::CGCtx, func, arg_types::Vector,
                          block_args::Vector{Value}, block_type_ids::Vector{TypeId})
    # 1. Resolve method instance
    argtuple = Tuple{arg_types...}
    world = ctx.cache.world
    mi = @something(
        method_instance(func, argtuple;
                        world, method_table=cuTileMethodTable),
        method_instance(func, argtuple; world),
        error("No method found for $func($(join(arg_types, ", ")))")
    )

    # 2. Compile through cuTile pipeline (cached)
    if !haskey(ctx.cache, mi)
        error("Expected $func($(join(arg_types, ", "))) to be cached already by inference.")
    end
    sci, _ = emit_ir(ctx.cache, mi)

    # 3. Create sub-context
    sub_ctx = CGCtx(; ctx.cb, ctx.tt, sci,
                      ctx.token, ctx.token_type,
                      ctx.type_cache, ctx.sm_arch,
                      ctx.cache)

    # 4. Map arguments dynamically: ghost args get ghost_value, non-ghost args
    #    consume block_args sequentially.
    n_argtypes = length(sci.argtypes)
    block_idx = 1  # cursor into block_args

    if mi.def.isva
        # Varargs: fixed argtypes are 1:n_argtypes-1, last is the varargs tuple.
        # Map fixed args (ghost or non-ghost), then pack remaining block_args
        # into a tuple CGVal for the varargs argument.
        for i in 1:(n_argtypes - 1)
            argtype = sci.argtypes[i]
            if is_ghost_type(CC.widenconst(argtype))
                sub_ctx[Argument(i)] = ghost_value(argtype)
            else
                sub_ctx[Argument(i)] = CGVal(block_args[block_idx], block_type_ids[block_idx], arg_types[block_idx])
                block_idx += 1
            end
        end
        # Pack remaining block_args into a virtual tuple for the varargs argument
        va_offset = n_argtypes + length(block_args)  # high indices to avoid collision
        tuple_components = Any[]
        for j in block_idx:length(block_args)
            sub_ctx[Argument(va_offset + j - block_idx + 1)] = CGVal(block_args[j], block_type_ids[j], arg_types[j])
            push!(tuple_components, Argument(va_offset + j - block_idx + 1))
        end
        constants = Vector{Any}(fill(nothing, length(tuple_components)))
        sub_ctx[Argument(n_argtypes)] = tuple_value(sci.argtypes[end], tuple_components, constants)
    else
        for i in 1:n_argtypes
            argtype = sci.argtypes[i]
            if is_ghost_type(CC.widenconst(argtype))
                sub_ctx[Argument(i)] = ghost_value(argtype)
            else
                sub_ctx[Argument(i)] = CGVal(block_args[block_idx], block_type_ids[block_idx], arg_types[block_idx])
                block_idx += 1
            end
        end
    end

    # 5. Emit body (skip terminator — we yield manually)
    emit_block!(sub_ctx, sci.entry; skip_terminator=true)

    # 6. Extract return value and yield
    ret = sci.entry.terminator::ReturnNode
    tv = emit_value!(sub_ctx, ret.val)
    if tv.tuple !== nothing
        # Tuple return: resolve each component to a concrete Value
        results = Value[]
        for ref in tv.tuple
            component = emit_value!(sub_ctx, ref)
            component === nothing && throw(IRError("Cannot resolve tuple component in subprogram return"))
            push!(results, component.v::Value)
        end
    else
        results = tv.v isa Vector ? tv.v : [tv.v]
    end
    encode_YieldOp!(ctx.cb, results)
    return results
end
