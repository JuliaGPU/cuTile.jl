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
                      sm_arch::Union{VersionNumber, Nothing} = nothing,
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
    param_mapping = Tuple{Int, Vector{Int}}[]

    for (i, argtype) in enumerate(sci.argtypes)
        argtype_unwrapped = CC.widenconst(argtype)
        if is_ghost_type(argtype_unwrapped)
            # No kernel parameter, but register a ghost CGVal so codegen
            # can resolve the value (e.g. get_constant on function singletons)
            tv = ghost_value(argtype_unwrapped,
                             Base.issingletontype(argtype_unwrapped) ?
                                 argtype_unwrapped.instance : nothing)
            ctx[SlotNumber(i)] = tv
            ctx[Argument(i)] = tv
            continue
        elseif is_const_arg(i)
            continue  # const arg: no kernel parameter
        elseif isprimitivetype(argtype_unwrapped)
            push!(param_types, tile_type_for_julia!(ctx, argtype_unwrapped))
            push!(param_mapping, (i, Int[]))
        else
            flatten_struct_params!(ctx, param_types, param_mapping, i, argtype_unwrapped, Int[])
            ctx.arg_types[i] = argtype_unwrapped
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
    field_values = Dict{Tuple{Int, Vector{Int}}, Vector{Value}}()
    for (param_idx, val) in enumerate(arg_values)
        key = param_mapping[param_idx]
        if !haskey(field_values, key)
            field_values[key] = Value[]
        end
        push!(field_values[key], val)
    end

    # Store in context and set up slot/argument mappings
    for (key, values) in field_values
        arg_idx, path = key
        ctx.arg_flat_values[key] = values

        if isempty(path) && !haskey(ctx.arg_types, arg_idx)
            # Regular (non-destructured) argument - create concrete CGVal
            if length(values) != 1
                throw(IRError("Expected exactly one value for argument $arg_idx, got $(length(values))"))
            end
            val = values[1]
            argtype = CC.widenconst(sci.argtypes[arg_idx])
            type_id = tile_type_for_julia!(ctx, argtype)
            # Promote scalar kernel args to 0D tile jltype at the boundary.
            # sci.argtypes retains the Julia signature (Int32), but the IR
            # body is uniformly tile-typed after scalar_elim_pass!.
            jltype = argtype <: Number ? Tile{argtype, Tuple{}} : sci.argtypes[arg_idx]
            tv = CGVal(val, type_id, jltype)
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
                # Primitive: emit ConstantOp (jltype promoted to 0D tile)
                bytes = constant_to_bytes(val, T)
                v = encode_ConstantOp!(ctx.cb, type_id, bytes)
                tv = CGVal(v, type_id, Tile{T, Tuple{}}, ScalarShape(), nothing, Some(val), nothing)
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
        tv = arg_ref_value(arg_idx, Int[], argtype)
        ctx[SlotNumber(arg_idx)] = tv
        ctx[Argument(arg_idx)] = tv
    end

    # Create TensorViews for all TileArray arguments at kernel entry
    for (arg_idx, argtype) in ctx.arg_types
        create_tensor_views!(ctx, arg_idx, argtype, Int[])
    end

    # Hoist early returns BEFORE token ordering — hoist_returns! rewrites
    # ReturnNode terminators to YieldOp, which the token pass then extends.
    hoist_returns!(ctx.sci.entry)

    # Run the pass pipeline (normalize, optimize, token ordering, DCE).
    run_passes!(sci)

    # Cache the token bytecode type for codegen
    ctx.token_type = Token(tt)

    # Emit the structured IR (uses original Julia SSA indices everywhere)
    emit_block!(ctx, ctx.sci.entry)

    finalize_function!(func_buf, cb, writer.debug_info)
end

"""
    create_tensor_views!(ctx, arg_idx, T, path)

Walk the type tree and create TensorViews for all nested TileArrays.
"""
function create_tensor_views!(ctx::CGCtx, arg_idx::Int, @nospecialize(T), path::Vector{Int})
    if T <: TileArray
        cache_tensor_view!(ctx, arg_idx, path, T)
    else
        for fi in 1:fieldcount(T)
            ftype = fieldtype(T, fi)
            (is_ghost_type(ftype) || isprimitivetype(ftype)) && continue
            field_path = [path..., fi]
            create_tensor_views!(ctx, arg_idx, ftype, field_path)
        end
    end
end

"""
    flatten_struct_params!(ctx, param_types, param_mapping, arg_idx, T, path)

Recursively flatten a struct type into kernel parameters.
"""
function flatten_struct_params!(ctx, param_types, param_mapping, arg_idx, @nospecialize(T), path::Vector{Int})
    for fi in 1:fieldcount(T)
        ftype = fieldtype(T, fi)
        field_path = [path..., fi]
        if is_ghost_type(ftype)
            continue
        elseif isprimitivetype(ftype)
            type_id = tile_type_for_julia!(ctx, ftype; throw_error=false)
            type_id === nothing && continue
            push!(param_types, type_id)
            push!(param_mapping, (arg_idx, field_path))
        else
            flatten_struct_params!(ctx, param_types, param_mapping, arg_idx, ftype, field_path)
        end
    end
end

# getfield for destructured arguments (lazy chain extension)
function emit_getfield!(ctx::CGCtx, args, @nospecialize(result_type))
    length(args) >= 2 || return nothing

    # special case: multi-valued loops rely on getfield to extract values
    tv = emit_loop_getfield!(ctx, args)
    tv !== nothing && return tv

    obj_arg = args[1]
    field_arg = args[2]

    field = @something get_constant(ctx, field_arg) return nothing

    obj_tv = emit_value!(ctx, obj_arg)

    # Tuple indexing: extract component by integer index
    if obj_tv !== nothing && obj_tv.tuple !== nothing && field isa Integer
        return emit_value!(ctx, obj_tv.tuple[field])
    end

    # If obj is a lazy arg_ref, extend the chain
    if obj_tv !== nothing && is_arg_ref(obj_tv)
        arg_idx, chain = obj_tv.arg_ref

        # Convert field to integer index
        idx = if field isa Symbol
            obj_type = CC.widenconst(obj_tv.jltype)
            Base.fieldindex(obj_type, field)
        elseif field isa Integer
            Int(field)
        else
            nothing
        end
        idx === nothing && return nothing

        return resolve_arg_ref(ctx, arg_idx, chain, idx, CC.widenconst(result_type))
    end

    nothing
end

# getindex for tuple field access (lazy chain extension)
function emit_getindex!(ctx::CGCtx, args, @nospecialize(result_type))
    length(args) >= 2 || return nothing

    obj_arg = args[1]
    index_arg = args[2]

    index = @something get_constant(ctx, index_arg) return nothing
    index isa Integer || return nothing

    # Try to get the object as a CGVal
    obj_tv = emit_value!(ctx, obj_arg)
    obj_tv === nothing && return nothing

    # If obj is a lazy arg_ref, extend the chain with the index
    if is_arg_ref(obj_tv)
        arg_idx, chain = obj_tv.arg_ref
        return resolve_arg_ref(ctx, arg_idx, chain, Int(index), CC.widenconst(result_type))
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
    F = typeof(func)
    if !is_ghost_type(F)
        throw(IRError("emit_subprogram!: function argument $(F) (sizeof=$(sizeof(F))) is not " *
                      "a zero-size type. All non-tile arguments must be zero-size."))
    end

    # 1. Resolve method instance
    argtuple = Tuple{arg_types...}
    world = ctx.cache.world
    mi = lookup_method_instance(func, argtuple; world)

    # 2. Compile through cuTile pipeline (cached)
    if !haskey(ctx.cache, mi)
        error("Expected $func($(join(arg_types, ", "))) to be cached already by inference.")
    end
    # Suppress compile_hook to avoid @device_code_tiled treating
    # region bodies (e.g. reduce combiners) as standalone entries.
    old_hook = compile_hook[]
    compile_hook[] = nothing
    sci, _, _ = try
        emit_structured!(ctx.cache, mi)
    finally
        compile_hook[] = old_hook
    end

    # 2b. Run the pass pipeline on subprogram IR
    run_passes!(sci)

    # 3. Create sub-context
    sub_ctx = CGCtx(; ctx.cb, ctx.tt, sci,
                      ctx.token_type,
                      ctx.type_cache, ctx.sm_arch,
                      ctx.cache)

    # 4. Map arguments dynamically: ghost args get ghost_value, non-ghost args
    #    consume block_args sequentially.
    n_argtypes = length(sci.argtypes)
    block_idx = 1  # cursor into block_args

    # Helper to promote scalar arg types to 0D tile at the boundary.
    # sci.argtypes retains the Julia signature; the IR body is tile-typed.
    _arg_jltype(T) = let U = CC.widenconst(T); U <: Number ? Tile{U, Tuple{}} : T end

    if mi.def.isva
        # Varargs: fixed argtypes are 1:n_argtypes-1, last is the varargs tuple.
        # Map fixed args (ghost or non-ghost), then pack remaining block_args
        # into a tuple CGVal for the varargs argument.
        for i in 1:(n_argtypes - 1)
            argtype = sci.argtypes[i]
            if is_ghost_type(CC.widenconst(argtype))
                sub_ctx[Argument(i)] = ghost_value(argtype)
            else
                sub_ctx[Argument(i)] = CGVal(block_args[block_idx], block_type_ids[block_idx], _arg_jltype(arg_types[block_idx]))
                block_idx += 1
            end
        end
        # Pack remaining block_args into a virtual tuple for the varargs argument
        va_offset = n_argtypes + length(block_args)  # high indices to avoid collision
        tuple_components = Any[]
        for j in block_idx:length(block_args)
            sub_ctx[Argument(va_offset + j - block_idx + 1)] = CGVal(block_args[j], block_type_ids[j], _arg_jltype(arg_types[j]))
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
                sub_ctx[Argument(i)] = CGVal(block_args[block_idx], block_type_ids[block_idx], _arg_jltype(arg_types[block_idx]))
                block_idx += 1
            end
        end
    end

    # 5. Emit body (skip terminator — we yield manually)
    emit_block!(sub_ctx, sci.entry; skip_terminator=true)

    # 6. Extract return value and yield
    ret = terminator(sci.entry)::ReturnNode
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
