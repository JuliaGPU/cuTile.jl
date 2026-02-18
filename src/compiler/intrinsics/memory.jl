# Memory

# TODO: cuda_tile.join_tokens

# cuda_tile.load_ptr_tko
@intrinsic load_ptr_tko(ptrs, latency=nothing, mask=nothing, padding=nothing)
function tfunc(𝕃, ::typeof(Intrinsics.load_ptr_tko), @nospecialize(ptrs), @nospecialize args...)
    ptrs_type = CC.widenconst(ptrs)
    ptrs_type isa DataType && ptrs_type <: Tile || return nothing
    ptr_type = eltype(ptrs_type)
    ptr_type <: Ptr || return nothing
    T = eltype(ptr_type)
    S = ptrs_type.parameters[2]
    return Tile{T, S}
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.load_ptr_tko), args)
    cb = ctx.cb
    tt = ctx.tt

    # args: (ptrs, latency, mask?, padding?)
    # Get pointer tile (arg 1)
    ptrs_tv = emit_value!(ctx, args[1])
    ptrs_tv === nothing && throw(IRError("load_ptr_tko: cannot resolve pointer tile"))
    pointers = ptrs_tv.v
    tile_shape = ptrs_tv.shape

    # Get element type from pointer tile type (Tile{Ptr{T}, S})
    ptrs_type = CC.widenconst(ptrs_tv.jltype)
    ptr_type = eltype(ptrs_type)  # Ptr{T} from Tile{Ptr{T}, S}
    elem_type = eltype(ptr_type)  # T from Ptr{T}
    dtype = julia_to_tile_dtype!(tt, elem_type)
    result_tile_type = tile_type!(tt, dtype, tile_shape)
    token_type = Token(tt)

    # Extract latency hint (args[2])
    latency = @something get_constant(ctx, args[2]) throw(IRError("latency must be a compile-time constant"))

    optimization_hints = create_optimization_hints(ctx, latency)

    mask_tv, has_mask = emit_optional_mask(ctx, args, 3)

    # Get alias set use global token if unknown
    alias_set = get_alias_set(ctx, args[1])
    input_token = if alias_set isa AliasUniverse
        ctx.token
    else
        last_store_key_val = last_store_key(alias_set)
        first(get_input_token!(ctx, last_store_key_val, nothing))
    end

    if has_mask
        mask = mask_tv.v

        # Get padding tile (arg 4)
        padding_tv = emit_value!(ctx, args[4])
        padding_tv === nothing && throw(IRError("load_ptr_tko: cannot resolve padding tile"))
        padding = padding_tv.v

        # Load with mask and padding
        tile_val, new_token = encode_LoadPtrTkoOp!(
            cb, result_tile_type, token_type, pointers;
            mask = mask,
            padding_value = padding,
            token = input_token,
            optimization_hints
        )
    else
        # Load without mask
        tile_val, new_token = encode_LoadPtrTkoOp!(
            cb, result_tile_type, token_type, pointers;
            token = input_token,
            optimization_hints
        )
    end

    # Only track alias if we have a real alias set
    if alias_set isa AliasUniverse
        ctx.token = new_token
    else
        last_op_key_val = last_op_key(alias_set)
        last_op_token = get(ctx.token_map, last_op_key_val, nothing)
        if last_op_token === nothing || last_op_token === input_token || last_op_token === new_token
            new_last_op_token = new_token
        else
            new_last_op_token = encode_JoinTokensOp!(ctx.cb, token_type, [last_op_token, new_token])
        end
        ctx.token_map[last_op_key_val] = new_last_op_token
    end

    julia_shape = ColMajorShape(tile_shape)
    result_jltype = Tile{elem_type, TupleType(julia_shape)}
    return CGVal(tile_val, result_tile_type, result_jltype, tile_shape)
end

# TODO: cuda_tile.make_token

# cuda_tile.store_ptr_tko
@intrinsic store_ptr_tko(ptrs::Tile{Ptr{T}, S}, values::Tile{T, S},
                                   latency::Union{Int, Nothing},
                                   mask::Union{Tile{Bool, S}, Nothing}=nothing) where {T, S}
tfunc(𝕃, ::typeof(Intrinsics.store_ptr_tko), @nospecialize args...) = Nothing
efunc(::typeof(Intrinsics.store_ptr_tko), effects::CC.Effects) =
    CC.Effects(effects; effect_free=CC.ALWAYS_FALSE)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.store_ptr_tko), args)
    cb = ctx.cb
    tt = ctx.tt

    # args: (ptrs, values, latency, mask?)
    # Get pointer tile (arg 1)
    ptrs_tv = emit_value!(ctx, args[1])
    ptrs_tv === nothing && throw(IRError("store_ptr_tko: cannot resolve pointer tile"))
    pointers = ptrs_tv.v

    # Get value tile (arg 2)
    values_tv = emit_value!(ctx, args[2])
    values_tv === nothing && throw(IRError("store_ptr_tko: cannot resolve values tile"))
    values = values_tv.v

    token_type = Token(tt)

    latency = @something get_constant(ctx, args[3]) throw(IRError("latency must be a compile-time constant"))

    optimization_hints = create_optimization_hints(ctx, latency)

    mask_tv, has_mask = emit_optional_mask(ctx, args, 4)

    alias_set = get_alias_set(ctx, args[1])

    if alias_set isa AliasUniverse
        # Baseline behavior: use global token directly, no alias tracking overhead
        if has_mask
            mask = mask_tv.v

            # Store with mask
            new_token = encode_StorePtrTkoOp!(
                cb, token_type, pointers, values;
                mask = mask,
                token = ctx.token,
                optimization_hints
            )
        else
            new_token = encode_StorePtrTkoOp!(
                cb, token_type, pointers, values;
                token = ctx.token,
                optimization_hints
            )
        end
        ctx.token = new_token
    else
        last_op_key_val = last_op_key(alias_set)
        last_store_key_val = last_store_key(alias_set)

        # Store depends on LAST_OP (write after read/write)
        input_token, _ = get_input_token!(ctx, last_op_key_val, nothing)

        if has_mask
            mask = mask_tv.v

            new_token = encode_StorePtrTkoOp!(
                cb, token_type, pointers, values;
                mask = mask,
                token = input_token,
                optimization_hints
            )
        else
            new_token = encode_StorePtrTkoOp!(
                cb, token_type, pointers, values;
                token = input_token,
                optimization_hints
            )
        end

        # Update both LAST_OP and LAST_STORE.
        # Do NOT update ctx.token — alias-aware path uses token_map only.
        ctx.token_map[last_op_key_val] = new_token
        ctx.token_map[last_store_key_val] = new_token
    end

    nothing
end
