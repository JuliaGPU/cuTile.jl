#=============================================================================
 8.6. Memory
 cuda_tile.join_tokens, cuda_tile.load_ptr_tko, cuda_tile.make_token,
 cuda_tile.store_ptr_tko
=============================================================================#


## TODO: cuda_tile.join_tokens


## cuda_tile.load_ptr_tko

@eval Intrinsics begin
    """
        load_ptr_tko(ptrs, mask=nothing, padding=nothing)

    Load values from a tile of pointers.
    If mask is provided, masked-out positions return the padding value.
    Compiled to cuda_tile.load_ptr_tko.
    """
    @noinline function load_ptr_tko(ptrs::Tile{Ptr{T}, S},
                                     mask::Union{Tile{Bool, S}, Nothing}=nothing,
                                     padding::Union{Tile{T, S}, Nothing}=nothing) where {T, S}
        Base.donotdelete(ptrs, mask, padding)
        Tile{T, S}()
    end
end

function emit_intrinsic!(ctx::CodegenContext, ::typeof(Intrinsics.load_ptr_tko), args, @nospecialize(result_type))
    cb = ctx.cb
    tt = ctx.tt

    # Get pointer tile (arg 1)
    ptrs_tv = emit_value!(ctx, args[1])
    ptrs_tv === nothing && error("load_ptr_tko: cannot resolve pointer tile")
    pointers = ptrs_tv.v
    tile_shape = ptrs_tv.shape

    # Get element type from result_type (Tile{T, S})
    result_type_unwrapped = unwrap_type(result_type)
    elem_type = result_type_unwrapped.parameters[1]
    dtype = julia_to_tile_dtype!(tt, elem_type)
    result_tile_type = tile_type!(tt, dtype, tile_shape)
    token_type = Token(tt)

    # Check if mask is provided (arg 2 is not nothing)
    has_mask = length(args) >= 2 && get_constant(ctx, args[2]) !== nothing

    if has_mask
        # Get mask tile (arg 2)
        mask_tv = emit_value!(ctx, args[2])
        mask_tv === nothing && error("load_ptr_tko: cannot resolve mask tile")
        mask = mask_tv.v

        # Get padding tile (arg 3)
        padding_tv = emit_value!(ctx, args[3])
        padding_tv === nothing && error("load_ptr_tko: cannot resolve padding tile")
        padding = padding_tv.v

        # Load with mask and padding
        tile_val, new_token = encode_LoadPtrTkoOp!(cb, result_tile_type, token_type, pointers;
                                                    mask=mask,
                                                    padding_value=padding,
                                                    token=ctx.token)
    else
        # Load without mask
        tile_val, new_token = encode_LoadPtrTkoOp!(cb, result_tile_type, token_type, pointers;
                                                    token=ctx.token)
    end
    ctx.token = new_token

    CGVal(tile_val, result_tile_type, result_type_unwrapped, tile_shape)
end


## TODO: cuda_tile.make_token


## cuda_tile.store_ptr_tko

@eval Intrinsics begin
    """
        store_ptr_tko(ptrs, values, mask=nothing)

    Store values to a tile of pointers.
    If mask is provided, masked-out positions are not written.
    Compiled to cuda_tile.store_ptr_tko.
    """
    @noinline function store_ptr_tko(ptrs::Tile{Ptr{T}, S}, values::Tile{T, S},
                                      mask::Union{Tile{Bool, S}, Nothing}=nothing) where {T, S}
        Base.donotdelete(ptrs, values, mask)
        nothing
    end
end

function emit_intrinsic!(ctx::CodegenContext, ::typeof(Intrinsics.store_ptr_tko), args, @nospecialize(result_type))
    cb = ctx.cb
    tt = ctx.tt

    # Get pointer tile (arg 1)
    ptrs_tv = emit_value!(ctx, args[1])
    ptrs_tv === nothing && error("store_ptr_tko: cannot resolve pointer tile")
    pointers = ptrs_tv.v

    # Get value tile (arg 2)
    values_tv = emit_value!(ctx, args[2])
    values_tv === nothing && error("store_ptr_tko: cannot resolve values tile")
    values = values_tv.v

    token_type = Token(tt)

    # Check if mask is provided (arg 3 is not nothing)
    has_mask = length(args) >= 3 && get_constant(ctx, args[3]) !== nothing

    if has_mask
        # Get mask tile (arg 3)
        mask_tv = emit_value!(ctx, args[3])
        mask_tv === nothing && error("store_ptr_tko: cannot resolve mask tile")
        mask = mask_tv.v

        # Store with mask
        new_token = encode_StorePtrTkoOp!(cb, token_type, pointers, values;
                                           mask=mask,
                                           token=ctx.token)
    else
        # Store without mask
        new_token = encode_StorePtrTkoOp!(cb, token_type, pointers, values;
                                           token=ctx.token)
    end
    ctx.token = new_token

    nothing
end
