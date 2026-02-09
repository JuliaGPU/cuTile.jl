# Memory

# TODO: cuda_tile.join_tokens

# cuda_tile.load_ptr_tko
@eval Intrinsics begin
    """
        load_ptr_tko(ptrs, latency, mask=nothing, padding=nothing)

    Load values from a tile of pointers.
    If mask is provided, masked-out positions return the padding value.
    Compiled to cuda_tile.load_ptr_tko.

    Note: TMA (allow_tma) is not applicable for pointer-based loads as they
    support irregular access patterns incompatible with TMA requirements.
    """
    @noinline function load_ptr_tko(ptrs::Tile{Ptr{T}, S},
                                     latency::Union{Int, Nothing}=nothing,
                                     mask::Union{Tile{Bool, S}, Nothing}=nothing,
                                     padding::Union{Tile{T, S}, Nothing}=nothing) where {T, S}
        Tile{T, S}()
    end
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
    latency = get_constant(ctx, args[2])

    # Create optimization hints if provided
    optimization_hints = create_optimization_hints(ctx, latency)

    # Check if mask is provided (arg 3 is not nothing)
    has_mask = length(args) >= 3 && get_constant(ctx, args[3]) !== nothing

    if has_mask
        # Get mask tile (arg 3)
        mask_tv = emit_value!(ctx, args[3])
        mask_tv === nothing && throw(IRError("load_ptr_tko: cannot resolve mask tile"))
        mask = mask_tv.v

        # Get padding tile (arg 4)
        padding_tv = emit_value!(ctx, args[4])
        padding_tv === nothing && throw(IRError("load_ptr_tko: cannot resolve padding tile"))
        padding = padding_tv.v

        # Load with mask and padding
        tile_val, new_token = encode_LoadPtrTkoOp!(cb, result_tile_type, token_type, pointers;
                                                    mask=mask,
                                                    padding_value=padding,
                                                    token=ctx.token,
                                                    optimization_hints)
    else
        # Load without mask
        tile_val, new_token = encode_LoadPtrTkoOp!(cb, result_tile_type, token_type, pointers;
                                                    token=ctx.token,
                                                    optimization_hints)
    end
    ctx.token = new_token

    result_jltype = Tile{elem_type, Tuple{tile_shape...}}
    CGVal(tile_val, result_tile_type, result_jltype, tile_shape)
end

# TODO: cuda_tile.make_token

# cuda_tile.store_ptr_tko
@eval Intrinsics begin
    """
        store_ptr_tko(ptrs, values, latency, mask=nothing)

    Store values to a tile of pointers.
    If mask is provided, masked-out positions are not written.
    Compiled to cuda_tile.store_ptr_tko.

    Note: TMA (allow_tma) is not applicable for pointer-based stores as they
    support irregular access patterns incompatible with TMA requirements.
    """
    @noinline function store_ptr_tko(ptrs::Tile{Ptr{T}, S}, values::Tile{T, S},
                                      latency::Union{Int, Nothing},
                                      mask::Union{Tile{Bool, S}, Nothing}=nothing) where {T, S}
        donotdelete()
        nothing
    end
end
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

    # Extract latency hint (args[3])
    latency = get_constant(ctx, args[3])

    # Create optimization hints if provided
    optimization_hints = create_optimization_hints(ctx, latency)

    # Check if mask is provided (arg 4 is not nothing)
    has_mask = length(args) >= 4 && get_constant(ctx, args[4]) !== nothing

    if has_mask
        # Get mask tile (arg 4)
        mask_tv = emit_value!(ctx, args[4])
        mask_tv === nothing && throw(IRError("store_ptr_tko: cannot resolve mask tile"))
        mask = mask_tv.v

        # Store with mask
        new_token = encode_StorePtrTkoOp!(cb, token_type, pointers, values;
                                           mask=mask,
                                           token=ctx.token,
                                           optimization_hints)
    else
        # Store without mask
        new_token = encode_StorePtrTkoOp!(cb, token_type, pointers, values;
                                           token=ctx.token,
                                           optimization_hints)
    end
    ctx.token = new_token

    nothing
end
