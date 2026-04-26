# Memory

"""
    Intrinsics.load_ptr_tko(ptrs::Tile{Ptr{T},S},
                            latency::Union{Int,Nothing}=nothing,
                            mask::Union{Tile{Bool,S},Nothing}=nothing,
                            padding::Union{Tile{T,S},Nothing}=nothing) -> Tile{T,S}

Token-ordered gather load from a tile of pointers; lowers to
`cuda_tile.load_ptr_tko`.

`latency` is a compile-time hint (1–10 or `nothing`). The token argument
is appended by `token_order_pass!` and is not part of the user-visible
signature.
"""
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

    # Extract input token from last arg (added by token_order_pass!)
    input_token = extract_token_arg!(ctx, args)

    # args: (ptrs, latency, mask?, padding?)
    ptrs_tv = emit_value!(ctx, args[1])
    ptrs_tv === nothing && throw(IRError("load_ptr_tko: cannot resolve pointer tile"))
    pointers = ptrs_tv.v
    tile_shape = ptrs_tv.shape

    ptrs_type = CC.widenconst(ptrs_tv.jltype)
    ptr_type = eltype(ptrs_type)
    elem_type = eltype(ptr_type)
    dtype = julia_to_tile_dtype!(tt, elem_type)
    result_tile_type = tile_type!(tt, dtype, tile_shape)
    token_type = Token(tt)

    latency = @something get_constant(ctx, args[2]) throw(IRError("latency must be a compile-time constant"))
    optimization_hints = create_optimization_hints(ctx, latency)
    mask_tv, has_mask = emit_optional_mask(ctx, args, 3)

    if has_mask
        mask = mask_tv.v
        padding_tv = emit_value!(ctx, args[4])
        padding_tv === nothing && throw(IRError("load_ptr_tko: cannot resolve padding tile"))
        padding = padding_tv.v

        tile_val, new_token = encode_LoadPtrTkoOp!(
            cb, result_tile_type, token_type, pointers;
            mask, padding_value = padding, token = input_token, optimization_hints
        )
    else
        tile_val, new_token = encode_LoadPtrTkoOp!(
            cb, result_tile_type, token_type, pointers;
            token = input_token, optimization_hints
        )
    end

    # Store result token for TokenResultNode
    ctx.result_tokens[ctx.current_ssa_idx] = new_token

    julia_shape = ColMajorShape(tile_shape)
    result_jltype = Tile{elem_type, TupleType(julia_shape)}
    return CGVal(tile_val, result_tile_type, result_jltype, tile_shape)
end

"""
    Intrinsics.store_ptr_tko(ptrs::Tile{Ptr{T},S}, values::Tile{T,S},
                             latency::Union{Int,Nothing},
                             mask::Union{Tile{Bool,S},Nothing}=nothing) -> Nothing  where {T,S}

Token-ordered scatter store to a tile of pointers; lowers to
`cuda_tile.store_ptr_tko`.

`latency` is a compile-time hint (1–10 or `nothing`). The token argument
is appended by `token_order_pass!` and is not part of the user-visible
signature.
"""
@intrinsic store_ptr_tko(ptrs::Tile{Ptr{T}, S}, values::Tile{T, S},
                                   latency::Union{Int, Nothing},
                                   mask::Union{Tile{Bool, S}, Nothing}=nothing) where {T, S}
tfunc(𝕃, ::typeof(Intrinsics.store_ptr_tko), @nospecialize args...) = Nothing
efunc(::typeof(Intrinsics.store_ptr_tko), effects::CC.Effects) =
    CC.Effects(effects; effect_free=CC.ALWAYS_FALSE)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.store_ptr_tko), args)
    cb = ctx.cb
    tt = ctx.tt

    # Extract input token from last arg (added by token_order_pass!)
    input_token = extract_token_arg!(ctx, args)

    ptrs_tv = emit_value!(ctx, args[1])
    ptrs_tv === nothing && throw(IRError("store_ptr_tko: cannot resolve pointer tile"))
    pointers = ptrs_tv.v

    values_tv = emit_value!(ctx, args[2])
    values_tv === nothing && throw(IRError("store_ptr_tko: cannot resolve values tile"))
    values = values_tv.v

    token_type = Token(tt)

    latency = @something get_constant(ctx, args[3]) throw(IRError("latency must be a compile-time constant"))
    optimization_hints = create_optimization_hints(ctx, latency)
    mask_tv, has_mask = emit_optional_mask(ctx, args, 4)

    if has_mask
        mask = mask_tv.v
        new_token = encode_StorePtrTkoOp!(
            cb, token_type, pointers, values;
            mask, token = input_token, optimization_hints
        )
    else
        new_token = encode_StorePtrTkoOp!(
            cb, token_type, pointers, values;
            token = input_token, optimization_hints
        )
    end

    # Store result token for TokenResultNode
    ctx.result_tokens[ctx.current_ssa_idx] = new_token

    nothing
end

"""
    extract_token_arg!(ctx, args) -> Value

Extract the token argument (last arg, added by token_order_pass!) from a memory op call.
Pops the token from args and returns the resolved bytecode Value.
"""
function extract_token_arg!(ctx::CGCtx, args)
    isempty(args) && throw(IRError("Memory op has no arguments"))
    last_arg = args[end]
    tv = emit_value!(ctx, last_arg)
    if tv !== nothing && tv.jltype === TokenType
        pop!(args)
        return tv.v
    end
    throw(IRError("Memory op missing token argument (token_order_pass! not run?)"))
end
