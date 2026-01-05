# bitwise intrinsics


## cuda_tile.andi

@eval Intrinsics begin
    """Element-wise logical AND for boolean tiles."""
    @noinline function andi(a::Tile{Bool, S}, b::Tile{Bool, S}) where {S}
        Base.donotdelete(a, b)
        Tile{Bool, S}()
    end
end

function emit_intrinsic!(ctx::CodegenContext, ::typeof(Intrinsics.andi), args, @nospecialize(result_type))
    cb = ctx.cb
    tt = ctx.tt

    # Get both boolean tiles
    a_tv = emit_value!(ctx, args[1])
    a_tv === nothing && error("andi: cannot resolve first argument")

    b_tv = emit_value!(ctx, args[2])
    b_tv === nothing && error("andi: cannot resolve second argument")

    tile_shape = a_tv.shape
    bool_tile_type = tile_type!(tt, I1(tt), tile_shape)

    result = encode_AndIOp!(cb, bool_tile_type, a_tv.v, b_tv.v)

    result_type_unwrapped = unwrap_type(result_type)
    CGVal(result, bool_tile_type, result_type_unwrapped, tile_shape)
end


## XXX: not_int via cuda_tile.xori
# Boolean negation using XOR with true (1).
# Called from Core.IntrinsicFunction handler in math.jl.

function emit_not_int!(ctx::CodegenContext, args)
    cb = ctx.cb
    tt = ctx.tt

    source = emit_value!(ctx, args[1])
    source === nothing && error("Cannot resolve source operand for not_int")

    # not_int is applied to Bool (i1) values
    # XOR with true (1) to invert: x XOR 1 = NOT x
    source_v = source isa CGVal ? source.v : source
    source_shape = source isa CGVal ? source.shape : Int[]

    bool_dtype = I1(tt)
    bool_type = tile_type!(tt, bool_dtype, source_shape)

    # Create constant true (0xff for i1)
    true_bytes = UInt8[0xff]
    true_scalar = encode_ConstantOp!(cb, tile_type!(tt, bool_dtype, Int[]), true_bytes)

    # Broadcast if needed
    if !isempty(source_shape)
        ones_shape = fill(1, length(source_shape))
        reshaped_type = tile_type!(tt, bool_dtype, ones_shape)
        true_reshaped = encode_ReshapeOp!(cb, reshaped_type, true_scalar)
        true_tile = encode_BroadcastOp!(cb, bool_type, true_reshaped)
    else
        true_tile = true_scalar
    end

    # XOR to invert
    result_v = encode_XOrIOp!(cb, bool_type, source_v, true_tile)
    CGVal(result_v, bool_type, Bool, source_shape)
end
