#=============================================================================
 8.4. Conversions
 cuda_tile.bitcast, cuda_tile.exti, cuda_tile.ftof, cuda_tile.ftoi,
 cuda_tile.itof, cuda_tile.trunci
=============================================================================#

#-----------------------------------------------------------------------------
# cuda_tile.ftof, cuda_tile.ftoi, cuda_tile.itof, cuda_tile.exti, cuda_tile.trunci
#-----------------------------------------------------------------------------

function emit_intrinsic!(ctx::CodegenContext, ::typeof(Intrinsics.astype), args, @nospecialize(result_type))
    cb = ctx.cb
    tt = ctx.tt

    # Get source tile
    source = emit_value!(ctx, args[1])
    source === nothing && error("Cannot resolve source operand for astype()")

    # Get source element type and shape
    source_type = unwrap_type(source.jltype)
    source_elem = source_type <: Tile ? source_type.parameters[1] : source_type
    tile_shape = source.shape

    # Get target element type from the Type argument
    target_elem = @something get_constant(ctx, args[2]) error("astype() requires a compile-time constant type")
    target_elem isa Type || error("astype() second argument must be a Type")

    # Same type? Return source unchanged
    if source_elem === target_elem
        return source
    end

    # Create target type
    target_dtype = julia_to_tile_dtype!(tt, target_elem)
    target_tile_type = tile_type!(tt, target_dtype, tile_shape)

    # Determine signedness for integer types
    function is_signed_int(T)
        T <: Signed || T === Int32 || T === Int64 || T === Int16 || T === Int8
    end

    # Emit conversion based on source and target types
    result = if source_elem <: AbstractFloat && target_elem <: AbstractFloat
        # Float -> Float
        encode_FToFOp!(cb, target_tile_type, source.v)
    elseif source_elem <: Integer && target_elem <: AbstractFloat
        # Integer -> Float
        signedness = is_signed_int(source_elem) ? SignednessSigned : SignednessUnsigned
        encode_IToFOp!(cb, target_tile_type, source.v; signedness)
    elseif source_elem <: AbstractFloat && target_elem <: Integer
        # Float -> Integer
        signedness = is_signed_int(target_elem) ? SignednessSigned : SignednessUnsigned
        encode_FToIOp!(cb, target_tile_type, source.v; signedness)
    elseif source_elem <: Integer && target_elem <: Integer
        # Integer -> Integer
        source_size = sizeof(source_elem)
        target_size = sizeof(target_elem)
        if source_size == target_size
            # Same size - no conversion needed (just reinterpret)
            source.v
        elseif target_size > source_size
            # Extension (upsize)
            signedness = is_signed_int(source_elem) ? SignednessSigned : SignednessUnsigned
            encode_ExtIOp!(cb, target_tile_type, source.v; signedness)
        else
            # Truncation (downsize)
            encode_TruncIOp!(cb, target_tile_type, source.v)
        end
    else
        error("astype() unsupported conversion: $source_elem -> $target_elem")
    end

    CGVal(result, target_tile_type, Tile{target_elem, Tuple(tile_shape)}, tile_shape)
end

#-----------------------------------------------------------------------------
# Core.IntrinsicFunction: sitofp, uitofp
#-----------------------------------------------------------------------------

# Signed integer to floating point conversion
function emit_sitofp!(ctx::CodegenContext, args)
    cb = ctx.cb
    tt = ctx.tt

    # args[1] is the target type (e.g., Float32), args[2] is the value
    target_type = args[1]
    source = emit_value!(ctx, args[2])
    source === nothing && error("Cannot resolve source operand for sitofp")

    # Get the target float type
    dtype = julia_to_tile_dtype!(tt, target_type)
    result_shape = source isa CGVal ? source.shape : Int[]
    result_type = tile_type!(tt, dtype, result_shape)

    result_v = encode_IToFOp!(cb, result_type, source isa CGVal ? source.v : source;
                              signedness=SignednessSigned)
    CGVal(result_v, result_type, target_type, result_shape)
end

# Unsigned integer to floating point conversion
function emit_uitofp!(ctx::CodegenContext, args)
    cb = ctx.cb
    tt = ctx.tt

    # args[1] is the target type (e.g., Float32), args[2] is the value
    target_type = args[1]
    source = emit_value!(ctx, args[2])
    source === nothing && error("Cannot resolve source operand for uitofp")

    # Get the target float type
    dtype = julia_to_tile_dtype!(tt, target_type)
    result_shape = source isa CGVal ? source.shape : Int[]
    result_type = tile_type!(tt, dtype, result_shape)

    result_v = encode_IToFOp!(cb, result_type, source isa CGVal ? source.v : source;
                              signedness=SignednessUnsigned)
    CGVal(result_v, result_type, target_type, result_shape)
end
