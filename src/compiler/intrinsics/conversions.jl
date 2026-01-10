# Type conversions

# cuda_tile.astype (high-level tile conversion)
@eval Intrinsics begin
    """
        astype(tile, T2)

    Convert tile element type from T1 to T2.
    Compiled to cuda_tile.ftof, cuda_tile.ftoi, cuda_tile.itof,
    cuda_tile.exti, or cuda_tile.trunci based on source/target types.
    """
    @noinline function astype(tile::Tile{T1, Shape}, ::Type{T2}) where {T1, Shape, T2}
        donotdelete(tile)
        Tile{T2, Shape}()
    end
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.astype), args)
    cb = ctx.cb
    tt = ctx.tt

    # Get source tile
    source = emit_value!(ctx, args[1])
    source === nothing && error("Cannot resolve source operand for astype()")

    # Get source element type and shape
    source_type = unwrap_type(source.jltype)
    source_elem = source_type.parameters[1]
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

# TODO: cuda_tile.bitcast

# cuda_tile.exti (scalar integer extension)
@eval Intrinsics begin
    @noinline function exti(x::I, ::Type{T}, s::Signedness) where {I<:Integer, T<:Integer}
        s === SignednessSigned ? Core.Intrinsics.sext_int(T, x) : Core.Intrinsics.zext_int(T, x)
    end
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.exti), args)
    cb = ctx.cb
    tt = ctx.tt

    source = emit_value!(ctx, args[1])
    target_type = @something get_constant(ctx, args[2]) error("exti requires compile-time target type")
    signedness = @something get_constant(ctx, args[3]) error("exti requires compile-time signedness")

    source === nothing && error("Cannot resolve source operand for exti")

    source_v = source isa CGVal ? source.v : source
    result_shape = source isa CGVal ? source.shape : Int[]

    dtype = julia_to_tile_dtype!(tt, target_type)
    result_type_id = tile_type!(tt, dtype, result_shape)

    result_v = encode_ExtIOp!(cb, result_type_id, source_v; signedness)
    CGVal(result_v, result_type_id, target_type, result_shape)
end

# cuda_tile.ftof (scalar float to float)
@eval Intrinsics begin
    @noinline function ftof(x::F1, ::Type{F2}) where {F1<:AbstractFloat, F2<:AbstractFloat}
        sizeof(F2) > sizeof(F1) ? Core.Intrinsics.fpext(F2, x) : Core.Intrinsics.fptrunc(F2, x)
    end
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.ftof), args)
    cb = ctx.cb
    tt = ctx.tt

    source = emit_value!(ctx, args[1])
    target_type = @something get_constant(ctx, args[2]) error("ftof requires compile-time target type")

    source === nothing && error("Cannot resolve source operand for ftof")

    source_v = source isa CGVal ? source.v : source
    result_shape = source isa CGVal ? source.shape : Int[]

    dtype = julia_to_tile_dtype!(tt, target_type)
    result_type_id = tile_type!(tt, dtype, result_shape)

    result_v = encode_FToFOp!(cb, result_type_id, source_v)
    CGVal(result_v, result_type_id, target_type, result_shape)
end

# cuda_tile.ftoi (scalar float to integer)
@eval Intrinsics begin
    @noinline function ftoi(x::AbstractFloat, ::Type{I}, s::Signedness) where {I<:Integer}
        s === SignednessSigned ? Core.Intrinsics.fptosi(I, x) : Core.Intrinsics.fptoui(I, x)
    end
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.ftoi), args)
    cb = ctx.cb
    tt = ctx.tt

    source = emit_value!(ctx, args[1])
    target_type = @something get_constant(ctx, args[2]) error("ftoi requires compile-time target type")
    signedness = @something get_constant(ctx, args[3]) error("ftoi requires compile-time signedness")

    source === nothing && error("Cannot resolve source operand for ftoi")

    source_v = source isa CGVal ? source.v : source
    result_shape = source isa CGVal ? source.shape : Int[]

    dtype = julia_to_tile_dtype!(tt, target_type)
    result_type_id = tile_type!(tt, dtype, result_shape)

    result_v = encode_FToIOp!(cb, result_type_id, source_v; signedness)
    CGVal(result_v, result_type_id, target_type, result_shape)
end

# cuda_tile.itof (scalar integer to float)
@eval Intrinsics begin
    @noinline function itof(x::Integer, ::Type{F}, s::Signedness) where {F<:AbstractFloat}
        s === SignednessSigned ? Core.Intrinsics.sitofp(F, x) : Core.Intrinsics.uitofp(F, x)
    end
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.itof), args)
    cb = ctx.cb
    tt = ctx.tt

    source = emit_value!(ctx, args[1])
    target_type = @something get_constant(ctx, args[2]) error("itof requires compile-time target type")
    signedness = @something get_constant(ctx, args[3]) error("itof requires compile-time signedness")

    source === nothing && error("Cannot resolve source operand for itof")

    source_v = source isa CGVal ? source.v : source
    result_shape = source isa CGVal ? source.shape : Int[]

    dtype = julia_to_tile_dtype!(tt, target_type)
    result_type_id = tile_type!(tt, dtype, result_shape)

    result_v = encode_IToFOp!(cb, result_type_id, source_v; signedness)
    CGVal(result_v, result_type_id, target_type, result_shape)
end

# cuda_tile.trunci (scalar integer truncation)
@eval Intrinsics begin
    @noinline trunci(x::Integer, ::Type{T}) where {T<:Integer} = Core.Intrinsics.trunc_int(T, x)
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.trunci), args)
    cb = ctx.cb
    tt = ctx.tt

    source = emit_value!(ctx, args[1])
    target_type = @something get_constant(ctx, args[2]) error("trunci requires compile-time target type")

    source === nothing && error("Cannot resolve source operand for trunci")

    source_v = source isa CGVal ? source.v : source
    result_shape = source isa CGVal ? source.shape : Int[]

    dtype = julia_to_tile_dtype!(tt, target_type)
    result_type_id = tile_type!(tt, dtype, result_shape)

    result_v = encode_TruncIOp!(cb, result_type_id, source_v)
    CGVal(result_v, result_type_id, target_type, result_shape)
end

# cuda_tile.int_to_ptr, cuda_tile.ptr_to_int# NOTE: Used internally by atomic operations, not exposed as user intrinsics

# TODO: cuda_tile.ptr_to_ptr
