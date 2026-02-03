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
    source = @something emit_value!(ctx, args[1]) throw(IRError("astype: cannot resolve source"))

    # Get source element type and shape
    source_type = CC.widenconst(source.jltype)
    source_elem = eltype(source_type)
    tile_shape = source.shape

    # Get target element type from the Type argument
    target_elem = @something get_constant(ctx, args[2]) throw(IRError("astype() requires a compile-time constant type"))
    target_elem isa Type || throw(IRError("astype() second argument must be a Type"))

    # Same type? Return source unchanged
    if source_elem === target_elem
        return source
    end

    # Create target type
    target_dtype = julia_to_tile_dtype!(tt, target_elem)
    target_tile_type = tile_type!(tt, target_dtype, tile_shape)

    # Emit conversion based on source and target types
    result = if source_elem <: AbstractFloat && target_elem <: AbstractFloat
        # Float -> Float
        encode_FToFOp!(cb, target_tile_type, source.v)
    elseif source_elem <: Integer && target_elem <: AbstractFloat
        # Integer -> Float
        signedness = source_elem <: Signed ? SignednessSigned : SignednessUnsigned
        encode_IToFOp!(cb, target_tile_type, source.v; signedness)
    elseif source_elem <: AbstractFloat && target_elem <: Integer
        # Float -> Integer
        signedness = target_elem <: Signed ? SignednessSigned : SignednessUnsigned
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
            signedness = source_elem <: Signed ? SignednessSigned : SignednessUnsigned
            encode_ExtIOp!(cb, target_tile_type, source.v; signedness)
        else
            # Truncation (downsize)
            encode_TruncIOp!(cb, target_tile_type, source.v)
        end
    else
        throw(IRError("astype() unsupported conversion: $source_elem -> $target_elem"))
    end

    CGVal(result, target_tile_type, replace_eltype(source_type, target_elem), tile_shape)
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

    source = @something emit_value!(ctx, args[1]) throw(IRError("exti: cannot resolve source"))
    target_type = @something get_constant(ctx, args[2]) throw(IRError("exti: requires compile-time target type"))
    signedness = @something get_constant(ctx, args[3]) throw(IRError("exti: requires compile-time signedness"))

    dtype = julia_to_tile_dtype!(tt, target_type)
    result_type_id = tile_type!(tt, dtype, source.shape)

    result_v = encode_ExtIOp!(cb, result_type_id, source.v; signedness)
    src_type = CC.widenconst(source.jltype)
    result_jltype = replace_eltype(src_type, target_type)
    CGVal(result_v, result_type_id, result_jltype, source.shape)
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

    source = @something emit_value!(ctx, args[1]) throw(IRError("ftof: cannot resolve source"))
    target_type = @something get_constant(ctx, args[2]) throw(IRError("ftof: requires compile-time target type"))

    dtype = julia_to_tile_dtype!(tt, target_type)
    result_type_id = tile_type!(tt, dtype, source.shape)

    result_v = encode_FToFOp!(cb, result_type_id, source.v)
    src_type = CC.widenconst(source.jltype)
    result_jltype = replace_eltype(src_type, target_type)
    CGVal(result_v, result_type_id, result_jltype, source.shape)
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

    source = @something emit_value!(ctx, args[1]) throw(IRError("ftoi: cannot resolve source"))
    target_type = @something get_constant(ctx, args[2]) throw(IRError("ftoi: requires compile-time target type"))
    signedness = @something get_constant(ctx, args[3]) throw(IRError("ftoi: requires compile-time signedness"))

    dtype = julia_to_tile_dtype!(tt, target_type)
    result_type_id = tile_type!(tt, dtype, source.shape)

    result_v = encode_FToIOp!(cb, result_type_id, source.v; signedness)
    src_type = CC.widenconst(source.jltype)
    result_jltype = replace_eltype(src_type, target_type)
    CGVal(result_v, result_type_id, result_jltype, source.shape)
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

    source = @something emit_value!(ctx, args[1]) throw(IRError("itof: cannot resolve source"))
    target_type = @something get_constant(ctx, args[2]) throw(IRError("itof: requires compile-time target type"))
    signedness = @something get_constant(ctx, args[3]) throw(IRError("itof: requires compile-time signedness"))

    dtype = julia_to_tile_dtype!(tt, target_type)
    result_type_id = tile_type!(tt, dtype, source.shape)

    result_v = encode_IToFOp!(cb, result_type_id, source.v; signedness)
    src_type = CC.widenconst(source.jltype)
    result_jltype = replace_eltype(src_type, target_type)
    CGVal(result_v, result_type_id, result_jltype, source.shape)
end

# cuda_tile.trunci (scalar integer truncation)
@eval Intrinsics begin
    @noinline trunci(x::Integer, ::Type{T}) where {T<:Integer} = Core.Intrinsics.trunc_int(T, x)
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.trunci), args)
    cb = ctx.cb
    tt = ctx.tt

    source = @something emit_value!(ctx, args[1]) throw(IRError("trunci: cannot resolve source"))
    target_type = @something get_constant(ctx, args[2]) throw(IRError("trunci: requires compile-time target type"))

    dtype = julia_to_tile_dtype!(tt, target_type)
    result_type_id = tile_type!(tt, dtype, source.shape)

    result_v = encode_TruncIOp!(cb, result_type_id, source.v)
    src_type = CC.widenconst(source.jltype)
    result_jltype = replace_eltype(src_type, target_type)
    CGVal(result_v, result_type_id, result_jltype, source.shape)
end

# cuda_tile.int_to_ptr, cuda_tile.ptr_to_int# NOTE: Used internally by atomic operations, not exposed as user intrinsics

# TODO: cuda_tile.ptr_to_ptr
