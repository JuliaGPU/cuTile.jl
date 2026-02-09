# Type conversions

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
    result_jltype = similar_type(src_type, target_type)
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
    result_jltype = similar_type(src_type, target_type)
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
    result_jltype = similar_type(src_type, target_type)
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
    result_jltype = similar_type(src_type, target_type)
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
    result_jltype = similar_type(src_type, target_type)
    CGVal(result_v, result_type_id, result_jltype, source.shape)
end

# cuda_tile.int_to_ptr, cuda_tile.ptr_to_int# NOTE: Used internally by atomic operations, not exposed as user intrinsics

# TODO: cuda_tile.ptr_to_ptr
