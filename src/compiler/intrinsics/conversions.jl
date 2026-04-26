# Type conversions

"""
    Intrinsics.bitcast(x::Tile, ::Type{T}) -> Tile{T}

Reinterprets the bits of `x` element-wise as type `T`; lowers to
`cuda_tile.bitcast`.

Also invocable with a scalar, promoted to a 0-D tile before codegen. `T`
must be a compile-time constant. The op is elided when source and target
map to the same Tile IR type (e.g. `Int32`/`UInt32`, since Tile IR integers
are signless).
"""
@intrinsic bitcast(x, ::Type{T}) where {T}
function tfunc(𝕃, ::typeof(Intrinsics.bitcast), @nospecialize(x), @nospecialize(target_type))
    T = instanceof_tfunc(target_type)
    T === nothing && return nothing
    src = CC.widenconst(x)
    src <: Tile ? similar_type(src, T) : T
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.bitcast), args)
    cb = ctx.cb
    tt = ctx.tt

    source = @something emit_value!(ctx, args[1]) throw(IRError("bitcast: cannot resolve source"))
    target_type = @something get_constant(ctx, args[2]) throw(IRError("bitcast: requires compile-time target type"))

    dtype = julia_to_tile_dtype!(tt, target_type)
    result_type_id = tile_type!(tt, dtype, source.shape)

    src_type = CC.widenconst(source.jltype)
    result_jltype = similar_type(src_type, target_type)

    # No-op when source and target map to the same Tile IR type (e.g., Int32 ↔ UInt32).
    # Tile IR integers are signless, so these are the same type.
    if result_type_id == source.type_id
        return CGVal(source.v, source.type_id, result_jltype, source.shape)
    end

    result_v = encode_BitcastOp!(cb, result_type_id, source.v)
    CGVal(result_v, result_type_id, result_jltype, source.shape)
end

"""
    Intrinsics.exti(x::Tile{<:Integer}, ::Type{T}, s::Signedness.T) -> Tile{T}     where {T<:Integer}

Element-wise integer extension; lowers to `cuda_tile.exti`.

Also invocable with a scalar, promoted to a 0-D tile before codegen. `s`
and `T` are compile-time constants.
"""
@intrinsic exti(x::I, ::Type{T}, s::Signedness.T) where {I<:Integer, T<:Integer}
function tfunc(𝕃, ::typeof(Intrinsics.exti), @nospecialize(x), @nospecialize(target_type), @nospecialize(s))
    T = instanceof_tfunc(target_type)
    T === nothing && return nothing
    src = CC.widenconst(x)
    src <: Tile ? similar_type(src, T) : T
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

"""
    Intrinsics.ftof(x::Tile{<:AbstractFloat}, ::Type{F2}) -> Tile{F2}     where {F2<:AbstractFloat}

Element-wise floating-point to floating-point conversion; lowers to
`cuda_tile.ftof`.

Also invocable with a scalar, promoted to a 0-D tile before codegen. `F2`
must be a compile-time constant. The current emit does not pass a
`rounding_mode` and so uses Tile IR's default.
"""
@intrinsic ftof(x::F1, ::Type{F2}) where {F1<:AbstractFloat, F2<:AbstractFloat}
function tfunc(𝕃, ::typeof(Intrinsics.ftof), @nospecialize(x), @nospecialize(target_type))
    T = instanceof_tfunc(target_type)
    T === nothing && return nothing
    src = CC.widenconst(x)
    src <: Tile ? similar_type(src, T) : T
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

"""
    Intrinsics.ftoi(x::Tile{<:AbstractFloat}, ::Type{I}, s::Signedness.T) -> Tile{I}     where {I<:Integer}

Element-wise floating-point to integer conversion; lowers to
`cuda_tile.ftoi`.

Also invocable with a scalar, promoted to a 0-D tile before codegen. `s`
and `I` are compile-time constants.
"""
@intrinsic ftoi(x::AbstractFloat, ::Type{I}, s::Signedness.T) where {I<:Integer}
function tfunc(𝕃, ::typeof(Intrinsics.ftoi), @nospecialize(x), @nospecialize(target_type), @nospecialize(s))
    T = instanceof_tfunc(target_type)
    T === nothing && return nothing
    src = CC.widenconst(x)
    src <: Tile ? similar_type(src, T) : T
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

"""
    Intrinsics.itof(x::Tile{<:Integer}, ::Type{F}, s::Signedness.T) -> Tile{F}     where {F<:AbstractFloat}

Element-wise integer to floating-point conversion; lowers to
`cuda_tile.itof`.

Also invocable with a scalar, promoted to a 0-D tile before codegen. `s`
and `F` are compile-time constants.
"""
@intrinsic itof(x::Integer, ::Type{F}, s::Signedness.T) where {F<:AbstractFloat}
function tfunc(𝕃, ::typeof(Intrinsics.itof), @nospecialize(x), @nospecialize(target_type), @nospecialize(s))
    T = instanceof_tfunc(target_type)
    T === nothing && return nothing
    src = CC.widenconst(x)
    src <: Tile ? similar_type(src, T) : T
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

"""
    Intrinsics.trunci(x::Tile{<:Integer}, ::Type{T}) -> Tile{T}     where {T<:Integer}

Element-wise integer truncation; lowers to `cuda_tile.trunci`.

Also invocable with a scalar, promoted to a 0-D tile before codegen. `T`
must be a compile-time constant. The current emit does not pass an
`overflow` flag and so uses Tile IR's default.
"""
@intrinsic trunci(x::Integer, ::Type{T}) where {T<:Integer}
function tfunc(𝕃, ::typeof(Intrinsics.trunci), @nospecialize(x), @nospecialize(target_type))
    T = instanceof_tfunc(target_type)
    T === nothing && return nothing
    src = CC.widenconst(x)
    src <: Tile ? similar_type(src, T) : T
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
