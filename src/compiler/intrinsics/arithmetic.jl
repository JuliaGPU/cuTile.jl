# Integer, floating-point, and boolean arithmetic

## Helpers

function emit_binop!(ctx::CGCtx, args, encoder::Function; kwargs...)
    cb = ctx.cb
    tt = ctx.tt

    lhs_tv = emit_value!(ctx, args[1])
    rhs_tv = emit_value!(ctx, args[2])

    (lhs_tv === nothing || rhs_tv === nothing) && return missing

    # Determine what kind of operands we have
    lhs_type = CC.widenconst(lhs_tv.jltype)
    rhs_type = CC.widenconst(rhs_tv.jltype)
    lhs_is_tile = lhs_type <: Tile
    rhs_is_tile = rhs_type <: Tile

    if lhs_is_tile && rhs_is_tile
        # Tile + Tile: shapes should be identical
        lhs_elem = eltype(lhs_type)
        rhs_elem = eltype(rhs_type)
        lhs_elem === rhs_elem || throw(IRError("Binary op type mismatch: lhs element type $lhs_elem != rhs element type $rhs_elem"))
        elem_type = lhs_elem
        result_shape = lhs_tv.shape
        result_jltype = lhs_tv.jltype
    elseif !lhs_is_tile && !rhs_is_tile
        # Scalar + Scalar: for integer intrinsics on index calculations
        lhs_type === rhs_type || throw(IRError("Binary op type mismatch: lhs type $lhs_type != rhs type $rhs_type"))
        elem_type = lhs_type

        # Shape propagation: scalar Julia values may carry an IR-side shape
        # (via to_scalar). Broadcast shapeless operands (constants) to match.
        if !isempty(lhs_tv.shape) || !isempty(rhs_tv.shape)
            result_shape = !isempty(lhs_tv.shape) ? lhs_tv.shape : rhs_tv.shape
            dtype = julia_to_tile_dtype!(tt, elem_type)
            if isempty(lhs_tv.shape)
                bv = broadcast_tile_to_shape!(cb, tt, lhs_tv, result_shape, dtype)
                lhs_tv = CGVal(bv, tile_type!(tt, dtype, result_shape), elem_type,
                               result_shape, nothing, lhs_tv.constant, nothing)
            elseif isempty(rhs_tv.shape)
                bv = broadcast_tile_to_shape!(cb, tt, rhs_tv, result_shape, dtype)
                rhs_tv = CGVal(bv, tile_type!(tt, dtype, result_shape), elem_type,
                               result_shape, nothing, rhs_tv.constant, nothing)
            end
        else
            result_shape = Int[]
        end
        result_jltype = lhs_tv.jltype
    else
        throw(IRError("Mixed tile/scalar operations should be handled at intrinsic level via Tile() and broadcast_to()"))
    end

    dtype = julia_to_tile_dtype!(tt, elem_type)
    result_type_id = tile_type!(tt, dtype, result_shape)

    result_v = encoder(cb, result_type_id, lhs_tv.v, rhs_tv.v; kwargs...)

    CGVal(result_v, result_type_id, result_jltype, result_shape)
end

function emit_unop!(ctx::CGCtx, args, encoder::Function; kwargs...)
    cb = ctx.cb
    tt = ctx.tt

    source = emit_value!(ctx, args[1])
    source === nothing && return missing

    source_type = CC.widenconst(source.jltype)
    elem_type = eltype(source_type)
    result_shape = source.shape
    result_jltype = source.jltype

    dtype = julia_to_tile_dtype!(tt, elem_type)
    result_type_id = tile_type!(tt, dtype, result_shape)

    result_v = encoder(cb, result_type_id, source.v; kwargs...)

    CGVal(result_v, result_type_id, result_jltype, result_shape)
end


## Integer arithmetic

# cuda_tile.absi
@intrinsic absi(x::Integer)
@intrinsic absi(x::Tile{<:Integer})
tfunc(𝕃, ::typeof(Intrinsics.absi), @nospecialize(x)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.absi), args)
    emit_unop!(ctx, args, encode_AbsIOp!)
end

# cuda_tile.addi
@intrinsic addi(x::T, y::T) where {T<:Integer}
@intrinsic addi(a::Tile{T}, b::Tile{T}) where {T<:Integer}
tfunc(𝕃, ::typeof(Intrinsics.addi), @nospecialize(x), @nospecialize(y)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.addi), args)
    emit_binop!(ctx, args, encode_AddIOp!)
end

# cuda_tile.cldi (ceiling division, toward positive infinity)
@intrinsic cldi(x::T, y::T, s::Signedness) where {T<:Integer}
@intrinsic cldi(a::Tile{T}, b::Tile{T}, s::Signedness) where {T<:Integer}
tfunc(𝕃, ::typeof(Intrinsics.cldi), @nospecialize(x), @nospecialize(y), @nospecialize(s)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.cldi), args)
    signedness = @something get_constant(ctx, args[3]) throw(IRError("cldi requires compile-time signedness"))
    emit_binop!(ctx, args, encode_DivIOp!; signedness, rounding=RoundingPositiveInf)
end

# cuda_tile.cmpi
@intrinsic function cmpi(x::T, y::T, pred::ComparisonPredicate, s::Signedness) where {T<:Integer}
    if pred === CmpLessThan
        s === SignednessSigned ? Core.Intrinsics.slt_int(x, y) : Core.Intrinsics.ult_int(x, y)
    elseif pred === CmpLessThanOrEqual
        s === SignednessSigned ? Core.Intrinsics.sle_int(x, y) : Core.Intrinsics.ule_int(x, y)
    elseif pred === CmpGreaterThan
        s === SignednessSigned ? Core.Intrinsics.slt_int(y, x) : Core.Intrinsics.ult_int(y, x)
    elseif pred === CmpGreaterThanOrEqual
        s === SignednessSigned ? Core.Intrinsics.sle_int(y, x) : Core.Intrinsics.ule_int(y, x)
    elseif pred === CmpEqual
        Core.Intrinsics.eq_int(x, y)
    else  # CmpNotEqual
        Core.Intrinsics.ne_int(x, y)
    end
end
@intrinsic cmpi(a::Tile{T}, b::Tile{T}, pred::ComparisonPredicate, s::Signedness) where {T<:Integer}
function tfunc(𝕃, ::typeof(Intrinsics.cmpi), @nospecialize(x), @nospecialize(y), @nospecialize(pred), @nospecialize(s))
    t = CC.widenconst(x)
    if t <: Tile
        S = t.parameters[2]
        return Tile{Bool, S}
    end
    return Bool
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.cmpi), args)
    cb = ctx.cb
    tt = ctx.tt

    lhs = @something emit_value!(ctx, args[1]) throw(IRError("cmpi: cannot resolve lhs"))
    rhs = @something emit_value!(ctx, args[2]) throw(IRError("cmpi: cannot resolve rhs"))
    predicate = @something get_constant(ctx, args[3]) throw(IRError("cmpi: requires compile-time predicate"))
    signedness = @something get_constant(ctx, args[4]) throw(IRError("cmpi: requires compile-time signedness"))

    # Validate type match
    lhs.type_id == rhs.type_id || throw(IRError("cmpi type mismatch: lhs type $(lhs.jltype) != rhs type $(rhs.jltype)"))

    result_shape = lhs.shape

    bool_dtype = I1(tt)
    result_type_id = tile_type!(tt, bool_dtype, result_shape)

    result_v = encode_CmpIOp!(cb, result_type_id, lhs.v, rhs.v; predicate, signedness)
    lhs_type = CC.widenconst(lhs.jltype)
    result_jltype = similar_type(lhs_type, Bool)
    CGVal(result_v, result_type_id, result_jltype, result_shape)
end

# cuda_tile.divi (truncating division, toward zero)
@intrinsic function divi(x::T, y::T, s::Signedness) where {T<:Integer}
    s === SignednessSigned ? Core.Intrinsics.sdiv_int(x, y) : Core.Intrinsics.udiv_int(x, y)
end
@intrinsic divi(a::Tile{T}, b::Tile{T}, s::Signedness) where {T<:Integer}
tfunc(𝕃, ::typeof(Intrinsics.divi), @nospecialize(x), @nospecialize(y), @nospecialize(s)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.divi), args)
    signedness = @something get_constant(ctx, args[3]) throw(IRError("divi requires compile-time signedness"))
    emit_binop!(ctx, args, encode_DivIOp!; signedness, rounding=RoundingZero)
end

# cuda_tile.fldi (floor division, toward negative infinity)
@intrinsic fldi(x::T, y::T, s::Signedness) where {T<:Integer}
@intrinsic fldi(a::Tile{T}, b::Tile{T}, s::Signedness) where {T<:Integer}
tfunc(𝕃, ::typeof(Intrinsics.fldi), @nospecialize(x), @nospecialize(y), @nospecialize(s)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.fldi), args)
    signedness = @something get_constant(ctx, args[3]) throw(IRError("fldi requires compile-time signedness"))
    emit_binop!(ctx, args, encode_DivIOp!; signedness, rounding=RoundingNegativeInf)
end

# cuda_tile.maxi
@intrinsic function maxi(x::T, y::T, s::Signedness) where {T<:Integer}
    lt = s === SignednessSigned ? Core.Intrinsics.slt_int(x, y) : Core.Intrinsics.ult_int(x, y)
    ifelse(lt, y, x)
end
@intrinsic maxi(a::Tile{T}, b::Tile{T}, s::Signedness) where {T<:Integer}
tfunc(𝕃, ::typeof(Intrinsics.maxi), @nospecialize(x), @nospecialize(y), @nospecialize(s)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.maxi), args)
    signedness = @something get_constant(ctx, args[3]) throw(IRError("maxi requires compile-time signedness"))
    emit_binop!(ctx, args, encode_MaxIOp!; signedness)
end

# cuda_tile.mini
@intrinsic mini(x::T, y::T, s::Signedness) where {T<:Integer}
@intrinsic mini(a::Tile{T}, b::Tile{T}, s::Signedness) where {T<:Integer}
tfunc(𝕃, ::typeof(Intrinsics.mini), @nospecialize(x), @nospecialize(y), @nospecialize(s)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.mini), args)
    signedness = @something get_constant(ctx, args[3]) throw(IRError("mini requires compile-time signedness"))
    emit_binop!(ctx, args, encode_MinIOp!; signedness)
end

# cuda_tile.muli
@intrinsic muli(x::T, y::T) where {T<:Integer} = Core.Intrinsics.mul_int(x, y)
@intrinsic muli(a::Tile{T}, b::Tile{T}) where {T<:Integer}
tfunc(𝕃, ::typeof(Intrinsics.muli), @nospecialize(x), @nospecialize(y)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.muli), args)
    emit_binop!(ctx, args, encode_MulIOp!)
end

# cuda_tile.mulhii
@intrinsic mulhii(x::T, y::T, s::Signedness) where {T<:Integer}
@intrinsic mulhii(a::Tile{T}, b::Tile{T}, s::Signedness) where {T<:Integer}
tfunc(𝕃, ::typeof(Intrinsics.mulhii), @nospecialize(x), @nospecialize(y), @nospecialize(s)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.mulhii), args)
    emit_binop!(ctx, args, encode_MulhiIOp!)
end

# cuda_tile.negi
@intrinsic negi(x::T) where {T<:Integer}
@intrinsic negi(a::Tile{<:Integer})
tfunc(𝕃, ::typeof(Intrinsics.negi), @nospecialize(x)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.negi), args)
    emit_unop!(ctx, args, encode_NegIOp!; overflow=OverflowNone)
end

# cuda_tile.remi
@intrinsic remi(x::T, y::T, s::Signedness) where {T<:Integer}
@intrinsic remi(a::Tile{T}, b::Tile{T}, s::Signedness) where {T<:Integer}
tfunc(𝕃, ::typeof(Intrinsics.remi), @nospecialize(x), @nospecialize(y), @nospecialize(s)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.remi), args)
    signedness = @something get_constant(ctx, args[3]) throw(IRError("remi requires compile-time signedness"))
    emit_binop!(ctx, args, encode_RemIOp!; signedness)
end

# cuda_tile.shli
@intrinsic shli(x::T, y::Integer) where {T<:Integer} = Core.Intrinsics.shl_int(x, y % T)
@intrinsic shli(a::Tile{T}, b::Tile{T}) where {T<:Integer}
tfunc(𝕃, ::typeof(Intrinsics.shli), @nospecialize(x), @nospecialize(y)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.shli), args)
    emit_binop!(ctx, args, encode_ShLIOp!)
end

# cuda_tile.shri
@intrinsic shri(x::T, y::Integer, s::Signedness) where {T<:Integer}
@intrinsic shri(a::Tile{T}, b::Tile{T}, s::Signedness) where {T<:Integer}
tfunc(𝕃, ::typeof(Intrinsics.shri), @nospecialize(x), @nospecialize(y), @nospecialize(s)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.shri), args)
    signedness = @something get_constant(ctx, args[3]) throw(IRError("shri requires compile-time signedness"))
    emit_binop!(ctx, args, encode_ShRIOp!; signedness)
end

# cuda_tile.subi
@intrinsic subi(x::T, y::T) where {T<:Integer} = Core.Intrinsics.sub_int(x, y)
@intrinsic subi(a::Tile{T}, b::Tile{T}) where {T<:Integer}
tfunc(𝕃, ::typeof(Intrinsics.subi), @nospecialize(x), @nospecialize(y)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.subi), args)
    emit_binop!(ctx, args, encode_SubIOp!)
end


## Floating-point arithmetic

# cuda_tile.absf
@intrinsic absf(x::T) where {T<:AbstractFloat}
@intrinsic absf(a::Tile{<:AbstractFloat})
tfunc(𝕃, ::typeof(Intrinsics.absf), @nospecialize(x)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.absf), args)
    emit_unop!(ctx, args, encode_AbsFOp!)
end

# cuda_tile.addf
@intrinsic addf(x::T, y::T) where {T<:AbstractFloat}
@intrinsic addf(a::Tile{T}, b::Tile{T}) where {T<:AbstractFloat}
tfunc(𝕃, ::typeof(Intrinsics.addf), @nospecialize(x), @nospecialize(y)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.addf), args)
    emit_binop!(ctx, args, encode_AddFOp!)
end

# cuda_tile.cmpf
@intrinsic cmpf(x::T, y::T, pred::ComparisonPredicate) where {T<:AbstractFloat}
@intrinsic cmpf(a::Tile{T}, b::Tile{T}, pred::ComparisonPredicate) where {T<:AbstractFloat}
function tfunc(𝕃, ::typeof(Intrinsics.cmpf), @nospecialize(x), @nospecialize(y), @nospecialize(pred))
    t = CC.widenconst(x)
    if t <: Tile
        S = t.parameters[2]
        return Tile{Bool, S}
    end
    return Bool
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.cmpf), args)
    cb = ctx.cb
    tt = ctx.tt

    lhs = @something emit_value!(ctx, args[1]) throw(IRError("cmpf: cannot resolve lhs"))
    rhs = @something emit_value!(ctx, args[2]) throw(IRError("cmpf: cannot resolve rhs"))
    predicate = @something get_constant(ctx, args[3]) throw(IRError("cmpf: requires compile-time predicate"))

    # Validate type match
    lhs.type_id == rhs.type_id || throw(IRError("cmpf type mismatch: lhs type $(lhs.jltype) != rhs type $(rhs.jltype)"))

    result_shape = lhs.shape

    bool_dtype = I1(tt)
    result_type_id = tile_type!(tt, bool_dtype, result_shape)

    result_v = encode_CmpFOp!(cb, result_type_id, lhs.v, rhs.v; predicate)
    lhs_type = CC.widenconst(lhs.jltype)
    result_jltype = similar_type(lhs_type, Bool)
    CGVal(result_v, result_type_id, result_jltype, result_shape)
end

# cuda_tile.divf
@intrinsic divf(x::T, y::T) where {T<:AbstractFloat}
@intrinsic divf(a::Tile{T}, b::Tile{T}) where {T<:AbstractFloat}
tfunc(𝕃, ::typeof(Intrinsics.divf), @nospecialize(x), @nospecialize(y)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.divf), args)
    emit_binop!(ctx, args, encode_DivFOp!)
end

# cuda_tile.mulf
@intrinsic mulf(x::T, y::T) where {T<:AbstractFloat}
@intrinsic mulf(a::Tile{T}, b::Tile{T}) where {T<:AbstractFloat}
tfunc(𝕃, ::typeof(Intrinsics.mulf), @nospecialize(x), @nospecialize(y)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.mulf), args)
    emit_binop!(ctx, args, encode_MulFOp!)
end

# cuda_tile.negf
@intrinsic negf(x::T) where {T<:AbstractFloat}
@intrinsic negf(a::Tile{<:AbstractFloat})
tfunc(𝕃, ::typeof(Intrinsics.negf), @nospecialize(x)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.negf), args)
    emit_unop!(ctx, args, encode_NegFOp!)
end

# cuda_tile.subf
@intrinsic subf(x::T, y::T) where {T<:AbstractFloat}
@intrinsic subf(a::Tile{T}, b::Tile{T}) where {T<:AbstractFloat}
tfunc(𝕃, ::typeof(Intrinsics.subf), @nospecialize(x), @nospecialize(y)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.subf), args)
    emit_binop!(ctx, args, encode_SubFOp!)
end


## Boolean arithmetic

# cuda_tile.andi
@intrinsic andi(x::T, y::T) where {T<:Integer} = Core.Intrinsics.and_int(x, y)
@intrinsic andi(a::Tile{T}, b::Tile{T}) where {T<:Integer}
tfunc(𝕃, ::typeof(Intrinsics.andi), @nospecialize(x), @nospecialize(y)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.andi), args)
    cb = ctx.cb
    tt = ctx.tt

    lhs = @something emit_value!(ctx, args[1]) throw(IRError("andi: cannot resolve lhs"))
    rhs = @something emit_value!(ctx, args[2]) throw(IRError("andi: cannot resolve rhs"))

    lhs_type = CC.widenconst(lhs.jltype)
    dtype = julia_to_tile_dtype!(tt, eltype(lhs_type))
    result_type_id = tile_type!(tt, dtype, lhs.shape)

    result = encode_AndIOp!(cb, result_type_id, lhs.v, rhs.v)
    CGVal(result, result_type_id, lhs.jltype, lhs.shape)
end

# cuda_tile.ori
@intrinsic ori(x::T, y::T) where {T<:Integer}
@intrinsic ori(a::Tile{T}, b::Tile{T}) where {T<:Integer}
tfunc(𝕃, ::typeof(Intrinsics.ori), @nospecialize(x), @nospecialize(y)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.ori), args)
    cb = ctx.cb
    tt = ctx.tt

    lhs = @something emit_value!(ctx, args[1]) throw(IRError("ori: cannot resolve lhs"))
    rhs = @something emit_value!(ctx, args[2]) throw(IRError("ori: cannot resolve rhs"))

    lhs_type = CC.widenconst(lhs.jltype)
    dtype = julia_to_tile_dtype!(tt, eltype(lhs_type))
    result_type_id = tile_type!(tt, dtype, lhs.shape)

    result = encode_OrIOp!(cb, result_type_id, lhs.v, rhs.v)
    CGVal(result, result_type_id, lhs.jltype, lhs.shape)
end

# cuda_tile.xori
@intrinsic xori(x::T, y::T) where {T<:Integer}
@intrinsic xori(a::Tile{T}, b::Tile{T}) where {T<:Integer}
tfunc(𝕃, ::typeof(Intrinsics.xori), @nospecialize(x), @nospecialize(y)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.xori), args)
    cb = ctx.cb
    tt = ctx.tt

    lhs = @something emit_value!(ctx, args[1]) throw(IRError("xori: cannot resolve lhs"))
    rhs = @something emit_value!(ctx, args[2]) throw(IRError("xori: cannot resolve rhs"))

    lhs_type = CC.widenconst(lhs.jltype)
    dtype = julia_to_tile_dtype!(tt, eltype(lhs_type))
    result_type_id = tile_type!(tt, dtype, lhs.shape)

    result = encode_XOrIOp!(cb, result_type_id, lhs.v, rhs.v)
    CGVal(result, result_type_id, lhs.jltype, lhs.shape)
end
