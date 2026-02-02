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
        lhs_elem === rhs_elem || error("Binary op type mismatch: lhs element type $lhs_elem != rhs element type $rhs_elem")
        elem_type = lhs_elem
        result_shape = lhs_tv.shape
        result_jltype = lhs_tv.jltype
    elseif !lhs_is_tile && !rhs_is_tile
        # Scalar + Scalar: for integer intrinsics on index calculations
        lhs_type === rhs_type || error("Binary op type mismatch: lhs type $lhs_type != rhs type $rhs_type")
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
        error("Mixed tile/scalar operations should be handled at intrinsic level via Tile() and broadcast_to()")
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
    elem_type = source_type <: Tile ? eltype(source_type) : source_type
    result_shape = source.shape  # Propagate IR-side shape from to_scalar
    result_jltype = source.jltype

    dtype = julia_to_tile_dtype!(tt, elem_type)
    result_type_id = tile_type!(tt, dtype, result_shape)

    result_v = encoder(cb, result_type_id, source.v; kwargs...)

    CGVal(result_v, result_type_id, result_jltype, result_shape)
end


## Integer arithmetic

# cuda_tile.absi
@eval Intrinsics begin
    """Integer absolute value. Compiled to cuda_tile.absi."""
    @noinline absi(x::T) where {T<:Integer} =
        ifelse(Core.Intrinsics.slt_int(x, zero(T)), Core.Intrinsics.neg_int(x), x)
    @noinline absi(a::Tile{T, S}) where {T<:Integer, S} = compilerbarrier(:const, a)
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.absi), args)
    emit_unop!(ctx, args, encode_AbsIOp!)
end

# cuda_tile.addi
@eval Intrinsics begin
    @noinline addi(x::T, y::T) where {T<:Integer} = Core.Intrinsics.add_int(x, y)
    @noinline addi(a::Tile{T, S}, b::Tile{T, S}) where {T<:Integer, S} = (donotdelete(a, b); Tile{T, S}())
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.addi), args)
    emit_binop!(ctx, args, encode_AddIOp!)
end

# cuda_tile.cldi (ceiling division, toward positive infinity)
@eval Intrinsics begin
    @noinline cldi(x::T, y::T, s::Signedness) where {T<:Integer} = (donotdelete(x, y, s); compilerbarrier(:const, zero(T)))
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.cldi), args)
    signedness = @something get_constant(ctx, args[3]) error("cldi requires compile-time signedness")
    emit_binop!(ctx, args, encode_DivIOp!; signedness, rounding=RoundingPositiveInf)
end

# cuda_tile.cmpi
@eval Intrinsics begin
    @noinline function cmpi(x::T, y::T, pred::ComparisonPredicate, s::Signedness) where {T<:Integer}
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
    @noinline cmpi(a::Tile{T, S}, b::Tile{T, S}, ::ComparisonPredicate, ::Signedness) where {T<:Integer, S} =
        (donotdelete(a, b); Tile{Bool, S}())
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.cmpi), args)
    cb = ctx.cb
    tt = ctx.tt

    lhs = emit_value!(ctx, args[1])
    rhs = emit_value!(ctx, args[2])
    predicate = @something get_constant(ctx, args[3]) error("cmpi requires compile-time predicate")
    signedness = @something get_constant(ctx, args[4]) error("cmpi requires compile-time signedness")

    (lhs === nothing || rhs === nothing) && error("Cannot resolve operands for cmpi")

    # Validate type match
    lhs.type_id == rhs.type_id || error("cmpi type mismatch: lhs type $(lhs.jltype) != rhs type $(rhs.jltype)")

    result_shape = lhs isa CGVal ? lhs.shape : Int[]

    bool_dtype = I1(tt)
    result_type_id = tile_type!(tt, bool_dtype, result_shape)

    result_v = encode_CmpIOp!(cb, result_type_id, lhs.v, rhs.v; predicate, signedness)
    lhs_type = CC.widenconst(lhs.jltype)
    result_jltype = lhs_type <: Tile ? replace_eltype(lhs_type, Bool) : Bool
    CGVal(result_v, result_type_id, result_jltype, result_shape)
end

# cuda_tile.divi (truncating division, toward zero)
@eval Intrinsics begin
    @noinline function divi(x::T, y::T, s::Signedness) where {T<:Integer}
        s === SignednessSigned ? Core.Intrinsics.sdiv_int(x, y) : Core.Intrinsics.udiv_int(x, y)
    end
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.divi), args)
    signedness = @something get_constant(ctx, args[3]) error("divi requires compile-time signedness")
    emit_binop!(ctx, args, encode_DivIOp!; signedness, rounding=RoundingZero)
end

# cuda_tile.fldi (floor division, toward negative infinity)
@eval Intrinsics begin
    @noinline fldi(x::T, y::T, s::Signedness) where {T<:Integer} = (donotdelete(x, y, s); compilerbarrier(:const, zero(T)))
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.fldi), args)
    signedness = @something get_constant(ctx, args[3]) error("fldi requires compile-time signedness")
    emit_binop!(ctx, args, encode_DivIOp!; signedness, rounding=RoundingNegativeInf)
end

# cuda_tile.maxi
@eval Intrinsics begin
    @noinline function maxi(x::T, y::T, s::Signedness) where {T<:Integer}
        lt = s === SignednessSigned ? Core.Intrinsics.slt_int(x, y) : Core.Intrinsics.ult_int(x, y)
        ifelse(lt, y, x)
    end
    @noinline maxi(a::Tile{T, S}, b::Tile{T, S}, s::Signedness) where {T<:Integer, S} =
        (donotdelete(a, b); Tile{T, S}())
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.maxi), args)
    signedness = @something get_constant(ctx, args[3]) error("maxi requires compile-time signedness")
    emit_binop!(ctx, args, encode_MaxIOp!; signedness)
end

# cuda_tile.mini
@eval Intrinsics begin
    @noinline function mini(x::T, y::T, s::Signedness) where {T<:Integer}
        lt = s === SignednessSigned ? Core.Intrinsics.slt_int(x, y) : Core.Intrinsics.ult_int(x, y)
        ifelse(lt, x, y)
    end
    @noinline mini(a::Tile{T, S}, b::Tile{T, S}, s::Signedness) where {T<:Integer, S} =
        (donotdelete(a, b); Tile{T, S}())
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.mini), args)
    signedness = @something get_constant(ctx, args[3]) error("mini requires compile-time signedness")
    emit_binop!(ctx, args, encode_MinIOp!; signedness)
end

# cuda_tile.muli
@eval Intrinsics begin
    @noinline muli(x::T, y::T) where {T<:Integer} = Core.Intrinsics.mul_int(x, y)
    @noinline muli(a::Tile{T, S}, b::Tile{T, S}) where {T<:Integer, S} = (donotdelete(a, b); Tile{T, S}())
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.muli), args)
    emit_binop!(ctx, args, encode_MulIOp!)
end

# cuda_tile.mulhii
@eval Intrinsics begin
    """High bits of integer multiply (for extended precision arithmetic). Compiled to cuda_tile.mulhii."""
    @noinline function mulhii(x::T, y::T, s::Signedness) where {T<:Integer}
        ((widen(x) * widen(y)) >>> (8 * sizeof(T))) % T
    end
    @noinline mulhii(a::Tile{T, S}, b::Tile{T, S}, s::Signedness) where {T<:Integer, S} = (donotdelete(a, b); Tile{T, S}())
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.mulhii), args)
    emit_binop!(ctx, args, encode_MulhiIOp!)
end

# cuda_tile.negi
@eval Intrinsics begin
    @noinline negi(x::T) where {T<:Integer} = Core.Intrinsics.neg_int(x)
    @noinline negi(a::Tile{T, S}) where {T<:Integer, S} = compilerbarrier(:const, a)
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.negi), args)
    emit_unop!(ctx, args, encode_NegIOp!; overflow=OverflowNone)
end

# cuda_tile.remi
@eval Intrinsics begin
    @noinline function remi(x::T, y::T, s::Signedness) where {T<:Integer}
        s === SignednessSigned ? Core.Intrinsics.srem_int(x, y) : Core.Intrinsics.urem_int(x, y)
    end
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.remi), args)
    signedness = @something get_constant(ctx, args[3]) error("remi requires compile-time signedness")
    emit_binop!(ctx, args, encode_RemIOp!; signedness)
end

# cuda_tile.shli
@eval Intrinsics begin
    @noinline shli(x::T, y::Integer) where {T<:Integer} = Core.Intrinsics.shl_int(x, y % T)
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.shli), args)
    emit_binop!(ctx, args, encode_ShLIOp!)
end

# cuda_tile.shri
@eval Intrinsics begin
    @noinline function shri(x::T, y::Integer, s::Signedness) where {T<:Integer}
        s === SignednessSigned ? Core.Intrinsics.ashr_int(x, y % T) : Core.Intrinsics.lshr_int(x, y % T)
    end
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.shri), args)
    signedness = @something get_constant(ctx, args[3]) error("shri requires compile-time signedness")
    emit_binop!(ctx, args, encode_ShRIOp!; signedness)
end

# cuda_tile.subi
@eval Intrinsics begin
    @noinline subi(x::T, y::T) where {T<:Integer} = Core.Intrinsics.sub_int(x, y)
    @noinline subi(a::Tile{T, S}, b::Tile{T, S}) where {T<:Integer, S} = (donotdelete(a, b); Tile{T, S}())
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.subi), args)
    emit_binop!(ctx, args, encode_SubIOp!)
end


## Floating-point arithmetic

# cuda_tile.absf
@eval Intrinsics begin
    @noinline absf(x::T) where {T<:AbstractFloat} = Core.Intrinsics.abs_float(x)
    @noinline absf(a::Tile{T, S}) where {T<:AbstractFloat, S} = compilerbarrier(:const, a)
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.absf), args)
    emit_unop!(ctx, args, encode_AbsFOp!)
end

# cuda_tile.addf
@eval Intrinsics begin
    @noinline addf(x::T, y::T) where {T<:AbstractFloat} = Core.Intrinsics.add_float(x, y)
    @noinline addf(a::Tile{T, S}, b::Tile{T, S}) where {T<:AbstractFloat, S} = (donotdelete(a, b); Tile{T, S}())
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.addf), args)
    emit_binop!(ctx, args, encode_AddFOp!)
end

# cuda_tile.cmpf
@eval Intrinsics begin
    @noinline function cmpf(x::T, y::T, pred::ComparisonPredicate) where {T<:AbstractFloat}
        if pred === CmpLessThan
            Core.Intrinsics.lt_float(x, y)
        elseif pred === CmpLessThanOrEqual
            Core.Intrinsics.le_float(x, y)
        elseif pred === CmpGreaterThan
            Core.Intrinsics.lt_float(y, x)
        elseif pred === CmpGreaterThanOrEqual
            Core.Intrinsics.le_float(y, x)
        elseif pred === CmpEqual
            Core.Intrinsics.eq_float(x, y)
        else  # CmpNotEqual
            Core.Intrinsics.ne_float(x, y)
        end
    end
    @noinline cmpf(a::Tile{T, S}, b::Tile{T, S}, ::ComparisonPredicate) where {T<:AbstractFloat, S} =
        (donotdelete(a, b); Tile{Bool, S}())
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.cmpf), args)
    cb = ctx.cb
    tt = ctx.tt

    lhs = emit_value!(ctx, args[1])
    rhs = emit_value!(ctx, args[2])
    predicate = @something get_constant(ctx, args[3]) error("cmpf requires compile-time predicate")

    (lhs === nothing || rhs === nothing) && error("Cannot resolve operands for cmpf")

    # Validate type match
    lhs.type_id == rhs.type_id || error("cmpf type mismatch: lhs type $(lhs.jltype) != rhs type $(rhs.jltype)")

    result_shape = lhs isa CGVal ? lhs.shape : Int[]

    bool_dtype = I1(tt)
    result_type_id = tile_type!(tt, bool_dtype, result_shape)

    result_v = encode_CmpFOp!(cb, result_type_id, lhs.v, rhs.v; predicate)
    lhs_type = CC.widenconst(lhs.jltype)
    result_jltype = lhs_type <: Tile ? replace_eltype(lhs_type, Bool) : Bool
    CGVal(result_v, result_type_id, result_jltype, result_shape)
end

# cuda_tile.divf
@eval Intrinsics begin
    @noinline divf(x::T, y::T) where {T<:AbstractFloat} = Core.Intrinsics.div_float(x, y)
    @noinline divf(a::Tile{T, S}, b::Tile{T, S}) where {T<:AbstractFloat, S} = (donotdelete(a, b); Tile{T, S}())
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.divf), args)
    emit_binop!(ctx, args, encode_DivFOp!)
end

# cuda_tile.mulf
@eval Intrinsics begin
    @noinline mulf(x::T, y::T) where {T<:AbstractFloat} = Core.Intrinsics.mul_float(x, y)
    @noinline mulf(a::Tile{T, S}, b::Tile{T, S}) where {T<:AbstractFloat, S} = (donotdelete(a, b); Tile{T, S}())
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.mulf), args)
    emit_binop!(ctx, args, encode_MulFOp!)
end

# cuda_tile.negf
@eval Intrinsics begin
    @noinline negf(x::T) where {T<:AbstractFloat} = Core.Intrinsics.neg_float(x)
    @noinline negf(a::Tile{T, S}) where {T<:AbstractFloat, S} = compilerbarrier(:const, a)
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.negf), args)
    emit_unop!(ctx, args, encode_NegFOp!)
end

# cuda_tile.subf
@eval Intrinsics begin
    @noinline subf(x::T, y::T) where {T<:AbstractFloat} = Core.Intrinsics.sub_float(x, y)
    @noinline subf(a::Tile{T, S}, b::Tile{T, S}) where {T<:AbstractFloat, S} = (donotdelete(a, b); Tile{T, S}())
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.subf), args)
    emit_binop!(ctx, args, encode_SubFOp!)
end


## Boolean arithmetic

# cuda_tile.andi
@eval Intrinsics begin
    @noinline andi(x::T, y::T) where {T<:Integer} = Core.Intrinsics.and_int(x, y)
    """Element-wise logical AND for boolean tiles."""
    @noinline andi(a::Tile{Bool, S}, b::Tile{Bool, S}) where {S} = (donotdelete(a, b); Tile{Bool, S}())
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.andi), args)
    cb = ctx.cb
    tt = ctx.tt

    lhs = emit_value!(ctx, args[1])
    rhs = emit_value!(ctx, args[2])

    (lhs === nothing || rhs === nothing) && error("andi: cannot resolve arguments")
    lhs isa CGVal || error("andi: lhs must be a CGVal")

    lhs_v = lhs.v
    rhs_v = rhs isa CGVal ? rhs.v : rhs
    result_shape = lhs.shape
    lhs_type = CC.widenconst(lhs.jltype)
    elem = lhs_type <: Tile ? eltype(lhs_type) : lhs_type

    dtype = julia_to_tile_dtype!(tt, elem)
    result_type_id = tile_type!(tt, dtype, result_shape)

    result = encode_AndIOp!(cb, result_type_id, lhs_v, rhs_v)
    CGVal(result, result_type_id, lhs.jltype, result_shape)
end

# cuda_tile.ori
@eval Intrinsics begin
    @noinline ori(x::T, y::T) where {T<:Integer} = Core.Intrinsics.or_int(x, y)
    """Element-wise logical OR for boolean tiles."""
    @noinline ori(a::Tile{Bool, S}, b::Tile{Bool, S}) where {S} = (donotdelete(a, b); Tile{Bool, S}())
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.ori), args)
    cb = ctx.cb
    tt = ctx.tt

    lhs = emit_value!(ctx, args[1])
    rhs = emit_value!(ctx, args[2])

    (lhs === nothing || rhs === nothing) && error("ori: cannot resolve arguments")
    lhs isa CGVal || error("ori: lhs must be a CGVal")

    lhs_v = lhs.v
    rhs_v = rhs isa CGVal ? rhs.v : rhs
    result_shape = lhs.shape
    lhs_type = CC.widenconst(lhs.jltype)
    elem = lhs_type <: Tile ? eltype(lhs_type) : lhs_type

    dtype = julia_to_tile_dtype!(tt, elem)
    result_type_id = tile_type!(tt, dtype, result_shape)

    result = encode_OrIOp!(cb, result_type_id, lhs_v, rhs_v)
    CGVal(result, result_type_id, lhs.jltype, result_shape)
end

# cuda_tile.xori
@eval Intrinsics begin
    @noinline xori(x::T, y::T) where {T<:Integer} = Core.Intrinsics.xor_int(x, y)
    """Element-wise logical XOR for boolean tiles."""
    @noinline xori(a::Tile{Bool, S}, b::Tile{Bool, S}) where {S} = (donotdelete(a, b); Tile{Bool, S}())
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.xori), args)
    cb = ctx.cb
    tt = ctx.tt

    lhs = emit_value!(ctx, args[1])
    rhs = emit_value!(ctx, args[2])

    (lhs === nothing || rhs === nothing) && error("xori: cannot resolve arguments")
    lhs isa CGVal || error("xori: lhs must be a CGVal")

    lhs_v = lhs.v
    rhs_v = rhs isa CGVal ? rhs.v : rhs
    result_shape = lhs.shape
    lhs_type = CC.widenconst(lhs.jltype)
    elem = lhs_type <: Tile ? eltype(lhs_type) : lhs_type

    dtype = julia_to_tile_dtype!(tt, elem)
    result_type_id = tile_type!(tt, dtype, result_shape)

    result = encode_XOrIOp!(cb, result_type_id, lhs_v, rhs_v)
    CGVal(result, result_type_id, lhs.jltype, result_shape)
end
