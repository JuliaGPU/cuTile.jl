# Integer, floating-point, and boolean arithmetic

## Helpers

# Broadcast the smaller-shaped operand to match the larger when shapes differ.
# This handles 0D constants meeting shaped tiles after constant folding.
function _broadcast_match_shapes!(cb, tt, lhs::CGVal, rhs::CGVal)
    lhs.shape == rhs.shape && return (lhs, rhs)
    elem_type = eltype(CC.widenconst(lhs.jltype))
    dtype = julia_to_tile_dtype!(tt, elem_type)
    if length(lhs.shape) < length(rhs.shape)
        bv = broadcast_tile_to_shape!(cb, tt, lhs, rhs.shape, dtype)
        lhs = CGVal(bv, tile_type!(tt, dtype, rhs.shape), rhs.jltype, rhs.shape,
                    nothing, lhs.constant, nothing)
    else
        bv = broadcast_tile_to_shape!(cb, tt, rhs, lhs.shape, dtype)
        rhs = CGVal(bv, tile_type!(tt, dtype, lhs.shape), lhs.jltype, lhs.shape,
                    nothing, rhs.constant, nothing)
    end
    return (lhs, rhs)
end

# Build rounding_mode/flush_to_zero kwargs from the active @fpmode scope.
function fpmode_kwargs(ctx::CGCtx)
    mode = current_fpmode(ctx)
    kwargs = NamedTuple()
    mode.rounding_mode !== nothing && (kwargs = (; kwargs..., rounding_mode=mode.rounding_mode))
    mode.flush_to_zero && (kwargs = (; kwargs..., flush_to_zero=true))
    kwargs
end

# Constant-fold scalar operations at codegen time.
# Returns a CGVal if all operands are compile-time constants and `op` is
# applicable to them, nothing otherwise.
function try_const_fold(ctx::CGCtx, op, args)
    vals = Any[]
    for arg in args
        c = get_constant(ctx, arg)
        c === nothing && return nothing
        val = something(c)
        val isa Number || return nothing
        push!(vals, val)
    end
    applicable(op, vals...) || return nothing
    emit_value!(ctx, op(vals...))
end

function emit_binop!(ctx::CGCtx, args, encoder::Function; kwargs...)
    cb = ctx.cb
    tt = ctx.tt

    lhs_tv = emit_value!(ctx, args[1])
    rhs_tv = emit_value!(ctx, args[2])

    (lhs_tv === nothing || rhs_tv === nothing) && return missing

    # After scalar_elim_pass!, all values are tile-typed.
    lhs_type = CC.widenconst(lhs_tv.jltype)
    rhs_type = CC.widenconst(rhs_tv.jltype)
    elem_type = eltype(lhs_type)
    eltype(rhs_type) === elem_type || throw(IRError("Binary op type mismatch: lhs element type $elem_type != rhs element type $(eltype(rhs_type))"))

    # Broadcast smaller-shaped operand to match the larger (e.g., 0D constant
    # meeting a shaped tile after constant folding removes reshape/broadcast)
    lhs_tv, rhs_tv = _broadcast_match_shapes!(cb, tt, lhs_tv, rhs_tv)
    result_shape = lhs_tv.shape
    result_jltype = lhs_tv.jltype

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

"""
    Intrinsics.absi(x::Tile{<:Integer}) -> Tile

Element-wise integer absolute value; lowers to `cuda_tile.absi`.

Also invocable with a scalar, promoted to a 0-D tile before codegen.
"""
@intrinsic absi(x::Integer)
@intrinsic absi(x::Tile{<:Integer})
tfunc(𝕃, ::typeof(Intrinsics.absi), @nospecialize(x)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.absi), args)
    @something try_const_fold(ctx, abs, args) emit_unop!(ctx, args, encode_AbsIOp!)
end

"""
    Intrinsics.addi(a::Tile{T}, b::Tile{T}) -> Tile{T}  where {T<:Integer}

Element-wise integer addition; lowers to `cuda_tile.addi`.

Also invocable with scalars, promoted to 0-D tiles before codegen.
Mismatched-shape operands are broadcast to a common shape.
"""
@intrinsic addi(x::T, y::T) where {T<:Integer}
@intrinsic addi(a::Tile{T}, b::Tile{T}) where {T<:Integer}
tfunc(𝕃, ::typeof(Intrinsics.addi), @nospecialize(x), @nospecialize(y)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.addi), args)
    @something try_const_fold(ctx, +, args) emit_binop!(ctx, args, encode_AddIOp!)
end

"""
    Intrinsics.cldi(a::Tile{T}, b::Tile{T}, s::Signedness.T) -> Tile{T}  where {T<:Integer}

Element-wise integer division rounded toward positive infinity (ceiling
division); lowers to `cuda_tile.divi` with `rounding=positive_inf`.

Also invocable with scalars, promoted to 0-D tiles before codegen. `s` is
a compile-time `Signedness.T` flag. Mismatched-shape operands are
broadcast to a common shape.
"""
@intrinsic cldi(x::T, y::T, s::Signedness.T) where {T<:Integer}
@intrinsic cldi(a::Tile{T}, b::Tile{T}, s::Signedness.T) where {T<:Integer}
tfunc(𝕃, ::typeof(Intrinsics.cldi), @nospecialize(x), @nospecialize(y), @nospecialize(s)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.cldi), args)
    signedness = @something get_constant(ctx, args[3]) throw(IRError("cldi requires compile-time signedness"))
    emit_binop!(ctx, args, encode_DivIOp!; signedness, rounding=RoundingMode.PositiveInf)
end

"""
    Intrinsics.cmpi(a::Tile{T}, b::Tile{T}, pred::ComparisonPredicate.T, s::Signedness.T) -> Tile{Bool}  where {T<:Integer}

Element-wise integer comparison; lowers to `cuda_tile.cmpi`.

Also invocable with scalars, promoted to 0-D tiles before codegen. `pred`
and `s` are compile-time `ComparisonPredicate.T` and `Signedness.T` flags.
Mismatched-shape operands are broadcast to a common shape.
"""
@intrinsic cmpi(x::T, y::T, pred::ComparisonPredicate.T, s::Signedness.T) where {T<:Integer}
@intrinsic cmpi(a::Tile{T}, b::Tile{T}, pred::ComparisonPredicate.T, s::Signedness.T) where {T<:Integer}
function tfunc(𝕃, ::typeof(Intrinsics.cmpi), @nospecialize(x), @nospecialize(y), @nospecialize(pred), @nospecialize(s))
    t = CC.widenconst(x)
    if t isa DataType && t <: Tile
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

    # Broadcast mismatched shapes (e.g., 0D constant vs shaped tile)
    lhs, rhs = _broadcast_match_shapes!(cb, tt, lhs, rhs)

    result_shape = lhs.shape

    bool_dtype = I1(tt)
    result_type_id = tile_type!(tt, bool_dtype, result_shape)

    result_v = encode_CmpIOp!(cb, result_type_id, lhs.v, rhs.v; predicate, signedness)
    lhs_type = CC.widenconst(lhs.jltype)
    result_jltype = similar_type(lhs_type, Bool)
    CGVal(result_v, result_type_id, result_jltype, result_shape)
end

"""
    Intrinsics.divi(a::Tile{T}, b::Tile{T}, s::Signedness.T) -> Tile{T}  where {T<:Integer}

Element-wise integer division rounded toward zero (truncating division);
lowers to `cuda_tile.divi` with `rounding=zero`.

Also invocable with scalars, promoted to 0-D tiles before codegen. `s` is
a compile-time `Signedness.T` flag. Mismatched-shape operands are
broadcast to a common shape.
"""
@intrinsic divi(x::T, y::T, s::Signedness.T) where {T<:Integer}
@intrinsic divi(a::Tile{T}, b::Tile{T}, s::Signedness.T) where {T<:Integer}
tfunc(𝕃, ::typeof(Intrinsics.divi), @nospecialize(x), @nospecialize(y), @nospecialize(s)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.divi), args)
    signedness = @something get_constant(ctx, args[3]) throw(IRError("divi requires compile-time signedness"))
    emit_binop!(ctx, args, encode_DivIOp!; signedness, rounding=RoundingMode.Zero)
end

"""
    Intrinsics.fldi(a::Tile{T}, b::Tile{T}, s::Signedness.T) -> Tile{T}  where {T<:Integer}

Element-wise integer division rounded toward negative infinity (floor
division); lowers to `cuda_tile.divi` with `rounding=negative_inf`.

Also invocable with scalars, promoted to 0-D tiles before codegen. `s` is
a compile-time `Signedness.T` flag (using `unsigned` is invalid for floor
division). Mismatched-shape operands are broadcast to a common shape.
"""
@intrinsic fldi(x::T, y::T, s::Signedness.T) where {T<:Integer}
@intrinsic fldi(a::Tile{T}, b::Tile{T}, s::Signedness.T) where {T<:Integer}
tfunc(𝕃, ::typeof(Intrinsics.fldi), @nospecialize(x), @nospecialize(y), @nospecialize(s)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.fldi), args)
    signedness = @something get_constant(ctx, args[3]) throw(IRError("fldi requires compile-time signedness"))
    emit_binop!(ctx, args, encode_DivIOp!; signedness, rounding=RoundingMode.NegativeInf)
end

"""
    Intrinsics.maxi(a::Tile{T}, b::Tile{T}, s::Signedness.T) -> Tile{T}  where {T<:Integer}

Element-wise integer maximum; lowers to `cuda_tile.maxi`.

Also invocable with scalars, promoted to 0-D tiles before codegen. `s` is
a compile-time `Signedness.T` flag. Mismatched-shape operands are
broadcast to a common shape.
"""
@intrinsic maxi(x::T, y::T, s::Signedness.T) where {T<:Integer}
@intrinsic maxi(a::Tile{T}, b::Tile{T}, s::Signedness.T) where {T<:Integer}
tfunc(𝕃, ::typeof(Intrinsics.maxi), @nospecialize(x), @nospecialize(y), @nospecialize(s)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.maxi), args)
    signedness = @something get_constant(ctx, args[3]) throw(IRError("maxi requires compile-time signedness"))
    emit_binop!(ctx, args, encode_MaxIOp!; signedness)
end

"""
    Intrinsics.mini(a::Tile{T}, b::Tile{T}, s::Signedness.T) -> Tile{T}  where {T<:Integer}

Element-wise integer minimum; lowers to `cuda_tile.mini`.

Also invocable with scalars, promoted to 0-D tiles before codegen. `s` is
a compile-time `Signedness.T` flag. Mismatched-shape operands are
broadcast to a common shape.
"""
@intrinsic mini(x::T, y::T, s::Signedness.T) where {T<:Integer}
@intrinsic mini(a::Tile{T}, b::Tile{T}, s::Signedness.T) where {T<:Integer}
tfunc(𝕃, ::typeof(Intrinsics.mini), @nospecialize(x), @nospecialize(y), @nospecialize(s)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.mini), args)
    signedness = @something get_constant(ctx, args[3]) throw(IRError("mini requires compile-time signedness"))
    emit_binop!(ctx, args, encode_MinIOp!; signedness)
end

"""
    Intrinsics.muli(a::Tile{T}, b::Tile{T}) -> Tile{T}  where {T<:Integer}

Element-wise integer multiplication, returning the low half of the
double-width product; lowers to `cuda_tile.muli`.

Also invocable with scalars, promoted to 0-D tiles before codegen.
Mismatched-shape operands are broadcast to a common shape.
"""
@intrinsic muli(x::T, y::T) where {T<:Integer}
@intrinsic muli(a::Tile{T}, b::Tile{T}) where {T<:Integer}
tfunc(𝕃, ::typeof(Intrinsics.muli), @nospecialize(x), @nospecialize(y)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.muli), args)
    @something try_const_fold(ctx, *, args) emit_binop!(ctx, args, encode_MulIOp!)
end

"""
    Intrinsics.mulhii(a::Tile{T}, b::Tile{T}, s::Signedness.T) -> Tile{T}  where {T<:Integer}

Element-wise integer multiplication, returning the high half of the
double-width product; lowers to `cuda_tile.mulhii`.

Also invocable with scalars, promoted to 0-D tiles before codegen. The
Tile IR op is unsigned-only, so the `s` operand is accepted for signature
uniformity with other integer ops but ignored at codegen. Mismatched-shape
operands are broadcast to a common shape.
"""
@intrinsic mulhii(x::T, y::T, s::Signedness.T) where {T<:Integer}
@intrinsic mulhii(a::Tile{T}, b::Tile{T}, s::Signedness.T) where {T<:Integer}
tfunc(𝕃, ::typeof(Intrinsics.mulhii), @nospecialize(x), @nospecialize(y), @nospecialize(s)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.mulhii), args)
    emit_binop!(ctx, args, encode_MulhiIOp!)
end

"""
    Intrinsics.negi(a::Tile{<:Integer}) -> Tile

Element-wise integer negation; lowers to `cuda_tile.negi` with no overflow
assumption (`overflow=none`).

Also invocable with a scalar, promoted to a 0-D tile before codegen.
"""
@intrinsic negi(x::T) where {T<:Integer}
@intrinsic negi(a::Tile{<:Integer})
tfunc(𝕃, ::typeof(Intrinsics.negi), @nospecialize(x)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.negi), args)
    @something try_const_fold(ctx, -, args) emit_unop!(ctx, args, encode_NegIOp!; overflow=IntegerOverflow.None)
end

"""
    Intrinsics.remi(a::Tile{T}, b::Tile{T}, s::Signedness.T) -> Tile{T}  where {T<:Integer}

Element-wise integer remainder using truncated division (sign of dividend);
lowers to `cuda_tile.remi`.

Also invocable with scalars, promoted to 0-D tiles before codegen. `s` is
a compile-time `Signedness.T` flag. Mismatched-shape operands are
broadcast to a common shape.
"""
@intrinsic remi(x::T, y::T, s::Signedness.T) where {T<:Integer}
@intrinsic remi(a::Tile{T}, b::Tile{T}, s::Signedness.T) where {T<:Integer}
tfunc(𝕃, ::typeof(Intrinsics.remi), @nospecialize(x), @nospecialize(y), @nospecialize(s)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.remi), args)
    signedness = @something get_constant(ctx, args[3]) throw(IRError("remi requires compile-time signedness"))
    emit_binop!(ctx, args, encode_RemIOp!; signedness)
end

"""
    Intrinsics.shli(a::Tile{T}, b::Tile{T}) -> Tile{T}    where {T<:Integer}

Element-wise integer left shift, zero-filling the low bits; lowers to
`cuda_tile.shli`.

Also invocable with scalars, promoted to 0-D tiles before codegen. The
shift amount is interpreted as unsigned. Mismatched-shape operands are
broadcast to a common shape.
"""
@intrinsic shli(x::T, y::Integer) where {T<:Integer}
@intrinsic shli(a::Tile{T}, b::Tile{T}) where {T<:Integer}
tfunc(𝕃, ::typeof(Intrinsics.shli), @nospecialize(x), @nospecialize(y)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.shli), args)
    @something try_const_fold(ctx, <<, args) emit_binop!(ctx, args, encode_ShLIOp!)
end

"""
    Intrinsics.shri(a::Tile{T}, b::Tile{T}, s::Signedness.T) -> Tile{T}    where {T<:Integer}

Element-wise integer right shift; lowers to `cuda_tile.shri`.

Also invocable with scalars, promoted to 0-D tiles before codegen. `s` is
a compile-time `Signedness.T` flag controlling whether the shift is
arithmetic (signed) or logical (unsigned). The shift amount is always
interpreted as unsigned. Mismatched-shape operands are broadcast to a
common shape.
"""
@intrinsic shri(x::T, y::Integer, s::Signedness.T) where {T<:Integer}
@intrinsic shri(a::Tile{T}, b::Tile{T}, s::Signedness.T) where {T<:Integer}
tfunc(𝕃, ::typeof(Intrinsics.shri), @nospecialize(x), @nospecialize(y), @nospecialize(s)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.shri), args)
    signedness = @something get_constant(ctx, args[3]) throw(IRError("shri requires compile-time signedness"))
    emit_binop!(ctx, args, encode_ShRIOp!; signedness)
end

"""
    Intrinsics.subi(a::Tile{T}, b::Tile{T}) -> Tile{T}  where {T<:Integer}

Element-wise integer subtraction; lowers to `cuda_tile.subi`.

Also invocable with scalars, promoted to 0-D tiles before codegen.
Mismatched-shape operands are broadcast to a common shape.
"""
@intrinsic subi(x::T, y::T) where {T<:Integer}
@intrinsic subi(a::Tile{T}, b::Tile{T}) where {T<:Integer}
tfunc(𝕃, ::typeof(Intrinsics.subi), @nospecialize(x), @nospecialize(y)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.subi), args)
    @something try_const_fold(ctx, -, args) emit_binop!(ctx, args, encode_SubIOp!)
end


## Floating-point arithmetic

"""
    Intrinsics.absf(a::Tile{<:AbstractFloat}) -> Tile

Element-wise floating-point absolute value; lowers to `cuda_tile.absf`.

Also invocable with a scalar, promoted to a 0-D tile before codegen.
"""
@intrinsic absf(x::T) where {T<:AbstractFloat}
@intrinsic absf(a::Tile{<:AbstractFloat})
tfunc(𝕃, ::typeof(Intrinsics.absf), @nospecialize(x)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.absf), args)
    @something try_const_fold(ctx, abs, args) emit_unop!(ctx, args, encode_AbsFOp!)
end

"""
    Intrinsics.addf(a::Tile{T}, b::Tile{T}) -> Tile{T}  where {T<:AbstractFloat}

Element-wise floating-point addition; lowers to `cuda_tile.addf`.

Also invocable with scalars, promoted to 0-D tiles before codegen. The
active `@fpmode` scope supplies the `rounding_mode` and `flush_to_zero`
attributes. Mismatched-shape operands are broadcast to a common shape.
"""
@intrinsic addf(x::T, y::T) where {T<:AbstractFloat}
@intrinsic addf(a::Tile{T}, b::Tile{T}) where {T<:AbstractFloat}
tfunc(𝕃, ::typeof(Intrinsics.addf), @nospecialize(x), @nospecialize(y)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.addf), args)
    @something try_const_fold(ctx, +, args) emit_binop!(ctx, args, encode_AddFOp!; fpmode_kwargs(ctx)...)
end

"""
    Intrinsics.cmpf(a::Tile{T}, b::Tile{T}, pred::ComparisonPredicate.T,
                    ord::ComparisonOrdering.T = Ordered) -> Tile{Bool}  where {T<:AbstractFloat}

Element-wise floating-point comparison; lowers to `cuda_tile.cmpf`.

Also invocable with scalars, promoted to 0-D tiles before codegen. `pred`
and `ord` are compile-time `ComparisonPredicate.T` and
`ComparisonOrdering.T` flags. Mismatched-shape operands are broadcast to
a common shape.
"""
@intrinsic cmpf(x::T, y::T, pred::ComparisonPredicate.T) where {T<:AbstractFloat}
@intrinsic cmpf(a::Tile{T}, b::Tile{T}, pred::ComparisonPredicate.T) where {T<:AbstractFloat}
@intrinsic cmpf(x::T, y::T, pred::ComparisonPredicate.T, ord::ComparisonOrdering.T) where {T<:AbstractFloat}
@intrinsic cmpf(a::Tile{T}, b::Tile{T}, pred::ComparisonPredicate.T, ord::ComparisonOrdering.T) where {T<:AbstractFloat}
function tfunc(𝕃, ::typeof(Intrinsics.cmpf), @nospecialize(x), @nospecialize(y), @nospecialize(pred))
    t = CC.widenconst(x)
    if t isa DataType && t <: Tile
        S = t.parameters[2]
        return Tile{Bool, S}
    end
    return Bool
end
function tfunc(𝕃, ::typeof(Intrinsics.cmpf), @nospecialize(x), @nospecialize(y), @nospecialize(pred), @nospecialize(ord))
    t = CC.widenconst(x)
    if t isa DataType && t <: Tile
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
    ordering = if length(args) >= 4
        @something get_constant(ctx, args[4]) throw(IRError("cmpf: requires compile-time ordering"))
    else
        ComparisonOrdering.Ordered
    end

    # Broadcast mismatched shapes (e.g., 0D constant vs shaped tile)
    lhs, rhs = _broadcast_match_shapes!(cb, tt, lhs, rhs)

    result_shape = lhs.shape

    bool_dtype = I1(tt)
    result_type_id = tile_type!(tt, bool_dtype, result_shape)

    result_v = encode_CmpFOp!(cb, result_type_id, lhs.v, rhs.v; predicate, ordering)
    lhs_type = CC.widenconst(lhs.jltype)
    result_jltype = similar_type(lhs_type, Bool)
    CGVal(result_v, result_type_id, result_jltype, result_shape)
end

"""
    Intrinsics.divf(a::Tile{T}, b::Tile{T}) -> Tile{T}  where {T<:AbstractFloat}

Element-wise floating-point division; lowers to `cuda_tile.divf`.

Also invocable with scalars, promoted to 0-D tiles before codegen. The
active `@fpmode` scope supplies the `rounding_mode` and `flush_to_zero`
attributes. Mismatched-shape operands are broadcast to a common shape.
"""
@intrinsic divf(x::T, y::T) where {T<:AbstractFloat}
@intrinsic divf(a::Tile{T}, b::Tile{T}) where {T<:AbstractFloat}
tfunc(𝕃, ::typeof(Intrinsics.divf), @nospecialize(x), @nospecialize(y)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.divf), args)
    @something try_const_fold(ctx, /, args) emit_binop!(ctx, args, encode_DivFOp!; fpmode_kwargs(ctx)...)
end

"""
    Intrinsics.mulf(a::Tile{T}, b::Tile{T}) -> Tile{T}  where {T<:AbstractFloat}

Element-wise floating-point multiplication; lowers to `cuda_tile.mulf`.

Also invocable with scalars, promoted to 0-D tiles before codegen. The
active `@fpmode` scope supplies the `rounding_mode` and `flush_to_zero`
attributes. Mismatched-shape operands are broadcast to a common shape.
"""
@intrinsic mulf(x::T, y::T) where {T<:AbstractFloat}
@intrinsic mulf(a::Tile{T}, b::Tile{T}) where {T<:AbstractFloat}
tfunc(𝕃, ::typeof(Intrinsics.mulf), @nospecialize(x), @nospecialize(y)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.mulf), args)
    @something try_const_fold(ctx, *, args) emit_binop!(ctx, args, encode_MulFOp!; fpmode_kwargs(ctx)...)
end

"""
    Intrinsics.negf(a::Tile{<:AbstractFloat}) -> Tile

Element-wise floating-point negation; lowers to `cuda_tile.negf`.

Also invocable with a scalar, promoted to a 0-D tile before codegen.
"""
@intrinsic negf(x::T) where {T<:AbstractFloat}
@intrinsic negf(a::Tile{<:AbstractFloat})
tfunc(𝕃, ::typeof(Intrinsics.negf), @nospecialize(x)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.negf), args)
    @something try_const_fold(ctx, -, args) emit_unop!(ctx, args, encode_NegFOp!)
end

"""
    Intrinsics.subf(a::Tile{T}, b::Tile{T}) -> Tile{T}  where {T<:AbstractFloat}

Element-wise floating-point subtraction; lowers to `cuda_tile.subf`.

Also invocable with scalars, promoted to 0-D tiles before codegen. The
active `@fpmode` scope supplies the `rounding_mode` and `flush_to_zero`
attributes. Mismatched-shape operands are broadcast to a common shape.
"""
@intrinsic subf(x::T, y::T) where {T<:AbstractFloat}
@intrinsic subf(a::Tile{T}, b::Tile{T}) where {T<:AbstractFloat}
tfunc(𝕃, ::typeof(Intrinsics.subf), @nospecialize(x), @nospecialize(y)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.subf), args)
    @something try_const_fold(ctx, -, args) emit_binop!(ctx, args, encode_SubFOp!; fpmode_kwargs(ctx)...)
end


## Boolean arithmetic

"""
    Intrinsics.andi(a::Tile{T}, b::Tile{T}) -> Tile{T}  where {T<:Integer}

Element-wise bitwise AND; lowers to `cuda_tile.andi`.

Also invocable with scalars, promoted to 0-D tiles before codegen. The
tfunc short-circuits to `Const(false)` when one operand is constant
`false` (avoiding unnecessary inference of the other operand).
Mismatched-shape operands are broadcast to a common shape, and
"""
@intrinsic andi(x::T, y::T) where {T<:Integer}
@intrinsic andi(a::Tile{T}, b::Tile{T}) where {T<:Integer}
function tfunc(𝕃, ::typeof(Intrinsics.andi), @nospecialize(x), @nospecialize(y))
    if isa(x, CC.Const) && x.val === false && CC.widenconst(y) === Bool
        return CC.Const(false)
    elseif isa(y, CC.Const) && y.val === false && CC.widenconst(x) === Bool
        return CC.Const(false)
    end
    return CC.widenconst(x)
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.andi), args)
    @something try_const_fold(ctx, &, args) emit_binop!(ctx, args, encode_AndIOp!)
end

"""
    Intrinsics.ori(a::Tile{T}, b::Tile{T}) -> Tile{T}  where {T<:Integer}

Element-wise bitwise OR; lowers to `cuda_tile.ori`.

Also invocable with scalars, promoted to 0-D tiles before codegen. The
tfunc short-circuits to `Const(true)` when one operand is constant `true`
(avoiding unnecessary inference of the other operand). Mismatched-shape
operands are broadcast to a common shape.
"""
@intrinsic ori(x::T, y::T) where {T<:Integer}
@intrinsic ori(a::Tile{T}, b::Tile{T}) where {T<:Integer}
function tfunc(𝕃, ::typeof(Intrinsics.ori), @nospecialize(x), @nospecialize(y))
    if isa(x, CC.Const) && x.val === true && CC.widenconst(y) === Bool
        return CC.Const(true)
    elseif isa(y, CC.Const) && y.val === true && CC.widenconst(x) === Bool
        return CC.Const(true)
    end
    return CC.widenconst(x)
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.ori), args)
    @something try_const_fold(ctx, |, args) emit_binop!(ctx, args, encode_OrIOp!)
end

"""
    Intrinsics.xori(a::Tile{T}, b::Tile{T}) -> Tile{T}  where {T<:Integer}

Element-wise bitwise XOR; lowers to `cuda_tile.xori`.

Also invocable with scalars, promoted to 0-D tiles before codegen.
Mismatched-shape operands are broadcast to a common shape.
"""
@intrinsic xori(x::T, y::T) where {T<:Integer}
@intrinsic xori(a::Tile{T}, b::Tile{T}) where {T<:Integer}
tfunc(𝕃, ::typeof(Intrinsics.xori), @nospecialize(x), @nospecialize(y)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.xori), args)
    @something try_const_fold(ctx, xor, args) emit_binop!(ctx, args, encode_XOrIOp!)
end
