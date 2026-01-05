"""
    emit_binop!(ctx, args, float_encoder, int_encoder)

Binary operation emitter.

Handles:
- Tile + Tile (same shapes - broadcasting is done at intrinsic level via broadcast_to)
- Scalar + Scalar (for integer intrinsics on index calculations)

Note: tile+scalar operations are handled at the intrinsic level via Tile(scalar) and
broadcast_to(), so by the time we reach tile_add etc., both operands are already tiles.
"""
function emit_binop!(ctx::CodegenContext, args, float_encoder::Function, int_encoder::Function)
    cb = ctx.cb
    tt = ctx.tt

    # Emit both operands
    lhs_tv = emit_value!(ctx, args[1])
    rhs_tv = emit_value!(ctx, args[2])

    # Both operands must resolve to CGVals
    if lhs_tv === nothing || rhs_tv === nothing
        return missing
    end

    # Determine what kind of operands we have
    lhs_is_tile = unwrap_type(lhs_tv.jltype) <: Tile
    rhs_is_tile = unwrap_type(rhs_tv.jltype) <: Tile

    if lhs_is_tile && rhs_is_tile
        # Tile + Tile: shapes should be identical (broadcasting via broadcast_to at intrinsic level)
        elem_type = unwrap_type(lhs_tv.jltype).parameters[1]
        result_shape = lhs_tv.shape
        lhs_v = lhs_tv.v
        rhs_v = rhs_tv.v
        result_jltype = Tile{elem_type, Tuple(result_shape)}
    elseif !lhs_is_tile && !rhs_is_tile
        # Scalar + Scalar: for integer intrinsics on index calculations
        elem_type = unwrap_type(lhs_tv.jltype)
        result_shape = Int[]
        lhs_v = lhs_tv.v
        rhs_v = rhs_tv.v
        result_jltype = elem_type
    else
        error("Mixed tile/scalar operations should be handled at intrinsic level via Tile() and broadcast_to()")
    end

    dtype = julia_to_tile_dtype!(tt, elem_type)
    result_type_id = tile_type!(tt, dtype, result_shape)

    # Emit the binary operation
    if elem_type <: AbstractFloat
        result_v = float_encoder(cb, result_type_id, lhs_v, rhs_v)
    else
        result_v = int_encoder(cb, result_type_id, lhs_v, rhs_v)
    end

    CGVal(result_v, result_type_id, result_jltype, result_shape)
end

# Helper to emit binary operation with pre-resolved operands
function emit_binop_inner!(ctx::CodegenContext, lhs_tv::CGVal, rhs_tv::CGVal,
                           float_encoder::Function, int_encoder::Function)
    cb = ctx.cb
    tt = ctx.tt

    # Determine element type
    elem_type = unwrap_type(lhs_tv.jltype)
    if elem_type <: Tile
        elem_type = elem_type.parameters[1]
    end

    result_shape = lhs_tv.shape
    result_jltype = Tile{elem_type, Tuple(result_shape)}

    dtype = julia_to_tile_dtype!(tt, elem_type)
    result_type_id = tile_type!(tt, dtype, result_shape)

    lhs_v = lhs_tv.v
    rhs_v = rhs_tv.v

    result_v = if elem_type <: AbstractFloat
        float_encoder(cb, result_type_id, lhs_v, rhs_v)
    else
        int_encoder(cb, result_type_id, lhs_v, rhs_v)
    end

    CGVal(result_v, result_type_id, result_jltype, result_shape)
end

#=============================================================================
 8.7. Floating Point + 8.8. Integer Arithmetic
=============================================================================#

#=============================================================================
 8.7. Floating Point
 cuda_tile.addf, cuda_tile.divf, cuda_tile.mulf, cuda_tile.pow,
 cuda_tile.rsqrt, cuda_tile.sqrt, cuda_tile.subf
=============================================================================#

#-----------------------------------------------------------------------------
# cuda_tile.sqrt
#-----------------------------------------------------------------------------

function emit_intrinsic!(ctx::CodegenContext, ::typeof(Intrinsics.sqrt), args, @nospecialize(result_type))
    cb = ctx.cb

    source = emit_value!(ctx, args[1])
    source === nothing && error("Cannot resolve operand for sqrt()")

    result = encode_SqrtOp!(cb, source.type_id, source.v)

    CGVal(result, source.type_id, source.jltype, source.shape)
end

#=============================================================================
 8.8. Integer
 cuda_tile.addi, cuda_tile.divi, cuda_tile.muli, cuda_tile.subi
=============================================================================#

#-----------------------------------------------------------------------------
# cuda_tile.divi (cdiv, floordiv)
#-----------------------------------------------------------------------------

function emit_intrinsic!(ctx::CodegenContext, ::typeof(Intrinsics.cdiv), args, @nospecialize(result_type))
    cb = ctx.cb
    tt = ctx.tt

    scalar_i32 = tile_type!(tt, I32(tt), Int[])

    a = resolve_or_constant(ctx, args[1], scalar_i32)
    b = resolve_or_constant(ctx, args[2], scalar_i32)

    one_val = encode_ConstantOp!(cb, scalar_i32, collect(reinterpret(UInt8, [Int32(1)])))

    sum1 = encode_AddIOp!(cb, scalar_i32, a, b)
    sum2 = encode_SubIOp!(cb, scalar_i32, sum1, one_val)
    result = encode_DivIOp!(cb, scalar_i32, sum2, b; signedness=SignednessSigned, rounding=RoundingZero)

    CGVal(result, scalar_i32, Int32)
end

function emit_intrinsic!(ctx::CodegenContext, ::typeof(Intrinsics.floordiv), args, @nospecialize(result_type))
    cb = ctx.cb
    tt = ctx.tt

    scalar_i32 = tile_type!(tt, I32(tt), Int[])

    a = resolve_or_constant(ctx, args[1], scalar_i32)
    b = resolve_or_constant(ctx, args[2], scalar_i32)

    result = encode_DivIOp!(cb, scalar_i32, a, b; signedness=SignednessSigned, rounding=RoundingZero)

    CGVal(result, scalar_i32, Int32)
end

#-----------------------------------------------------------------------------
# Base.rem, Base.min (integer operations)
#-----------------------------------------------------------------------------

function emit_intrinsic!(ctx::CodegenContext, ::typeof(Base.rem), args, @nospecialize(result_type))
    cb = ctx.cb
    tt = ctx.tt

    scalar_i32 = tile_type!(tt, I32(tt), Int[])

    a = resolve_or_constant(ctx, args[1], scalar_i32)
    b = resolve_or_constant(ctx, args[2], scalar_i32)

    result = encode_RemIOp!(cb, scalar_i32, a, b; signedness=SignednessSigned)

    CGVal(result, scalar_i32, Int32)
end

function emit_intrinsic!(ctx::CodegenContext, ::typeof(Base.min), args, @nospecialize(result_type))
    cb = ctx.cb
    tt = ctx.tt

    scalar_i32 = tile_type!(tt, I32(tt), Int[])

    a = resolve_or_constant(ctx, args[1], scalar_i32)
    b = resolve_or_constant(ctx, args[2], scalar_i32)

    result = encode_MinIOp!(cb, scalar_i32, a, b; signedness=SignednessSigned)

    CGVal(result, scalar_i32, Int32)
end

#-----------------------------------------------------------------------------
# Intrinsics.arith - unified arithmetic (dispatches on operator, branches on element type)
#-----------------------------------------------------------------------------

function emit_intrinsic!(ctx::CodegenContext, ::typeof(Intrinsics.arith), args, @nospecialize(_))
    cb = ctx.cb
    tt = ctx.tt

    lhs_tv = emit_value!(ctx, args[1])
    rhs_tv = emit_value!(ctx, args[2])

    (lhs_tv === nothing || rhs_tv === nothing) && error("Cannot resolve operands for arith")

    # Get the operator from the third argument (a ghost value with the function)
    op_tv = emit_value!(ctx, args[3])
    op = op_tv.constant

    # Power is float-only
    if op === (^)
        elem_type = unwrap_type(lhs_tv.jltype)
        if elem_type <: Tile
            elem_type = elem_type.parameters[1]
        end
        elem_type <: AbstractFloat || error("power (^) only supports float types, got $elem_type")

        result_shape = lhs_tv.shape
        result_jltype = Tile{elem_type, Tuple(result_shape)}

        dtype = julia_to_tile_dtype!(tt, elem_type)
        result_type_id = tile_type!(tt, dtype, result_shape)

        result_v = encode_PowOp!(cb, result_type_id, lhs_tv.v, rhs_tv.v)

        return CGVal(result_v, result_type_id, result_jltype, result_shape)
    end

    # Map operator to encoder functions
    float_encoder, int_encoder = if op === (+)
        (encode_AddFOp!, encode_AddIOp!)
    elseif op === (-)
        (encode_SubFOp!, encode_SubIOp!)
    elseif op === (*)
        (encode_MulFOp!, encode_MulIOp!)
    elseif op === (/)
        (encode_DivFOp!, encode_DivIOp!)
    else
        error("Unknown arithmetic operator: $op")
    end

    emit_binop_inner!(ctx, lhs_tv, rhs_tv, float_encoder, int_encoder)
end

#-----------------------------------------------------------------------------
# Core.IntrinsicFunction - Julia's integer intrinsics
#-----------------------------------------------------------------------------

function emit_intrinsic!(ctx::CodegenContext, func::Core.IntrinsicFunction, args, @nospecialize(_))
    # Integer arithmetic
    if func === Base.add_int
        return emit_binop!(ctx, args, encode_AddFOp!, encode_AddIOp!)
    elseif func === Base.sub_int
        return emit_binop!(ctx, args, encode_SubFOp!, encode_SubIOp!)
    elseif func === Base.mul_int
        return emit_binop!(ctx, args, encode_MulFOp!, encode_MulIOp!)
    # Integer to float conversions
    elseif func === Base.sitofp
        return emit_sitofp!(ctx, args)
    elseif func === Base.uitofp
        return emit_uitofp!(ctx, args)
    # Integer comparison intrinsics (signed and unsigned use same predicate, signedness is separate)
    elseif func === Base.slt_int
        return emit_int_cmp!(ctx, args, CmpLessThan, SignednessSigned)
    elseif func === Base.sle_int
        return emit_int_cmp!(ctx, args, CmpLessThanOrEqual, SignednessSigned)
    elseif func === Base.ult_int
        return emit_int_cmp!(ctx, args, CmpLessThan, SignednessUnsigned)
    elseif func === Base.ule_int
        return emit_int_cmp!(ctx, args, CmpLessThanOrEqual, SignednessUnsigned)
    elseif func === Base.eq_int
        return emit_int_cmp!(ctx, args, CmpEqual, SignednessSigned)
    elseif func === Base.ne_int
        return emit_int_cmp!(ctx, args, CmpNotEqual, SignednessSigned)
    elseif func === Base.not_int
        return emit_not_int!(ctx, args)
    end
    missing  # Unknown intrinsic
end
