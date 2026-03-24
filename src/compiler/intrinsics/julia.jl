# Julia intrinsics

# Handle Julia Core.Intrinsics that IRStructurizer uses for control flow transformations.
# These are: add_int / sub_int (loop increments), slt_int / sle_int / ult_int (loop bounds),
# and not_int (bitwise NOT, used by `for` loop iteration).
function emit_intrinsic!(ctx::CGCtx, func::Core.IntrinsicFunction, args)
    if func === Core.Intrinsics.add_int
        emit_intrinsic!(ctx, Intrinsics.addi, args)
    elseif func === Core.Intrinsics.sub_int
        emit_intrinsic!(ctx, Intrinsics.subi, args)
    elseif func === Core.Intrinsics.slt_int
        emit_intrinsic!(ctx, Intrinsics.cmpi, [args..., ComparisonPredicate.LessThan, Signedness.Signed])
    elseif func === Core.Intrinsics.sle_int
        emit_intrinsic!(ctx, Intrinsics.cmpi, [args..., ComparisonPredicate.LessThanOrEqual, Signedness.Signed])
    elseif func === Core.Intrinsics.ult_int
        emit_intrinsic!(ctx, Intrinsics.cmpi, [args..., ComparisonPredicate.LessThan, Signedness.Unsigned])
    elseif func === Core.Intrinsics.not_int
        emit_not_int!(ctx, args)
    else
        throw(IRError("Unhandled Julia intrinsic: $func"))
    end
end

# not_int(x) — bitwise NOT.
# Julia's `for` loops generate `not_int` to negate loop-exit conditions.
# Emitted as xori(x, allones) where allones is the bitwise complement identity:
#   Bool    → xori(x, true)     (logical negation)
#   Integer → xori(x, -1)       (bitwise complement, all bits set)
function emit_not_int!(ctx::CGCtx, args)
    cb = ctx.cb
    tt = ctx.tt

    operand = @something emit_value!(ctx, args[1]) throw(IRError("not_int: cannot resolve operand"))
    jltype = CC.widenconst(operand.jltype)

    # Build the all-ones constant for xori: true for Bool, -1 (all bits set) for integers
    allones_val = jltype === Bool ? true : jltype(-1)
    type_id = tile_type_for_julia!(ctx, jltype)
    allones_bytes = reinterpret(UInt8, [allones_val])
    allones_v = encode_ConstantOp!(cb, type_id, collect(allones_bytes))

    result = encode_XOrIOp!(cb, type_id, operand.v, allones_v)
    CGVal(result, type_id, operand.jltype, operand.shape)
end

# built-in: ===
function emit_intrinsic!(ctx::CGCtx, ::typeof(===), args)
    cb = ctx.cb
    tt = ctx.tt

    lhs = @something emit_value!(ctx, args[1]) throw(IRError("===: cannot resolve lhs"))
    rhs = @something emit_value!(ctx, args[2]) throw(IRError("===: cannot resolve rhs"))

    result_type_id = tile_type!(tt, I1(tt), Int[])

    result_v = encode_CmpIOp!(cb, result_type_id, lhs.v, rhs.v;
                              predicate=ComparisonPredicate.Equal, signedness=Signedness.Signed)

    CGVal(result_v, result_type_id, Bool, Int[])
end

# built-in: tuple
emit_intrinsic!(ctx::CGCtx, ::typeof(Core.tuple), args) = nothing

# built-in: isa
emit_intrinsic!(ctx::CGCtx, ::typeof(isa), args) = nothing

# built-in: donotdelete
emit_intrinsic!(ctx::CGCtx, ::typeof(donotdelete), args) = nothing

# built-in: ifelse
emit_intrinsic!(ctx::CGCtx, ::typeof(Core.ifelse), args) =
    emit_intrinsic!(ctx, Intrinsics.select, args)

# XXX: Tile constructor
function emit_intrinsic!(ctx::CGCtx, func::Type{<:Tile}, args)
    # Emit the scalar value
    source = emit_value!(ctx, args[1])

    # Get element type from the constructor type (Tile{T, S})
    # If func is fully parametric, extract T; otherwise infer from source
    elem_type = if func !== Tile && length(func.parameters) >= 1
        func.parameters[1]
    elseif source.constant !== nothing
        typeof(something(source.constant))
    else
        CC.widenconst(source.jltype)
    end

    # Return as 0D tile type with element type from the constructor
    result_jltype = Tile{elem_type, Tuple{}}
    CGVal(source.v, source.type_id, result_jltype, source.shape, nothing, source.constant, nothing)
end
