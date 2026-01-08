# high-level intrinsics
#
# These handle Julia built-ins and map them to Tile IR intrinsics.


## Julia intrinsics

# Handle Julia Core.Intrinsics that IRStructurizer uses for control flow transformations.
# These are: add_int (loop increments), slt_int, sle_int, ult_int (loop bounds).

function emit_intrinsic!(ctx::CGCtx, func::Core.IntrinsicFunction, args)
    if func === Core.Intrinsics.add_int
        emit_intrinsic!(ctx, Intrinsics.addi, args)
    elseif func === Core.Intrinsics.sub_int
        emit_intrinsic!(ctx, Intrinsics.subi, args)
    elseif func === Core.Intrinsics.slt_int
        emit_intrinsic!(ctx, Intrinsics.cmpi, [args..., CmpLessThan, SignednessSigned])
    elseif func === Core.Intrinsics.sle_int
        emit_intrinsic!(ctx, Intrinsics.cmpi, [args..., CmpLessThanOrEqual, SignednessSigned])
    elseif func === Core.Intrinsics.ult_int
        emit_intrinsic!(ctx, Intrinsics.cmpi, [args..., CmpLessThan, SignednessUnsigned])
    else
        error("Unhandled Julia intrinsic: $func")
    end
end


## Built-in functions

# We cannot overlay built-in functions

function emit_intrinsic!(ctx::CGCtx, ::typeof(===), args)
    cb = ctx.cb
    tt = ctx.tt

    lhs = emit_value!(ctx, args[1])
    rhs = emit_value!(ctx, args[2])

    lhs === nothing && error("Cannot resolve LHS operand for ===")
    rhs === nothing && error("Cannot resolve RHS operand for ===")

    result_type_id = tile_type!(tt, I1(tt), Int[])

    lhs_v = lhs isa CGVal ? lhs.v : lhs
    rhs_v = rhs isa CGVal ? rhs.v : rhs

    result_v = encode_CmpIOp!(cb, result_type_id, lhs_v, rhs_v;
                              predicate=CmpEqual, signedness=SignednessSigned)

    CGVal(result_v, result_type_id, Bool, Int[])
end

emit_intrinsic!(ctx::CGCtx, ::typeof(Core.tuple), args) = nothing

emit_intrinsic!(ctx::CGCtx, ::typeof(isa), args) = nothing

emit_intrinsic!(ctx::CGCtx, ::typeof(Base.donotdelete), args) = nothing


## Other

## XXX: Tile constructor
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
        unwrap_type(source.jltype)
    end

    # Return as 0D tile type with element type from the constructor
    result_jltype = Tile{elem_type, ()}
    CGVal(source.v, source.type_id, result_jltype, source.shape)
end
