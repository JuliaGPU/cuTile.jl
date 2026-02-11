# miscellaneous intrinsics

# cuda_tile.assert
@intrinsic assert(cond::Bool, message::String)
tfunc(𝕃, ::typeof(Intrinsics.assert), @nospecialize(cond), @nospecialize(message)) = Nothing
efunc(::typeof(Intrinsics.assert), effects::CC.Effects) =
    CC.Effects(effects; effect_free=CC.ALWAYS_FALSE)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.assert), args)
    cond = @something emit_value!(ctx, args[1]) throw(IRError("assert: cannot resolve condition"))
    message = @something get_constant(ctx, args[2]) throw(IRError("assert: requires constant message"))
    encode_AssertOp!(ctx.cb, cond.v, message)
    nothing  # no result value
end

# XXX: cuda_tile.assume
# make this a pass?
function emit_assume_ops!(ctx::CGCtx, array_val::Value, size_vals::Vector{Value},
                          stride_vals::Vector{Value}, array_spec::ArraySpec, dtype::TypeId, scalar_type::TypeId;
                          tv_strides::Union{Vector{Int64}, Nothing}=nothing)
    cb = ctx.cb
    tt = ctx.tt

    # Pointer alignment
    if array_spec.alignment > 0
        ptr_dtype = pointer_type!(tt, dtype)
        ptr_tile_type = tile_type!(tt, ptr_dtype, Int[])
        array_val = encode_AssumeOp!(cb, ptr_tile_type, array_val, DivBy(array_spec.alignment))
    end

    # Bounds assumes for sizes
    size_vals = Value[encode_AssumeOp!(cb, scalar_type, v, Bounded(0, nothing)) for v in size_vals]

    # Bounds assumes for strides - only for dynamic strides
    if tv_strides !== nothing
        stride_vals = Value[tv_strides[i] == DYNAMIC_SHAPE ?
                       encode_AssumeOp!(cb, scalar_type, v, Bounded(0, nothing)) : v
                       for (i, v) in enumerate(stride_vals)]
    else
        stride_vals = Value[encode_AssumeOp!(cb, scalar_type, v, Bounded(0, nothing)) for v in stride_vals]
    end

    # Divisibility assumes for sizes
    for (i, div_by) in enumerate(array_spec.shape_div_by)
        if div_by > 0 && i <= length(size_vals)
            size_vals[i] = encode_AssumeOp!(cb, scalar_type, size_vals[i], DivBy(div_by))
        end
    end

    # Divisibility assumes for strides - only for dynamic strides
    for (i, div_by) in enumerate(array_spec.stride_div_by)
        if div_by > 0 && i <= length(stride_vals)
            # Skip if this stride is static (not DYNAMIC_SHAPE)
            if tv_strides === nothing || tv_strides[i] == DYNAMIC_SHAPE
                stride_vals[i] = encode_AssumeOp!(cb, scalar_type, stride_vals[i], DivBy(div_by))
            end
        end
    end

    return array_val, size_vals, stride_vals
end

# TODO: cuda_tile.print_tko
