# Mathematical intrinsics

## Floating-point math

# cuda_tile.ceil
@intrinsic ceil(x)
tfunc(𝕃, ::typeof(Intrinsics.ceil), @nospecialize(x)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.ceil), args)
    emit_unop!(ctx, args, encode_CeilOp!)
end

# cuda_tile.cos
@intrinsic cos(x)
tfunc(𝕃, ::typeof(Intrinsics.cos), @nospecialize(x)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.cos), args)
    emit_unop!(ctx, args, encode_CosOp!)
end

# cuda_tile.cosh
@intrinsic cosh(x)
tfunc(𝕃, ::typeof(Intrinsics.cosh), @nospecialize(x)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.cosh), args)
    emit_unop!(ctx, args, encode_CosHOp!)
end

# cuda_tile.exp2
@intrinsic exp2(x, flush_to_zero=false)
tfunc(𝕃, ::typeof(Intrinsics.exp2), @nospecialize(x), @nospecialize args...) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.exp2), args)
    cb = ctx.cb

    source = emit_value!(ctx, args[1])
    source === nothing && throw(IRError("Cannot resolve operand for exp2()"))

    flush_to_zero = length(args) > 1 ? args[2]::Bool : false

    result = encode_Exp2Op!(cb, source.type_id, source.v; flush_to_zero)

    CGVal(result, source.type_id, source.jltype, source.shape)
end

# cuda_tile.exp
@intrinsic exp(x)
tfunc(𝕃, ::typeof(Intrinsics.exp), @nospecialize(x)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.exp), args)
    cb = ctx.cb

    source = emit_value!(ctx, args[1])
    source === nothing && throw(IRError("Cannot resolve operand for exp()"))

    result = encode_ExpOp!(cb, source.type_id, source.v)

    CGVal(result, source.type_id, source.jltype, source.shape)
end

# cuda_tile.floor
@intrinsic floor(x)
tfunc(𝕃, ::typeof(Intrinsics.floor), @nospecialize(x)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.floor), args)
    emit_unop!(ctx, args, encode_FloorOp!)
end

# cuda_tile.fma
@intrinsic fma(x, y, z)
tfunc(𝕃, ::typeof(Intrinsics.fma), @nospecialize(x), @nospecialize(y), @nospecialize(z)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.fma), args)
    cb = ctx.cb

    a = emit_value!(ctx, args[1])
    b = emit_value!(ctx, args[2])
    c = emit_value!(ctx, args[3])

    (a === nothing || b === nothing || c === nothing) && throw(IRError("Cannot resolve operands for fma"))

    result_v = encode_FmaOp!(cb, a.type_id, a.v, b.v, c.v)

    CGVal(result_v, a.type_id, a.jltype, a.shape)
end

# cuda_tile.log2
@intrinsic log2(x)
tfunc(𝕃, ::typeof(Intrinsics.log2), @nospecialize(x)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.log2), args)
    cb = ctx.cb

    source = emit_value!(ctx, args[1])
    source === nothing && throw(IRError("Cannot resolve operand for log2()"))

    result = encode_Log2Op!(cb, source.type_id, source.v)

    CGVal(result, source.type_id, source.jltype, source.shape)
end

# cuda_tile.log
@intrinsic log(x)
tfunc(𝕃, ::typeof(Intrinsics.log), @nospecialize(x)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.log), args)
    cb = ctx.cb

    source = emit_value!(ctx, args[1])
    source === nothing && throw(IRError("Cannot resolve operand for log()"))

    result = encode_LogOp!(cb, source.type_id, source.v)

    CGVal(result, source.type_id, source.jltype, source.shape)
end

# cuda_tile.maxf
@intrinsic maxf(x, y)
tfunc(𝕃, ::typeof(Intrinsics.maxf), @nospecialize(x), @nospecialize(y)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.maxf), args)
    emit_binop!(ctx, args, encode_MaxFOp!)
end

# cuda_tile.minf
@intrinsic minf(x, y)
tfunc(𝕃, ::typeof(Intrinsics.minf), @nospecialize(x), @nospecialize(y)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.minf), args)
    emit_binop!(ctx, args, encode_MinFOp!)
end

# cuda_tile.pow
@intrinsic pow(x, y)
tfunc(𝕃, ::typeof(Intrinsics.pow), @nospecialize(x), @nospecialize(y)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.pow), args)
    emit_binop!(ctx, args, encode_PowOp!)
end

# cuda_tile.remf
@intrinsic remf(x, y)
tfunc(𝕃, ::typeof(Intrinsics.remf), @nospecialize(x), @nospecialize(y)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.remf), args)
    emit_binop!(ctx, args, encode_RemFOp!)
end

# cuda_tile.rsqrt
@intrinsic rsqrt(x, flush_to_zero=false)
tfunc(𝕃, ::typeof(Intrinsics.rsqrt), @nospecialize(x), @nospecialize args...) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.rsqrt), args)
    cb = ctx.cb

    source = emit_value!(ctx, args[1])
    source === nothing && throw(IRError("Cannot resolve operand for rsqrt()"))

    flush_to_zero = length(args) > 1 ? args[2]::Bool : false

    result = encode_RSqrtOp!(cb, source.type_id, source.v; flush_to_zero)

    CGVal(result, source.type_id, source.jltype, source.shape)
end

# cuda_tile.sin
@intrinsic sin(x)
tfunc(𝕃, ::typeof(Intrinsics.sin), @nospecialize(x)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.sin), args)
    emit_unop!(ctx, args, encode_SinOp!)
end

# cuda_tile.sinh
@intrinsic sinh(x)
tfunc(𝕃, ::typeof(Intrinsics.sinh), @nospecialize(x)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.sinh), args)
    emit_unop!(ctx, args, encode_SinHOp!)
end

# cuda_tile.sqrt
@intrinsic sqrt(x)
tfunc(𝕃, ::typeof(Intrinsics.sqrt), @nospecialize(x)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.sqrt), args)
    cb = ctx.cb

    source = emit_value!(ctx, args[1])
    source === nothing && throw(IRError("Cannot resolve operand for sqrt()"))

    result = encode_SqrtOp!(cb, source.type_id, source.v)

    CGVal(result, source.type_id, source.jltype, source.shape)
end

# cuda_tile.tan
@intrinsic tan(x)
tfunc(𝕃, ::typeof(Intrinsics.tan), @nospecialize(x)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.tan), args)
    emit_unop!(ctx, args, encode_TanOp!)
end

# cuda_tile.tanh
@intrinsic tanh(x)
tfunc(𝕃, ::typeof(Intrinsics.tanh), @nospecialize(x)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.tanh), args)
    emit_unop!(ctx, args, encode_TanHOp!)
end
