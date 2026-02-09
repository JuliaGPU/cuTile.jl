# Mathematical intrinsics

## Floating-point math

# cuda_tile.ceil
@intrinsic ceil(x)
tfunc(::typeof(Intrinsics.ceil), argtypes::Vector{Any}) = CC.widenconst(argtypes[2])
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.ceil), args)
    emit_unop!(ctx, args, encode_CeilOp!)
end

# cuda_tile.cos
@intrinsic cos(x)
tfunc(::typeof(Intrinsics.cos), argtypes::Vector{Any}) = CC.widenconst(argtypes[2])
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.cos), args)
    emit_unop!(ctx, args, encode_CosOp!)
end

# cuda_tile.cosh
@intrinsic cosh(x)
tfunc(::typeof(Intrinsics.cosh), argtypes::Vector{Any}) = CC.widenconst(argtypes[2])
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.cosh), args)
    emit_unop!(ctx, args, encode_CosHOp!)
end

# cuda_tile.exp2
@intrinsic exp2(x, flush_to_zero=false)
tfunc(::typeof(Intrinsics.exp2), argtypes::Vector{Any}) = CC.widenconst(argtypes[2])
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
tfunc(::typeof(Intrinsics.exp), argtypes::Vector{Any}) = CC.widenconst(argtypes[2])
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.exp), args)
    cb = ctx.cb

    source = emit_value!(ctx, args[1])
    source === nothing && throw(IRError("Cannot resolve operand for exp()"))

    result = encode_ExpOp!(cb, source.type_id, source.v)

    CGVal(result, source.type_id, source.jltype, source.shape)
end

# cuda_tile.floor
@intrinsic floor(x)
tfunc(::typeof(Intrinsics.floor), argtypes::Vector{Any}) = CC.widenconst(argtypes[2])
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.floor), args)
    emit_unop!(ctx, args, encode_FloorOp!)
end

# cuda_tile.fma
@intrinsic fma(x, y, z)
tfunc(::typeof(Intrinsics.fma), argtypes::Vector{Any}) = CC.widenconst(argtypes[2])
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
tfunc(::typeof(Intrinsics.log2), argtypes::Vector{Any}) = CC.widenconst(argtypes[2])
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.log2), args)
    cb = ctx.cb

    source = emit_value!(ctx, args[1])
    source === nothing && throw(IRError("Cannot resolve operand for log2()"))

    result = encode_Log2Op!(cb, source.type_id, source.v)

    CGVal(result, source.type_id, source.jltype, source.shape)
end

# cuda_tile.log
@intrinsic log(x)
tfunc(::typeof(Intrinsics.log), argtypes::Vector{Any}) = CC.widenconst(argtypes[2])
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.log), args)
    cb = ctx.cb

    source = emit_value!(ctx, args[1])
    source === nothing && throw(IRError("Cannot resolve operand for log()"))

    result = encode_LogOp!(cb, source.type_id, source.v)

    CGVal(result, source.type_id, source.jltype, source.shape)
end

# cuda_tile.maxf
@intrinsic maxf(x, y)
tfunc(::typeof(Intrinsics.maxf), argtypes::Vector{Any}) = CC.widenconst(argtypes[2])
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.maxf), args)
    emit_binop!(ctx, args, encode_MaxFOp!)
end

# cuda_tile.minf
@intrinsic minf(x, y)
tfunc(::typeof(Intrinsics.minf), argtypes::Vector{Any}) = CC.widenconst(argtypes[2])
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.minf), args)
    emit_binop!(ctx, args, encode_MinFOp!)
end

# cuda_tile.pow
@intrinsic pow(x, y)
tfunc(::typeof(Intrinsics.pow), argtypes::Vector{Any}) = CC.widenconst(argtypes[2])
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.pow), args)
    emit_binop!(ctx, args, encode_PowOp!)
end

# cuda_tile.remf
@intrinsic remf(x, y)
tfunc(::typeof(Intrinsics.remf), argtypes::Vector{Any}) = CC.widenconst(argtypes[2])
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.remf), args)
    emit_binop!(ctx, args, encode_RemFOp!)
end

# cuda_tile.rsqrt
@intrinsic rsqrt(x, flush_to_zero=false)
tfunc(::typeof(Intrinsics.rsqrt), argtypes::Vector{Any}) = CC.widenconst(argtypes[2])
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
tfunc(::typeof(Intrinsics.sin), argtypes::Vector{Any}) = CC.widenconst(argtypes[2])
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.sin), args)
    emit_unop!(ctx, args, encode_SinOp!)
end

# cuda_tile.sinh
@intrinsic sinh(x)
tfunc(::typeof(Intrinsics.sinh), argtypes::Vector{Any}) = CC.widenconst(argtypes[2])
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.sinh), args)
    emit_unop!(ctx, args, encode_SinHOp!)
end

# cuda_tile.sqrt
@intrinsic sqrt(x)
tfunc(::typeof(Intrinsics.sqrt), argtypes::Vector{Any}) = CC.widenconst(argtypes[2])
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.sqrt), args)
    cb = ctx.cb

    source = emit_value!(ctx, args[1])
    source === nothing && throw(IRError("Cannot resolve operand for sqrt()"))

    result = encode_SqrtOp!(cb, source.type_id, source.v)

    CGVal(result, source.type_id, source.jltype, source.shape)
end

# cuda_tile.tan
@intrinsic tan(x)
tfunc(::typeof(Intrinsics.tan), argtypes::Vector{Any}) = CC.widenconst(argtypes[2])
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.tan), args)
    emit_unop!(ctx, args, encode_TanOp!)
end

# cuda_tile.tanh
@intrinsic tanh(x)
tfunc(::typeof(Intrinsics.tanh), argtypes::Vector{Any}) = CC.widenconst(argtypes[2])
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.tanh), args)
    emit_unop!(ctx, args, encode_TanHOp!)
end
