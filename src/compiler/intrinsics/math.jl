# Mathematical intrinsics

## Floating-point math

# cuda_tile.ceil
@eval Intrinsics begin
    """Ceiling (round toward positive infinity). Compiled to cuda_tile.ceil."""
    @noinline ceil(x::T) where {T<:AbstractFloat} = compilerbarrier(:const, x)
    @noinline ceil(tile::Tile{T, S}) where {T<:AbstractFloat, S} = compilerbarrier(:const, tile)
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.ceil), args)
    emit_unop!(ctx, args, encode_CeilOp!)
end

# cuda_tile.cos
@eval Intrinsics begin
    """Cosine. Compiled to cuda_tile.cos."""
    @noinline cos(x::T) where {T<:AbstractFloat} = compilerbarrier(:const, x)
    @noinline cos(tile::Tile{T, S}) where {T<:AbstractFloat, S} = compilerbarrier(:const, tile)
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.cos), args)
    emit_unop!(ctx, args, encode_CosOp!)
end

# cuda_tile.cosh
@eval Intrinsics begin
    """Hyperbolic cosine. Compiled to cuda_tile.cosh."""
    @noinline cosh(x::T) where {T<:AbstractFloat} = compilerbarrier(:const, x)
    @noinline cosh(tile::Tile{T, S}) where {T<:AbstractFloat, S} = compilerbarrier(:const, tile)
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.cosh), args)
    emit_unop!(ctx, args, encode_CosHOp!)
end

# cuda_tile.exp2
@eval Intrinsics begin
    """Base-2 exponential (2^x). Compiled to cuda_tile.exp2."""
    @noinline exp2(x::T) where {T<:AbstractFloat} = compilerbarrier(:const, x)
    @noinline exp2(tile::Tile{T, S}, flush_to_zero::Bool=false) where {T<:AbstractFloat, S} = compilerbarrier(:const, tile)
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.exp2), args)
    cb = ctx.cb

    source = emit_value!(ctx, args[1])
    source === nothing && throw(IRError("Cannot resolve operand for exp2()"))

    flush_to_zero = length(args) > 1 ? args[2]::Bool : false

    result = encode_Exp2Op!(cb, source.type_id, source.v; flush_to_zero)

    CGVal(result, source.type_id, source.jltype, source.shape)
end

# cuda_tile.exp
@eval Intrinsics begin
    """Natural exponential (e^x). Compiled to cuda_tile.exp."""
    @noinline exp(x::T) where {T<:AbstractFloat} = compilerbarrier(:const, x)
    @noinline exp(tile::Tile{T, S}) where {T<:AbstractFloat, S} = compilerbarrier(:const, tile)
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.exp), args)
    cb = ctx.cb

    source = emit_value!(ctx, args[1])
    source === nothing && throw(IRError("Cannot resolve operand for exp()"))

    result = encode_ExpOp!(cb, source.type_id, source.v)

    CGVal(result, source.type_id, source.jltype, source.shape)
end

# cuda_tile.floor
@eval Intrinsics begin
    """Floor (round toward negative infinity). Compiled to cuda_tile.floor."""
    @noinline floor(x::T) where {T<:AbstractFloat} = compilerbarrier(:const, x)
    @noinline floor(tile::Tile{T, S}) where {T<:AbstractFloat, S} = compilerbarrier(:const, tile)
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.floor), args)
    emit_unop!(ctx, args, encode_FloorOp!)
end

# cuda_tile.fma
@eval Intrinsics begin
    """Fused multiply-add: a * b + c. Compiled to cuda_tile.fma."""
    @noinline fma(x::T, y::T, z::T) where {T<:AbstractFloat} = compilerbarrier(:const, x)
    @noinline fma(a::Tile{T, S}, b::Tile{T, S}, c::Tile{T, S}) where {T<:AbstractFloat, S} = Tile{T, S}()
end
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
@eval Intrinsics begin
    """Base-2 logarithm. Compiled to cuda_tile.log2."""
    @noinline log2(x::T) where {T<:AbstractFloat} = compilerbarrier(:const, x)
    @noinline log2(tile::Tile{T, S}) where {T<:AbstractFloat, S} = compilerbarrier(:const, tile)
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.log2), args)
    cb = ctx.cb

    source = emit_value!(ctx, args[1])
    source === nothing && throw(IRError("Cannot resolve operand for log2()"))

    result = encode_Log2Op!(cb, source.type_id, source.v)

    CGVal(result, source.type_id, source.jltype, source.shape)
end

# cuda_tile.log
@eval Intrinsics begin
    """Element-wise natural logarithm. Compiled to cuda_tile.log."""
    @noinline log(x::T) where {T<:AbstractFloat} = compilerbarrier(:const, x)
    @noinline log(tile::Tile{T, S}) where {T<:AbstractFloat, S} = compilerbarrier(:const, tile)
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.log), args)
    cb = ctx.cb

    source = emit_value!(ctx, args[1])
    source === nothing && throw(IRError("Cannot resolve operand for log()"))

    result = encode_LogOp!(cb, source.type_id, source.v)

    CGVal(result, source.type_id, source.jltype, source.shape)
end

# cuda_tile.maxf
@eval Intrinsics begin
    @noinline maxf(x::T, y::T) where {T<:AbstractFloat} = ifelse(x > y || isnan(x), x, y)
    @noinline maxf(a::Tile{T, S}, b::Tile{T, S}) where {T<:AbstractFloat, S} = Tile{T, S}()
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.maxf), args)
    emit_binop!(ctx, args, encode_MaxFOp!)
end

# cuda_tile.minf
@eval Intrinsics begin
    @noinline minf(x::T, y::T) where {T<:AbstractFloat} = ifelse(x < y || isnan(x), x, y)
    @noinline minf(a::Tile{T, S}, b::Tile{T, S}) where {T<:AbstractFloat, S} = Tile{T, S}()
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.minf), args)
    emit_binop!(ctx, args, encode_MinFOp!)
end

# cuda_tile.pow
@eval Intrinsics begin
    """Element-wise power. Compiled to cuda_tile.pow."""
    @noinline pow(x::T, y::T) where {T<:AbstractFloat} = compilerbarrier(:const, x)
    @noinline pow(a::Tile{T, S}, b::Tile{T, S}) where {T<:AbstractFloat, S} = Tile{T, S}()
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.pow), args)
    emit_binop!(ctx, args, encode_PowOp!)
end

# cuda_tile.remf
@eval Intrinsics begin
    """Element-wise floating-point remainder. Compiled to cuda_tile.remf."""
    @noinline remf(x::T, y::T) where {T<:AbstractFloat} = compilerbarrier(:const, x)
    @noinline remf(a::Tile{T, S}, b::Tile{T, S}) where {T<:AbstractFloat, S} = Tile{T, S}()
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.remf), args)
    emit_binop!(ctx, args, encode_RemFOp!)
end

# cuda_tile.rsqrt
@eval Intrinsics begin
    """Element-wise reciprocal square root. Compiled to cuda_tile.rsqrt."""
    @noinline rsqrt(x::T) where {T<:AbstractFloat} = compilerbarrier(:const, x)
    @noinline rsqrt(tile::Tile{T, S}, flush_to_zero::Bool=false) where {T<:AbstractFloat, S} = compilerbarrier(:const, tile)
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.rsqrt), args)
    cb = ctx.cb

    source = emit_value!(ctx, args[1])
    source === nothing && throw(IRError("Cannot resolve operand for rsqrt()"))

    flush_to_zero = length(args) > 1 ? args[2]::Bool : false

    result = encode_RSqrtOp!(cb, source.type_id, source.v; flush_to_zero)

    CGVal(result, source.type_id, source.jltype, source.shape)
end

# cuda_tile.sin
@eval Intrinsics begin
    """Element-wise sine. Compiled to cuda_tile.sin."""
    @noinline sin(x::T) where {T<:AbstractFloat} = compilerbarrier(:const, x)
    @noinline sin(tile::Tile{T, S}) where {T<:AbstractFloat, S} = compilerbarrier(:const, tile)
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.sin), args)
    emit_unop!(ctx, args, encode_SinOp!)
end

# cuda_tile.sinh
@eval Intrinsics begin
    """Element-wise hyperbolic sine. Compiled to cuda_tile.sinh."""
    @noinline sinh(x::T) where {T<:AbstractFloat} = compilerbarrier(:const, x)
    @noinline sinh(tile::Tile{T, S}) where {T<:AbstractFloat, S} = compilerbarrier(:const, tile)
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.sinh), args)
    emit_unop!(ctx, args, encode_SinHOp!)
end

# cuda_tile.sqrt
@eval Intrinsics begin
    """Element-wise square root. Compiled to cuda_tile.sqrt."""
    @noinline sqrt(x::T) where {T<:AbstractFloat} = compilerbarrier(:const, x)
    @noinline sqrt(tile::Tile{T, S}) where {T<:AbstractFloat, S} = compilerbarrier(:const, tile)
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.sqrt), args)
    cb = ctx.cb

    source = emit_value!(ctx, args[1])
    source === nothing && throw(IRError("Cannot resolve operand for sqrt()"))

    result = encode_SqrtOp!(cb, source.type_id, source.v)

    CGVal(result, source.type_id, source.jltype, source.shape)
end

# cuda_tile.tan
@eval Intrinsics begin
    """Element-wise tangent. Compiled to cuda_tile.tan."""
    @noinline tan(x::T) where {T<:AbstractFloat} = compilerbarrier(:const, x)
    @noinline tan(tile::Tile{T, S}) where {T<:AbstractFloat, S} = compilerbarrier(:const, tile)
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.tan), args)
    emit_unop!(ctx, args, encode_TanOp!)
end

# cuda_tile.tanh
@eval Intrinsics begin
    """Element-wise hyperbolic tangent. Compiled to cuda_tile.tanh."""
    @noinline tanh(x::T) where {T<:AbstractFloat} = compilerbarrier(:const, x)
    @noinline tanh(tile::Tile{T, S}) where {T<:AbstractFloat, S} = compilerbarrier(:const, tile)
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.tanh), args)
    emit_unop!(ctx, args, encode_TanHOp!)
end
