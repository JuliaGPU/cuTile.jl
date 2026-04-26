# Mathematical intrinsics

## Floating-point math

"""
    Intrinsics.ceil(x::Tile{<:AbstractFloat}) -> Tile

Element-wise floor toward positive infinity (ceiling); lowers to
`cuda_tile.ceil`.

Also invocable with a scalar, promoted to a 0-D tile before codegen.
"""
@intrinsic ceil(x::AbstractFloat)
@intrinsic ceil(x::Tile{<:AbstractFloat})
tfunc(𝕃, ::typeof(Intrinsics.ceil), @nospecialize(x)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.ceil), args)
    emit_unop!(ctx, args, encode_CeilOp!)
end

"""
    Intrinsics.cos(x::Tile{<:AbstractFloat}) -> Tile

Element-wise cosine; lowers to `cuda_tile.cos`.

Also invocable with a scalar, promoted to a 0-D tile before codegen.
"""
@intrinsic cos(x::AbstractFloat)
@intrinsic cos(x::Tile{<:AbstractFloat})
tfunc(𝕃, ::typeof(Intrinsics.cos), @nospecialize(x)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.cos), args)
    emit_unop!(ctx, args, encode_CosOp!)
end

"""
    Intrinsics.cosh(x::Tile{<:AbstractFloat}) -> Tile

Element-wise hyperbolic cosine; lowers to `cuda_tile.cosh`.

Also invocable with a scalar, promoted to a 0-D tile before codegen.
"""
@intrinsic cosh(x::AbstractFloat)
@intrinsic cosh(x::Tile{<:AbstractFloat})
tfunc(𝕃, ::typeof(Intrinsics.cosh), @nospecialize(x)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.cosh), args)
    emit_unop!(ctx, args, encode_CosHOp!)
end

"""
    Intrinsics.exp2(x::Tile{<:AbstractFloat}) -> Tile

Element-wise base-2 exponential (`2^x`); lowers to `cuda_tile.exp2`.

Also invocable with a scalar, promoted to a 0-D tile before codegen. The
active `@fpmode` scope supplies `flush_to_zero`.
"""
@intrinsic exp2(x::AbstractFloat)
@intrinsic exp2(x::Tile{<:AbstractFloat})
tfunc(𝕃, ::typeof(Intrinsics.exp2), @nospecialize(x)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.exp2), args)
    flush_to_zero = current_fpmode(ctx).flush_to_zero
    emit_unop!(ctx, args, encode_Exp2Op!; flush_to_zero)
end

"""
    Intrinsics.exp(x::Tile{<:AbstractFloat}) -> Tile

Element-wise natural exponential (`e^x`); lowers to `cuda_tile.exp`.

Also invocable with a scalar, promoted to a 0-D tile before codegen.
"""
@intrinsic exp(x::AbstractFloat)
@intrinsic exp(x::Tile{<:AbstractFloat})
tfunc(𝕃, ::typeof(Intrinsics.exp), @nospecialize(x)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.exp), args)
    cb = ctx.cb

    source = emit_value!(ctx, args[1])
    source === nothing && throw(IRError("Cannot resolve operand for exp()"))

    result = encode_ExpOp!(cb, source.type_id, source.v)

    CGVal(result, source.type_id, source.jltype, source.shape)
end

"""
    Intrinsics.floor(x::Tile{<:AbstractFloat}) -> Tile

Element-wise floor toward negative infinity; lowers to `cuda_tile.floor`.

Also invocable with a scalar, promoted to a 0-D tile before codegen.
"""
@intrinsic floor(x::AbstractFloat)
@intrinsic floor(x::Tile{<:AbstractFloat})
tfunc(𝕃, ::typeof(Intrinsics.floor), @nospecialize(x)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.floor), args)
    emit_unop!(ctx, args, encode_FloorOp!)
end

"""
    Intrinsics.fma(x::Tile{T}, y::Tile{T}, z::Tile{T}) -> Tile{T}    where {T<:AbstractFloat}

Element-wise fused multiply-add `x*y + z` performed as a single rounded
operation; lowers to `cuda_tile.fma`.

Also invocable with scalars, promoted to 0-D tiles before codegen. The
active `@fpmode` scope supplies the `rounding_mode` and `flush_to_zero`
attributes.
"""
@intrinsic fma(x::T, y::T, z::T) where {T<:AbstractFloat}
@intrinsic fma(x::Tile{T}, y::Tile{T}, z::Tile{T}) where {T<:AbstractFloat}
tfunc(𝕃, ::typeof(Intrinsics.fma), @nospecialize(x), @nospecialize(y), @nospecialize(z)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.fma), args)
    cb = ctx.cb

    a = emit_value!(ctx, args[1])
    b = emit_value!(ctx, args[2])
    c = emit_value!(ctx, args[3])

    (a === nothing || b === nothing || c === nothing) && throw(IRError("Cannot resolve operands for fma"))

    result_v = encode_FmaOp!(cb, a.type_id, a.v, b.v, c.v; fpmode_kwargs(ctx)...)

    CGVal(result_v, a.type_id, a.jltype, a.shape)
end

"""
    Intrinsics.log2(x::Tile{<:AbstractFloat}) -> Tile

Element-wise base-2 logarithm; lowers to `cuda_tile.log2`.

Also invocable with a scalar, promoted to a 0-D tile before codegen.
"""
@intrinsic log2(x::AbstractFloat)
@intrinsic log2(x::Tile{<:AbstractFloat})
tfunc(𝕃, ::typeof(Intrinsics.log2), @nospecialize(x)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.log2), args)
    cb = ctx.cb

    source = emit_value!(ctx, args[1])
    source === nothing && throw(IRError("Cannot resolve operand for log2()"))

    result = encode_Log2Op!(cb, source.type_id, source.v)

    CGVal(result, source.type_id, source.jltype, source.shape)
end

"""
    Intrinsics.log(x::Tile{<:AbstractFloat}) -> Tile

Element-wise natural logarithm; lowers to `cuda_tile.log`.

Also invocable with a scalar, promoted to a 0-D tile before codegen.
"""
@intrinsic log(x::AbstractFloat)
@intrinsic log(x::Tile{<:AbstractFloat})
tfunc(𝕃, ::typeof(Intrinsics.log), @nospecialize(x)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.log), args)
    cb = ctx.cb

    source = emit_value!(ctx, args[1])
    source === nothing && throw(IRError("Cannot resolve operand for log()"))

    result = encode_LogOp!(cb, source.type_id, source.v)

    CGVal(result, source.type_id, source.jltype, source.shape)
end

"""
    Intrinsics.maxf(x::Tile{T}, y::Tile{T}) -> Tile{T}  where {T<:AbstractFloat}

Element-wise floating-point maximum; lowers to `cuda_tile.maxf`.

Also invocable with scalars, promoted to 0-D tiles before codegen. The
active `@fpmode` scope supplies `flush_to_zero`. Mismatched-shape operands
are broadcast to a common shape.
"""
@intrinsic maxf(x::T, y::T) where {T<:AbstractFloat}
@intrinsic maxf(x::Tile{T}, y::Tile{T}) where {T<:AbstractFloat}
tfunc(𝕃, ::typeof(Intrinsics.maxf), @nospecialize(x), @nospecialize(y)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.maxf), args)
    flush_to_zero = current_fpmode(ctx).flush_to_zero
    emit_binop!(ctx, args, encode_MaxFOp!; flush_to_zero)
end

"""
    Intrinsics.minf(x::Tile{T}, y::Tile{T}) -> Tile{T}  where {T<:AbstractFloat}

Element-wise floating-point minimum; lowers to `cuda_tile.minf`.

Also invocable with scalars, promoted to 0-D tiles before codegen. The
active `@fpmode` scope supplies `flush_to_zero`. Mismatched-shape operands
are broadcast to a common shape.
"""
@intrinsic minf(x::T, y::T) where {T<:AbstractFloat}
@intrinsic minf(x::Tile{T}, y::Tile{T}) where {T<:AbstractFloat}
tfunc(𝕃, ::typeof(Intrinsics.minf), @nospecialize(x), @nospecialize(y)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.minf), args)
    flush_to_zero = current_fpmode(ctx).flush_to_zero
    emit_binop!(ctx, args, encode_MinFOp!; flush_to_zero)
end

"""
    Intrinsics.pow(x::Tile{T}, y::Tile{T}) -> Tile{T}  where {T<:AbstractFloat}

Element-wise floating-point exponentiation (`x^y`); lowers to
`cuda_tile.pow`.

Also invocable with scalars, promoted to 0-D tiles before codegen.
Mismatched-shape operands are broadcast to a common shape.
"""
@intrinsic pow(x::T, y::T) where {T<:AbstractFloat}
@intrinsic pow(x::Tile{T}, y::Tile{T}) where {T<:AbstractFloat}
tfunc(𝕃, ::typeof(Intrinsics.pow), @nospecialize(x), @nospecialize(y)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.pow), args)
    emit_binop!(ctx, args, encode_PowOp!)
end

"""
    Intrinsics.remf(x::Tile{T}, y::Tile{T}) -> Tile{T}  where {T<:AbstractFloat}

Element-wise floating-point remainder using truncated division (sign of
dividend); lowers to `cuda_tile.remf`.

Also invocable with scalars, promoted to 0-D tiles before codegen.
Mismatched-shape operands are broadcast to a common shape.
"""
@intrinsic remf(x::T, y::T) where {T<:AbstractFloat}
@intrinsic remf(x::Tile{T}, y::Tile{T}) where {T<:AbstractFloat}
tfunc(𝕃, ::typeof(Intrinsics.remf), @nospecialize(x), @nospecialize(y)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.remf), args)
    emit_binop!(ctx, args, encode_RemFOp!)
end

"""
    Intrinsics.rsqrt(x::Tile{<:AbstractFloat}) -> Tile

Element-wise reciprocal square root (`1/sqrt(x)`); lowers to
`cuda_tile.rsqrt`.

Also invocable with a scalar, promoted to a 0-D tile before codegen. The
active `@fpmode` scope supplies `flush_to_zero`.
"""
@intrinsic rsqrt(x::AbstractFloat)
@intrinsic rsqrt(x::Tile{<:AbstractFloat})
tfunc(𝕃, ::typeof(Intrinsics.rsqrt), @nospecialize(x)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.rsqrt), args)
    flush_to_zero = current_fpmode(ctx).flush_to_zero
    emit_unop!(ctx, args, encode_RSqrtOp!; flush_to_zero)
end

"""
    Intrinsics.sin(x::Tile{<:AbstractFloat}) -> Tile

Element-wise sine; lowers to `cuda_tile.sin`.

Also invocable with a scalar, promoted to a 0-D tile before codegen.
"""
@intrinsic sin(x::AbstractFloat)
@intrinsic sin(x::Tile{<:AbstractFloat})
tfunc(𝕃, ::typeof(Intrinsics.sin), @nospecialize(x)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.sin), args)
    emit_unop!(ctx, args, encode_SinOp!)
end

"""
    Intrinsics.sinh(x::Tile{<:AbstractFloat}) -> Tile

Element-wise hyperbolic sine; lowers to `cuda_tile.sinh`.

Also invocable with a scalar, promoted to a 0-D tile before codegen.
"""
@intrinsic sinh(x::AbstractFloat)
@intrinsic sinh(x::Tile{<:AbstractFloat})
tfunc(𝕃, ::typeof(Intrinsics.sinh), @nospecialize(x)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.sinh), args)
    emit_unop!(ctx, args, encode_SinHOp!)
end

"""
    Intrinsics.sqrt(x::Tile{<:AbstractFloat}) -> Tile

Element-wise floating-point square root; lowers to `cuda_tile.sqrt`.

Also invocable with a scalar, promoted to a 0-D tile before codegen. The
active `@fpmode` scope supplies the `rounding_mode` and `flush_to_zero`
attributes.
"""
@intrinsic sqrt(x::AbstractFloat)
@intrinsic sqrt(x::Tile{<:AbstractFloat})
tfunc(𝕃, ::typeof(Intrinsics.sqrt), @nospecialize(x)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.sqrt), args)
    emit_unop!(ctx, args, encode_SqrtOp!; fpmode_kwargs(ctx)...)
end

"""
    Intrinsics.tan(x::Tile{<:AbstractFloat}) -> Tile

Element-wise tangent; lowers to `cuda_tile.tan`.

Also invocable with a scalar, promoted to a 0-D tile before codegen.
"""
@intrinsic tan(x::AbstractFloat)
@intrinsic tan(x::Tile{<:AbstractFloat})
tfunc(𝕃, ::typeof(Intrinsics.tan), @nospecialize(x)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.tan), args)
    emit_unop!(ctx, args, encode_TanOp!)
end

"""
    Intrinsics.tanh(x::Tile{<:AbstractFloat}) -> Tile

Element-wise hyperbolic tangent; lowers to `cuda_tile.tanh`.

Also invocable with a scalar, promoted to a 0-D tile before codegen.
"""
@intrinsic tanh(x::AbstractFloat)
@intrinsic tanh(x::Tile{<:AbstractFloat})
tfunc(𝕃, ::typeof(Intrinsics.tanh), @nospecialize(x)) = CC.widenconst(x)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.tanh), args)
    emit_unop!(ctx, args, encode_TanHOp!)
end
