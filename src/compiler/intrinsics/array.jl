# TileArray construction
#
# Generic ArrayValue-from-components packaging primitive. Used by any
# language-level operation that derives a new TileArray from existing
# pointer/sizes/strides Values (slicing today; reshape / transpose /
# permutedims when those land). The result type is computed by the caller
# and passed in explicitly so each operation can apply its own ArraySpec
# rules (e.g. slice drops alignment, transpose flips the contiguity flag).

@intrinsic make_array(::Type{T}, base, sizes, strides) where {T}

function tfunc(𝕃, ::typeof(Intrinsics.make_array),
               @nospecialize(T_arg), @nospecialize args...)
    T_outer = CC.widenconst(T_arg)
    T_outer isa DataType && T_outer <: Type || return nothing
    T = T_outer.parameters[1]
    T isa Type && T <: TileArray ? T : nothing
end

function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.make_array), args)
    T_arg, base_arg, sizes_arg, strides_arg = args

    T = @something get_constant(ctx, T_arg) throw(IRError("make_array: result type must be a compile-time constant"))
    T isa Type && T <: TileArray ||
        throw(IRError("make_array: result type must be a TileArray, got $T"))

    base_tv = emit_value!(ctx, base_arg)
    base_tv === nothing && throw(IRError("make_array: cannot resolve base"))
    base = base_tv.v::Value
    sizes = Value[tv.v for tv in resolve_tuple(ctx, sizes_arg, "make_array: sizes")]
    strides = Value[tv.v for tv in resolve_tuple(ctx, strides_arg, "make_array: strides")]

    av = ArrayValue(base, sizes, strides, T)
    return array_value_cgval(av, T)
end
