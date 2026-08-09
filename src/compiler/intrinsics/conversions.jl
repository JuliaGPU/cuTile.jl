# Type conversions

# Result type for an element-wise conversion intrinsic with target type `T`.
# - `src <: Tile`: result is `Tile` of the same shape with element type `T`.
# - `src` is a concrete scalar: result is `T` (the scalar overload).
# - otherwise (e.g. `Any`, abstract, or `Union`): return `nothing` so inference
#   falls back to the intrinsic body's `Any`. Returning `T` here would lie about
#   the result being scalar when the runtime value may well be a `Tile`.
function convert_result_type(@nospecialize(src), @nospecialize(T))
    src <: Tile && return similar_type(src, T)
    src isa DataType && isconcretetype(src) && return T
    return nothing
end

"""
    Intrinsics.bitcast(x::Tile, ::Type{T}) -> Tile{T}

Reinterprets the bits of `x` element-wise as type `T`; lowers to
`cuda_tile.bitcast`.

Also invocable with a scalar, promoted to a 0-D tile before codegen. `T`
must be a compile-time constant. The op is elided when source and target
map to the same Tile IR type (e.g. `Int32`/`UInt32`, since Tile IR integers
are signless).
"""
@intrinsic bitcast(x, ::Type{T}) where {T}
function tfunc(𝕃, ::typeof(Intrinsics.bitcast), @nospecialize(x), @nospecialize(target_type))
    T = instanceof_tfunc(target_type)
    T === nothing && return nothing
    src = CC.widenconst(x)
    convert_result_type(src, T)
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.bitcast), args)
    cb = ctx.cb
    tt = ctx.tt

    source = @something emit_value!(ctx, args[1]) throw(IRError("bitcast: cannot resolve source"))
    target_type = @something get_constant(ctx, args[2]) throw(IRError("bitcast: requires compile-time target type"))

    dtype = lookup_dtype!(tt, target_type)
    result_type_id = tile_type!(tt, dtype, source.shape)

    src_type = CC.widenconst(source.jltype)
    result_jltype = similar_type(src_type, target_type)

    # No-op when source and target map to the same Tile IR type (e.g., Int32 ↔ UInt32).
    # Tile IR integers are signless, so these are the same type.
    if result_type_id == source.type_id
        return CGVal(source.v, source.type_id, result_jltype, source.shape)
    end

    result_v = encode_BitcastOp!(cb, result_type_id, source.v)
    CGVal(result_v, result_type_id, result_jltype, source.shape)
end

@inline lookup_bitwidth(@nospecialize(T::Type)) =
    Base.invokelatest(bitwidth, T)::Int

"""
    Intrinsics.pack(x::Tile{S,Tuple{N}}) -> Tile{UInt8,Tuple{N*bitwidth(S)÷8}}

Pack a rank-1 numeric tile into a rank-1 `UInt8` tile (the tile's bits viewed as
a byte array); lowers to `cuda_tile.pack`. `S` must not be 8-bit (use `bitcast`).
Requires Tile IR bytecode v13.3+.
"""
@intrinsic pack(x)
function tfunc(𝕃, ::typeof(Intrinsics.pack), @nospecialize(x))
    src = CC.widenconst(x)
    src <: Tile || return nothing
    S = src.parameters[1]
    Shape = src.parameters[2]
    (S isa Type && Shape isa Type) || return nothing
    dims = Shape.parameters
    length(dims) == 1 || return nothing
    n = dims[1]::Int
    bs = lookup_bitwidth(S)
    return Tile{UInt8, Tuple{fld(n * bs, 8)}}
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.pack), args)
    cb = ctx.cb
    tt = ctx.tt

    source = @something emit_value!(ctx, args[1]) throw(IRError("pack: cannot resolve source"))
    bytecode_version(tt) >= v"13.3" ||
        throw(IRError("cuda_tile.pack requires Tile IR bytecode v13.3+, got v$(bytecode_version(tt))"))
    length(source.shape) == 1 ||
        throw(IRError("pack: requires a rank-1 tile, got a $(length(source.shape))-D tile"))

    src_type = CC.widenconst(source.jltype)
    S = eltype(src_type)
    sbits = lookup_bitwidth(S)
    sbits == 8 &&
        throw(IRError("pack: 8-bit element type $S should be reinterpreted via bitcast, not packed"))
    n = source.shape[1]
    (n * sbits) % 8 == 0 ||
        throw(IRError("pack: a $n-element $S tile ($(n * sbits) bits) is not a whole number of bytes"))
    new_n = (n * sbits) ÷ 8

    new_shape = RowMajorShape([new_n])
    result_type_id = tile_type!(tt, lookup_dtype!(tt, UInt8), new_shape)
    result_v = encode_PackOp!(cb, result_type_id, source.v)
    CGVal(result_v, result_type_id, Tile{UInt8, Tuple{new_n}}, new_shape)
end

"""
    Intrinsics.unpack(x::Tile{UInt8,Tuple{N}}, ::Type{T}) -> Tile{T,Tuple{N*8÷bitwidth(T)}}

Unpack a rank-1 `UInt8` tile into a rank-1 numeric tile of element type `T` (the
inverse of [`pack`](@ref Intrinsics.pack)); lowers to `cuda_tile.unpack`. `T`
must be a compile-time constant and must not be 8-bit (use `bitcast`). Requires
Tile IR bytecode v13.3+.
"""
@intrinsic unpack(x, ::Type{T}) where {T}
function tfunc(𝕃, ::typeof(Intrinsics.unpack), @nospecialize(x), @nospecialize(target_type))
    T = instanceof_tfunc(target_type)
    T === nothing && return nothing
    src = CC.widenconst(x)
    src <: Tile || return nothing
    Shape = src.parameters[2]
    Shape isa Type || return nothing
    dims = Shape.parameters
    length(dims) == 1 || return nothing
    n = dims[1]::Int
    bt = lookup_bitwidth(T)
    return Tile{T, Tuple{fld(n * 8, bt)}}
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.unpack), args)
    cb = ctx.cb
    tt = ctx.tt

    source = @something emit_value!(ctx, args[1]) throw(IRError("unpack: cannot resolve source"))
    target_type = @something get_constant(ctx, args[2]) throw(IRError("unpack: requires compile-time target type"))
    bytecode_version(tt) >= v"13.3" ||
        throw(IRError("cuda_tile.unpack requires Tile IR bytecode v13.3+, got v$(bytecode_version(tt))"))
    length(source.shape) == 1 ||
        throw(IRError("unpack: requires a rank-1 tile, got a $(length(source.shape))-D tile"))

    src_type = CC.widenconst(source.jltype)
    eltype(src_type) === UInt8 ||
        throw(IRError("unpack: requires a UInt8 tile, got $(eltype(src_type))"))
    tbits = lookup_bitwidth(target_type)
    tbits == 8 &&
        throw(IRError("unpack: 8-bit target $target_type should be reinterpreted via bitcast, not unpacked"))
    n = source.shape[1]
    (n * 8) % tbits == 0 ||
        throw(IRError("unpack: $n bytes ($(n * 8) bits) do not evenly divide into $target_type ($tbits-bit) elements"))
    new_n = (n * 8) ÷ tbits

    new_shape = RowMajorShape([new_n])
    result_type_id = tile_type!(tt, lookup_dtype!(tt, target_type), new_shape)
    result_v = encode_UnpackOp!(cb, result_type_id, source.v)
    CGVal(result_v, result_type_id, Tile{target_type, Tuple{new_n}}, new_shape)
end

"""
    Intrinsics.exti(x::Tile{<:Integer}, ::Type{T}, s::Signedness.T) -> Tile{T}     where {T<:Integer}

Element-wise integer extension; lowers to `cuda_tile.exti`.

Also invocable with a scalar, promoted to a 0-D tile before codegen. `s`
and `T` are compile-time constants.
"""
@intrinsic exti(x::I, ::Type{T}, s::Signedness.T) where {I<:Integer, T<:Integer}
function tfunc(𝕃, ::typeof(Intrinsics.exti), @nospecialize(x), @nospecialize(target_type), @nospecialize(s))
    T = instanceof_tfunc(target_type)
    T === nothing && return nothing
    src = CC.widenconst(x)
    convert_result_type(src, T)
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.exti), args)
    cb = ctx.cb
    tt = ctx.tt

    source = @something emit_value!(ctx, args[1]) throw(IRError("exti: cannot resolve source"))
    target_type = @something get_constant(ctx, args[2]) throw(IRError("exti: requires compile-time target type"))
    signedness = @something get_constant(ctx, args[3]) throw(IRError("exti: requires compile-time signedness"))

    dtype = lookup_dtype!(tt, target_type)
    result_type_id = tile_type!(tt, dtype, source.shape)

    result_v = encode_ExtIOp!(cb, result_type_id, source.v; signedness)
    src_type = CC.widenconst(source.jltype)
    result_jltype = similar_type(src_type, target_type)
    CGVal(result_v, result_type_id, result_jltype, source.shape)
end

"""
    ftof_rounding_mode(::Type) -> RoundingMode.T

Rounding mode used by [`Intrinsics.ftof`](@ref) when converting *to* this
target type. Defaults to `NearestEven`; extensions override for types
whose tileiras verifier rejects nearest-even (e.g. `Float8_E8M0FNU`,
which only accepts `Zero` or `PositiveInf`). Codegen reaches this via
[`lookup_ftof_rounding_mode`](@ref) so extension methods land in the
latest world.
"""
ftof_rounding_mode(::Type) = RoundingMode.NearestEven

@inline lookup_ftof_rounding_mode(@nospecialize(T::Type)) =
    Base.invokelatest(ftof_rounding_mode, T)::RoundingMode.T

const E8M0_LATE_SOURCES = (SimpleType.F64, SimpleType.F8E5M2, SimpleType.F8E4M3FN)
const F16_DIRECTED_SOURCES = (
    SimpleType.F16, SimpleType.BF16, SimpleType.F8E5M2,
    SimpleType.F8E4M3FN, SimpleType.F4E2M1FN,
)

function ftof_rounding_min_version(from::UInt8, to::UInt8, mode::RoundingMode.T)
    if mode == RoundingMode.NearestEven
        return to == SimpleType.F8E8M0FNU ? nothing : v"13.0"
    elseif mode == RoundingMode.Zero
        to in (SimpleType.F64, SimpleType.F32, SimpleType.TF32,
               SimpleType.F16, SimpleType.BF16) && return v"13.4"
        to == SimpleType.F8E8M0FNU &&
            return (from in E8M0_LATE_SOURCES ? v"13.4" : v"13.3")
    elseif mode in (RoundingMode.NegativeInf, RoundingMode.PositiveInf)
        to == SimpleType.F64 && return v"13.4"
        to == SimpleType.F32 && from != SimpleType.F8E8M0FNU && return v"13.4"
        to == SimpleType.F16 && from in F16_DIRECTED_SOURCES && return v"13.4"
        mode == RoundingMode.PositiveInf && to == SimpleType.F8E8M0FNU &&
            return (from in E8M0_LATE_SOURCES ? v"13.4" : v"13.3")
    elseif mode == RoundingMode.NearestAway
        to == SimpleType.F64 && return v"13.4"
        to in (SimpleType.F32, SimpleType.TF32) &&
            from ∉ (SimpleType.F64, SimpleType.F8E8M0FNU) && return v"13.4"
        to == SimpleType.F16 && from in F16_DIRECTED_SOURCES && return v"13.4"
    end
    return nothing
end

bytecode_rounding_mode(::Base.Rounding.RoundingMode{:Nearest}) = RoundingMode.NearestEven
bytecode_rounding_mode(::Base.Rounding.RoundingMode{:ToZero}) = RoundingMode.Zero
bytecode_rounding_mode(::Base.Rounding.RoundingMode{:Down}) = RoundingMode.NegativeInf
bytecode_rounding_mode(::Base.Rounding.RoundingMode{:Up}) = RoundingMode.PositiveInf
bytecode_rounding_mode(::Base.Rounding.RoundingMode{:NearestTiesAway}) = RoundingMode.NearestAway
bytecode_rounding_mode(mode::Base.Rounding.RoundingMode) =
    throw(IRError("rounding mode $mode is not supported for float conversion"))

const FLOAT_DTYPE_NAMES = Dict(
    SimpleType.F16 => "Float16", SimpleType.BF16 => "BFloat16",
    SimpleType.F32 => "Float32", SimpleType.TF32 => "TFloat32",
    SimpleType.F64 => "Float64", SimpleType.F8E4M3FN => "Float8_E4M3FN",
    SimpleType.F8E5M2 => "Float8_E5M2", SimpleType.F8E8M0FNU => "Float8_E8M0FNU",
    SimpleType.F4E2M1FN => "Float4_E2M1FN",
)
float_dtype_name(tag::UInt8) = get(FLOAT_DTYPE_NAMES, tag, string(tag))

function validate_ftof_rounding(from_tag::UInt8, to_tag::UInt8, mode::RoundingMode.T,
                                version::VersionNumber)
    from = float_dtype_name(from_tag)
    to = float_dtype_name(to_tag)
    min_version = ftof_rounding_min_version(from_tag, to_tag, mode)
    if min_version === nothing
        supported = RoundingMode.T[
            candidate for candidate in (RoundingMode.NearestEven, RoundingMode.Zero,
                                         RoundingMode.NegativeInf, RoundingMode.PositiveInf,
                                         RoundingMode.NearestAway)
            if ftof_rounding_min_version(from_tag, to_tag, candidate) !== nothing
        ]
        isempty(supported) && throw(IRError("float conversion from $from to $to is not supported"))
        throw(IRError("rounding mode $mode is not supported for conversion from $from to $to; supported modes are $(Tuple(supported))"))
    end
    version >= min_version ||
        throw(IRError("conversion from $from to $to with rounding mode $mode requires Tile IR bytecode v$min_version+, got v$version"))
    return
end

"""
    Intrinsics.ftof(x::Tile{<:AbstractFloat}, ::Type{F2}, rounding=nothing) -> Tile{F2}

Element-wise floating-point to floating-point conversion; lowers to
`cuda_tile.ftof`.

Also invocable with a scalar, promoted to a 0-D tile before codegen. `F2`
must be a compile-time constant. `rounding` accepts Base's `RoundNearest`,
`RoundToZero`, `RoundDown`, `RoundUp`, and `RoundNearestTiesAway`; when omitted,
the mode is picked from [`ftof_rounding_mode`](@ref).
"""
@intrinsic ftof(x::F1, ::Type{F2}, rounding=nothing) where {F1<:AbstractFloat, F2<:AbstractFloat}
function tfunc(𝕃, ::typeof(Intrinsics.ftof), @nospecialize(x), @nospecialize(target_type), @nospecialize(rest...))
    T = instanceof_tfunc(target_type)
    T === nothing && return nothing
    src = CC.widenconst(x)
    convert_result_type(src, T)
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.ftof), args)
    cb = ctx.cb
    tt = ctx.tt

    source = @something emit_value!(ctx, args[1]) throw(IRError("ftof: cannot resolve source"))
    target_type = @something get_constant(ctx, args[2]) throw(IRError("ftof: requires compile-time target type"))

    dtype = lookup_dtype!(tt, target_type)
    result_type_id = tile_type!(tt, dtype, source.shape)

    rounding_mode = if length(args) == 3
        mode = @something get_constant(ctx, args[3]) throw(IRError("ftof: requires a compile-time rounding mode"))
        mode isa Base.Rounding.RoundingMode || throw(IRError("ftof: invalid rounding mode $mode"))
        bytecode_rounding_mode(mode)
    else
        lookup_ftof_rounding_mode(target_type)
    end
    source_tag = simple_type_tag(tt, tile_eltype_id(tt, source.type_id))
    target_tag = simple_type_tag(tt, dtype)
    validate_ftof_rounding(source_tag, target_tag, rounding_mode, bytecode_version(tt))
    result_v = encode_FToFOp!(cb, result_type_id, source.v; rounding_mode)
    src_type = CC.widenconst(source.jltype)
    result_jltype = similar_type(src_type, target_type)
    CGVal(result_v, result_type_id, result_jltype, source.shape)
end

"""
    Intrinsics.ftoi(x::Tile{<:AbstractFloat}, ::Type{I}, s::Signedness.T) -> Tile{I}     where {I<:Integer}

Element-wise floating-point to integer conversion; lowers to
`cuda_tile.ftoi`.

Also invocable with a scalar, promoted to a 0-D tile before codegen. `s`
and `I` are compile-time constants.
"""
@intrinsic ftoi(x::AbstractFloat, ::Type{I}, s::Signedness.T) where {I<:Integer}
function tfunc(𝕃, ::typeof(Intrinsics.ftoi), @nospecialize(x), @nospecialize(target_type), @nospecialize(s))
    T = instanceof_tfunc(target_type)
    T === nothing && return nothing
    src = CC.widenconst(x)
    convert_result_type(src, T)
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.ftoi), args)
    cb = ctx.cb
    tt = ctx.tt

    source = @something emit_value!(ctx, args[1]) throw(IRError("ftoi: cannot resolve source"))
    target_type = @something get_constant(ctx, args[2]) throw(IRError("ftoi: requires compile-time target type"))
    signedness = @something get_constant(ctx, args[3]) throw(IRError("ftoi: requires compile-time signedness"))

    dtype = lookup_dtype!(tt, target_type)
    result_type_id = tile_type!(tt, dtype, source.shape)

    result_v = encode_FToIOp!(cb, result_type_id, source.v; signedness)
    src_type = CC.widenconst(source.jltype)
    result_jltype = similar_type(src_type, target_type)
    CGVal(result_v, result_type_id, result_jltype, source.shape)
end

"""
    Intrinsics.itof(x::Tile{<:Integer}, ::Type{F}, s::Signedness.T) -> Tile{F}     where {F<:AbstractFloat}

Element-wise integer to floating-point conversion; lowers to
`cuda_tile.itof`.

Also invocable with a scalar, promoted to a 0-D tile before codegen. `s`
and `F` are compile-time constants.
"""
@intrinsic itof(x::Integer, ::Type{F}, s::Signedness.T) where {F<:AbstractFloat}
function tfunc(𝕃, ::typeof(Intrinsics.itof), @nospecialize(x), @nospecialize(target_type), @nospecialize(s))
    T = instanceof_tfunc(target_type)
    T === nothing && return nothing
    src = CC.widenconst(x)
    convert_result_type(src, T)
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.itof), args)
    cb = ctx.cb
    tt = ctx.tt

    source = @something emit_value!(ctx, args[1]) throw(IRError("itof: cannot resolve source"))
    target_type = @something get_constant(ctx, args[2]) throw(IRError("itof: requires compile-time target type"))
    signedness = @something get_constant(ctx, args[3]) throw(IRError("itof: requires compile-time signedness"))

    dtype = lookup_dtype!(tt, target_type)
    result_type_id = tile_type!(tt, dtype, source.shape)

    result_v = encode_IToFOp!(cb, result_type_id, source.v; signedness)
    src_type = CC.widenconst(source.jltype)
    result_jltype = similar_type(src_type, target_type)
    CGVal(result_v, result_type_id, result_jltype, source.shape)
end

"""
    Intrinsics.trunci(x::Tile{<:Integer}, ::Type{T}) -> Tile{T}     where {T<:Integer}

Element-wise integer truncation; lowers to `cuda_tile.trunci`.

Also invocable with a scalar, promoted to a 0-D tile before codegen. `T`
must be a compile-time constant. The current emit does not pass an
`overflow` flag and so uses Tile IR's default.
"""
@intrinsic trunci(x::Integer, ::Type{T}) where {T<:Integer}
function tfunc(𝕃, ::typeof(Intrinsics.trunci), @nospecialize(x), @nospecialize(target_type))
    T = instanceof_tfunc(target_type)
    T === nothing && return nothing
    src = CC.widenconst(x)
    convert_result_type(src, T)
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.trunci), args)
    cb = ctx.cb
    tt = ctx.tt

    source = @something emit_value!(ctx, args[1]) throw(IRError("trunci: cannot resolve source"))
    target_type = @something get_constant(ctx, args[2]) throw(IRError("trunci: requires compile-time target type"))

    dtype = lookup_dtype!(tt, target_type)
    result_type_id = tile_type!(tt, dtype, source.shape)

    result_v = encode_TruncIOp!(cb, result_type_id, source.v)
    src_type = CC.widenconst(source.jltype)
    result_jltype = similar_type(src_type, target_type)
    CGVal(result_v, result_type_id, result_jltype, source.shape)
end

# cuda_tile.int_to_ptr, cuda_tile.ptr_to_int# NOTE: Used internally by atomic operations, not exposed as user intrinsics

# TODO: cuda_tile.ptr_to_ptr
