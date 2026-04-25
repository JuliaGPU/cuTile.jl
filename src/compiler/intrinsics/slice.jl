# Slicing of TileArrays
#
# Arithmetic and the new shape / strides tuples are built at the language
# level in `unsafe_view`; this intrinsic is a packaging step that bundles
# `(new_base, new_sizes, new_strides)` into an `ArrayValue` and attaches it
# to the result CGVal. Subsequent `arr.ptr` / `arr.sizes[i]` / `make_tensor_view(arr)`
# read off the aggregate directly.

"""
    slice_spec(spec::ArraySpec{N}) -> ArraySpec{N}

Conservative ArraySpec for a sliced TileArray: drops alignment to 0 and zeros
all per-axis `shape_div_by`. Strides and contiguity are preserved (slicing
doesn't change `stride[1] == 1` for column-major arrays). A future
divisibility analysis can recover tighter facts from the IR, at which point
this function (and the `slice_spec`-bearing return type from `tfunc`) becomes
redundant.
"""
function slice_spec(@nospecialize(spec::ArraySpec{N})) where N
    ArraySpec{N, 0, spec.contiguous, spec.stride_div_by, ntuple(_ -> 0, N)}()
end

@intrinsic slice(arr, new_base, new_sizes, new_strides)

function tfunc(𝕃, ::typeof(Intrinsics.slice),
               @nospecialize(arr), @nospecialize args...)
    T = CC.widenconst(arr)
    T <: TileArray || return nothing
    T isa DataType || return nothing

    spec = array_spec(T)
    elem_T = eltype(T)
    N = ndims(T)
    spec === nothing && return TileArray{elem_T, N}
    return TileArray{elem_T, N, slice_spec(spec)}
end

function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.slice), args)
    arr_arg, new_base_arg, new_sizes_arg, new_strides_arg = args

    arr_tv = emit_value!(ctx, arr_arg)
    arr_tv === nothing && throw(IRError("slice: cannot resolve array argument"))
    src_type = CC.widenconst(arr_tv.jltype)
    src_type <: TileArray || throw(IRError("slice: expected TileArray, got $src_type"))

    new_base_tv = emit_value!(ctx, new_base_arg)
    new_base_tv === nothing && throw(IRError("slice: cannot resolve new_base"))
    new_base = new_base_tv.v::Value
    new_sizes = Value[tv.v for tv in resolve_tuple(ctx, new_sizes_arg, "slice: new_sizes")]
    new_strides = Value[tv.v for tv in resolve_tuple(ctx, new_strides_arg, "slice: new_strides")]

    elem_T = eltype(src_type)
    ndim = ndims(src_type)
    src_spec = array_spec(src_type)
    result_type = src_spec === nothing ?
        TileArray{elem_T, ndim} :
        TileArray{elem_T, ndim, slice_spec(src_spec)}

    av = ArrayValue(new_base, new_sizes, new_strides, result_type)
    return array_value_cgval(av, result_type)
end
