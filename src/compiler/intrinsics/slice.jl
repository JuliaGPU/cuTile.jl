# Slicing of TileArrays
#
# Arithmetic and the new shape / strides tuples are built at the language level
# in `unsafe_view`; intrinsics just package them as a virtual destructured
# argument (negative `arg_idx` to stay disjoint from real kernel params) and
# pre-creates the `TensorView` so a later `make_tensor_view(sub)` finds it.

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

# Package an already-derived `(new_base, new_sizes, new_strides)` triple into
# a sliced TileArray aggregate. `arr` is carried purely for type information.
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

    # Resolve the source array purely for its type — the new flat values
    # arrive as explicit arguments below.
    arr_tv = emit_value!(ctx, arr_arg)
    arr_tv === nothing && throw(IRError("slice: cannot resolve array argument"))
    src_type = CC.widenconst(arr_tv.jltype)
    src_type <: TileArray || throw(IRError("slice: expected TileArray, got $src_type"))
    ndim = ndims(src_type)

    ptr_fi = Base.fieldindex(src_type, :ptr)
    sizes_fi = Base.fieldindex(src_type, :sizes)
    strides_fi = Base.fieldindex(src_type, :strides)

    # Pre-derived base pointer (scalar Ptr) and shape / stride tuples.
    new_base_tv = emit_value!(ctx, new_base_arg)
    new_base_tv === nothing && throw(IRError("slice: cannot resolve new_base"))
    new_base = new_base_tv.v::Value
    new_sizes = Value[tv.v for tv in resolve_tuple(ctx, new_sizes_arg, "slice: new_sizes")]
    new_strides = Value[tv.v for tv in resolve_tuple(ctx, new_strides_arg, "slice: new_strides")]

    # Result type: conservative spec (see `slice_spec`).
    elem_T = eltype(src_type)
    src_spec = array_spec(src_type)
    result_type = src_spec === nothing ?
        TileArray{elem_T, ndim} :
        TileArray{elem_T, ndim, slice_spec(src_spec)}

    # Register the slice result as a virtual destructured argument. Using
    # `-current_ssa_idx` keeps the namespace disjoint from real kernel params
    # (positive indices) while reusing the lazy arg_ref plumbing.
    slice_arg_idx = -ctx.current_ssa_idx
    ctx.arg_types[slice_arg_idx] = result_type
    ctx.arg_flat_values[(slice_arg_idx, [ptr_fi])] = Value[new_base]
    for i in 1:ndim
        ctx.arg_flat_values[(slice_arg_idx, [sizes_fi, i])] = Value[new_sizes[i]]
        ctx.arg_flat_values[(slice_arg_idx, [strides_fi, i])] = Value[new_strides[i]]
    end

    # Emit the TensorView now so a later make_tensor_view(sub) can fetch it.
    create_tensor_view!(ctx, tensor_view_key(slice_arg_idx, Int[]),
                        new_base, new_sizes, new_strides, result_type)

    arg_ref_value(slice_arg_idx, Int[], result_type)
end
