# Slicing of TileArrays — codegen side.
#
# The arithmetic (`subi`, `muli`, pointer `offset`) lives at the language level
# in `_view_chain` (src/language/operations.jl) so it goes through the regular
# arithmetic intrinsics and benefits from constant folding, strength reduction,
# and — once it lands — the divisibility dataflow analysis (see DIVBY.md).
#
# What remains in codegen is aggregate synthesis: register the slice result as
# a virtual destructured argument (negative `arg_idx` to stay disjoint from
# real kernel params) with `new_base`/`new_sizes`/source strides filed under
# `ctx.arg_flat_values`, and pre-create the `TensorView` keyed so that a later
# `make_tensor_view(sub)` finds it. This mirrors Python's
# `unflatten_aggregates` + `make_aggregate` in `_m_array_slice`.

"""
    slice_spec(spec::ArraySpec{N}, axis::Int) -> ArraySpec{N}

Conservative ArraySpec for a sliced TileArray: drops alignment to 0 and resets
`shape_div_by[axis]` to 0 (unknown). Strides and other axes' divisibility are
preserved.
"""
function slice_spec(@nospecialize(spec::ArraySpec{N}), axis::Int) where N
    new_shape_div_by = ntuple(N) do i
        i == axis ? 0 : spec.shape_div_by[i]
    end
    ArraySpec{N, 0, spec.contiguous, spec.stride_div_by, new_shape_div_by}()
end

# cuda_tile slice: package an already-derived `(new_base, new_size)` pair into
# a sliced TileArray aggregate. Arithmetic is done at the language level; see
# `_view_chain` in src/language/operations.jl. `axis_v::Val{Axis}` makes the
# axis compile-time.
@intrinsic slice(arr, axis_v::Val, new_base, new_size)

function tfunc(𝕃, ::typeof(Intrinsics.slice),
               @nospecialize(arr), @nospecialize(axis_v),
               @nospecialize(new_base), @nospecialize(new_size))
    T = CC.widenconst(arr)
    T <: TileArray || return nothing
    T isa DataType || return nothing

    axisT = CC.widenconst(axis_v)
    axisT <: Val || return nothing
    axisT isa DataType && length(axisT.parameters) >= 1 || return nothing
    axis = axisT.parameters[1]
    axis isa Integer || return nothing
    N = ndims(T)
    1 <= axis <= N || return nothing

    spec = array_spec(T)
    elem_T = eltype(T)
    if spec === nothing
        return TileArray{elem_T, N}
    else
        return TileArray{elem_T, N, slice_spec(spec, Int(axis))}
    end
end

function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.slice), args)
    # args: (arr, Val(Axis), new_base, new_size) — arithmetic already emitted
    arr_arg, axis_arg, new_base_arg, new_size_arg = args[1], args[2], args[3], args[4]

    # Resolve axis from Val type parameter
    axis_val = @something get_constant(ctx, axis_arg) throw(IRError("slice: axis must be a Val{N}"))
    axis = if axis_val isa Val
        first(typeof(axis_val).parameters)
    elseif axis_val isa Type{<:Val}
        first(axis_val.parameters)
    else
        throw(IRError("slice: axis must be Val{Int}, got $axis_val"))
    end
    axis isa Integer || throw(IRError("slice: axis must be an Integer, got $(typeof(axis))"))
    axis = Int(axis)

    # Resolve source array as a lazy arg_ref (must point to a TileArray).
    arr_tv = emit_value!(ctx, arr_arg)
    arr_tv === nothing && throw(IRError("slice: cannot resolve array argument"))
    is_arg_ref(arr_tv) || throw(IRError("slice: array must be a TileArray (kernel parameter or slice result)"))

    src_arg_idx, src_path = arr_tv.arg_ref
    src_type = CC.widenconst(arr_tv.jltype)
    src_type <: TileArray || throw(IRError("slice: expected TileArray, got $src_type"))

    ndim = ndims(src_type)
    1 <= axis <= ndim || throw(IRError("slice: axis $axis out of range for $ndim-D array"))

    ptr_fi = Base.fieldindex(src_type, :ptr)
    sizes_fi = Base.fieldindex(src_type, :sizes)
    strides_fi = Base.fieldindex(src_type, :strides)

    sizes_path = [src_path..., sizes_fi]
    strides_path = [src_path..., strides_fi]

    old_sizes = collect_child_values(ctx, src_arg_idx, sizes_path, ndim)
    old_sizes === nothing && throw(IRError("slice: cannot resolve sizes of source array"))
    old_strides = collect_child_values(ctx, src_arg_idx, strides_path, ndim)
    old_strides === nothing && throw(IRError("slice: cannot resolve strides of source array"))

    # Pre-computed new base pointer and new axis size (emitted at language level).
    new_base_tv = emit_value!(ctx, new_base_arg)
    new_size_tv = emit_value!(ctx, new_size_arg)
    (new_base_tv === nothing || new_size_tv === nothing) &&
        throw(IRError("slice: cannot resolve new_base / new_size"))
    new_base = new_base_tv.v::Value
    new_size_axis = new_size_tv.v::Value

    # Build new sizes (axis replaced).
    new_sizes = Value[i == axis ? new_size_axis : old_sizes[i] for i in 1:ndim]

    # Result TileArray type. slice_spec drops alignment and the axis's
    # shape_div_by; a future dataflow pass will recover tighter facts.
    elem_T = eltype(src_type)
    src_spec = array_spec(src_type)
    result_type = src_spec === nothing ?
        TileArray{elem_T, ndim} :
        TileArray{elem_T, ndim, slice_spec(src_spec, axis)}

    # Register the slice result as a virtual destructured argument. Using
    # `-current_ssa_idx` keeps the namespace disjoint from real kernel params
    # (positive indices) while reusing the lazy arg_ref plumbing.
    slice_arg_idx = -ctx.current_ssa_idx
    ctx.arg_types[slice_arg_idx] = result_type
    ctx.arg_flat_values[(slice_arg_idx, [ptr_fi])] = Value[new_base]
    for i in 1:ndim
        ctx.arg_flat_values[(slice_arg_idx, [sizes_fi, i])] = Value[new_sizes[i]]
        ctx.arg_flat_values[(slice_arg_idx, [strides_fi, i])] = Value[old_strides[i]]
    end

    # Emit the TensorView now so that downstream make_tensor_view(sliced_arr)
    # can fetch it. Caches under the bare slice_arg_idx (empty path).
    key = tensor_view_key(slice_arg_idx, Int[])
    create_tensor_view!(ctx, key, new_base, new_sizes, Value[s for s in old_strides], result_type)

    arg_ref_value(slice_arg_idx, Int[], result_type)
end
