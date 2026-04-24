# Slicing of TileArrays
#
# Implements `Intrinsics.slice(arr, Val(axis), start, stop)` that mirrors cuTile
# Python's `_m_array_slice`. Produces a new TileArray whose base pointer is offset
# by `start * stride[axis]` and whose `sizes[axis]` is `stop - start`. Other
# dimensions are unchanged. `start`/`stop` are 0-indexed half-open bounds.
#
# The result is registered as a "virtual" destructured argument using a negative
# `arg_idx` (derived from the current SSA index). This reuses the existing lazy
# arg_ref machinery so downstream intrinsics (make_tensor_view, chained slice,
# field access via `.ptr`/`.sizes`/`.strides`) work without special-casing.

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

# cuda_tile slice: narrow a TileArray along a single axis to `[start, stop)`.
# Half-open bounds matching cuTile Python's `Array.slice(axis, start, stop)`
# (`res/cutile-python/src/cuda/tile/_ir/ops.py:_m_array_slice`). `axis_v::Val{Axis}`
# makes the axis compile-time without requiring CC.Const propagation through
# unrelated paths.
@intrinsic slice(arr, axis_v::Val, start, stop)

function tfunc(𝕃, ::typeof(Intrinsics.slice),
               @nospecialize(arr), @nospecialize(axis_v),
               @nospecialize(start), @nospecialize(stop))
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
    cb = ctx.cb
    tt = ctx.tt

    # args: (arr, Val(Axis), start, stop) — half-open [start, stop)
    arr_arg, axis_arg, start_arg, stop_arg = args[1], args[2], args[3], args[4]

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

    elem_T = eltype(src_type)
    size_elem_T = eltype(fieldtype(src_type, :sizes))
    scalar_size_type = tile_type_for_julia!(ctx, size_elem_T)

    ptr_fi = Base.fieldindex(src_type, :ptr)
    sizes_fi = Base.fieldindex(src_type, :sizes)
    strides_fi = Base.fieldindex(src_type, :strides)

    # Resolve ptr / sizes / strides of source
    ptr_path = [src_path..., ptr_fi]
    sizes_path = [src_path..., sizes_fi]
    strides_path = [src_path..., strides_fi]

    ptr_vals = get_arg_flat_values(ctx, src_arg_idx, ptr_path)
    (ptr_vals === nothing || isempty(ptr_vals)) &&
        throw(IRError("slice: cannot resolve base pointer of source array"))
    base_ptr = ptr_vals[1]

    old_sizes = collect_child_values(ctx, src_arg_idx, sizes_path, ndim)
    old_sizes === nothing && throw(IRError("slice: cannot resolve sizes of source array"))
    old_strides = collect_child_values(ctx, src_arg_idx, strides_path, ndim)
    old_strides === nothing && throw(IRError("slice: cannot resolve strides of source array"))

    # Resolve start and stop scalars (both in elements).
    start_tv = emit_value!(ctx, start_arg)
    stop_tv = emit_value!(ctx, stop_arg)
    (start_tv === nothing || stop_tv === nothing) &&
        throw(IRError("slice: cannot resolve start/stop"))
    start_val = start_tv.v::Value
    stop_val = stop_tv.v::Value

    # new_size = stop - start
    new_size_axis = encode_SubIOp!(cb, scalar_size_type, stop_val, start_val)

    # Read divby of start/stop from local assume_div_by wrappers (inserted by
    # insert_divby_assumes!) or literals.
    start_div = operand_divby_local(ctx.sci, start_arg)
    stop_div = operand_divby_local(ctx.sci, stop_arg)
    size_div = gcd(start_div, stop_div)

    if size_div > 1
        new_size_axis = encode_AssumeOp!(cb, scalar_size_type, new_size_axis, DivBy(size_div))
    end

    # new_base = base + start * stride[axis]
    stride_axis = old_strides[axis]
    offset_val = encode_MulIOp!(cb, scalar_size_type, start_val, stride_axis)

    elem_dtype = julia_to_tile_dtype!(tt, elem_T)
    ptr_dtype = pointer_type!(tt, elem_dtype)
    ptr_tile_type = tile_type!(tt, ptr_dtype, RowMajorShape(()))
    new_base = encode_OffsetOp!(cb, ptr_tile_type, base_ptr, offset_val)

    # Source ArraySpec (if known) feeds both divby computation and the result
    # type below. `stride_div_by[axis]` and `alignment` are compile-time facts.
    src_spec = array_spec(src_type)

    # new_base divby = gcd(src_alignment_bytes, start_div * stride_axis_div * elem_bytes).
    # `stride_div_by[i] == 0` in ArraySpec means "unknown"; treat as 1 for the
    # purposes of this product (the trivial bound — every integer is divisible by 1).
    stride_axis_div = src_spec === nothing ? 1 : max(src_spec.stride_div_by[axis], 1)
    src_alignment = src_spec === nothing ? 0 : src_spec.alignment
    elem_bytes = sizeof(elem_T)
    if src_alignment > 0 && start_div > 0
        offset_bytes_div = cap_divby(Int128(start_div) * Int128(stride_axis_div) * Int128(elem_bytes))
        new_base_div = gcd(src_alignment, offset_bytes_div)
        if new_base_div > 1
            new_base = encode_AssumeOp!(cb, ptr_tile_type, new_base, DivBy(new_base_div))
        end
    end

    # Build new sizes (axis replaced)
    new_sizes = Value[i == axis ? new_size_axis : old_sizes[i] for i in 1:ndim]

    # Compute the result TileArray type (same as the tfunc).
    result_type = src_spec === nothing ?
        TileArray{elem_T, ndim} :
        TileArray{elem_T, ndim, slice_spec(src_spec, axis)}

    # Register the slice result as a virtual destructured argument. Using
    # `-current_ssa_idx` keeps the namespace disjoint from real kernel params
    # (which are positive indices) while reusing the lazy arg_ref plumbing.
    slice_arg_idx = -ctx.current_ssa_idx
    ctx.arg_types[slice_arg_idx] = result_type
    ctx.arg_flat_values[(slice_arg_idx, [ptr_fi])] = Value[new_base]
    for i in 1:ndim
        ctx.arg_flat_values[(slice_arg_idx, [sizes_fi, i])] = Value[new_sizes[i]]
        ctx.arg_flat_values[(slice_arg_idx, [strides_fi, i])] = Value[old_strides[i]]
    end

    # Emit the TensorView now so that downstream make_tensor_view(sliced_arr) can
    # fetch it. Caches under the bare slice_arg_idx (empty path).
    key = tensor_view_key(slice_arg_idx, Int[])
    create_tensor_view!(ctx, key, new_base, new_sizes, Value[s for s in old_strides], result_type)

    arg_ref_value(slice_arg_idx, Int[], result_type)
end
