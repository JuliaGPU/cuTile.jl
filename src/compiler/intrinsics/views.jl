# views



# cuda_tile.get_index_space_shape
@intrinsic get_index_space_shape(pv, axis)
tfunc(𝕃, ::typeof(Intrinsics.get_index_space_shape), @nospecialize(pv), @nospecialize(axis)) = Int32
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.get_index_space_shape), args)
    cb = ctx.cb
    tt = ctx.tt

    # args: (partition_view, axis)
    pv_arg = emit_value!(ctx, args[1])
    pv_arg === nothing && throw(IRError("get_index_space_shape() requires a PartitionView argument"))
    pv_arg.v === nothing && throw(IRError("get_index_space_shape() requires a materialized PartitionView"))

    # Get axis (0-indexed Julia) and flip to Tile IR order
    axis = @something get_constant(ctx, args[2]) throw(IRError("get_index_space_shape() axis must be a compile-time constant"))
    axis = Int(axis)

    # Get ndim from the PartitionView constant field
    pv_arg.constant === nothing && throw(IRError("get_index_space_shape(): PartitionView missing ndim info"))
    ndim = something(pv_arg.constant)

    # Flip axis for row-major Tile IR: Julia dim 0 → Tile IR dim ndim-1
    tileir_axis = ndim - 1 - axis

    # Create result types for all dimensions
    scalar_i32 = tile_type!(tt, I32(tt), RowMajorShape(()))
    result_types = fill(scalar_i32, ndim)

    # Emit GetIndexSpaceShapeOp
    shape_vals = encode_GetIndexSpaceShapeOp!(cb, result_types, pv_arg.v)

    # Return the value for the requested axis (in Tile IR order)
    # shape_vals is a single Value when ndim == 1, otherwise a Tuple
    result_val = ndim == 1 ? shape_vals : shape_vals[tileir_axis + 1]
    CGVal(result_val, scalar_i32, Tile{Int32, Tuple{}})
end

# TODO: cuda_tile.get_tensor_shape

# cuda_tile.load_view_tko
@intrinsic load_partition_view(pv, latency, allow_tma, indices)
function tfunc(𝕃, ::typeof(Intrinsics.load_partition_view), @nospecialize(pv), @nospecialize args...)
    pv_type = CC.widenconst(pv)
    pv_type <: PartitionView || return nothing
    pv_type isa DataType || return nothing
    length(pv_type.parameters) >= 3 || return nothing
    T = eltype(pv_type)
    Shape = pv_type.parameters[3]
    Shape isa Type || return nothing
    return Tile{T, Shape}
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.load_partition_view), args)
    cb = ctx.cb
    tt = ctx.tt

    # Extract input token from last arg (added by token_order_pass!)
    input_token = extract_token_arg!(ctx, args)

    # args: (partition_view, latency, allow_tma, indices)
    pv_arg = emit_value!(ctx, args[1])
    pv_arg === nothing && throw(IRError("load_partition_view() requires a PartitionView argument"))
    pv_arg.v === nothing && throw(IRError("load_partition_view() requires a materialized PartitionView"))

    # Get ndim from PartitionView constant field
    pv_arg.constant === nothing && throw(IRError("load_partition_view(): PartitionView missing ndim info"))
    ndim = something(pv_arg.constant)

    # Extract tile shape from PartitionView type (PartitionView{T, N, Shape})
    # Reverse to Tile IR row-major order
    pv_type = CC.widenconst(pv_arg.jltype)
    elem_type = eltype(pv_type)
    tile_shape = RowMajorShape(ColMajorShape(size(pv_type)))

    dtype = julia_to_tile_dtype!(tt, elem_type)
    tile_type = tile_type!(tt, dtype, tile_shape)
    token_type = Token(tt)

    latency = @something get_constant(ctx, args[2]) throw(IRError("load_partition_view(): latency must be a compile-time constant"))
    allow_tma = @something get_constant(ctx, args[3]) throw(IRError("load_partition_view(): allow_tma must be a compile-time constant"))
    allow_tma_val = allow_tma isa Bool ? allow_tma : true

    # Extract indices
    index_tvs = resolve_tuple(ctx, args[4], "load_partition_view indices")
    index_vals = Value[tv.v for tv in index_tvs]
    index_jl_types = Type[tv.jltype for tv in index_tvs]

    unique_types = unique(index_jl_types)
    length(unique_types) <= 1 || throw(IRError("All index types must match, got: $unique_types"))
    isempty(unique_types) && ndim > 0 && throw(IRError("load_partition_view(): indices required for $(ndim)D view"))
    index_jl_type = isempty(unique_types) ? Int32 : unique_types[1]  # Int32 only for 0D case
    index_type = tile_type_for_julia!(ctx, index_jl_type)

    # Pad indices if needed, then reverse for Tile IR row-major order
    index_vals = pad_indices(ctx, index_vals, ndim, index_type, index_jl_type)
    reverse!(index_vals)

    # Create optimization hints if provided
    optimization_hints = create_optimization_hints(ctx, latency, allow_tma_val)

    tile_val, result_token = encode_LoadViewTkoOp!(
        cb, tile_type, token_type, pv_arg.v, index_vals;
        token = input_token, optimization_hints
    )

    # Store result token for TokenResultNode
    ctx.result_tokens[ctx.current_ssa_idx] = result_token

    julia_shape = ColMajorShape(tile_shape)
    return CGVal(tile_val, tile_type, Tile{elem_type, TupleType(julia_shape)}, tile_shape)
end

function pad_indices(ctx::CGCtx, index_vals::Vector{Value}, ndim::Int, idx_type::TypeId, idx_jl_type::Type)
    while length(index_vals) < ndim
        idx_bytes = reinterpret(UInt8, [eltype(idx_jl_type)(0)])
        push!(index_vals, encode_ConstantOp!(ctx.cb, idx_type, collect(idx_bytes)))
    end
    return index_vals
end

# cuda_tile.make_partition_view
@intrinsic make_partition_view(tv, shape, padding_mode, order)
function tfunc(𝕃, ::typeof(Intrinsics.make_partition_view), @nospecialize(tv), @nospecialize(shape_arg), @nospecialize args...)
    tv_type = CC.widenconst(tv)
    tv_type <: TensorView || return nothing
    isa(shape_arg, CC.Const) || return nothing
    shape = shape_arg.val
    T = eltype(tv_type)
    N = ndims(tv_type)
    return PartitionView{T, N, Tuple{shape...}}
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.make_partition_view), args)
    tv = emit_value!(ctx, args[1])
    tv === nothing && throw(IRError("make_partition_view() requires a TensorView argument"))

    # Shape from user call (e.g., load(arr, idx, (16,)))
    # Reverse to Tile IR row-major order
    shape = @something get_constant(ctx, args[2]) throw(IRError("make_partition_view() shape must be a compile-time constant"))
    shape isa Tuple || throw(IRError("make_partition_view() shape must be a tuple, got $(typeof(shape))"))
    validate_tile_shape(collect(Int, shape), "load")
    tile_shape = RowMajorShape(ColMajorShape(shape))

    padding_value = if length(args) >= 3
        convert_enum(PaddingValue, @something get_constant(ctx, args[3]) throw(IRError("padding_mode must be a compile-time constant")))
    else
        PaddingValue.Missing
    end

    tensor_view = tv.v
    tv_type = tv.type_id
    elem_type = eltype(tv.jltype)
    ndim = length(tile_shape)

    # Extract order (arg 4) and reverse for Tile IR row-major order
    # nothing → identity dim_map, (2,1) → [1, 0] (1-indexed → 0-indexed)
    order_val = @something get_constant(ctx, args[4]) throw(IRError("make_partition_view() order must be a compile-time constant"))
    if order_val === nothing
        dim_map = collect(0:ndim-1)
    else
        # Convert Julia dim_map to Tile IR: reverse and remap indices
        julia_dim_map = collect(Int, map(p -> p - 1, order_val))
        dim_map = [ndim - 1 - julia_dim_map[ndim - i] for i in 0:ndim-1]
    end

    pv_type = partition_view_type!(ctx.tt, tile_shape, tv_type, dim_map, padding_value)
    partition = encode_MakePartitionViewOp!(ctx.cb, pv_type, tensor_view)

    CGVal(partition, pv_type, PartitionView{elem_type, ndim, Tuple{shape...}}, RowMajorShape(()), nothing, Some(ndim), nothing, nothing)
end

"""
    build_array_value!(ctx, arg_idx, path, tilearray_type) -> ArrayValue

Pull the destructured base pointer, sizes, and strides for a TileArray arg
out of `ctx.arg_flat_values` at `path` and package them as an `ArrayValue`.
For dimensions whose stride was destructured as a literal (i.e. not a kernel
parameter), synthesize column-major stride values from the sizes.

Used for both top-level TileArray args (`path = []`) and nested TileArrays
inside struct args. The returned ArrayValue has its `tensor_view` field
unset; `emit_tensor_view!` fills it lazily on first use.
"""
function build_array_value!(ctx::CGCtx, arg_idx::Int, path::Vector{Int},
                            @nospecialize(tilearray_type::Type))
    cb = ctx.cb
    ndim = ndims(tilearray_type)
    size_elem_type = eltype(fieldtype(tilearray_type, :sizes))
    scalar_size_type = tile_type_for_julia!(ctx, size_elem_type)

    ptr_fi = Base.fieldindex(tilearray_type, :ptr)
    sizes_fi = Base.fieldindex(tilearray_type, :sizes)
    strides_fi = Base.fieldindex(tilearray_type, :strides)

    ptr_vals = get_arg_flat_values(ctx, arg_idx, [path..., ptr_fi])
    (ptr_vals === nothing || isempty(ptr_vals)) &&
        throw(IRError("Cannot get ptr from TileArray argument at path $path"))
    base_ptr = ptr_vals[1]

    sizes_from_arg = collect_child_values(ctx, arg_idx, [path..., sizes_fi], ndim)
    strides_from_arg = collect_child_values(ctx, arg_idx, [path..., strides_fi], ndim)

    sizes_from_arg === nothing && throw(IRError("TileArray at kernel entry requires explicit sizes"))
    length(sizes_from_arg) < ndim && throw(IRError("TileArray sizes don't match ndim"))

    sizes = Value[sizes_from_arg[i] for i in 1:ndim]

    if strides_from_arg !== nothing && length(strides_from_arg) >= ndim
        strides = Value[strides_from_arg[i] for i in 1:ndim]
    else
        # Synthesize column-major: stride[1]=1, stride[i]=stride[i-1]*size[i-1]
        strides = Value[]
        for dim in 1:ndim
            if dim == 1
                stride_bytes = reinterpret(UInt8, [size_elem_type(1)])
                push!(strides, encode_ConstantOp!(cb, scalar_size_type, collect(stride_bytes)))
            else
                push!(strides, encode_MulIOp!(cb, scalar_size_type, strides[end], sizes[dim-1]))
            end
        end
    end

    return ArrayValue(base_ptr, sizes, strides, tilearray_type)
end

"""
    emit_tensor_view!(ctx, av::ArrayValue) -> (Value, TypeId)

Emit a `MakeTensorViewOp` from `av`'s base pointer, sizes, and strides, and
cache the result on `av.tensor_view`. Idempotent — subsequent calls return
the cached pair.

`av.sizes`/`av.strides` are Julia column-major (reversed for Tile IR).
"""
function emit_tensor_view!(ctx::CGCtx, av::ArrayValue)
    av.tensor_view !== nothing && return av.tensor_view

    cb = ctx.cb
    tt = ctx.tt

    elem_type = eltype(av.tilearray_type)
    ndim = ndims(av.tilearray_type)
    spec = array_spec(av.tilearray_type)
    dtype = julia_to_tile_dtype!(tt, elem_type)
    size_elem_type = eltype(fieldtype(av.tilearray_type, :sizes))
    scalar_size_type = tile_type_for_julia!(ctx, size_elem_type)

    base_ptr = av.base_ptr
    size_vals = reverse(av.sizes)
    stride_vals = reverse(av.strides)

    tv_shape = RowMajorShape(fill(DYNAMIC_SHAPE, ndim))
    tv_strides = compute_tensor_view_strides(spec, ndim)
    tv_type = tensor_view_type!(tt, dtype, tv_shape, tv_strides)

    if spec !== nothing
        base_ptr, size_vals, stride_vals = emit_assume_ops!(ctx, base_ptr, size_vals, stride_vals,
                                                            spec, dtype, scalar_size_type; tv_strides)
    end

    dynamic_stride_vals = filter_dynamic_strides(stride_vals, tv_strides)

    tensor_view = encode_MakeTensorViewOp!(cb, tv_type, base_ptr, size_vals, dynamic_stride_vals)
    av.tensor_view = (tensor_view, tv_type)
    return av.tensor_view
end

"""
    find_array_value(ctx, cv::CGVal) -> Union{ArrayValue, Nothing}

Resolve a TileArray-typed CGVal to its underlying `ArrayValue`. Returns the
aggregate carried directly on the CGVal (slice results, top-level args), or
falls back to looking it up via the arg_ref → `ctx.array_values` map (nested
TileArray fields inside struct kernel args). Returns `nothing` if the CGVal
is neither.
"""
function find_array_value(ctx::CGCtx, cv::CGVal)
    cv.array_value !== nothing && return cv.array_value
    if cv.arg_ref !== nothing
        return get(ctx.array_values, cv.arg_ref, nothing)
    end
    return nothing
end

"""
    compute_tensor_view_strides(array_spec, ndim) -> Vector{Int64}

Compute the stride values for a TensorView type based on ArraySpec.
Returns static stride values where known, DYNAMIC_SHAPE where dynamic.

For contiguous column-major arrays (matching Julia's memory layout),
stride[1] = 1 is statically known. Higher dimensions are typically dynamic.
"""
function compute_tensor_view_strides(array_spec::Union{ArraySpec, Nothing}, ndim::Int)
    strides = fill(DYNAMIC_SHAPE, ndim)

    if array_spec !== nothing && array_spec.contiguous && ndim >= 1
        # Contiguous column-major array: Julia stride[1]=1 becomes Tile IR stride[ndim]=1
        strides[ndim] = 1
    end

    return strides
end

"""
    filter_dynamic_strides(stride_vals, tv_strides) -> Vector{Value}

Filter stride values to only include those corresponding to dynamic dimensions.
Only pass operands for dimensions where tv_strides[i] == DYNAMIC_SHAPE.
"""
function filter_dynamic_strides(stride_vals::Vector{Value}, tv_strides::Vector{Int64})
    dynamic_vals = Value[]
    for (i, stride_type_val) in enumerate(tv_strides)
        if stride_type_val == DYNAMIC_SHAPE && i <= length(stride_vals)
            push!(dynamic_vals, stride_vals[i])
        end
    end
    return dynamic_vals
end

# cuda_tile.make_tensor_view
@intrinsic make_tensor_view(arr::TileArray{T, N}) where {T, N}
function tfunc(𝕃, ::typeof(Intrinsics.make_tensor_view), @nospecialize(arr))
    t = CC.widenconst(arr)
    t <: TileArray || return nothing
    TensorView{eltype(t), ndims(t)}
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.make_tensor_view), args)
    cv = emit_value!(ctx, args[1])
    cv === nothing && throw(IRError("make_tensor_view() requires a TileArray argument"))
    av = find_array_value(ctx, cv)
    av === nothing && throw(IRError("make_tensor_view(): no ArrayValue for $(CC.widenconst(cv.jltype))"))
    tensor_view, tv_type = emit_tensor_view!(ctx, av)
    result_jltype = TensorView{eltype(av.tilearray_type), ndims(av.tilearray_type)}
    return CGVal(tensor_view, tv_type, result_jltype)
end

# cuda_tile.store_view_tko
@intrinsic store_partition_view(pv::PartitionView{T, N, Shape},
                                          tile::Tile{T},
                                          latency::Union{Int, Nothing},
                                          allow_tma::Bool,
                                          indices::NTuple{M, <:Integer}) where {T, N, Shape, M}
tfunc(𝕃, ::typeof(Intrinsics.store_partition_view), @nospecialize args...) = Nothing
efunc(::typeof(Intrinsics.store_partition_view), effects::CC.Effects) =
    CC.Effects(effects; effect_free=CC.ALWAYS_FALSE)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.store_partition_view), args)
    cb = ctx.cb
    tt = ctx.tt

    # Extract input token from last arg (added by token_order_pass!)
    input_token = extract_token_arg!(ctx, args)

    # args: (partition_view, tile, latency, allow_tma, indices)
    pv_arg = emit_value!(ctx, args[1])
    pv_arg === nothing && throw(IRError("store_partition_view() requires a PartitionView argument"))
    pv_arg.v === nothing && throw(IRError("store_partition_view() requires a materialized PartitionView"))

    # Get ndim from PartitionView constant field
    pv_arg.constant === nothing && throw(IRError("store_partition_view(): PartitionView missing ndim info"))
    ndim = something(pv_arg.constant)

    # Get tile value
    tile_tv = emit_value!(ctx, args[2])
    tile_tv === nothing && throw(IRError("store_partition_view() requires a tile argument"))
    tile_shape = tile_tv.shape
    tile_shape === nothing && throw(IRError("Cannot determine tile shape for store_partition_view()"))

    elem_type = eltype(CC.widenconst(tile_tv.jltype))
    dtype = julia_to_tile_dtype!(tt, elem_type)

    # Handle 0D scalar stores by reshaping to 1D (partition views require at least 1D)
    tile_val = tile_tv.v
    actual_ndim = ndim
    actual_tile_shape = tile_shape
    if length(tile_shape) == 0
        actual_ndim = 1
        actual_tile_shape = RowMajorShape([1])
        tile_1d_type = tile_type!(tt, dtype, actual_tile_shape)
        tile_val = encode_ReshapeOp!(cb, tile_1d_type, tile_val)
    end

    # Extract optimization hints (args[3] = latency, args[4] = allow_tma)
    latency = @something get_constant(ctx, args[3]) throw(IRError("store_partition_view(): latency must be a compile-time constant"))
    allow_tma = @something get_constant(ctx, args[4]) throw(IRError("store_partition_view(): allow_tma must be a compile-time constant"))
    allow_tma_val = allow_tma isa Bool ? allow_tma : true

    # Extract indices
    index_tvs = resolve_tuple(ctx, args[5], "store_partition_view indices")
    index_vals = Value[tv.v for tv in index_tvs]
    index_jl_types = Type[tv.jltype for tv in index_tvs]

    unique_types = unique(index_jl_types)
    length(unique_types) <= 1 || throw(IRError("All index types must match, got: $unique_types"))
    isempty(unique_types) && actual_ndim > 0 && throw(IRError("store_partition_view(): indices required for $(actual_ndim)D view"))
    index_jl_type = isempty(unique_types) ? Int32 : unique_types[1]  # Int32 only for 0D case
    index_type = tile_type_for_julia!(ctx, index_jl_type)

    # Pad indices if needed, then reverse for Tile IR row-major order
    index_vals = pad_indices(ctx, index_vals, actual_ndim, index_type, index_jl_type)
    reverse!(index_vals)

    # Create optimization hints if provided
    optimization_hints = create_optimization_hints(ctx, latency, allow_tma_val)

    token_type = Token(tt)

    result_token = encode_StoreViewTkoOp!(
        cb, token_type, tile_val, pv_arg.v, index_vals;
        token = input_token, optimization_hints
    )

    # Store result token for TokenResultNode
    ctx.result_tokens[ctx.current_ssa_idx] = result_token

    return nothing
end
