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
    scalar_i32 = tile_type!(tt, I32(tt), ScalarShape())
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
    tuple_arg = emit_value!(ctx, args[4])
    tuple_arg === nothing && throw(IRError("load_partition_view(): cannot resolve index tuple argument"))

    index_vals = Value[]
    index_jl_types = Type[]

    # Get tuple element refs
    tuple_arg.tuple !== nothing || throw(IRError("load_partition_view(): index tuple must have component refs"))
    for ref in tuple_arg.tuple
        tv = emit_value!(ctx, ref)
        tv === nothing && throw(IRError("load_partition_view(): cannot resolve index element"))
        push!(index_vals, tv.v)
        push!(index_jl_types, tv.jltype)
    end

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

    CGVal(partition, pv_type, PartitionView{elem_type, ndim, Tuple{shape...}}, ScalarShape(), nothing, Some(ndim), nothing)
end

"""
    cache_tensor_view!(ctx, arg_idx[, path, tilearray_type])

Create a TensorView for a TileArray argument at kernel entry.
Uses TileArray's ndim from type and requires explicit sizes/strides from parameters.

When `path` is provided (non-empty), the TileArray is nested inside a destructured struct
and its flat values are found at paths like `[path..., fieldindex(T, :ptr)]`.
"""
function cache_tensor_view!(ctx::CGCtx, arg_idx::Int,
                            path::Vector{Int}=Int[],
                            @nospecialize(tilearray_type::Type=Nothing))
    cb = ctx.cb
    tt = ctx.tt

    if tilearray_type === Nothing
        tilearray_type = get_arg_type(ctx, arg_idx)
    end
    elem_type = eltype(tilearray_type)
    ndim = ndims(tilearray_type)
    spec = array_spec(tilearray_type)
    dtype = julia_to_tile_dtype!(tt, elem_type)

    # Build paths for nested fields using field indices
    ptr_fi = Base.fieldindex(tilearray_type, :ptr)
    sizes_fi = Base.fieldindex(tilearray_type, :sizes)
    strides_fi = Base.fieldindex(tilearray_type, :strides)

    ptr_path = [path..., ptr_fi]
    sizes_path = [path..., sizes_fi]
    strides_path = [path..., strides_fi]

    ptr_vals = get_arg_flat_values(ctx, arg_idx, ptr_path)
    (ptr_vals === nothing || isempty(ptr_vals)) && throw(IRError("Cannot get ptr from TileArray argument at path $path"))
    array_val = ptr_vals[1]

    # Get sizes and strides from parameters (required at kernel entry)
    # Tuple fields are destructured per-element, so collect children
    sizes_from_arg = collect_child_values(ctx, arg_idx, sizes_path, ndim)
    strides_from_arg = collect_child_values(ctx, arg_idx, strides_path, ndim)

    sizes_from_arg === nothing && throw(IRError("TileArray at kernel entry requires explicit sizes"))
    length(sizes_from_arg) < ndim && throw(IRError("TileArray sizes don't match ndim"))

    # Deduce size/stride type from TileArray fields
    size_elem_type = eltype(fieldtype(tilearray_type, :sizes))
    scalar_size_type = tile_type_for_julia!(ctx, size_elem_type)

    # Sizes in Julia column-major order from parameters
    julia_size_vals = Value[sizes_from_arg[i] for i in 1:ndim]

    # Strides from parameters or compute for contiguous arrays (Julia column-major order)
    if strides_from_arg !== nothing && length(strides_from_arg) >= ndim
        julia_stride_vals = Value[strides_from_arg[i] for i in 1:ndim]
    else
        # Compute column-major strides: stride[1]=1, stride[i]=stride[i-1]*size[i-1]
        julia_stride_vals = Value[]
        for dim in 1:ndim
            if dim == 1
                stride_bytes = reinterpret(UInt8, [size_elem_type(1)])
                push!(julia_stride_vals, encode_ConstantOp!(cb, scalar_size_type, collect(stride_bytes)))
            else
                push!(julia_stride_vals, encode_MulIOp!(cb, scalar_size_type, julia_stride_vals[end], julia_size_vals[dim-1]))
            end
        end
    end

    # Reverse sizes and strides for Tile IR row-major order
    size_vals = reverse(julia_size_vals)
    stride_vals = reverse(julia_stride_vals)

    # TensorView type (strides also in Tile IR order: last dim = contiguous)
    tv_shape = RowMajorShape(fill(DYNAMIC_SHAPE, ndim))
    tv_strides = compute_tensor_view_strides(spec, ndim)
    tv_type = tensor_view_type!(tt, dtype, tv_shape, tv_strides)

    # Emit AssumeOps for optimization hints
    if spec !== nothing
        array_val, size_vals, stride_vals = emit_assume_ops!(ctx, array_val, size_vals, stride_vals, spec, dtype, scalar_size_type; tv_strides)
    end

    # Filter strides to only pass dynamic ones as operands
    dynamic_stride_vals = filter_dynamic_strides(stride_vals, tv_strides)

    # Create tensor view
    tensor_view = encode_MakeTensorViewOp!(cb, tv_type, array_val, size_vals, dynamic_stride_vals)

    # Cache it: use arg_idx for top-level, (arg_idx, path) for nested
    key = isempty(path) ? arg_idx : (arg_idx, path)
    ctx.tensor_views[key] = (tensor_view, tv_type)
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
    array_arg = args[1]

    # Case 1: Direct top-level argument
    arg_idx = extract_argument_index(array_arg)
    if arg_idx !== nothing && is_destructured_arg(ctx, arg_idx)
        haskey(ctx.tensor_views, arg_idx) || throw(IRError("TensorView not found for arg $arg_idx"))
        tensor_view, tv_type = ctx.tensor_views[arg_idx]
        tilearray_type = get_arg_type(ctx, arg_idx)
        result_jltype = TensorView{eltype(tilearray_type), ndims(tilearray_type)}
        return CGVal(tensor_view, tv_type, result_jltype)
    end

    # Case 2: Lazy arg ref (nested TileArray inside a struct)
    tv = emit_value!(ctx, array_arg)
    if tv !== nothing && is_arg_ref(tv)
        arg_idx, chain = tv.arg_ref
        key = (arg_idx, chain)
        haskey(ctx.tensor_views, key) || throw(IRError("TensorView not found for arg $arg_idx at path $chain"))
        tensor_view, tv_type = ctx.tensor_views[key]
        ta_type = CC.widenconst(tv.jltype)
        result_jltype = TensorView{eltype(ta_type), ndims(ta_type)}
        return CGVal(tensor_view, tv_type, result_jltype)
    end

    throw(IRError("make_tensor_view() requires a TileArray argument (direct or nested)"))
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
    tuple_arg = emit_value!(ctx, args[5])
    tuple_arg === nothing && throw(IRError("store_partition_view(): cannot resolve index tuple argument"))

    index_vals = Value[]
    index_jl_types = Type[]

    # Get tuple element refs
    tuple_arg.tuple !== nothing || throw(IRError("store_partition_view(): index tuple must have component refs"))
    for ref in tuple_arg.tuple
        tv = emit_value!(ctx, ref)
        tv === nothing && throw(IRError("store_partition_view(): cannot resolve index element"))
        push!(index_vals, tv.v)
        push!(index_jl_types, tv.jltype)
    end

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
