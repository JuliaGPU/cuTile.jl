#=============================================================================
 8.11. Views
 cuda_tile.get_index_space_shape, cuda_tile.get_tensor_shape,
 cuda_tile.load_view_tko, cuda_tile.make_partition_view,
 cuda_tile.make_tensor_view, cuda_tile.store_view_tko
=============================================================================#

#-----------------------------------------------------------------------------
# Helpers
#-----------------------------------------------------------------------------

"""
Convert integer padding mode value to bytecode PaddingValue enum.
Maps from cuTile.PaddingMode constants to bytecode PaddingValue.
"""
function padding_mode_to_padding_value(mode::Int)
    if mode == 0  # Undetermined
        PaddingMissing
    elseif mode == 1  # Zero
        PaddingZero
    elseif mode == 2  # NegZero
        PaddingNegZero
    elseif mode == 3  # Nan
        PaddingNan
    elseif mode == 4  # PosInf
        PaddingPosInf
    else  # 5 = NegInf
        PaddingNegInf
    end
end

#-----------------------------------------------------------------------------
# cuda_tile.make_tensor_view
#-----------------------------------------------------------------------------

function emit_intrinsic!(ctx::CodegenContext, ::typeof(Intrinsics.make_tensor_view), args, @nospecialize(result_type))
    array_arg = args[1]

    # Extract TileArray argument index
    arg_idx = extract_argument_index(array_arg)
    (arg_idx === nothing || !is_destructured_arg(ctx, arg_idx)) &&
        error("make_tensor_view() requires a TileArray argument")

    # Return ghost value with arg_idx stored as constant
    # The actual MakeTensorViewOp will be emitted by make_partition_view
    ghost_value(unwrap_type(result_type), arg_idx)
end

#-----------------------------------------------------------------------------
# cuda_tile.make_partition_view
#-----------------------------------------------------------------------------

function emit_intrinsic!(ctx::CodegenContext, ::typeof(Intrinsics.make_partition_view), args, @nospecialize(result_type))
    cb = ctx.cb
    tt = ctx.tt

    # args: (tensor_view, Val{shape}, padding_mode)
    tv_arg = emit_value!(ctx, args[1])
    tv_arg === nothing && error("make_partition_view() requires a TensorView argument")

    # Extract arg_idx from TensorView ghost value
    arg_idx = tv_arg.constant
    arg_idx === nothing && error("make_partition_view(): TensorView must come from make_tensor_view()")
    !is_destructured_arg(ctx, arg_idx) && error("make_partition_view(): invalid TensorView")

    # Get tile shape from Val argument
    tile_shape = get_constant(ctx, args[2])
    tile_shape isa Tuple || error("make_partition_view() shape must be a compile-time constant tuple")
    tile_shape = collect(Int, tile_shape)
    ndim = length(tile_shape)

    # Get padding mode
    padding_mode_int = 0  # Default: Undetermined
    if length(args) >= 3
        pm = get_constant(ctx, args[3])
        if pm isa Integer
            padding_mode_int = Int(pm)
        end
    end
    padding_value = padding_mode_to_padding_value(padding_mode_int)

    # Get TileArray info
    tilearray_type = get_arg_type(ctx, arg_idx)
    elem_type = eltype(tilearray_type)
    array_spec = get_array_spec(tilearray_type)

    ptr_vals = get_arg_flat_values(ctx, arg_idx, :ptr)
    isempty(ptr_vals) && error("Cannot get ptr from TileArray argument")
    array_val = ptr_vals[1]

    dtype = julia_to_tile_dtype!(tt, elem_type)

    # TensorView type
    tv_shape = fill(DYNAMIC_SHAPE, ndim)
    tv_strides = compute_tensor_view_strides(array_spec, ndim)
    tv_type = tensor_view_type!(tt, dtype, tv_shape, tv_strides)

    # PartitionView type
    dim_map = collect(0:ndim-1)
    pv_type = partition_view_type!(tt, tile_shape, tv_type, dim_map, padding_value)

    scalar_i32 = tile_type!(tt, I32(tt), Int[])

    # Get size and stride values (no indices for partition view creation)
    size_vals, stride_vals = get_size_stride_vals(ctx, arg_idx, true, ndim, tile_shape, Value[], scalar_i32)

    # Emit AssumeOps for optimization hints
    if array_spec !== nothing
        array_val, size_vals, stride_vals = emit_assume_ops!(ctx, array_val, size_vals, stride_vals, array_spec, dtype, scalar_i32; tv_strides)
    end

    # Filter strides to only pass dynamic ones as operands
    dynamic_stride_vals = filter_dynamic_strides(stride_vals, tv_strides)

    # Create tensor view
    tensor_view = encode_MakeTensorViewOp!(cb, tv_type, array_val, size_vals, dynamic_stride_vals)

    # Create partition view
    partition = encode_MakePartitionViewOp!(cb, pv_type, tensor_view)

    # Return CGVal with partition view value
    # Store ndim in constant for get_index_space_shape to use
    CGVal(partition, pv_type, unwrap_type(result_type), Int[], nothing, ndim)
end

#-----------------------------------------------------------------------------
# cuda_tile.get_index_space_shape
#-----------------------------------------------------------------------------

function emit_intrinsic!(ctx::CodegenContext, ::typeof(Intrinsics.get_index_space_shape), args, @nospecialize(result_type))
    cb = ctx.cb
    tt = ctx.tt

    # args: (partition_view, axis)
    pv_arg = emit_value!(ctx, args[1])
    pv_arg === nothing && error("get_index_space_shape() requires a PartitionView argument")
    pv_arg.v === nothing && error("get_index_space_shape() requires a materialized PartitionView")

    # Get axis (0-indexed)
    axis = get_constant(ctx, args[2])
    axis === nothing && error("get_index_space_shape() axis must be a compile-time constant")
    axis = Int(axis)

    # Get ndim from the PartitionView constant field
    ndim = pv_arg.constant
    ndim === nothing && error("get_index_space_shape(): PartitionView missing ndim info")

    # Create result types for all dimensions
    scalar_i32 = tile_type!(tt, I32(tt), Int[])
    result_types = fill(scalar_i32, ndim)

    # Emit GetIndexSpaceShapeOp
    shape_vals = encode_GetIndexSpaceShapeOp!(cb, result_types, pv_arg.v)

    # Return the value for the requested axis
    # shape_vals is a single Value when ndim == 1, otherwise a Tuple
    result_val = ndim == 1 ? shape_vals : shape_vals[axis + 1]
    CGVal(result_val, scalar_i32, Int32)
end

#-----------------------------------------------------------------------------
# cuda_tile.load_view_tko
#-----------------------------------------------------------------------------

function emit_intrinsic!(ctx::CodegenContext, ::typeof(Intrinsics.load_partition_view), args, @nospecialize(result_type))
    cb = ctx.cb
    tt = ctx.tt

    # args: (partition_view, indices...)
    pv_arg = emit_value!(ctx, args[1])
    pv_arg === nothing && error("load_partition_view() requires a PartitionView argument")
    pv_arg.v === nothing && error("load_partition_view() requires a materialized PartitionView")

    # Get ndim from PartitionView constant field
    ndim = pv_arg.constant
    ndim === nothing && error("load_partition_view(): PartitionView missing ndim info")

    # Extract tile shape from result type
    result_type_unwrapped = unwrap_type(result_type)
    elem_type = result_type_unwrapped.parameters[1]
    tile_shape = collect(Int, result_type_unwrapped.parameters[2])

    dtype = julia_to_tile_dtype!(tt, elem_type)
    tile_type = tile_type!(tt, dtype, tile_shape)
    token_type = Token(tt)
    scalar_i32 = tile_type!(tt, I32(tt), Int[])

    # Extract indices from args[2:end]
    index_vals = Value[]
    for i in 2:length(args)
        tv = emit_value!(ctx, args[i])
        tv !== nothing && tv.v !== nothing && push!(index_vals, tv.v)
    end

    # Pad indices if needed
    index_vals = pad_indices(ctx, index_vals, ndim, scalar_i32)

    # Load tile with token
    tile_val, new_token = encode_LoadViewTkoOp!(cb, tile_type, token_type, pv_arg.v, index_vals; token=ctx.token)
    ctx.token = new_token

    CGVal(tile_val, tile_type, Tile{elem_type, Tuple(tile_shape)}, tile_shape)
end

#-----------------------------------------------------------------------------
# cuda_tile.store_view_tko
#-----------------------------------------------------------------------------

function emit_intrinsic!(ctx::CodegenContext, ::typeof(Intrinsics.store_partition_view), args, @nospecialize(result_type))
    cb = ctx.cb
    tt = ctx.tt

    # args: (partition_view, tile, indices...)
    pv_arg = emit_value!(ctx, args[1])
    pv_arg === nothing && error("store_partition_view() requires a PartitionView argument")
    pv_arg.v === nothing && error("store_partition_view() requires a materialized PartitionView")

    # Get ndim from PartitionView constant field
    ndim = pv_arg.constant
    ndim === nothing && error("store_partition_view(): PartitionView missing ndim info")

    # Get tile value
    tile_tv = emit_value!(ctx, args[2])
    tile_tv === nothing && error("store_partition_view() requires a tile argument")
    tile_shape = tile_tv.shape
    tile_shape === nothing && error("Cannot determine tile shape for store_partition_view()")

    elem_type = unwrap_type(tile_tv.jltype).parameters[1]
    dtype = julia_to_tile_dtype!(tt, elem_type)
    scalar_i32 = tile_type!(tt, I32(tt), Int[])

    # Handle 0D scalar stores by reshaping to 1D (partition views require at least 1D)
    tile_val = tile_tv.v
    actual_ndim = ndim
    actual_tile_shape = tile_shape
    if length(tile_shape) == 0
        actual_ndim = 1
        actual_tile_shape = Int[1]
        tile_1d_type = tile_type!(tt, dtype, actual_tile_shape)
        tile_val = encode_ReshapeOp!(cb, tile_1d_type, tile_val)
    end

    # Extract indices from args[3:end]
    index_vals = Value[]
    for i in 3:length(args)
        tv = emit_value!(ctx, args[i])
        tv !== nothing && tv.v !== nothing && push!(index_vals, tv.v)
    end

    # Pad indices if needed
    index_vals = pad_indices(ctx, index_vals, actual_ndim, scalar_i32)

    # Store tile with token
    token_type = Token(tt)
    new_token = encode_StoreViewTkoOp!(cb, token_type, tile_val, pv_arg.v, index_vals; token=ctx.token)
    ctx.token = new_token

    nothing
end
