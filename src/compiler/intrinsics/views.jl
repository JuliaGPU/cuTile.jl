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


## cuda_tile.get_index_space_shape

@eval Intrinsics begin
    """
        get_index_space_shape(pv::PartitionView, axis) -> Int32

    Get the number of tiles along the given axis (0-indexed).
    Compiled to cuda_tile.get_index_space_shape.
    """
    @noinline function get_index_space_shape(pv::PartitionView{T, N, Shape}, axis::Integer)::Int32 where {T, N, Shape}
        Base.donotdelete(pv)
        Base.inferencebarrier(zero(Int32))
    end
end

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


## TODO: cuda_tile.get_tensor_shape


## cuda_tile.load_view_tko

@eval Intrinsics begin
    """
        load_partition_view(pv::PartitionView, index...) -> Tile

    Load a tile from a partition view at the given 0-indexed tile coordinates.
    Compiled to cuda_tile.load_view_tko.
    """
    @noinline function load_partition_view(pv::PartitionView{T, N, Shape}, index::Vararg{Integer})::Tile{T, Shape} where {T, N, Shape}
        Base.donotdelete(pv)
        Base.inferencebarrier(Tile{T, Shape}())
    end
end

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

function pad_indices(ctx::CodegenContext, index_vals::Vector{Value}, ndim::Int, idx_type::TypeId)
    while length(index_vals) < ndim
        idx_bytes = reinterpret(UInt8, [Int32(0)])
        push!(index_vals, encode_ConstantOp!(ctx.cb, idx_type, collect(idx_bytes)))
    end
    return index_vals
end


## cuda_tile.make_partition_view

@eval Intrinsics begin
    """
        make_partition_view(tv::TensorView, shape_val, padding_mode) -> PartitionView

    Create a PartitionView from a TensorView with the given tile shape.
    Compiled to cuda_tile.make_partition_view.
    """
    @noinline function make_partition_view(tv::TensorView{T, N}, ::Val{Shape}, padding_mode::Int)::PartitionView{T, N, Shape} where {T, N, Shape}
        Base.donotdelete(tv)
        Base.inferencebarrier(PartitionView{T, N, Shape}())
    end
end

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

function get_array_spec(@nospecialize(T))
    if T <: TileArray && length(T.parameters) >= 3
        S = T.parameters[3]
        S isa ArraySpec && return S
    end
    nothing
end

"""
    compute_tensor_view_strides(array_spec, ndim) -> Vector{Int64}

Compute the stride values for a TensorView type based on ArraySpec.
Returns static stride values where known, DYNAMIC_SHAPE where dynamic.

For contiguous arrays (array_spec.contiguous == true), stride[1] = 1 is statically known.
Higher dimensions are typically dynamic unless we have explicit info.
"""
function compute_tensor_view_strides(array_spec::Union{ArraySpec, Nothing}, ndim::Int)
    strides = fill(DYNAMIC_SHAPE, ndim)

    if array_spec !== nothing && array_spec.contiguous && ndim >= 1
        # Contiguous array: first stride is statically known to be 1
        strides[1] = 1
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

function get_size_stride_vals(ctx::CodegenContext, arg_idx, is_tilearray::Bool, ndim::Int,
                               tile_shape::Vector{Int}, index_vals::Vector{Value}, scalar_i32::TypeId)
    cb = ctx.cb
    tt = ctx.tt
    size_vals = Value[]
    stride_vals = Value[]

    if is_tilearray
        sizes_from_arg = get_arg_flat_values(ctx, arg_idx, :sizes)
        strides_from_arg = get_arg_flat_values(ctx, arg_idx, :strides)

        if sizes_from_arg !== nothing && length(sizes_from_arg) >= ndim
            size_vals = Value[sizes_from_arg[i] for i in 1:ndim]
        end
        if strides_from_arg !== nothing && length(strides_from_arg) >= ndim
            stride_vals = Value[strides_from_arg[i] for i in 1:ndim]
        end
    end

    # Compute from grid if not available
    if isempty(size_vals)
        if ndim > 3
            error("4D+ tile operations require TileArray with explicit sizes (grid only provides 3D)")
        end
        nb_x, nb_y, nb_z = encode_GetNumTileBlocksOp!(cb, scalar_i32, scalar_i32, scalar_i32)
        grid_sizes = [nb_x, nb_y, nb_z]

        for dim in 1:ndim
            tile_size_bytes = reinterpret(UInt8, [Int32(tile_shape[dim])])
            tile_size_val = encode_ConstantOp!(cb, scalar_i32, collect(tile_size_bytes))
            size_val = encode_MulIOp!(cb, scalar_i32, grid_sizes[dim], tile_size_val)
            push!(size_vals, size_val)
        end
    end

    if isempty(stride_vals)
        for dim in 1:ndim
            if dim == 1
                stride_bytes = reinterpret(UInt8, [Int32(1)])
                stride_val = encode_ConstantOp!(cb, scalar_i32, collect(stride_bytes))
            else
                stride_val = encode_MulIOp!(cb, scalar_i32, stride_vals[end], size_vals[dim-1])
            end
            push!(stride_vals, stride_val)
        end
    end

    return size_vals, stride_vals
end


## cuda_tile.make_tensor_view

@eval Intrinsics begin
    """
        make_tensor_view(arr::TileArray) -> TensorView

    Create a TensorView from a TileArray.
    Compiled to cuda_tile.make_tensor_view.
    """
    @noinline function make_tensor_view(arr::TileArray{T, N})::TensorView{T, N} where {T, N}
        Base.donotdelete(arr)
        Base.inferencebarrier(TensorView{T, N}())
    end
end

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


## cuda_tile.store_view_tko

@eval Intrinsics begin
    """
        store_partition_view(pv::PartitionView, tile, index...) -> Nothing

    Store a tile to a partition view at the given 0-indexed tile coordinates.
    Compiled to cuda_tile.store_view_tko.
    """
    @noinline function store_partition_view(pv::PartitionView{T, N, Shape}, tile::Tile{T, Shape}, index::Vararg{Integer})::Nothing where {T, N, Shape}
        Base.donotdelete(pv)
        Base.donotdelete(tile)
        nothing
    end
end

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
