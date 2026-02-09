# views

"""
Convert integer padding mode value to bytecode PaddingValue enum.
"""
function padding_mode_to_padding_value(mode::Int)
    mode == 0 ? PaddingMissing :
    mode == 1 ? PaddingZero :
    mode == 2 ? PaddingNegZero :
    mode == 3 ? PaddingNan :
    mode == 4 ? PaddingPosInf : PaddingNegInf
end

"""
Get padding value from args, with default.
"""
function get_padding_value(ctx::CGCtx, args)
    mode = 0  # Default: Undetermined
    if length(args) >= 3
        pm = get_constant(ctx, args[3])
        pm isa Integer && (mode = Int(pm))
    end
    padding_mode_to_padding_value(mode)
end

# cuda_tile.get_index_space_shape
@eval Intrinsics begin
    """
        get_index_space_shape(pv::PartitionView, axis) -> Int32

    Get the number of tiles along the given axis (0-indexed).
    Compiled to cuda_tile.get_index_space_shape.
    """
    @noinline function get_index_space_shape(pv::PartitionView{T, N, Shape}, axis::Integer) where {T, N, Shape}
        compilerbarrier(:const, zero(Int32))
    end
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.get_index_space_shape), args)
    cb = ctx.cb
    tt = ctx.tt

    # args: (partition_view, axis)
    pv_arg = emit_value!(ctx, args[1])
    pv_arg === nothing && throw(IRError("get_index_space_shape() requires a PartitionView argument"))
    pv_arg.v === nothing && throw(IRError("get_index_space_shape() requires a materialized PartitionView"))

    # Get axis (0-indexed)
    axis = get_constant(ctx, args[2])
    axis === nothing && throw(IRError("get_index_space_shape() axis must be a compile-time constant"))
    axis = Int(axis)

    # Get ndim from the PartitionView constant field
    pv_arg.constant === nothing && throw(IRError("get_index_space_shape(): PartitionView missing ndim info"))
    ndim = something(pv_arg.constant)

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

# TODO: cuda_tile.get_tensor_shape

# cuda_tile.load_view_tko
@eval Intrinsics begin
    """
        load_partition_view(pv::PartitionView, latency, allow_tma, index...) -> Tile

    Load a tile from a partition view at the given 0-indexed tile coordinates.
    Compiled to cuda_tile.load_view_tko.
    """
    @noinline function load_partition_view(pv::PartitionView{T, N, Shape},
                                            latency::Union{Int, Nothing},
                                            allow_tma::Bool,
                                            indices::NTuple{M, <:Integer}) where {T, N, Shape, M}
        compilerbarrier(:type, nothing)
    end
end
function tfunc(::typeof(Intrinsics.load_partition_view), argtypes::Vector{Any})
    length(argtypes) >= 2 || return nothing
    pv_type = CC.widenconst(argtypes[2])
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

    # args: (partition_view, latency, allow_tma, indices)
    pv_arg = emit_value!(ctx, args[1])
    pv_arg === nothing && throw(IRError("load_partition_view() requires a PartitionView argument"))
    pv_arg.v === nothing && throw(IRError("load_partition_view() requires a materialized PartitionView"))

    # Get ndim from PartitionView constant field
    pv_arg.constant === nothing && throw(IRError("load_partition_view(): PartitionView missing ndim info"))
    ndim = something(pv_arg.constant)

    # Extract tile shape from PartitionView type (PartitionView{T, N, Shape})
    pv_type = CC.widenconst(pv_arg.jltype)
    elem_type = eltype(pv_type)
    tile_shape = collect(Int, size(pv_type))

    dtype = julia_to_tile_dtype!(tt, elem_type)
    tile_type = tile_type!(tt, dtype, tile_shape)
    token_type = Token(tt)

    # Extract optimization hints (args[2] = latency, args[3] = allow_tma)
    latency = get_constant(ctx, args[2])
    allow_tma = get_constant(ctx, args[3])

    # Verify we got compile-time constants
    if latency === nothing && allow_tma === nothing
        throw(IRError("load_partition_view(): latency and allow_tma must be compile-time constants"))
    end
    # allow_tma defaults to true if not provided
    allow_tma_val = allow_tma === nothing ? true : allow_tma::Bool

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

    # Pad indices if needed
    index_vals = pad_indices(ctx, index_vals, ndim, index_type, index_jl_type)

    # Create optimization hints if provided
    optimization_hints = create_optimization_hints(ctx, latency, allow_tma_val)

    # Load tile with token
    tile_val, new_token = encode_LoadViewTkoOp!(cb, tile_type, token_type, pv_arg.v, index_vals;
                                                 token=ctx.token, optimization_hints)
    ctx.token = new_token

    CGVal(tile_val, tile_type, Tile{elem_type, Tuple{tile_shape...}}, tile_shape)
end

function pad_indices(ctx::CGCtx, index_vals::Vector{Value}, ndim::Int, idx_type::TypeId, idx_jl_type::Type)
    while length(index_vals) < ndim
        idx_bytes = reinterpret(UInt8, [idx_jl_type(0)])
        push!(index_vals, encode_ConstantOp!(ctx.cb, idx_type, collect(idx_bytes)))
    end
    return index_vals
end

# cuda_tile.make_partition_view
@eval Intrinsics begin
    """
        make_partition_view(tv::TensorView, shape_val, padding_mode, order) -> PartitionView

    Create a PartitionView from a TensorView with the given tile shape.
    The `order` parameter (NTuple{N,Int} or nothing) specifies
    the logical-to-physical dimension mapping (1-indexed), or identity if nothing.
    Compiled to cuda_tile.make_partition_view.
    """
    @noinline function make_partition_view(tv::TensorView{T, N}, shape::NTuple{M, Int}, padding_mode::Int, order) where {T, N, M}
        compilerbarrier(:type, nothing)
    end
end
function tfunc(::typeof(Intrinsics.make_partition_view), argtypes::Vector{Any})
    length(argtypes) >= 3 || return nothing
    tv_type = CC.widenconst(argtypes[2])
    tv_type <: TensorView || return nothing
    shape_arg = argtypes[3]
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
    shape = get_constant(ctx, args[2])
    shape isa Tuple || throw(IRError("make_partition_view() shape must be a tuple, got $(typeof(shape))"))
    tile_shape = collect(Int, shape)
    validate_tile_shape(tile_shape, "load")

    padding_value = get_padding_value(ctx, args)

    tensor_view = tv.v
    tv_type = tv.type_id
    elem_type = eltype(tv.jltype)
    ndim = length(tile_shape)

    # Extract order (arg 4)
    # nothing → identity dim_map, (2,1) → [1, 0] (1-indexed → 0-indexed)
    order_val = get_constant(ctx, args[4])
    if order_val === nothing
        dim_map = collect(0:ndim-1)
    else
        dim_map = collect(Int, map(p -> p - 1, order_val))
    end

    pv_type = partition_view_type!(ctx.tt, tile_shape, tv_type, dim_map, padding_value)
    partition = encode_MakePartitionViewOp!(ctx.cb, pv_type, tensor_view)

    CGVal(partition, pv_type, PartitionView{elem_type, ndim, Tuple{tile_shape...}}, Int[], nothing, Some(ndim), nothing)
end

"""
    cache_tensor_view!(ctx, arg_idx)

Create a TensorView for a TileArray argument at kernel entry.
Uses TileArray's ndim from type and requires explicit sizes/strides from parameters.
"""
function cache_tensor_view!(ctx::CGCtx, arg_idx::Int)
    cb = ctx.cb
    tt = ctx.tt

    tilearray_type = get_arg_type(ctx, arg_idx)
    elem_type = eltype(tilearray_type)
    ndim = ndims(tilearray_type)
    spec = array_spec(tilearray_type)
    dtype = julia_to_tile_dtype!(tt, elem_type)

    ptr_vals = get_arg_flat_values(ctx, arg_idx, :ptr)
    isempty(ptr_vals) && throw(IRError("Cannot get ptr from TileArray argument"))
    array_val = ptr_vals[1]

    # Get sizes and strides from parameters (required at kernel entry)
    sizes_from_arg = get_arg_flat_values(ctx, arg_idx, :sizes)
    strides_from_arg = get_arg_flat_values(ctx, arg_idx, :strides)

    sizes_from_arg === nothing && throw(IRError("TileArray at kernel entry requires explicit sizes"))
    length(sizes_from_arg) < ndim && throw(IRError("TileArray sizes don't match ndim"))

    # Deduce size/stride type from TileArray fields
    size_elem_type = eltype(fieldtype(tilearray_type, :sizes))
    scalar_size_type = tile_type_for_julia!(ctx, size_elem_type)

    # Sizes are passed through directly from parameters
    size_vals = Value[sizes_from_arg[i] for i in 1:ndim]

    # Strides from parameters or compute for contiguous arrays
    if strides_from_arg !== nothing && length(strides_from_arg) >= ndim
        stride_vals = Value[strides_from_arg[i] for i in 1:ndim]
    else
        # Compute column-major strides: stride[1]=1, stride[i]=stride[i-1]*size[i-1]
        # This matches Julia's memory layout where the first dimension is contiguous
        stride_vals = Value[]
        for dim in 1:ndim
            if dim == 1
                stride_bytes = reinterpret(UInt8, [size_elem_type(1)])
                push!(stride_vals, encode_ConstantOp!(cb, scalar_size_type, collect(stride_bytes)))
            else
                push!(stride_vals, encode_MulIOp!(cb, scalar_size_type, stride_vals[end], size_vals[dim-1]))
            end
        end
    end

    # TensorView type
    tv_shape = fill(DYNAMIC_SHAPE, ndim)
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

    # Cache it
    ctx.tensor_views[arg_idx] = (tensor_view, tv_type)
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
        # Contiguous column-major array: first stride is statically known to be 1
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

# cuda_tile.make_tensor_view
@eval Intrinsics begin
    """
        make_tensor_view(arr::TileArray) -> TensorView

    Create a TensorView from a TileArray.
    Compiled to cuda_tile.make_tensor_view.
    """
    @noinline function make_tensor_view(arr::TileArray{T, N})::TensorView{T, N} where {T, N}
        TensorView{T, N}()
    end
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.make_tensor_view), args)
    array_arg = args[1]

    # Extract TileArray argument index
    arg_idx = extract_argument_index(array_arg)
    (arg_idx === nothing || !is_destructured_arg(ctx, arg_idx)) &&
        throw(IRError("make_tensor_view() requires a TileArray argument"))

    # Look up cached TensorView (created at kernel entry)
    haskey(ctx.tensor_views, arg_idx) || throw(IRError("TensorView not found for arg $arg_idx"))
    tensor_view, tv_type = ctx.tensor_views[arg_idx]

    tilearray_type = get_arg_type(ctx, arg_idx)
    result_jltype = TensorView{eltype(tilearray_type), ndims(tilearray_type)}

    CGVal(tensor_view, tv_type, result_jltype)
end

# cuda_tile.store_view_tko
@eval Intrinsics begin
    """
        store_partition_view(pv::PartitionView, tile, latency, allow_tma, index...) -> Nothing

    Store a tile to a partition view at the given 0-indexed tile coordinates.
    Compiled to cuda_tile.store_view_tko.
    """
    @noinline function store_partition_view(pv::PartitionView{T, N, Shape},
                                             tile::Tile{T},
                                             latency::Union{Int, Nothing},
                                             allow_tma::Bool,
                                             indices::NTuple{M, <:Integer}) where {T, N, Shape, M}
        donotdelete()
        nothing
    end
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.store_partition_view), args)
    cb = ctx.cb
    tt = ctx.tt

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
        actual_tile_shape = Int[1]
        tile_1d_type = tile_type!(tt, dtype, actual_tile_shape)
        tile_val = encode_ReshapeOp!(cb, tile_1d_type, tile_val)
    end

    # Extract optimization hints (args[3] = latency, args[4] = allow_tma)
    latency = get_constant(ctx, args[3])
    allow_tma = get_constant(ctx, args[4])

    # Verify we got compile-time constants
    if latency === nothing && allow_tma === nothing
        throw(IRError("store_partition_view(): latency and allow_tma must be compile-time constants"))
    end
    # allow_tma defaults to true if not provided
    allow_tma_val = allow_tma === nothing ? true : allow_tma::Bool

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

    # Pad indices if needed
    index_vals = pad_indices(ctx, index_vals, actual_ndim, index_type, index_jl_type)

    # Create optimization hints if provided
    optimization_hints = create_optimization_hints(ctx, latency, allow_tma_val)

    # Store tile with token
    token_type = Token(tt)
    new_token = encode_StoreViewTkoOp!(cb, token_type, tile_val, pv_arg.v, index_vals;
                                        token=ctx.token, optimization_hints)
    ctx.token = new_token

    nothing
end
