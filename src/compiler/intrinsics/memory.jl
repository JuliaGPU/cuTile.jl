#=============================================================================
 8.6. Memory
 cuda_tile.load_ptr_tko, cuda_tile.store_ptr_tko
=============================================================================#

#-----------------------------------------------------------------------------
# Helpers
#-----------------------------------------------------------------------------

function extract_pointer_elem_type(@nospecialize(jltype))
    jltype <: Ptr ? eltype(jltype) : Float32
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

"""
    extract_tile_shape(T) -> Vector{Int}

Extract shape from a Tile{T, Shape} type, returning Int[] if not a Tile type.
"""
function extract_tile_shape(@nospecialize(T))
    T = unwrap_type(T)
    if T <: Tile && length(T.parameters) >= 2
        shape = T.parameters[2]
        if shape isa Tuple
            return collect(Int, shape)
        end
    end
    Int[]
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

function emit_assume_ops!(ctx::CodegenContext, array_val::Value, size_vals::Vector{Value},
                          stride_vals::Vector{Value}, array_spec::ArraySpec, dtype::TypeId, scalar_i32::TypeId;
                          tv_strides::Union{Vector{Int64}, Nothing}=nothing)
    cb = ctx.cb
    tt = ctx.tt

    # Pointer alignment
    if array_spec.alignment > 0
        ptr_dtype = pointer_type!(tt, dtype)
        ptr_tile_type = tile_type!(tt, ptr_dtype, Int[])
        array_val = encode_AssumeOp!(cb, ptr_tile_type, array_val, DivBy(array_spec.alignment))
    end

    # Bounds assumes for sizes
    size_vals = Value[encode_AssumeOp!(cb, scalar_i32, v, Bounded(0, nothing)) for v in size_vals]

    # Bounds assumes for strides - only for dynamic strides
    if tv_strides !== nothing
        stride_vals = Value[tv_strides[i] == DYNAMIC_SHAPE ?
                       encode_AssumeOp!(cb, scalar_i32, v, Bounded(0, nothing)) : v
                       for (i, v) in enumerate(stride_vals)]
    else
        stride_vals = Value[encode_AssumeOp!(cb, scalar_i32, v, Bounded(0, nothing)) for v in stride_vals]
    end

    # Divisibility assumes for sizes
    if hasproperty(array_spec, :shape_div_by)
        for (i, div_by) in enumerate(array_spec.shape_div_by)
            if div_by > 0 && i <= length(size_vals)
                size_vals[i] = encode_AssumeOp!(cb, scalar_i32, size_vals[i], DivBy(div_by))
            end
        end
    end

    # Divisibility assumes for strides - only for dynamic strides
    if hasproperty(array_spec, :stride_div_by)
        for (i, div_by) in enumerate(array_spec.stride_div_by)
            if div_by > 0 && i <= length(stride_vals)
                # Skip if this stride is static (not DYNAMIC_SHAPE)
                if tv_strides === nothing || tv_strides[i] == DYNAMIC_SHAPE
                    stride_vals[i] = encode_AssumeOp!(cb, scalar_i32, stride_vals[i], DivBy(div_by))
                end
            end
        end
    end

    return array_val, size_vals, stride_vals
end

function pad_indices(ctx::CodegenContext, index_vals::Vector{Value}, ndim::Int, idx_type::TypeId)
    while length(index_vals) < ndim
        idx_bytes = reinterpret(UInt8, [Int32(0)])
        push!(index_vals, encode_ConstantOp!(ctx.cb, idx_type, collect(idx_bytes)))
    end
    return index_vals
end

#-----------------------------------------------------------------------------
# cuda_tile.load_ptr_tko
#-----------------------------------------------------------------------------

function emit_intrinsic!(ctx::CodegenContext, ::typeof(Intrinsics.load_ptr_tko), args, @nospecialize(result_type))
    cb = ctx.cb
    tt = ctx.tt

    # Get pointer tile (arg 1)
    ptrs_tv = emit_value!(ctx, args[1])
    ptrs_tv === nothing && error("load_ptr_tko: cannot resolve pointer tile")
    pointers = ptrs_tv.v
    tile_shape = ptrs_tv.shape

    # Get element type from result_type (Tile{T, S})
    result_type_unwrapped = unwrap_type(result_type)
    elem_type = result_type_unwrapped.parameters[1]
    dtype = julia_to_tile_dtype!(tt, elem_type)
    result_tile_type = tile_type!(tt, dtype, tile_shape)
    token_type = Token(tt)

    # Check if mask is provided (arg 2 is not nothing)
    has_mask = length(args) >= 2 && get_constant(ctx, args[2]) !== nothing

    if has_mask
        # Get mask tile (arg 2)
        mask_tv = emit_value!(ctx, args[2])
        mask_tv === nothing && error("load_ptr_tko: cannot resolve mask tile")
        mask = mask_tv.v

        # Get padding tile (arg 3)
        padding_tv = emit_value!(ctx, args[3])
        padding_tv === nothing && error("load_ptr_tko: cannot resolve padding tile")
        padding = padding_tv.v

        # Load with mask and padding
        tile_val, new_token = encode_LoadPtrTkoOp!(cb, result_tile_type, token_type, pointers;
                                                    mask=mask,
                                                    padding_value=padding,
                                                    token=ctx.token)
    else
        # Load without mask
        tile_val, new_token = encode_LoadPtrTkoOp!(cb, result_tile_type, token_type, pointers;
                                                    token=ctx.token)
    end
    ctx.token = new_token

    CGVal(tile_val, result_tile_type, result_type_unwrapped, tile_shape)
end

#-----------------------------------------------------------------------------
# cuda_tile.store_ptr_tko
#-----------------------------------------------------------------------------

function emit_intrinsic!(ctx::CodegenContext, ::typeof(Intrinsics.store_ptr_tko), args, @nospecialize(result_type))
    cb = ctx.cb
    tt = ctx.tt

    # Get pointer tile (arg 1)
    ptrs_tv = emit_value!(ctx, args[1])
    ptrs_tv === nothing && error("store_ptr_tko: cannot resolve pointer tile")
    pointers = ptrs_tv.v

    # Get value tile (arg 2)
    values_tv = emit_value!(ctx, args[2])
    values_tv === nothing && error("store_ptr_tko: cannot resolve values tile")
    values = values_tv.v

    token_type = Token(tt)

    # Check if mask is provided (arg 3 is not nothing)
    has_mask = length(args) >= 3 && get_constant(ctx, args[3]) !== nothing

    if has_mask
        # Get mask tile (arg 3)
        mask_tv = emit_value!(ctx, args[3])
        mask_tv === nothing && error("store_ptr_tko: cannot resolve mask tile")
        mask = mask_tv.v

        # Store with mask
        new_token = encode_StorePtrTkoOp!(cb, token_type, pointers, values;
                                           mask=mask,
                                           token=ctx.token)
    else
        # Store without mask
        new_token = encode_StorePtrTkoOp!(cb, token_type, pointers, values;
                                           token=ctx.token)
    end
    ctx.token = new_token

    nothing
end
