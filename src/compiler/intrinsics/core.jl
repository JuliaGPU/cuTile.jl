#=============================================================================
 8.3. Core Operations
 cuda_tile.broadcast, cuda_tile.cat, cuda_tile.cmpf, cuda_tile.cmpi,
 cuda_tile.constant, cuda_tile.extract, cuda_tile.get_num_tile_blocks,
 cuda_tile.get_tile_block_id, cuda_tile.iota, cuda_tile.mmaf, cuda_tile.mmai,
 cuda_tile.offset, cuda_tile.permute, cuda_tile.reduce, cuda_tile.reshape,
 cuda_tile.select
=============================================================================#

#-----------------------------------------------------------------------------
# cuda_tile.get_tile_block_id, cuda_tile.get_num_tile_blocks
#-----------------------------------------------------------------------------

function emit_intrinsic!(ctx::CodegenContext, ::typeof(Intrinsics.get_tile_block_id), args, @nospecialize(result_type))
    axis = @something get_constant(ctx, args[1]) error("get_tile_block_id() axis must be a compile-time constant")
    axis in (0, 1, 2) || error("get_tile_block_id() axis must be 0, 1, or 2, got $axis")

    res_type = tile_type!(ctx.tt, I32(ctx.tt), Int[])
    bid_x, bid_y, bid_z = encode_GetTileBlockIdOp!(ctx.cb, res_type, res_type, res_type)
    result = (bid_x, bid_y, bid_z)[axis + 1]

    CGVal(result, res_type, Int32)
end

function emit_intrinsic!(ctx::CodegenContext, ::typeof(Intrinsics.get_num_tile_blocks), args, @nospecialize(result_type))
    axis = @something get_constant(ctx, args[1]) error("get_num_tile_blocks() axis must be a compile-time constant")
    axis in (0, 1, 2) || error("get_num_tile_blocks() axis must be 0, 1, or 2, got $axis")

    res_type = tile_type!(ctx.tt, I32(ctx.tt), Int[])
    nb_x, nb_y, nb_z = encode_GetNumTileBlocksOp!(ctx.cb, res_type, res_type, res_type)

    CGVal((nb_x, nb_y, nb_z)[axis + 1], res_type, Int32)
end

#-----------------------------------------------------------------------------
# cuda_tile.broadcast
#-----------------------------------------------------------------------------

function emit_intrinsic!(ctx::CodegenContext, ::typeof(Intrinsics.broadcast), args, @nospecialize(result_type))
    cb = ctx.cb
    tt = ctx.tt

    # Get source tile
    source = emit_value!(ctx, args[1])
    source === nothing && error("Cannot resolve source operand for broadcast()")

    # Get source element type
    source_type = unwrap_type(source.jltype)
    source_elem = source_type <: Tile ? source_type.parameters[1] : source_type

    # Extract target shape from the constant tuple argument
    target_shape_tuple = get_constant(ctx, args[2])
    target_shape_tuple isa Tuple || error("broadcast() shape must be a compile-time constant tuple")
    target_shape = collect(Int, target_shape_tuple)

    # If already the right shape, return unchanged
    if source.shape == target_shape
        return source
    end

    # Use the existing broadcast helper
    dtype = julia_to_tile_dtype!(tt, source_elem)
    result_v = broadcast_tile_to_shape!(cb, tt, source, target_shape, dtype)
    result_type_id = tile_type!(tt, dtype, target_shape)

    CGVal(result_v, result_type_id, Tile{source_elem, Tuple(target_shape)}, target_shape)
end

"""
    broadcast_tile_to_shape!(cb, tt, tv::CGVal, target_shape::Vector{Int}, dtype::TypeId) -> Value

Broadcast a tile to a target shape by inserting ReshapeOp (for leading 1s) and BroadcastOp.
Returns the value after broadcasting, or the original value if shapes already match.
"""
function broadcast_tile_to_shape!(cb::CodeBuilder, tt::TypeTable, tv::CGVal,
                                   target_shape::Vector{Int}, dtype::TypeId)
    src_shape = tv.shape

    # Already the right shape?
    if src_shape == target_shape
        return tv.v
    end

    current_val = tv.v
    current_shape = src_shape

    # Step 1: Add leading 1s via ReshapeOp if needed (dimension mismatch)
    if length(current_shape) < length(target_shape)
        # Prepend 1s to match target ndim
        n_extra = length(target_shape) - length(current_shape)
        new_shape = vcat(fill(1, n_extra), current_shape)
        reshaped_type = tile_type!(tt, dtype, new_shape)
        current_val = encode_ReshapeOp!(cb, reshaped_type, current_val)
        current_shape = new_shape
    end

    # Step 2: Broadcast dimensions that are 1 to target size
    if current_shape != target_shape
        broadcast_type = tile_type!(tt, dtype, target_shape)
        current_val = encode_BroadcastOp!(cb, broadcast_type, current_val)
    end

    current_val
end

#-----------------------------------------------------------------------------
# cuda_tile.cat
#-----------------------------------------------------------------------------

function emit_intrinsic!(ctx::CodegenContext, ::typeof(Intrinsics.cat), args, @nospecialize(result_type))
    cb = ctx.cb
    tt = ctx.tt

    # args[1] is the tuple of tiles - need to trace back to Core.tuple call
    tuple_ref = args[1]
    if tuple_ref isa SSAValue
        stmt = code(ctx.target)[tuple_ref.id]
        if stmt isa Expr && stmt.head === :call
            callee = stmt.args[1]
            if callee isa GlobalRef && callee.mod === Core && callee.name === :tuple
                tile1_ref = stmt.args[2]
                tile2_ref = stmt.args[3]
            else
                error("cat() expects tuple created with Core.tuple, got call to $callee")
            end
        else
            error("cat() expects tuple SSA value pointing to Core.tuple call")
        end
    else
        error("cat() expects tuple SSA value, got $(typeof(tuple_ref))")
    end

    # Emit the two tiles
    lhs = emit_value!(ctx, tile1_ref)
    rhs = emit_value!(ctx, tile2_ref)
    (lhs === nothing || rhs === nothing) && error("Cannot resolve tile operands for cat()")

    # Get axis from Val{Axis}
    axis_val = get_constant(ctx, args[2])
    axis_val isa Integer || error("cat() axis must be a compile-time constant integer")

    # Handle negative axis
    lhs_shape = lhs.shape
    ndims = length(lhs_shape)
    axis = axis_val < 0 ? ndims + axis_val : axis_val

    # Compute output shape - concatenate along the axis
    rhs_shape = rhs.shape
    output_shape = collect(Int, lhs_shape)
    output_shape[axis + 1] += rhs_shape[axis + 1]  # 1-based indexing

    # Get element type
    elem_type = unwrap_type(lhs.jltype)
    if elem_type <: Tile
        elem_type = elem_type.parameters[1]
    end

    # Create output tile type
    dtype = julia_to_tile_dtype!(tt, elem_type)
    output_tile_type = tile_type!(tt, dtype, output_shape)

    # Emit CatOp (axis is 0-indexed for bytecode)
    result = encode_CatOp!(cb, output_tile_type, lhs.v, rhs.v, axis)

    CGVal(result, output_tile_type, Tile{elem_type, Tuple(output_shape)}, output_shape)
end

#-----------------------------------------------------------------------------
# cuda_tile.permute (includes transpose)
#-----------------------------------------------------------------------------

function emit_intrinsic!(ctx::CodegenContext, ::typeof(Intrinsics.transpose), args, @nospecialize(result_type))
    cb = ctx.cb
    tt = ctx.tt

    source = emit_value!(ctx, args[1])
    source === nothing && error("Cannot resolve operand for transpose()")

    input_shape = source.shape
    isempty(input_shape) && error("Cannot determine tile shape for transpose()")

    output_shape = reverse(input_shape)

    elem_type = unwrap_type(source.jltype)
    if elem_type <: Tile
        elem_type = elem_type.parameters[1]
    end

    dtype = julia_to_tile_dtype!(tt, elem_type)
    output_tile_type = tile_type!(tt, dtype, output_shape)

    ndim = length(output_shape)
    permutation = collect(ndim-1:-1:0)

    result = encode_PermuteOp!(cb, output_tile_type, source.v, permutation)

    CGVal(result, output_tile_type, Tile{elem_type, Tuple(output_shape)}, output_shape)
end

function emit_intrinsic!(ctx::CodegenContext, ::typeof(Intrinsics.reshape), args, @nospecialize(result_type))
    cb = ctx.cb
    tt = ctx.tt

    # Get source tile
    source = emit_value!(ctx, args[1])
    source === nothing && error("Cannot resolve source operand for reshape()")

    # Extract target shape from Val{Shape} argument
    target_shape_tuple = get_constant(ctx, args[2])
    target_shape_tuple isa Tuple || error("reshape() shape must be a compile-time constant tuple")
    target_shape = collect(Int, target_shape_tuple)

    # Get element type
    source_type = unwrap_type(source.jltype)
    elem_type = source_type <: Tile ? source_type.parameters[1] : source_type

    # Create target tile type
    dtype = julia_to_tile_dtype!(tt, elem_type)
    result_type_id = tile_type!(tt, dtype, target_shape)

    # Emit ReshapeOp
    result_v = encode_ReshapeOp!(cb, result_type_id, source.v)

    CGVal(result_v, result_type_id, Tile{elem_type, Tuple(target_shape)}, target_shape)
end

function emit_intrinsic!(ctx::CodegenContext, ::typeof(Intrinsics.permute), args, @nospecialize(result_type))
    cb = ctx.cb
    tt = ctx.tt

    # Get source tile
    source = emit_value!(ctx, args[1])
    source === nothing && error("Cannot resolve source operand for permute()")

    input_shape = source.shape
    isempty(input_shape) && error("Cannot determine tile shape for permute()")

    # Extract permutation from Val{Perm} argument
    perm_tuple = get_constant(ctx, args[2])
    perm_tuple isa Tuple || error("permute() permutation must be a compile-time constant tuple")

    # Convert to 0-indexed vector for bytecode
    permutation = collect(Int, perm_tuple)

    # Compute output shape based on permutation
    # permutation[i] tells us which input dimension goes to output position i
    output_shape = [input_shape[p + 1] for p in permutation]

    # Get element type
    elem_type = unwrap_type(source.jltype)
    if elem_type <: Tile
        elem_type = elem_type.parameters[1]
    end

    # Create output tile type
    dtype = julia_to_tile_dtype!(tt, elem_type)
    output_tile_type = tile_type!(tt, dtype, output_shape)

    # Emit PermuteOp
    result = encode_PermuteOp!(cb, output_tile_type, source.v, permutation)

    CGVal(result, output_tile_type, Tile{elem_type, Tuple(output_shape)}, output_shape)
end

#-----------------------------------------------------------------------------
# cuda_tile.extract
#-----------------------------------------------------------------------------

function emit_intrinsic!(ctx::CodegenContext, ::typeof(Intrinsics.extract), args, @nospecialize(result_type))
    cb = ctx.cb
    tt = ctx.tt

    # Get source tile
    source = emit_value!(ctx, args[1])
    source === nothing && error("Cannot resolve source operand for extract()")

    # Extract index from Val{Index} argument
    index_tuple = get_constant(ctx, args[2])
    index_tuple isa Tuple || error("extract() index must be a compile-time constant tuple")

    # Extract shape from Val{Shape} argument
    shape_tuple = get_constant(ctx, args[3])
    shape_tuple isa Tuple || error("extract() shape must be a compile-time constant tuple")
    output_shape = collect(Int, shape_tuple)

    # Get element type
    elem_type = unwrap_type(source.jltype)
    if elem_type <: Tile
        elem_type = elem_type.parameters[1]
    end

    # Create output tile type
    dtype = julia_to_tile_dtype!(tt, elem_type)
    output_tile_type = tile_type!(tt, dtype, output_shape)

    # Create constant index values (0D i32 tiles)
    scalar_i32 = tile_type!(tt, I32(tt), Int[])
    index_vals = Value[]
    for idx in index_tuple
        idx_bytes = collect(reinterpret(UInt8, [Int32(idx)]))
        idx_val = encode_ConstantOp!(cb, scalar_i32, idx_bytes)
        push!(index_vals, idx_val)
    end

    # Emit ExtractOp
    result = encode_ExtractOp!(cb, output_tile_type, source.v, index_vals)

    CGVal(result, output_tile_type, Tile{elem_type, Tuple(output_shape)}, output_shape)
end

#-----------------------------------------------------------------------------
# cuda_tile.constant (full/zeros)
#-----------------------------------------------------------------------------

function emit_intrinsic!(ctx::CodegenContext, ::typeof(Intrinsics.constant), args, @nospecialize(result_type))
    cb = ctx.cb
    tt = ctx.tt

    # Extract shape
    shape = get_constant(ctx, args[1])
    shape isa Tuple || error("full() shape must be a compile-time constant tuple")
    tile_shape = collect(Int, shape)

    # Extract value
    value = @something get_constant(ctx, args[2]) error("full() value must be a compile-time constant")

    # Extract dtype from result type
    result_type_unwrapped = unwrap_type(result_type)
    elem_type = Float32
    if result_type_unwrapped <: Tile
        elem_type = result_type_unwrapped.parameters[1]
    end

    dtype = julia_to_tile_dtype!(tt, elem_type)
    tile_type = tile_type!(tt, dtype, tile_shape)

    # Create scalar constant
    scalar_type = tile_type!(tt, dtype, Int[])
    value_bytes = constant_to_bytes(value, elem_type)
    scalar_val = encode_ConstantOp!(cb, scalar_type, value_bytes)

    # Reshape and broadcast
    ndims = length(tile_shape)
    if ndims > 0
        ones_shape = fill(1, ndims)
        reshaped_type = tile_type!(tt, dtype, ones_shape)
        reshaped_val = encode_ReshapeOp!(cb, reshaped_type, scalar_val)
    else
        reshaped_val = scalar_val
    end

    result = encode_BroadcastOp!(cb, tile_type, reshaped_val)

    CGVal(result, tile_type, Tile{elem_type, Tuple(tile_shape)}, tile_shape)
end

#-----------------------------------------------------------------------------
# cuda_tile.iota
#-----------------------------------------------------------------------------

function emit_intrinsic!(ctx::CodegenContext, ::typeof(Intrinsics.iota), args, @nospecialize(result_type))
    cb = ctx.cb
    tt = ctx.tt

    # Extract shape
    shape = get_constant(ctx, args[1])
    shape isa Tuple || error("iota() shape must be a compile-time constant tuple")
    tile_shape = collect(Int, shape)

    # Extract dtype from result type
    result_type_unwrapped = unwrap_type(result_type)
    elem_type = Int32
    if result_type_unwrapped <: Tile
        elem_type = result_type_unwrapped.parameters[1]
    end

    dtype = julia_to_tile_dtype!(tt, elem_type)
    tile_type = tile_type!(tt, dtype, tile_shape)

    # Emit IotaOp
    result = encode_IotaOp!(cb, tile_type)

    CGVal(result, tile_type, Tile{elem_type, Tuple(tile_shape)}, tile_shape)
end

#-----------------------------------------------------------------------------
# cuda_tile.mmaf, cuda_tile.mmai
#-----------------------------------------------------------------------------

function emit_intrinsic!(ctx::CodegenContext, ::typeof(Intrinsics.mma), args, @nospecialize(result_type))
    cb = ctx.cb

    lhs = emit_value!(ctx, args[1])
    rhs = emit_value!(ctx, args[2])
    acc = emit_value!(ctx, args[3])

    (lhs === nothing || rhs === nothing || acc === nothing) && error("Cannot resolve operands for mma()")

    result = encode_MmaFOp!(cb, acc.type_id, lhs.v, rhs.v, acc.v)

    CGVal(result, acc.type_id, acc.jltype, acc.shape)
end

#-----------------------------------------------------------------------------
# cuda_tile.offset
#-----------------------------------------------------------------------------

function emit_intrinsic!(ctx::CodegenContext, ::typeof(Intrinsics.offset), args, @nospecialize(result_type))
    cb = ctx.cb
    tt = ctx.tt

    # Get base pointer (arg 1)
    base_ptr_tv = emit_value!(ctx, args[1])
    base_ptr_tv === nothing && error("offset: cannot resolve base pointer")
    base_ptr = base_ptr_tv.v

    # Get offsets tile (arg 2)
    offsets_tv = emit_value!(ctx, args[2])
    offsets_tv === nothing && error("offset: cannot resolve offsets tile")
    offsets = offsets_tv.v
    tile_shape = offsets_tv.shape

    # Get pointer element type from result_type (Tile{Ptr{T}, S})
    result_type_unwrapped = unwrap_type(result_type)
    ptr_elem_type = eltype(result_type_unwrapped.parameters[1])  # T from Ptr{T}
    elem_dtype = julia_to_tile_dtype!(tt, ptr_elem_type)
    ptr_dtype = pointer_type!(tt, elem_dtype)
    ptr_tile_type = tile_type!(tt, ptr_dtype, tile_shape)

    # Broadcast base pointer to tile shape
    ndims = length(tile_shape)
    if ndims > 0
        ones_shape = fill(1, ndims)
        reshaped_ptr_type = tile_type!(tt, ptr_dtype, ones_shape)
        base_ptr_reshaped = encode_ReshapeOp!(cb, reshaped_ptr_type, base_ptr)
        base_ptr_tile = encode_BroadcastOp!(cb, ptr_tile_type, base_ptr_reshaped)
    else
        base_ptr_tile = base_ptr
    end

    # Compute offset pointers: base_ptr + offsets (element offset)
    pointers = encode_OffsetOp!(cb, ptr_tile_type, base_ptr_tile, offsets)

    CGVal(pointers, ptr_tile_type, result_type_unwrapped, tile_shape)
end

#-----------------------------------------------------------------------------
# cuda_tile.reduce
#-----------------------------------------------------------------------------

function emit_intrinsic!(ctx::CodegenContext, ::typeof(Intrinsics.reduce_sum), args, @nospecialize(result_type))
    emit_reduce!(ctx, args, result_type, :add)
end

function emit_intrinsic!(ctx::CodegenContext, ::typeof(Intrinsics.reduce_max), args, @nospecialize(result_type))
    emit_reduce!(ctx, args, result_type, :max)
end

function emit_reduce!(ctx::CodegenContext, args, @nospecialize(result_type), reduce_fn::Symbol)
    cb = ctx.cb
    tt = ctx.tt

    # Get input tile
    input_tv = emit_value!(ctx, args[1])
    input_tv === nothing && error("Cannot resolve input tile for reduction")

    # Get reduction axis
    axis = @something get_constant(ctx, args[2]) error("Reduction axis must be a compile-time constant")

    # Get element type and shapes
    input_type = unwrap_type(input_tv.jltype)
    elem_type = input_type <: Tile ? input_type.parameters[1] : input_type
    input_shape = input_tv.shape
    isempty(input_shape) && error("Cannot reduce scalar tile")

    # Compute output shape (dimension at axis is removed)
    output_shape = Int[input_shape[i] for i in eachindex(input_shape) if i != axis + 1]

    dtype = julia_to_tile_dtype!(tt, elem_type)

    # Output tile type
    output_tile_type = tile_type!(tt, dtype, output_shape)

    # Scalar type for reduction body (0D tile)
    scalar_tile_type = tile_type!(tt, dtype, Int[])

    # Create identity value - use simple dtype (f32), not tile type
    identity_val = reduce_fn == :add ? -0.0 : (reduce_fn == :max ? -Inf : 0.0)
    identity = FloatIdentity(identity_val, dtype, elem_type)

    # Emit ReduceOp
    results = encode_ReduceOp!(cb, [output_tile_type], [input_tv.v], axis, [identity], [scalar_tile_type]) do block_args
        acc, elem = block_args[1], block_args[2]

        if reduce_fn == :add
            res = encode_AddFOp!(cb, scalar_tile_type, acc, elem)
        elseif reduce_fn == :max
            res = encode_MaxFOp!(cb, scalar_tile_type, acc, elem)
        else
            error("Unsupported reduction function: $reduce_fn")
        end

        encode_YieldOp!(cb, [res])
    end

    CGVal(results[1], output_tile_type, Tile{elem_type, Tuple(output_shape)}, output_shape)
end

#-----------------------------------------------------------------------------
# cuda_tile.select
#-----------------------------------------------------------------------------

function emit_intrinsic!(ctx::CodegenContext, ::typeof(Intrinsics.select), args, @nospecialize(result_type))
    cb = ctx.cb

    cond_tv = emit_value!(ctx, args[1])
    x_tv = emit_value!(ctx, args[2])
    y_tv = emit_value!(ctx, args[3])

    (cond_tv === nothing || x_tv === nothing || y_tv === nothing) &&
        error("Cannot resolve operands for select()")

    result = encode_SelectOp!(cb, x_tv.type_id, cond_tv.v, x_tv.v, y_tv.v)

    CGVal(result, x_tv.type_id, x_tv.jltype, x_tv.shape)
end

#-----------------------------------------------------------------------------
# cuda_tile.cmpf, cuda_tile.cmpi (tile comparisons)
#-----------------------------------------------------------------------------

function emit_tile_cmp!(ctx::CodegenContext, args, predicate::ComparisonPredicate)
    cb = ctx.cb
    tt = ctx.tt

    lhs = emit_value!(ctx, args[1])
    rhs = emit_value!(ctx, args[2])

    (lhs === nothing || rhs === nothing) && error("Cannot resolve operands for tile comparison")

    # Result type is boolean tile with same shape
    tile_shape = lhs.shape
    bool_tile_type = tile_type!(tt, I1(tt), tile_shape)

    # Determine element type to choose CmpFOp vs CmpIOp
    elem_type = unwrap_type(lhs.jltype)
    if elem_type <: Tile
        elem_type = elem_type.parameters[1]
    end

    result_v = if elem_type <: AbstractFloat
        encode_CmpFOp!(cb, bool_tile_type, lhs.v, rhs.v;
                       predicate=predicate, ordering=CmpOrdered)
    else
        encode_CmpIOp!(cb, bool_tile_type, lhs.v, rhs.v;
                       predicate=predicate, signedness=SignednessSigned)
    end

    CGVal(result_v, bool_tile_type, Tile{Bool, Tuple(tile_shape)}, tile_shape)
end

# Unified comparison intrinsic - dispatches on comparator function type
function emit_intrinsic!(ctx::CodegenContext, ::typeof(Intrinsics.cmp), args, @nospecialize(_))
    # Get the comparator from the third argument (a ghost value with the function)
    cmp_tv = emit_value!(ctx, args[3])
    cmp_func = cmp_tv.constant

    # Map comparator to predicate
    predicate = if cmp_func === (<)
        CmpLessThan
    elseif cmp_func === (>)
        CmpGreaterThan
    elseif cmp_func === (<=)
        CmpLessThanOrEqual
    elseif cmp_func === (>=)
        CmpGreaterThanOrEqual
    elseif cmp_func === (==)
        CmpEqual
    elseif cmp_func === (!=)
        CmpNotEqual
    else
        error("Unknown comparison operator: $cmp_func")
    end

    emit_tile_cmp!(ctx, args, predicate)
end

#-----------------------------------------------------------------------------
# Tile(scalar) constructor - creates a 0D tile from a scalar value
#-----------------------------------------------------------------------------

function emit_intrinsic!(ctx::CodegenContext, ::Type{<:Tile}, args, @nospecialize(result_type))
    # Emit the scalar value
    source = emit_value!(ctx, args[1])

    # Get element type from result type, constant, or source jltype
    result_type_unwrapped = unwrap_type(result_type)
    elem_type = if result_type_unwrapped <: Tile
        result_type_unwrapped.parameters[1]
    elseif source.constant !== nothing
        typeof(source.constant)
    else
        unwrap_type(source.jltype)
    end

    # Return as 0D tile type
    result_jltype = Tile{elem_type, ()}
    CGVal(source.v, source.type_id, result_jltype, source.shape)
end
