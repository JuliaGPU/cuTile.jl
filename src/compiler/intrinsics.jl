#=============================================================================
 Intrinsic Dispatch
=============================================================================#

# Generic fallback
emit_intrinsic!(ctx::CodegenContext, @nospecialize(func), args, @nospecialize(result_type)) = missing

# Skip tuple construction
emit_intrinsic!(ctx::CodegenContext, ::typeof(Core.tuple), args, @nospecialize(result_type)) = nothing

# Skip isa type assertions (inserted by Julia during inlining)
emit_intrinsic!(ctx::CodegenContext, ::typeof(isa), args, @nospecialize(result_type)) = nothing

#-----------------------------------------------------------------------------
# cuTile intrinsics
#-----------------------------------------------------------------------------

function emit_intrinsic!(ctx::CodegenContext, ::typeof(_bid), args, @nospecialize(result_type))
    axis = @something get_constant(ctx, args[1]) error("bid() axis must be a compile-time constant")
    axis in (0, 1, 2) || error("bid() axis must be 0, 1, or 2, got $axis")

    res_type = tile_type!(ctx.tt, I32(ctx.tt), Int[])
    bid_x, bid_y, bid_z = encode_GetTileBlockIdOp!(ctx.cb, res_type, res_type, res_type)
    result = (bid_x, bid_y, bid_z)[axis + 1]

    CGVal(result, res_type, Int32)
end

function emit_intrinsic!(ctx::CodegenContext, ::typeof(_num_blocks), args, @nospecialize(result_type))
    axis = @something get_constant(ctx, args[1]) error("num_blocks() axis must be a compile-time constant")
    axis in (0, 1, 2) || error("num_blocks() axis must be 0, 1, or 2, got $axis")

    res_type = tile_type!(ctx.tt, I32(ctx.tt), Int[])
    nb_x, nb_y, nb_z = encode_GetNumTileBlocksOp!(ctx.cb, res_type, res_type, res_type)

    CGVal((nb_x, nb_y, nb_z)[axis + 1], res_type, Int32)
end

function emit_intrinsic!(ctx::CodegenContext, ::typeof(_load), args, @nospecialize(result_type))
    emit_load!(ctx, args, result_type)
end

function emit_intrinsic!(ctx::CodegenContext, ::typeof(_store), args, @nospecialize(result_type))
    emit_store!(ctx, args, result_type)
    nothing
end

function emit_intrinsic!(ctx::CodegenContext, ::typeof(_atomic_cas), args, @nospecialize(result_type))
    emit_atomic_cas!(ctx, args, result_type)
end

function emit_intrinsic!(ctx::CodegenContext, ::typeof(_atomic_xchg), args, @nospecialize(result_type))
    emit_atomic_rmw!(ctx, args, result_type, AtomicXCHG)
end

function emit_intrinsic!(ctx::CodegenContext, ::typeof(_atomic_add), args, @nospecialize(result_type))
    emit_atomic_rmw!(ctx, args, result_type, AtomicADD)
end

function emit_intrinsic!(ctx::CodegenContext, ::typeof(transpose), args, @nospecialize(result_type))
    emit_transpose!(ctx, args, result_type)
end

function emit_intrinsic!(ctx::CodegenContext, ::typeof(reshape), args, @nospecialize(result_type))
    emit_reshape!(ctx, args, result_type)
end

function emit_intrinsic!(ctx::CodegenContext, ::typeof(_permute), args, @nospecialize(result_type))
    emit_permute!(ctx, args, result_type)
end

function emit_intrinsic!(ctx::CodegenContext, ::typeof(_extract), args, @nospecialize(result_type))
    emit_extract!(ctx, args, result_type)
end

function emit_intrinsic!(ctx::CodegenContext, ::typeof(_cat), args, @nospecialize(result_type))
    emit_cat!(ctx, args, result_type)
end

function emit_intrinsic!(ctx::CodegenContext, ::typeof(mma), args, @nospecialize(result_type))
    emit_mma!(ctx, args, result_type)
end

function emit_intrinsic!(ctx::CodegenContext, ::typeof(full), args, @nospecialize(result_type))
    emit_full!(ctx, args, result_type)
end

function emit_intrinsic!(ctx::CodegenContext, ::typeof(_num_tiles), args, @nospecialize(result_type))
    emit_num_tiles!(ctx, args, result_type)
end

function emit_intrinsic!(ctx::CodegenContext, ::typeof(cdiv), args, @nospecialize(result_type))
    emit_cdiv!(ctx, args, result_type)
end

function emit_intrinsic!(ctx::CodegenContext, ::typeof(floordiv), args, @nospecialize(result_type))
    emit_floordiv!(ctx, args, result_type)
end

function emit_intrinsic!(ctx::CodegenContext, ::typeof(Base.rem), args, @nospecialize(result_type))
    emit_rem!(ctx, args, result_type)
end

function emit_intrinsic!(ctx::CodegenContext, ::typeof(Base.min), args, @nospecialize(result_type))
    emit_min!(ctx, args, result_type)
end

function emit_intrinsic!(ctx::CodegenContext, ::typeof(astype), args, @nospecialize(result_type))
    emit_astype!(ctx, args, result_type)
end

function emit_intrinsic!(ctx::CodegenContext, ::typeof(broadcast_to), args, @nospecialize(result_type))
    emit_broadcast_to!(ctx, args, result_type)
end

# Tile(scalar) constructor - creates a 0D tile from a scalar value
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

#-----------------------------------------------------------------------------
# Tile arithmetic
#-----------------------------------------------------------------------------

"""
    compute_broadcast_shape(s1::Vector{Int}, s2::Vector{Int}) -> Vector{Int}

Compute the NumPy-style broadcast shape from two shapes.
Shapes are compared from the right, dimensions must be equal or 1.
"""
function compute_broadcast_shape(s1::Vector{Int}, s2::Vector{Int})
    max_ndim = max(length(s1), length(s2))
    result = Vector{Int}(undef, max_ndim)
    for i in 1:max_ndim
        idx1 = length(s1) - max_ndim + i
        idx2 = length(s2) - max_ndim + i
        d1 = idx1 > 0 ? s1[idx1] : 1
        d2 = idx2 > 0 ? s2[idx2] : 1
        if d1 != d2 && d1 != 1 && d2 != 1
            error("Shapes $s1 and $s2 are not broadcastable")
        end
        result[i] = max(d1, d2)
    end
    result
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

"""
    emit_binop!(ctx, args, float_encoder, int_encoder)

Binary operation emitter.

Handles:
- Tile + Tile (same shapes - broadcasting is done at intrinsic level via broadcast_to)
- Scalar + Scalar (for integer intrinsics on index calculations)

Note: tile+scalar operations are handled at the intrinsic level via Tile(scalar) and
broadcast_to(), so by the time we reach tile_add etc., both operands are already tiles.
"""
function emit_binop!(ctx::CodegenContext, args, float_encoder::Function, int_encoder::Function)
    cb = ctx.cb
    tt = ctx.tt

    # Emit both operands
    lhs_tv = emit_value!(ctx, args[1])
    rhs_tv = emit_value!(ctx, args[2])

    # Both operands must resolve to CGVals
    if lhs_tv === nothing || rhs_tv === nothing
        return missing
    end

    # Determine what kind of operands we have
    lhs_is_tile = unwrap_type(lhs_tv.jltype) <: Tile
    rhs_is_tile = unwrap_type(rhs_tv.jltype) <: Tile

    if lhs_is_tile && rhs_is_tile
        # Tile + Tile: shapes should be identical (broadcasting via broadcast_to at intrinsic level)
        elem_type = unwrap_type(lhs_tv.jltype).parameters[1]
        result_shape = lhs_tv.shape
        lhs_v = lhs_tv.v
        rhs_v = rhs_tv.v
        result_jltype = Tile{elem_type, Tuple(result_shape)}
    elseif !lhs_is_tile && !rhs_is_tile
        # Scalar + Scalar: for integer intrinsics on index calculations
        elem_type = unwrap_type(lhs_tv.jltype)
        result_shape = Int[]
        lhs_v = lhs_tv.v
        rhs_v = rhs_tv.v
        result_jltype = elem_type
    else
        error("Mixed tile/scalar operations should be handled at intrinsic level via Tile() and broadcast_to()")
    end

    dtype = julia_to_tile_dtype!(tt, elem_type)
    result_type_id = tile_type!(tt, dtype, result_shape)

    # Emit the binary operation
    if elem_type <: AbstractFloat
        result_v = float_encoder(cb, result_type_id, lhs_v, rhs_v)
    else
        result_v = int_encoder(cb, result_type_id, lhs_v, rhs_v)
    end

    CGVal(result_v, result_type_id, result_jltype, result_shape)
end

# Same-shape tile operations - these emit the raw binary op
emit_intrinsic!(ctx::CodegenContext, ::typeof(tile_add), args, @nospecialize(_)) =
    emit_binop!(ctx, args, encode_AddFOp!, encode_AddIOp!)

emit_intrinsic!(ctx::CodegenContext, ::typeof(tile_sub), args, @nospecialize(_)) =
    emit_binop!(ctx, args, encode_SubFOp!, encode_SubIOp!)

emit_intrinsic!(ctx::CodegenContext, ::typeof(tile_mul), args, @nospecialize(_)) =
    emit_binop!(ctx, args, encode_MulFOp!, encode_MulIOp!)

emit_intrinsic!(ctx::CodegenContext, ::typeof(tile_div), args, @nospecialize(_)) =
    emit_binop!(ctx, args, encode_DivFOp!, encode_DivIOp!)

# Power operation (float only)
function emit_intrinsic!(ctx::CodegenContext, ::typeof(tile_pow), args, @nospecialize(_))
    cb = ctx.cb
    tt = ctx.tt

    lhs_tv = emit_value!(ctx, args[1])
    rhs_tv = emit_value!(ctx, args[2])

    (lhs_tv === nothing || rhs_tv === nothing) && error("Cannot resolve operands for tile_pow")

    # Power is float-only, so we expect tiles with float element types
    elem_type = unwrap_type(lhs_tv.jltype)
    if elem_type <: Tile
        elem_type = elem_type.parameters[1]
    end
    elem_type <: AbstractFloat || error("tile_pow only supports float types, got $elem_type")

    result_shape = lhs_tv.shape
    result_jltype = Tile{elem_type, Tuple(result_shape)}

    dtype = julia_to_tile_dtype!(tt, elem_type)
    result_type_id = tile_type!(tt, dtype, result_shape)

    result_v = encode_PowOp!(cb, result_type_id, lhs_tv.v, rhs_tv.v)

    CGVal(result_v, result_type_id, result_jltype, result_shape)
end

# Julia integer intrinsics (all are Core.IntrinsicFunction, so dispatch by value)
function emit_intrinsic!(ctx::CodegenContext, func::Core.IntrinsicFunction, args, @nospecialize(_))
    if func === Base.add_int
        return emit_binop!(ctx, args, encode_AddFOp!, encode_AddIOp!)
    elseif func === Base.sub_int
        return emit_binop!(ctx, args, encode_SubFOp!, encode_SubIOp!)
    elseif func === Base.mul_int
        return emit_binop!(ctx, args, encode_MulFOp!, encode_MulIOp!)
    elseif func === Base.sitofp
        return emit_sitofp!(ctx, args)
    elseif func === Base.uitofp
        return emit_uitofp!(ctx, args)
    # Integer comparison intrinsics (signed and unsigned use same predicate, signedness is separate)
    elseif func === Base.slt_int
        return emit_int_cmp!(ctx, args, CmpLessThan, SignednessSigned)
    elseif func === Base.sle_int
        return emit_int_cmp!(ctx, args, CmpLessThanOrEqual, SignednessSigned)
    elseif func === Base.ult_int
        return emit_int_cmp!(ctx, args, CmpLessThan, SignednessUnsigned)
    elseif func === Base.ule_int
        return emit_int_cmp!(ctx, args, CmpLessThanOrEqual, SignednessUnsigned)
    elseif func === Base.eq_int
        return emit_int_cmp!(ctx, args, CmpEqual, SignednessSigned)
    elseif func === Base.ne_int
        return emit_int_cmp!(ctx, args, CmpNotEqual, SignednessSigned)
    elseif func === Base.not_int
        return emit_not_int!(ctx, args)
    end
    missing  # Unknown intrinsic
end

# Boolean negation (not_int is used for iteration bounds checking in for-loops)
function emit_not_int!(ctx::CodegenContext, args)
    cb = ctx.cb
    tt = ctx.tt

    source = emit_value!(ctx, args[1])
    source === nothing && error("Cannot resolve source operand for not_int")

    # not_int is applied to Bool (i1) values
    # XOR with true (1) to invert: x XOR 1 = NOT x
    source_v = source isa CGVal ? source.v : source
    source_shape = source isa CGVal ? source.shape : Int[]

    bool_dtype = I1(tt)
    bool_type = tile_type!(tt, bool_dtype, source_shape)

    # Create constant true (0xff for i1)
    true_bytes = UInt8[0xff]
    true_scalar = encode_ConstantOp!(cb, tile_type!(tt, bool_dtype, Int[]), true_bytes)

    # Broadcast if needed
    if !isempty(source_shape)
        ones_shape = fill(1, length(source_shape))
        reshaped_type = tile_type!(tt, bool_dtype, ones_shape)
        true_reshaped = encode_ReshapeOp!(cb, reshaped_type, true_scalar)
        true_tile = encode_BroadcastOp!(cb, bool_type, true_reshaped)
    else
        true_tile = true_scalar
    end

    # XOR to invert
    result_v = encode_XOrIOp!(cb, bool_type, source_v, true_tile)
    CGVal(result_v, bool_type, Bool, source_shape)
end

# Signed integer to floating point conversion
function emit_sitofp!(ctx::CodegenContext, args)
    cb = ctx.cb
    tt = ctx.tt

    # args[1] is the target type (e.g., Float32), args[2] is the value
    target_type = args[1]
    source = emit_value!(ctx, args[2])
    source === nothing && error("Cannot resolve source operand for sitofp")

    # Get the target float type
    dtype = julia_to_tile_dtype!(tt, target_type)
    result_shape = source isa CGVal ? source.shape : Int[]
    result_type = tile_type!(tt, dtype, result_shape)

    result_v = encode_IToFOp!(cb, result_type, source isa CGVal ? source.v : source;
                              signedness=SignednessSigned)
    CGVal(result_v, result_type, target_type, result_shape)
end

# Unsigned integer to floating point conversion
function emit_uitofp!(ctx::CodegenContext, args)
    cb = ctx.cb
    tt = ctx.tt

    # args[1] is the target type (e.g., Float32), args[2] is the value
    target_type = args[1]
    source = emit_value!(ctx, args[2])
    source === nothing && error("Cannot resolve source operand for uitofp")

    # Get the target float type
    dtype = julia_to_tile_dtype!(tt, target_type)
    result_shape = source isa CGVal ? source.shape : Int[]
    result_type = tile_type!(tt, dtype, result_shape)

    result_v = encode_IToFOp!(cb, result_type, source isa CGVal ? source.v : source;
                              signedness=SignednessUnsigned)
    CGVal(result_v, result_type, target_type, result_shape)
end

#-----------------------------------------------------------------------------
# Comparison operators
#-----------------------------------------------------------------------------

function emit_cmp!(ctx::CodegenContext, args, predicate::ComparisonPredicate)
    emit_int_cmp!(ctx, args, predicate, SignednessSigned)
end

function emit_int_cmp!(ctx::CodegenContext, args, predicate::ComparisonPredicate, signedness::Signedness)
    cb = ctx.cb
    tt = ctx.tt

    lhs = emit_value!(ctx, args[1])
    rhs = emit_value!(ctx, args[2])

    lhs === nothing && error("Cannot resolve LHS operand for comparison")
    rhs === nothing && error("Cannot resolve RHS operand for comparison")

    # Result type is a boolean (i1) scalar
    result_type = tile_type!(tt, I1(tt), Int[])

    lhs_v = lhs isa CGVal ? lhs.v : lhs
    rhs_v = rhs isa CGVal ? rhs.v : rhs

    result_v = encode_CmpIOp!(cb, result_type, lhs_v, rhs_v;
                              predicate=predicate, signedness=signedness)

    CGVal(result_v, result_type, Bool, Int[])
end

emit_intrinsic!(ctx::CodegenContext, ::typeof(Base.:(>)), args, @nospecialize(_)) =
    emit_cmp!(ctx, args, CmpGreaterThan)

emit_intrinsic!(ctx::CodegenContext, ::typeof(Base.:(<)), args, @nospecialize(_)) =
    emit_cmp!(ctx, args, CmpLessThan)

emit_intrinsic!(ctx::CodegenContext, ::typeof(Base.:(>=)), args, @nospecialize(_)) =
    emit_cmp!(ctx, args, CmpGreaterThanOrEqual)

emit_intrinsic!(ctx::CodegenContext, ::typeof(Base.:(<=)), args, @nospecialize(_)) =
    emit_cmp!(ctx, args, CmpLessThanOrEqual)

emit_intrinsic!(ctx::CodegenContext, ::typeof(===), args, @nospecialize(_)) =
    emit_cmp!(ctx, args, CmpEqual)

emit_intrinsic!(ctx::CodegenContext, ::typeof(Base.:(==)), args, @nospecialize(_)) =
    emit_cmp!(ctx, args, CmpEqual)

emit_intrinsic!(ctx::CodegenContext, ::typeof(Base.:(!=)), args, @nospecialize(_)) =
    emit_cmp!(ctx, args, CmpNotEqual)

#-----------------------------------------------------------------------------
# getfield for destructured arguments (lazy chain extension)
#-----------------------------------------------------------------------------

function emit_intrinsic!(ctx::CodegenContext, ::typeof(Base.getfield), args, @nospecialize(result_type))
    length(args) >= 2 || return nothing

    obj_arg = args[1]
    field_arg = args[2]

    # Extract field name or index
    field = get_constant(ctx, field_arg)

    # Try to get the object as a CGVal
    obj_tv = emit_value!(ctx, obj_arg)

    # If obj is a lazy arg_ref, extend the chain
    if obj_tv !== nothing && is_arg_ref(obj_tv)
        arg_idx, chain = obj_tv.arg_ref

        if field isa Symbol
            # Field access: extend chain with symbol
            new_chain = Union{Symbol, Int}[chain..., field]
            # Check if this resolves to a scalar field (auto-materialize leaf)
            values = get_arg_flat_values(ctx, arg_idx, field)
            if values !== nothing && length(values) == 1
                # Scalar field - materialize immediately
                type_id = tile_type_for_julia!(ctx, unwrap_type(result_type))
                return CGVal(values[1], type_id, unwrap_type(result_type))
            end
            return arg_ref_value(arg_idx, new_chain, unwrap_type(result_type))
        elseif field isa Integer && !isempty(chain) && chain[end] isa Symbol
            # Tuple indexing: chain ends with field name, now indexing into it
            # This is a leaf - materialize immediately
            field_name = chain[end]
            values = get_arg_flat_values(ctx, arg_idx, field_name)
            if values !== nothing && 1 <= field <= length(values)
                type_id = tile_type_for_julia!(ctx, unwrap_type(result_type))
                return CGVal(values[field], type_id, unwrap_type(result_type))
            end
        end
    end

    nothing
end

function extract_argument_index(@nospecialize(arg))
    if arg isa SlotNumber
        return arg.id - 1
    elseif arg isa Argument
        return arg.n - 1
    end
    nothing
end

#-----------------------------------------------------------------------------
# getindex for tuple field access (lazy chain extension)
#-----------------------------------------------------------------------------

function emit_intrinsic!(ctx::CodegenContext, ::typeof(Base.getindex), args, @nospecialize(result_type))
    length(args) >= 2 || return nothing

    obj_arg = args[1]
    index_arg = args[2]

    # Extract constant index
    index = get_constant(ctx, index_arg)
    index isa Integer || return nothing

    # Try to get the object as a CGVal
    obj_tv = emit_value!(ctx, obj_arg)
    obj_tv === nothing && return nothing

    # If obj is a lazy arg_ref, extend the chain with the index
    if is_arg_ref(obj_tv)
        arg_idx, chain = obj_tv.arg_ref
        new_chain = Union{Symbol, Int}[chain..., Int(index)]
        return arg_ref_value(arg_idx, new_chain, unwrap_type(result_type))
    end

    # Not an arg_ref - not handled here
    nothing
end

#=============================================================================
 Load/Store Operations
=============================================================================#

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

function emit_load!(ctx::CodegenContext, args::AbstractVector, @nospecialize(result_type))
    cb = ctx.cb
    tt = ctx.tt

    array_arg = args[1]

    # Check if TileArray argument
    arg_idx = extract_argument_index(array_arg)
    is_tilearray = arg_idx !== nothing && is_destructured_arg(ctx, arg_idx)

    # Get pointer and element type
    array_spec = nothing
    if is_tilearray
        ptr_vals = get_arg_flat_values(ctx, arg_idx, :ptr)
        isempty(ptr_vals) && error("Cannot get ptr from TileArray argument")
        array_val = ptr_vals[1]
        tilearray_type = get_arg_type(ctx, arg_idx)
        elem_type = eltype(tilearray_type)
        array_spec = get_array_spec(tilearray_type)
    else
        tv = emit_value!(ctx, array_arg)
        tv === nothing && error("Cannot resolve array argument for load()")
        array_val = tv.v
        elem_type = extract_pointer_elem_type(tv.jltype)
    end

    # New argument order: arr, Val(shape), padding_mode, indices...
    # Parse shape from Val{shape} argument (args[2])
    tile_shape = Int[16]  # Default
    if length(args) >= 2
        shape = get_constant(ctx, args[2])
        if shape isa Tuple
            tile_shape = collect(Int, shape)
        end
    end

    ndim = length(tile_shape)

    # Parse padding_mode from args[3] (default: PaddingMode.Undetermined = 0)
    padding_mode_int = 0  # Default: Undetermined
    if length(args) >= 3
        pm = get_constant(ctx, args[3])
        if pm isa Integer
            padding_mode_int = Int(pm)
        end
    end
    padding_value = padding_mode_to_padding_value(padding_mode_int)

    # Extract indices directly from args[4:end] (no tuple decomposition needed!)
    index_vals = Value[]
    for i in 4:length(args)
        tv = emit_value!(ctx, args[i])
        tv !== nothing && tv.v !== nothing && push!(index_vals, tv.v)
    end

    dtype = julia_to_tile_dtype!(tt, elem_type)

    # Create types
    tile_type = tile_type!(tt, dtype, tile_shape)
    token_type = Token(tt)

    # TensorView type - use static strides where known from ArraySpec
    tv_shape = fill(DYNAMIC_SHAPE, ndim)
    tv_strides = compute_tensor_view_strides(array_spec, ndim)
    tv_type = tensor_view_type!(tt, dtype, tv_shape, tv_strides)

    # PartitionView type
    dim_map = collect(0:ndim-1)
    pv_type = partition_view_type!(tt, tile_shape, tv_type, dim_map, padding_value)

    scalar_i32 = tile_type!(tt, I32(tt), Int[])

    # Get size and stride values
    size_vals, stride_vals = get_size_stride_vals(ctx, arg_idx, is_tilearray, ndim, tile_shape, index_vals, scalar_i32)

    # Emit AssumeOps for optimization hints (skip stride assumes for static strides)
    if array_spec !== nothing
        array_val, size_vals, stride_vals = emit_assume_ops!(ctx, array_val, size_vals, stride_vals, array_spec, dtype, scalar_i32; tv_strides)
    end

    # Filter strides to only pass dynamic ones as operands
    dynamic_stride_vals = filter_dynamic_strides(stride_vals, tv_strides)

    # Create tensor view
    tensor_view = encode_MakeTensorViewOp!(cb, tv_type, array_val, size_vals, dynamic_stride_vals)

    # Create partition view
    partition = encode_MakePartitionViewOp!(cb, pv_type, tensor_view)

    # Pad indices if needed
    index_vals = pad_indices(ctx, index_vals, ndim, scalar_i32)

    # Load tile with token
    tile_val, new_token = encode_LoadViewTkoOp!(cb, tile_type, token_type, partition, index_vals; token=ctx.token)
    ctx.token = new_token

    CGVal(tile_val, tile_type, Tile{elem_type, Tuple(tile_shape)}, tile_shape)
end

function emit_store!(ctx::CodegenContext, args::AbstractVector, @nospecialize(result_type))
    cb = ctx.cb
    tt = ctx.tt

    array_arg = args[1]

    # Check if TileArray argument
    arg_idx = extract_argument_index(array_arg)
    is_tilearray = arg_idx !== nothing && is_destructured_arg(ctx, arg_idx)

    # Get pointer and element type
    array_spec = nothing
    if is_tilearray
        ptr_vals = get_arg_flat_values(ctx, arg_idx, :ptr)
        isempty(ptr_vals) && error("Cannot get ptr from TileArray argument")
        array_val = ptr_vals[1]
        tilearray_type = get_arg_type(ctx, arg_idx)
        elem_type = eltype(tilearray_type)
        array_spec = get_array_spec(tilearray_type)
    else
        tv = emit_value!(ctx, array_arg)
        tv === nothing && error("Cannot resolve array argument for store()")
        array_val = tv.v
        elem_type = extract_pointer_elem_type(tv.jltype)
    end

    # New argument order: arr, tile, indices...
    # Get tile value and shape (args[2])
    tile_tv = emit_value!(ctx, args[2])
    tile_tv === nothing && error("store() requires a tile argument")
    tile_shape = tile_tv.shape
    tile_shape === nothing && error("Cannot determine tile shape for store()")

    dtype = julia_to_tile_dtype!(tt, elem_type)
    ndim = length(tile_shape)

    # Handle 0D scalar stores by reshaping to 1D (partition views require at least 1D)
    tile_val = tile_tv.v
    if ndim == 0
        ndim = 1
        tile_shape = Int[1]
        tile_1d_type = tile_type!(tt, dtype, tile_shape)
        tile_val = encode_ReshapeOp!(cb, tile_1d_type, tile_val)
    end

    # Extract indices directly from args[3:end] (no tuple decomposition needed!)
    index_vals = Value[]
    for i in 3:length(args)
        tv = emit_value!(ctx, args[i])
        tv !== nothing && tv.v !== nothing && push!(index_vals, tv.v)
    end

    # TensorView type - use static strides where known from ArraySpec
    tv_shape = fill(DYNAMIC_SHAPE, ndim)
    tv_strides = compute_tensor_view_strides(array_spec, ndim)
    tv_type = tensor_view_type!(tt, dtype, tv_shape, tv_strides)

    # PartitionView type
    dim_map = collect(0:ndim-1)
    pv_type = partition_view_type!(tt, tile_shape, tv_type, dim_map, PaddingZero)

    scalar_i32 = tile_type!(tt, I32(tt), Int[])

    # Get size and stride values
    size_vals, stride_vals = get_size_stride_vals(ctx, arg_idx, is_tilearray, ndim, tile_shape, index_vals, scalar_i32)

    # Emit AssumeOps (skip stride assumes for static strides)
    if array_spec !== nothing
        array_val, size_vals, stride_vals = emit_assume_ops!(ctx, array_val, size_vals, stride_vals, array_spec, dtype, scalar_i32; tv_strides)
    end

    # Filter strides to only pass dynamic ones as operands
    dynamic_stride_vals = filter_dynamic_strides(stride_vals, tv_strides)

    # Create tensor view
    tensor_view = encode_MakeTensorViewOp!(cb, tv_type, array_val, size_vals, dynamic_stride_vals)

    # Create partition view
    partition = encode_MakePartitionViewOp!(cb, pv_type, tensor_view)

    # Pad indices
    index_vals = pad_indices(ctx, index_vals, ndim, scalar_i32)

    # Store tile with token
    token_type = Token(tt)
    new_token = encode_StoreViewTkoOp!(cb, token_type, tile_val, partition, index_vals; token=ctx.token)
    ctx.token = new_token

    nothing
end

#=============================================================================
 Atomic Operations
=============================================================================#

"""
Convert integer memory order value to bytecode MemoryOrderingSemantics enum
"""
function memory_order_to_semantics(order::Int)
    if order == 0  # Weak
        MemoryWeak
    elseif order == 1  # Relaxed
        MemoryRelaxed
    elseif order == 2  # Acquire
        MemoryAcquire
    elseif order == 3  # Release
        MemoryRelease
    else  # 4 = AcqRel
        MemoryAcqRel
    end
end

"""
Convert integer memory scope value to bytecode MemoryScope enum
"""
function memory_scope_to_scope(scope::Int)
    if scope == 0  # Block
        ScopeTLBlock
    elseif scope == 1  # Device
        ScopeDevice
    else  # 2 = System
        ScopeSystem
    end
end

function emit_atomic_cas!(ctx::CodegenContext, args::AbstractVector, @nospecialize(result_type))
    cb = ctx.cb
    tt = ctx.tt

    # args: (array, index, expected, desired, memory_order, memory_scope)
    array_arg = args[1]

    # Get array info
    arg_idx = extract_argument_index(array_arg)
    is_tilearray = arg_idx !== nothing && is_destructured_arg(ctx, arg_idx)

    if !is_tilearray
        error("atomic_cas requires a TileArray argument")
    end

    ptr_vals = get_arg_flat_values(ctx, arg_idx, :ptr)
    isempty(ptr_vals) && error("Cannot get ptr from TileArray argument")
    array_val = ptr_vals[1]
    tilearray_type = get_arg_type(ctx, arg_idx)
    elem_type = eltype(tilearray_type)

    # Get expected and desired values
    expected_tv = emit_value!(ctx, args[3])
    expected_tv === nothing && error("atomic_cas requires expected value")
    desired_tv = emit_value!(ctx, args[4])
    desired_tv === nothing && error("atomic_cas requires desired value")

    # Get memory order and scope from args
    memory_order = @something get_constant(ctx, args[5]) error("atomic_cas requires constant memory_order")
    memory_scope = @something get_constant(ctx, args[6]) error("atomic_cas requires constant memory_scope")

    # Create result type (0D tile of element type)
    dtype = julia_to_tile_dtype!(tt, elem_type)
    result_tile_type = tile_type!(tt, dtype, Int[])
    token_type = Token(tt)

    # Create pointer tile - compute pointer to index
    scalar_i64_type = tile_type!(tt, I64(tt), Int[])
    index_tv = emit_value!(ctx, args[2])

    # Create pointer type for element
    ptr_type = pointer_type!(tt, dtype)
    ptr_tile_type = tile_type!(tt, ptr_type, Int[])

    # Compute pointer: base + index * sizeof(elem)
    # Convert pointer to int64, do arithmetic, convert back
    ptr_as_int = encode_PtrToIntOp!(cb, scalar_i64_type, array_val)

    elem_size = sizeof(elem_type)
    elem_size_const_tv = emit_constant!(ctx, Int64(elem_size), Int64)
    elem_size_const = elem_size_const_tv.v

    # Extend index to 64-bit if needed
    index_i64 = encode_ExtIOp!(cb, scalar_i64_type, index_tv.v; signedness=SignednessSigned)
    index_scaled = encode_MulIOp!(cb, scalar_i64_type, index_i64, elem_size_const)
    ptr_int_offset = encode_AddIOp!(cb, scalar_i64_type, ptr_as_int, index_scaled)
    pointers = encode_IntToPtrOp!(cb, ptr_tile_type, ptr_int_offset)

    # Emit atomic CAS
    mem_ordering = memory_order_to_semantics(memory_order)
    mem_scope = memory_scope_to_scope(memory_scope)

    old_val, new_token = encode_AtomicCASPtrOp!(cb, result_tile_type, token_type, pointers,
                                         expected_tv.v, desired_tv.v;
                                         token=ctx.token,
                                         memory_ordering=mem_ordering,
                                         memory_scope=mem_scope)
    ctx.token = new_token

    # Return scalar type (not Tile) to match the intrinsic signature
    CGVal(old_val, result_tile_type, elem_type, Int[])
end

function emit_atomic_rmw!(ctx::CodegenContext, args::AbstractVector, @nospecialize(result_type), mode::AtomicRMWMode)
    cb = ctx.cb
    tt = ctx.tt

    # args: (array, index, val, memory_order, memory_scope)
    array_arg = args[1]

    # Get array info
    arg_idx = extract_argument_index(array_arg)
    is_tilearray = arg_idx !== nothing && is_destructured_arg(ctx, arg_idx)

    if !is_tilearray
        error("atomic operations require a TileArray argument")
    end

    ptr_vals = get_arg_flat_values(ctx, arg_idx, :ptr)
    isempty(ptr_vals) && error("Cannot get ptr from TileArray argument")
    array_val = ptr_vals[1]
    tilearray_type = get_arg_type(ctx, arg_idx)
    elem_type = eltype(tilearray_type)

    # Get update value
    val_tv = emit_value!(ctx, args[3])
    val_tv === nothing && error("atomic operation requires value")

    # Get memory order and scope from args
    memory_order = @something get_constant(ctx, args[4]) error("atomic operation requires constant memory_order")
    memory_scope = @something get_constant(ctx, args[5]) error("atomic operation requires constant memory_scope")

    # Create result type (0D tile of element type)
    dtype = julia_to_tile_dtype!(tt, elem_type)
    result_tile_type = tile_type!(tt, dtype, Int[])
    token_type = Token(tt)

    # Create pointer tile
    scalar_i64_type = tile_type!(tt, I64(tt), Int[])
    index_tv = emit_value!(ctx, args[2])

    ptr_type = pointer_type!(tt, dtype)
    ptr_tile_type = tile_type!(tt, ptr_type, Int[])

    # Compute pointer: base + index * sizeof(elem)
    # Convert pointer to int64, do arithmetic, convert back
    ptr_as_int = encode_PtrToIntOp!(cb, scalar_i64_type, array_val)

    elem_size = sizeof(elem_type)
    elem_size_const_tv = emit_constant!(ctx, Int64(elem_size), Int64)
    elem_size_const = elem_size_const_tv.v

    # Extend index to 64-bit if needed
    index_i64 = encode_ExtIOp!(cb, scalar_i64_type, index_tv.v; signedness=SignednessSigned)
    index_scaled = encode_MulIOp!(cb, scalar_i64_type, index_i64, elem_size_const)
    ptr_int_offset = encode_AddIOp!(cb, scalar_i64_type, ptr_as_int, index_scaled)
    pointers = encode_IntToPtrOp!(cb, ptr_tile_type, ptr_int_offset)

    # Use float add mode for floating point types
    actual_mode = mode
    if mode == AtomicADD && elem_type <: AbstractFloat
        actual_mode = AtomicADDF
    end

    # Emit atomic RMW
    mem_ordering = memory_order_to_semantics(memory_order)
    mem_scope = memory_scope_to_scope(memory_scope)

    old_val, new_token = encode_AtomicRMWPtrOp!(cb, result_tile_type, token_type, pointers,
                                         val_tv.v, actual_mode;
                                         token=ctx.token,
                                         memory_ordering=mem_ordering,
                                         memory_scope=mem_scope)
    ctx.token = new_token

    # Return scalar type (not Tile) to match the intrinsic signature
    CGVal(old_val, result_tile_type, elem_type, Int[])
end

#=============================================================================
 Load/Store Helpers
=============================================================================#

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

#=============================================================================
 Transpose Operation
=============================================================================#

function emit_transpose!(ctx::CodegenContext, args::AbstractVector, @nospecialize(result_type))
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

#=============================================================================
 Reshape Operation
=============================================================================#

function emit_reshape!(ctx::CodegenContext, args::AbstractVector, @nospecialize(result_type))
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

#=============================================================================
 Permute Operation (N-D generalization of transpose)
=============================================================================#

function emit_permute!(ctx::CodegenContext, args::AbstractVector, @nospecialize(result_type))
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

#=============================================================================
 Extract Operation (slice extraction)
=============================================================================#

function emit_extract!(ctx::CodegenContext, args::AbstractVector, @nospecialize(result_type))
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

#=============================================================================
 Concatenation
=============================================================================#

function emit_cat!(ctx::CodegenContext, args::AbstractVector, @nospecialize(result_type))
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

#=============================================================================
 Matrix Multiply-Accumulate
=============================================================================#

function emit_mma!(ctx::CodegenContext, args::AbstractVector, @nospecialize(result_type))
    cb = ctx.cb

    lhs = emit_value!(ctx, args[1])
    rhs = emit_value!(ctx, args[2])
    acc = emit_value!(ctx, args[3])

    (lhs === nothing || rhs === nothing || acc === nothing) && error("Cannot resolve operands for mma()")

    result = encode_MmaFOp!(cb, acc.type_id, lhs.v, rhs.v, acc.v)

    CGVal(result, acc.type_id, acc.jltype, acc.shape)
end

#=============================================================================
 Type Conversion
=============================================================================#

function emit_astype!(ctx::CodegenContext, args::AbstractVector, @nospecialize(result_type))
    cb = ctx.cb
    tt = ctx.tt

    # Get source tile
    source = emit_value!(ctx, args[1])
    source === nothing && error("Cannot resolve source operand for astype()")

    # Get source element type and shape
    source_type = unwrap_type(source.jltype)
    source_elem = source_type <: Tile ? source_type.parameters[1] : source_type
    tile_shape = source.shape

    # Get target element type from the Type argument
    target_elem = @something get_constant(ctx, args[2]) error("astype() requires a compile-time constant type")
    target_elem isa Type || error("astype() second argument must be a Type")

    # Same type? Return source unchanged
    if source_elem === target_elem
        return source
    end

    # Create target type
    target_dtype = julia_to_tile_dtype!(tt, target_elem)
    target_tile_type = tile_type!(tt, target_dtype, tile_shape)

    # Determine signedness for integer types
    function is_signed_int(T)
        T <: Signed || T === Int32 || T === Int64 || T === Int16 || T === Int8
    end

    # Emit conversion based on source and target types
    result = if source_elem <: AbstractFloat && target_elem <: AbstractFloat
        # Float -> Float
        encode_FToFOp!(cb, target_tile_type, source.v)
    elseif source_elem <: Integer && target_elem <: AbstractFloat
        # Integer -> Float
        signedness = is_signed_int(source_elem) ? SignednessSigned : SignednessUnsigned
        encode_IToFOp!(cb, target_tile_type, source.v; signedness)
    elseif source_elem <: AbstractFloat && target_elem <: Integer
        # Float -> Integer
        signedness = is_signed_int(target_elem) ? SignednessSigned : SignednessUnsigned
        encode_FToIOp!(cb, target_tile_type, source.v; signedness)
    elseif source_elem <: Integer && target_elem <: Integer
        # Integer -> Integer
        source_size = sizeof(source_elem)
        target_size = sizeof(target_elem)
        if source_size == target_size
            # Same size - no conversion needed (just reinterpret)
            source.v
        elseif target_size > source_size
            # Extension (upsize)
            signedness = is_signed_int(source_elem) ? SignednessSigned : SignednessUnsigned
            encode_ExtIOp!(cb, target_tile_type, source.v; signedness)
        else
            # Truncation (downsize)
            encode_TruncIOp!(cb, target_tile_type, source.v)
        end
    else
        error("astype() unsupported conversion: $source_elem -> $target_elem")
    end

    CGVal(result, target_tile_type, Tile{target_elem, Tuple(tile_shape)}, tile_shape)
end

#=============================================================================
 Explicit Broadcasting
=============================================================================#

function emit_broadcast_to!(ctx::CodegenContext, args::AbstractVector, @nospecialize(result_type))
    cb = ctx.cb
    tt = ctx.tt

    # Get source tile
    source = emit_value!(ctx, args[1])
    source === nothing && error("Cannot resolve source operand for broadcast_to()")

    # Get source element type
    source_type = unwrap_type(source.jltype)
    source_elem = source_type <: Tile ? source_type.parameters[1] : source_type

    # Extract target shape from the constant tuple argument
    target_shape_tuple = get_constant(ctx, args[2])
    target_shape_tuple isa Tuple || error("broadcast_to() shape must be a compile-time constant tuple")
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

#=============================================================================
 Tile Construction
=============================================================================#

function emit_full!(ctx::CodegenContext, args::AbstractVector, @nospecialize(result_type))
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

#=============================================================================
 Integer Arithmetic
=============================================================================#

function emit_num_tiles!(ctx::CodegenContext, args::AbstractVector, @nospecialize(result_type))
    cb = ctx.cb
    tt = ctx.tt

    array_arg = args[1]
    axis = @something get_constant(ctx, args[2]) error("num_tiles() axis must be a compile-time constant")

    shape = get_constant(ctx, args[3])
    shape isa Tuple || error("num_tiles() shape must be a compile-time constant tuple")

    tile_size = shape[axis + 1]

    arg_idx = extract_argument_index(array_arg)
    (arg_idx === nothing || !is_destructured_arg(ctx, arg_idx)) && error("num_tiles() requires a TileArray argument")

    sizes_vals = get_arg_flat_values(ctx, arg_idx, :sizes)
    (sizes_vals === nothing || length(sizes_vals) <= axis) && error("Cannot get size for axis $axis")

    array_size = sizes_vals[axis + 1]
    scalar_i32 = tile_type!(tt, I32(tt), Int[])

    # cdiv: (array_size + tile_size - 1) / tile_size
    tile_size_val = encode_ConstantOp!(cb, scalar_i32, collect(reinterpret(UInt8, [Int32(tile_size)])))
    one_val = encode_ConstantOp!(cb, scalar_i32, collect(reinterpret(UInt8, [Int32(1)])))

    sum1 = encode_AddIOp!(cb, scalar_i32, array_size, tile_size_val)
    sum2 = encode_SubIOp!(cb, scalar_i32, sum1, one_val)
    result = encode_DivIOp!(cb, scalar_i32, sum2, tile_size_val; signedness=SignednessSigned, rounding=RoundingZero)

    CGVal(result, scalar_i32, Int32)
end

function emit_cdiv!(ctx::CodegenContext, args::AbstractVector, @nospecialize(result_type))
    cb = ctx.cb
    tt = ctx.tt

    scalar_i32 = tile_type!(tt, I32(tt), Int[])

    a = resolve_or_constant(ctx, args[1], scalar_i32)
    b = resolve_or_constant(ctx, args[2], scalar_i32)

    one_val = encode_ConstantOp!(cb, scalar_i32, collect(reinterpret(UInt8, [Int32(1)])))

    sum1 = encode_AddIOp!(cb, scalar_i32, a, b)
    sum2 = encode_SubIOp!(cb, scalar_i32, sum1, one_val)
    result = encode_DivIOp!(cb, scalar_i32, sum2, b; signedness=SignednessSigned, rounding=RoundingZero)

    CGVal(result, scalar_i32, Int32)
end

function emit_floordiv!(ctx::CodegenContext, args::AbstractVector, @nospecialize(result_type))
    cb = ctx.cb
    tt = ctx.tt

    scalar_i32 = tile_type!(tt, I32(tt), Int[])

    a = resolve_or_constant(ctx, args[1], scalar_i32)
    b = resolve_or_constant(ctx, args[2], scalar_i32)

    result = encode_DivIOp!(cb, scalar_i32, a, b; signedness=SignednessSigned, rounding=RoundingZero)

    CGVal(result, scalar_i32, Int32)
end

function emit_rem!(ctx::CodegenContext, args::AbstractVector, @nospecialize(result_type))
    cb = ctx.cb
    tt = ctx.tt

    scalar_i32 = tile_type!(tt, I32(tt), Int[])

    a = resolve_or_constant(ctx, args[1], scalar_i32)
    b = resolve_or_constant(ctx, args[2], scalar_i32)

    result = encode_RemIOp!(cb, scalar_i32, a, b; signedness=SignednessSigned)

    CGVal(result, scalar_i32, Int32)
end

function emit_min!(ctx::CodegenContext, args::AbstractVector, @nospecialize(result_type))
    cb = ctx.cb
    tt = ctx.tt

    scalar_i32 = tile_type!(tt, I32(tt), Int[])

    a = resolve_or_constant(ctx, args[1], scalar_i32)
    b = resolve_or_constant(ctx, args[2], scalar_i32)

    result = encode_MinIOp!(cb, scalar_i32, a, b; signedness=SignednessSigned)

    CGVal(result, scalar_i32, Int32)
end

function resolve_or_constant(ctx::CodegenContext, @nospecialize(arg), type_id::TypeId)
    tv = emit_value!(ctx, arg)
    # If we have a runtime value, use it
    tv.v !== nothing && return tv.v
    # Otherwise emit a constant from the compile-time value
    val = @something tv.constant error("Cannot resolve argument")
    bytes = reinterpret(UInt8, [Int32(val)])
    encode_ConstantOp!(ctx.cb, type_id, collect(bytes))
end

#=============================================================================
 Math Operations
=============================================================================#

function emit_intrinsic!(ctx::CodegenContext, ::typeof(Base.sqrt), args, @nospecialize(result_type))
    cb = ctx.cb

    source = emit_value!(ctx, args[1])
    source === nothing && error("Cannot resolve operand for sqrt()")

    result = encode_SqrtOp!(cb, source.type_id, source.v)

    CGVal(result, source.type_id, source.jltype, source.shape)
end

#=============================================================================
 Tile Factory Operations
=============================================================================#

function emit_intrinsic!(ctx::CodegenContext, ::typeof(_arange), args, @nospecialize(result_type))
    cb = ctx.cb
    tt = ctx.tt

    # Extract shape
    shape = get_constant(ctx, args[1])
    shape isa Tuple || error("arange() shape must be a compile-time constant tuple")
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

#=============================================================================
 Reduction Operations
=============================================================================#

function emit_intrinsic!(ctx::CodegenContext, ::typeof(_reduce_sum), args, @nospecialize(result_type))
    emit_reduce!(ctx, args, result_type, :add)
end

function emit_intrinsic!(ctx::CodegenContext, ::typeof(_reduce_max), args, @nospecialize(result_type))
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

#=============================================================================
 Conditional Selection
=============================================================================#

function emit_intrinsic!(ctx::CodegenContext, ::typeof(where), args, @nospecialize(result_type))
    cb = ctx.cb

    cond_tv = emit_value!(ctx, args[1])
    x_tv = emit_value!(ctx, args[2])
    y_tv = emit_value!(ctx, args[3])

    (cond_tv === nothing || x_tv === nothing || y_tv === nothing) &&
        error("Cannot resolve operands for where()")

    result = encode_SelectOp!(cb, x_tv.type_id, cond_tv.v, x_tv.v, y_tv.v)

    CGVal(result, x_tv.type_id, x_tv.jltype, x_tv.shape)
end

#=============================================================================
 Tile Comparison Operations
=============================================================================#

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

emit_intrinsic!(ctx::CodegenContext, ::typeof(tile_lt), args, @nospecialize(_)) =
    emit_tile_cmp!(ctx, args, CmpLessThan)

emit_intrinsic!(ctx::CodegenContext, ::typeof(tile_gt), args, @nospecialize(_)) =
    emit_tile_cmp!(ctx, args, CmpGreaterThan)

emit_intrinsic!(ctx::CodegenContext, ::typeof(tile_le), args, @nospecialize(_)) =
    emit_tile_cmp!(ctx, args, CmpLessThanOrEqual)

emit_intrinsic!(ctx::CodegenContext, ::typeof(tile_ge), args, @nospecialize(_)) =
    emit_tile_cmp!(ctx, args, CmpGreaterThanOrEqual)

emit_intrinsic!(ctx::CodegenContext, ::typeof(tile_eq), args, @nospecialize(_)) =
    emit_tile_cmp!(ctx, args, CmpEqual)

emit_intrinsic!(ctx::CodegenContext, ::typeof(tile_ne), args, @nospecialize(_)) =
    emit_tile_cmp!(ctx, args, CmpNotEqual)

#=============================================================================
 Constants
=============================================================================#

function emit_constant!(ctx::CodegenContext, @nospecialize(value), @nospecialize(result_type))
    result_type_unwrapped = unwrap_type(result_type)

    # Ghost types have no runtime representation
    if is_ghost_type(result_type_unwrapped)
        return ghost_value(result_type_unwrapped)
    end

    # Skip non-primitive types
    if !(result_type_unwrapped <: Number || result_type_unwrapped === Bool)
        return nothing
    end

    type_id = tile_type_for_julia!(ctx, result_type_unwrapped)
    bytes = constant_to_bytes(value, result_type_unwrapped)
    v = encode_ConstantOp!(ctx.cb, type_id, bytes)

    CGVal(v, type_id, result_type_unwrapped)
end

function constant_to_bytes(@nospecialize(value), @nospecialize(T::Type))
    if T === Bool
        return UInt8[value ? 0xff : 0x00]
    elseif T === Int32 || T === UInt32
        return collect(reinterpret(UInt8, [Int32(value)]))
    elseif T === Int64 || T === UInt64
        return collect(reinterpret(UInt8, [Int64(value)]))
    elseif T === Float32
        return collect(reinterpret(UInt8, [Float32(value)]))
    elseif T === Float64
        return collect(reinterpret(UInt8, [Float64(value)]))
    else
        error("Cannot convert $T to constant bytes")
    end
end

#=============================================================================
 Gather/Scatter Operations
=============================================================================#

function emit_intrinsic!(ctx::CodegenContext, ::typeof(_gather), args, @nospecialize(result_type))
    if length(args) == 2
        emit_gather_1d!(ctx, args, result_type)
    elseif length(args) == 3
        emit_gather_2d!(ctx, args, result_type)
    else
        error("_gather: unexpected argument count $(length(args))")
    end
end

function emit_intrinsic!(ctx::CodegenContext, ::typeof(_scatter), args, @nospecialize(result_type))
    if length(args) == 3
        emit_scatter_1d!(ctx, args, result_type)
    elseif length(args) == 4
        emit_scatter_2d!(ctx, args, result_type)
    else
        error("_scatter: unexpected argument count $(length(args))")
    end
end

"""
    emit_gather_1d!(ctx, args, result_type)

Emit code for 1D gather: `gather(array, indices)`

Steps:
1. Get base pointer from TileArray
2. Compute pointer tile: OffsetOp(base_ptr, indices)
3. Compute bounds mask: (indices >= 0) & (indices < size)
4. Load via LoadPtrTkoOp with mask and zero padding
"""
function emit_gather_1d!(ctx::CodegenContext, args::AbstractVector, @nospecialize(result_type))
    cb = ctx.cb
    tt = ctx.tt

    # Get array info
    array_arg = args[1]
    arg_idx = extract_argument_index(array_arg)
    (arg_idx === nothing || !is_destructured_arg(ctx, arg_idx)) &&
        error("gather() requires a TileArray argument")

    ptr_vals = get_arg_flat_values(ctx, arg_idx, :ptr)
    isempty(ptr_vals) && error("Cannot get ptr from TileArray argument")
    base_ptr = ptr_vals[1]

    tilearray_type = get_arg_type(ctx, arg_idx)
    elem_type = eltype(tilearray_type)

    # Get array size
    sizes_vals = get_arg_flat_values(ctx, arg_idx, :sizes)
    (sizes_vals === nothing || isempty(sizes_vals)) && error("Cannot get size from TileArray")
    array_size = sizes_vals[1]

    # Get indices tile
    indices_tv = emit_value!(ctx, args[2])
    indices_tv === nothing && error("Cannot resolve indices for gather()")
    indices = indices_tv.v
    tile_shape = indices_tv.shape

    # Type setup
    dtype = julia_to_tile_dtype!(tt, elem_type)
    idx_dtype = julia_to_tile_dtype!(tt, Int32)
    ptr_dtype = pointer_type!(tt, dtype)

    scalar_i32 = tile_type!(tt, I32(tt), Int[])
    scalar_ptr_type = tile_type!(tt, ptr_dtype, Int[])
    ptr_tile_type = tile_type!(tt, ptr_dtype, tile_shape)
    result_tile_type = tile_type!(tt, dtype, tile_shape)
    bool_tile_type = tile_type!(tt, I1(tt), tile_shape)
    token_type = Token(tt)

    # Broadcast base pointer to tile shape
    # ReshapeOp to add dimensions, then BroadcastOp
    ndims = length(tile_shape)
    if ndims > 0
        ones_shape = fill(1, ndims)
        reshaped_ptr_type = tile_type!(tt, ptr_dtype, ones_shape)
        base_ptr_reshaped = encode_ReshapeOp!(cb, reshaped_ptr_type, base_ptr)
        base_ptr_tile = encode_BroadcastOp!(cb, ptr_tile_type, base_ptr_reshaped)
    else
        base_ptr_tile = base_ptr
    end

    # Compute offset pointers: base_ptr + indices (element offset)
    pointers = encode_OffsetOp!(cb, ptr_tile_type, base_ptr_tile, indices)

    # Compute bounds mask: (indices >= 0) & (indices < size)
    # Broadcast size to tile shape
    idx_tile_type = tile_type!(tt, idx_dtype, tile_shape)
    if ndims > 0
        ones_shape = fill(1, ndims)
        reshaped_size_type = tile_type!(tt, idx_dtype, ones_shape)
        size_reshaped = encode_ReshapeOp!(cb, reshaped_size_type, array_size)
        size_tile = encode_BroadcastOp!(cb, idx_tile_type, size_reshaped)
    else
        size_tile = array_size
    end

    # indices >= 0
    zero_bytes = collect(reinterpret(UInt8, [Int32(0)]))
    zero_scalar = encode_ConstantOp!(cb, scalar_i32, zero_bytes)
    if ndims > 0
        ones_shape = fill(1, ndims)
        reshaped_zero_type = tile_type!(tt, idx_dtype, ones_shape)
        zero_reshaped = encode_ReshapeOp!(cb, reshaped_zero_type, zero_scalar)
        zero_tile = encode_BroadcastOp!(cb, idx_tile_type, zero_reshaped)
    else
        zero_tile = zero_scalar
    end
    ge_zero = encode_CmpIOp!(cb, bool_tile_type, indices, zero_tile;
                             predicate=CmpGreaterThanOrEqual, signedness=SignednessSigned)

    # indices < size
    lt_size = encode_CmpIOp!(cb, bool_tile_type, indices, size_tile;
                             predicate=CmpLessThan, signedness=SignednessSigned)

    # Combined mask: ge_zero & lt_size
    mask = encode_AndIOp!(cb, bool_tile_type, ge_zero, lt_size)

    # Create padding value (zero)
    padding_bytes = if elem_type === Float32
        collect(reinterpret(UInt8, [Float32(0)]))
    elseif elem_type === Float64
        collect(reinterpret(UInt8, [Float64(0)]))
    elseif elem_type === Int32
        collect(reinterpret(UInt8, [Int32(0)]))
    else
        collect(reinterpret(UInt8, [Int32(0)]))  # Default
    end
    padding_scalar = encode_ConstantOp!(cb, tile_type!(tt, dtype, Int[]), padding_bytes)
    if ndims > 0
        ones_shape = fill(1, ndims)
        reshaped_pad_type = tile_type!(tt, dtype, ones_shape)
        padding_reshaped = encode_ReshapeOp!(cb, reshaped_pad_type, padding_scalar)
        padding_tile = encode_BroadcastOp!(cb, result_tile_type, padding_reshaped)
    else
        padding_tile = padding_scalar
    end

    # Load with mask
    tile_val, new_token = encode_LoadPtrTkoOp!(cb, result_tile_type, token_type, pointers;
                                                mask=mask,
                                                padding_value=padding_tile,
                                                token=ctx.token)
    ctx.token = new_token

    CGVal(tile_val, result_tile_type, Tile{elem_type, Tuple(tile_shape)}, tile_shape)
end

"""
    emit_gather_2d!(ctx, args, result_type)

Emit code for 2D gather: `gather(array, idx0, idx1)`

The indices are broadcast to a common shape. Linear index is computed as:
linear_idx = idx0 + idx1 * stride1
"""
function emit_gather_2d!(ctx::CodegenContext, args::AbstractVector, @nospecialize(result_type))
    cb = ctx.cb
    tt = ctx.tt

    # Get array info
    array_arg = args[1]
    arg_idx = extract_argument_index(array_arg)
    (arg_idx === nothing || !is_destructured_arg(ctx, arg_idx)) &&
        error("gather() requires a TileArray argument")

    ptr_vals = get_arg_flat_values(ctx, arg_idx, :ptr)
    isempty(ptr_vals) && error("Cannot get ptr from TileArray argument")
    base_ptr = ptr_vals[1]

    tilearray_type = get_arg_type(ctx, arg_idx)
    elem_type = eltype(tilearray_type)

    # Get array sizes and strides
    sizes_vals = get_arg_flat_values(ctx, arg_idx, :sizes)
    strides_vals = get_arg_flat_values(ctx, arg_idx, :strides)
    (sizes_vals === nothing || length(sizes_vals) < 2) && error("Cannot get sizes from 2D TileArray")
    (strides_vals === nothing || length(strides_vals) < 2) && error("Cannot get strides from 2D TileArray")

    size0, size1 = sizes_vals[1], sizes_vals[2]
    stride0, stride1 = strides_vals[1], strides_vals[2]

    # Get index tiles
    idx0_tv = emit_value!(ctx, args[2])
    idx1_tv = emit_value!(ctx, args[3])
    (idx0_tv === nothing || idx1_tv === nothing) && error("Cannot resolve indices for gather()")

    # Compute broadcast shape
    output_shape = compute_broadcast_shape(idx0_tv.shape, idx1_tv.shape)

    # Type setup
    dtype = julia_to_tile_dtype!(tt, elem_type)
    idx_dtype = julia_to_tile_dtype!(tt, Int32)
    ptr_dtype = pointer_type!(tt, dtype)

    scalar_i32 = tile_type!(tt, I32(tt), Int[])
    ptr_tile_type = tile_type!(tt, ptr_dtype, output_shape)
    result_tile_type = tile_type!(tt, dtype, output_shape)
    bool_tile_type = tile_type!(tt, I1(tt), output_shape)
    idx_tile_type = tile_type!(tt, idx_dtype, output_shape)
    token_type = Token(tt)

    # Broadcast indices to common shape
    idx0 = broadcast_tile_to_shape!(cb, tt, idx0_tv, output_shape, idx_dtype)
    idx1 = broadcast_tile_to_shape!(cb, tt, idx1_tv, output_shape, idx_dtype)

    # Compute linear index: idx0 * stride0 + idx1 * stride1
    # Broadcast strides
    ndims = length(output_shape)
    stride0_tile = broadcast_scalar_to_shape!(cb, tt, stride0, output_shape, idx_dtype)
    stride1_tile = broadcast_scalar_to_shape!(cb, tt, stride1, output_shape, idx_dtype)

    term0 = encode_MulIOp!(cb, idx_tile_type, idx0, stride0_tile)
    term1 = encode_MulIOp!(cb, idx_tile_type, idx1, stride1_tile)
    linear_idx = encode_AddIOp!(cb, idx_tile_type, term0, term1)

    # Broadcast base pointer
    base_ptr_tile = broadcast_scalar_to_shape!(cb, tt, base_ptr, output_shape, ptr_dtype)

    # Compute offset pointers
    pointers = encode_OffsetOp!(cb, ptr_tile_type, base_ptr_tile, linear_idx)

    # Compute bounds masks for each dimension
    size0_tile = broadcast_scalar_to_shape!(cb, tt, size0, output_shape, idx_dtype)
    size1_tile = broadcast_scalar_to_shape!(cb, tt, size1, output_shape, idx_dtype)

    zero_tile = broadcast_constant_to_shape!(cb, tt, Int32(0), output_shape, idx_dtype)

    # idx0 >= 0 && idx0 < size0
    ge_zero_0 = encode_CmpIOp!(cb, bool_tile_type, idx0, zero_tile;
                               predicate=CmpGreaterThanOrEqual, signedness=SignednessSigned)
    lt_size_0 = encode_CmpIOp!(cb, bool_tile_type, idx0, size0_tile;
                               predicate=CmpLessThan, signedness=SignednessSigned)
    mask_0 = encode_AndIOp!(cb, bool_tile_type, ge_zero_0, lt_size_0)

    # idx1 >= 0 && idx1 < size1
    ge_zero_1 = encode_CmpIOp!(cb, bool_tile_type, idx1, zero_tile;
                               predicate=CmpGreaterThanOrEqual, signedness=SignednessSigned)
    lt_size_1 = encode_CmpIOp!(cb, bool_tile_type, idx1, size1_tile;
                               predicate=CmpLessThan, signedness=SignednessSigned)
    mask_1 = encode_AndIOp!(cb, bool_tile_type, ge_zero_1, lt_size_1)

    # Combined mask
    mask = encode_AndIOp!(cb, bool_tile_type, mask_0, mask_1)

    # Create padding value (zero)
    padding_tile = broadcast_constant_to_shape!(cb, tt, zero_value(elem_type), output_shape, dtype)

    # Load with mask
    tile_val, new_token = encode_LoadPtrTkoOp!(cb, result_tile_type, token_type, pointers;
                                                mask=mask,
                                                padding_value=padding_tile,
                                                token=ctx.token)
    ctx.token = new_token

    CGVal(tile_val, result_tile_type, Tile{elem_type, Tuple(output_shape)}, output_shape)
end

"""
    emit_scatter_1d!(ctx, args, result_type)

Emit code for 1D scatter: `scatter(array, indices, tile)`

Steps:
1. Get base pointer from TileArray
2. Compute pointer tile: OffsetOp(base_ptr, indices)
3. Compute bounds mask: (indices >= 0) & (indices < size)
4. Store via StorePtrTkoOp with mask
"""
function emit_scatter_1d!(ctx::CodegenContext, args::AbstractVector, @nospecialize(result_type))
    cb = ctx.cb
    tt = ctx.tt

    # Get array info
    array_arg = args[1]
    arg_idx = extract_argument_index(array_arg)
    (arg_idx === nothing || !is_destructured_arg(ctx, arg_idx)) &&
        error("scatter() requires a TileArray argument")

    ptr_vals = get_arg_flat_values(ctx, arg_idx, :ptr)
    isempty(ptr_vals) && error("Cannot get ptr from TileArray argument")
    base_ptr = ptr_vals[1]

    tilearray_type = get_arg_type(ctx, arg_idx)
    elem_type = eltype(tilearray_type)

    # Get array size
    sizes_vals = get_arg_flat_values(ctx, arg_idx, :sizes)
    (sizes_vals === nothing || isempty(sizes_vals)) && error("Cannot get size from TileArray")
    array_size = sizes_vals[1]

    # Get indices tile
    indices_tv = emit_value!(ctx, args[2])
    indices_tv === nothing && error("Cannot resolve indices for scatter()")
    indices = indices_tv.v
    tile_shape = indices_tv.shape

    # Get value tile
    value_tv = emit_value!(ctx, args[3])
    value_tv === nothing && error("Cannot resolve value tile for scatter()")
    value = value_tv.v

    # Type setup
    dtype = julia_to_tile_dtype!(tt, elem_type)
    idx_dtype = julia_to_tile_dtype!(tt, Int32)
    ptr_dtype = pointer_type!(tt, dtype)

    scalar_i32 = tile_type!(tt, I32(tt), Int[])
    ptr_tile_type = tile_type!(tt, ptr_dtype, tile_shape)
    bool_tile_type = tile_type!(tt, I1(tt), tile_shape)
    idx_tile_type = tile_type!(tt, idx_dtype, tile_shape)
    token_type = Token(tt)

    # Broadcast base pointer to tile shape
    ndims = length(tile_shape)
    base_ptr_tile = broadcast_scalar_to_shape!(cb, tt, base_ptr, tile_shape, ptr_dtype)

    # Compute offset pointers
    pointers = encode_OffsetOp!(cb, ptr_tile_type, base_ptr_tile, indices)

    # Compute bounds mask
    size_tile = broadcast_scalar_to_shape!(cb, tt, array_size, tile_shape, idx_dtype)
    zero_tile = broadcast_constant_to_shape!(cb, tt, Int32(0), tile_shape, idx_dtype)

    ge_zero = encode_CmpIOp!(cb, bool_tile_type, indices, zero_tile;
                             predicate=CmpGreaterThanOrEqual, signedness=SignednessSigned)
    lt_size = encode_CmpIOp!(cb, bool_tile_type, indices, size_tile;
                             predicate=CmpLessThan, signedness=SignednessSigned)
    mask = encode_AndIOp!(cb, bool_tile_type, ge_zero, lt_size)

    # Store with mask
    new_token = encode_StorePtrTkoOp!(cb, token_type, pointers, value;
                                       mask=mask, token=ctx.token)
    ctx.token = new_token

    nothing
end

"""
    emit_scatter_2d!(ctx, args, result_type)

Emit code for 2D scatter: `scatter(array, idx0, idx1, tile)`
"""
function emit_scatter_2d!(ctx::CodegenContext, args::AbstractVector, @nospecialize(result_type))
    cb = ctx.cb
    tt = ctx.tt

    # Get array info
    array_arg = args[1]
    arg_idx = extract_argument_index(array_arg)
    (arg_idx === nothing || !is_destructured_arg(ctx, arg_idx)) &&
        error("scatter() requires a TileArray argument")

    ptr_vals = get_arg_flat_values(ctx, arg_idx, :ptr)
    isempty(ptr_vals) && error("Cannot get ptr from TileArray argument")
    base_ptr = ptr_vals[1]

    tilearray_type = get_arg_type(ctx, arg_idx)
    elem_type = eltype(tilearray_type)

    # Get array sizes and strides
    sizes_vals = get_arg_flat_values(ctx, arg_idx, :sizes)
    strides_vals = get_arg_flat_values(ctx, arg_idx, :strides)
    (sizes_vals === nothing || length(sizes_vals) < 2) && error("Cannot get sizes from 2D TileArray")
    (strides_vals === nothing || length(strides_vals) < 2) && error("Cannot get strides from 2D TileArray")

    size0, size1 = sizes_vals[1], sizes_vals[2]
    stride0, stride1 = strides_vals[1], strides_vals[2]

    # Get index tiles
    idx0_tv = emit_value!(ctx, args[2])
    idx1_tv = emit_value!(ctx, args[3])
    (idx0_tv === nothing || idx1_tv === nothing) && error("Cannot resolve indices for scatter()")

    # Get value tile
    value_tv = emit_value!(ctx, args[4])
    value_tv === nothing && error("Cannot resolve value tile for scatter()")

    # Compute broadcast shape from indices
    output_shape = compute_broadcast_shape(idx0_tv.shape, idx1_tv.shape)

    # Type setup
    dtype = julia_to_tile_dtype!(tt, elem_type)
    idx_dtype = julia_to_tile_dtype!(tt, Int32)
    ptr_dtype = pointer_type!(tt, dtype)

    scalar_i32 = tile_type!(tt, I32(tt), Int[])
    ptr_tile_type = tile_type!(tt, ptr_dtype, output_shape)
    bool_tile_type = tile_type!(tt, I1(tt), output_shape)
    idx_tile_type = tile_type!(tt, idx_dtype, output_shape)
    token_type = Token(tt)

    # Broadcast indices to common shape
    idx0 = broadcast_tile_to_shape!(cb, tt, idx0_tv, output_shape, idx_dtype)
    idx1 = broadcast_tile_to_shape!(cb, tt, idx1_tv, output_shape, idx_dtype)

    # Broadcast value tile if needed
    value = broadcast_tile_to_shape!(cb, tt, value_tv, output_shape, dtype)

    # Compute linear index
    ndims = length(output_shape)
    stride0_tile = broadcast_scalar_to_shape!(cb, tt, stride0, output_shape, idx_dtype)
    stride1_tile = broadcast_scalar_to_shape!(cb, tt, stride1, output_shape, idx_dtype)

    term0 = encode_MulIOp!(cb, idx_tile_type, idx0, stride0_tile)
    term1 = encode_MulIOp!(cb, idx_tile_type, idx1, stride1_tile)
    linear_idx = encode_AddIOp!(cb, idx_tile_type, term0, term1)

    # Broadcast base pointer
    base_ptr_tile = broadcast_scalar_to_shape!(cb, tt, base_ptr, output_shape, ptr_dtype)

    # Compute offset pointers
    pointers = encode_OffsetOp!(cb, ptr_tile_type, base_ptr_tile, linear_idx)

    # Compute bounds masks
    size0_tile = broadcast_scalar_to_shape!(cb, tt, size0, output_shape, idx_dtype)
    size1_tile = broadcast_scalar_to_shape!(cb, tt, size1, output_shape, idx_dtype)
    zero_tile = broadcast_constant_to_shape!(cb, tt, Int32(0), output_shape, idx_dtype)

    ge_zero_0 = encode_CmpIOp!(cb, bool_tile_type, idx0, zero_tile;
                               predicate=CmpGreaterThanOrEqual, signedness=SignednessSigned)
    lt_size_0 = encode_CmpIOp!(cb, bool_tile_type, idx0, size0_tile;
                               predicate=CmpLessThan, signedness=SignednessSigned)
    mask_0 = encode_AndIOp!(cb, bool_tile_type, ge_zero_0, lt_size_0)

    ge_zero_1 = encode_CmpIOp!(cb, bool_tile_type, idx1, zero_tile;
                               predicate=CmpGreaterThanOrEqual, signedness=SignednessSigned)
    lt_size_1 = encode_CmpIOp!(cb, bool_tile_type, idx1, size1_tile;
                               predicate=CmpLessThan, signedness=SignednessSigned)
    mask_1 = encode_AndIOp!(cb, bool_tile_type, ge_zero_1, lt_size_1)

    mask = encode_AndIOp!(cb, bool_tile_type, mask_0, mask_1)

    # Store with mask
    new_token = encode_StorePtrTkoOp!(cb, token_type, pointers, value;
                                       mask=mask, token=ctx.token)
    ctx.token = new_token

    nothing
end

# Helper: broadcast a scalar Value to a tile shape
function broadcast_scalar_to_shape!(cb::CodeBuilder, tt::TypeTable, scalar::Value,
                                     target_shape::Vector{Int}, dtype::TypeId)
    ndims = length(target_shape)
    if ndims == 0
        return scalar
    end

    ones_shape = fill(1, ndims)
    reshaped_type = tile_type!(tt, dtype, ones_shape)
    reshaped = encode_ReshapeOp!(cb, reshaped_type, scalar)

    target_type = tile_type!(tt, dtype, target_shape)
    encode_BroadcastOp!(cb, target_type, reshaped)
end

# Helper: create a constant and broadcast to shape
function broadcast_constant_to_shape!(cb::CodeBuilder, tt::TypeTable, @nospecialize(value),
                                       target_shape::Vector{Int}, dtype::TypeId)
    scalar_type = tile_type!(tt, dtype, Int[])
    bytes = if value isa Int32
        collect(reinterpret(UInt8, [value]))
    elseif value isa Float32
        collect(reinterpret(UInt8, [value]))
    elseif value isa Float64
        collect(reinterpret(UInt8, [value]))
    else
        collect(reinterpret(UInt8, [Int32(value)]))
    end
    scalar = encode_ConstantOp!(cb, scalar_type, bytes)

    broadcast_scalar_to_shape!(cb, tt, scalar, target_shape, dtype)
end

# Helper: get zero value for a type
function zero_value(@nospecialize(T::Type))
    if T === Float32
        Float32(0)
    elseif T === Float64
        Float64(0)
    elseif T <: Integer
        Int32(0)
    else
        Float32(0)
    end
end
