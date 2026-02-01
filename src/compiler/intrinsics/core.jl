# core Tile IR intrinsics

"""
    validate_tile_shape(shape, context::String)

Validate that all tile dimensions are powers of 2.
Tile IR requires all tile dimensions to be powers of 2.
Throws an error with a clear message if validation fails.
"""
function validate_tile_shape(shape, context::String)
    for (i, dim) in enumerate(shape)
        if dim <= 0
            error("$context: tile dimension $i must be positive, got $dim")
        end
        if !ispow2(dim)
            error("$context: tile dimension $i must be a power of 2, got $dim")
        end
    end
end

# cuda_tile.broadcast
@eval Intrinsics begin
    """
        broadcast(tile, shape_val)

    Explicitly broadcast a tile to a target shape.
    Compiled to cuda_tile.broadcast.
    """
    @noinline function broadcast(tile::Tile{T}, ::Val{Shape}) where {T, Shape}
        Tile{T, Shape}()
    end
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.broadcast), args)
    cb = ctx.cb
    tt = ctx.tt

    # Get source tile
    source = emit_value!(ctx, args[1])
    source === nothing && error("Cannot resolve source operand for broadcast()")

    # Get source element type
    source_type = unwrap_type(source.jltype)
    source_elem = source_type.parameters[1]

    # Extract target shape from the constant tuple argument
    target_shape_tuple = get_constant(ctx, args[2])
    target_shape_tuple isa Tuple || error("broadcast() shape must be a compile-time constant tuple")
    target_shape = collect(Int, target_shape_tuple)
    validate_tile_shape(target_shape, "broadcast")

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

# cuda_tile.cat
@eval Intrinsics begin
    """
        cat(tiles, axis_val)

    Concatenate two tiles along 0-indexed axis.
    Compiled to cuda_tile.cat.
    """
    @noinline function cat(tiles::Tuple{Tile{T, S1}, Tile{T, S2}}, ::Val{Axis}) where {T, S1, S2, Axis}
        ndims = length(S1)
        axis = Axis < 0 ? ndims + Axis : Axis
        result_shape = ntuple(ndims) do i
            if i == axis + 1  # 0-indexed axis, 1-indexed tuple access
                S1[i] + S2[i]
            else
                S1[i]
            end
        end
        Tile{T, result_shape}()
    end
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.cat), args)
    cb = ctx.cb
    tt = ctx.tt

    # Emit tuple value to get CGVal with component refs in .tuple
    tuple_tv = emit_value!(ctx, args[1])
    tuple_tv === nothing && error("cat() cannot resolve tuple argument")

    # Extract component refs from .tuple field
    tuple_tv.tuple !== nothing || error("cat() requires tuple with tracked components")
    length(tuple_tv.tuple) == 2 || error("cat() expects exactly 2 tiles, got $(length(tuple_tv.tuple))")

    # Emit tiles from refs (looks up ctx.values, not stmts!)
    lhs = emit_value!(ctx, tuple_tv.tuple[1])
    rhs = emit_value!(ctx, tuple_tv.tuple[2])
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
    validate_tile_shape(output_shape, "cat")

    # Get element type
    elem_type = unwrap_type(lhs.jltype).parameters[1]

    # Create output tile type
    dtype = julia_to_tile_dtype!(tt, elem_type)
    output_tile_type = tile_type!(tt, dtype, output_shape)

    # Emit CatOp (axis is 0-indexed for bytecode)
    result = encode_CatOp!(cb, output_tile_type, lhs.v, rhs.v, axis)

    CGVal(result, output_tile_type, Tile{elem_type, Tuple(output_shape)}, output_shape)
end

# cuda_tile.constant
@eval Intrinsics begin
    """
        constant(shape, value, T)

    Create a tile filled with a constant value.
    Compiled to cuda_tile.constant.
    """
    @noinline function constant(shape::NTuple{N, Int}, value, ::Type{T}) where {N, T}
        Tile{T, shape}()
    end
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.constant), args)
    cb = ctx.cb
    tt = ctx.tt

    # Extract shape
    shape = get_constant(ctx, args[1])
    shape isa Tuple || error("full() shape must be a compile-time constant tuple")
    tile_shape = collect(Int, shape)
    validate_tile_shape(tile_shape, "full")

    # Extract value
    value = @something get_constant(ctx, args[2]) error("full() value must be a compile-time constant")

    # Extract dtype from Type{T} argument
    elem_type = @something get_constant(ctx, args[3]) error("constant() requires a compile-time element type")

    dtype = julia_to_tile_dtype!(tt, elem_type)
    tile_type = tile_type!(tt, dtype, tile_shape)

    # Create constant directly at target shape
    value_bytes = constant_to_bytes(value, elem_type)
    result = encode_ConstantOp!(cb, tile_type, value_bytes)

    CGVal(result, tile_type, Tile{elem_type, Tuple(tile_shape)}, tile_shape)
end

# TODO: cuda_tile.entry

# cuda_tile.extract
@eval Intrinsics begin
    """
        extract(tile, index_val, shape_val)

    Extract a sub-tile from tile at 0-indexed slice indices.
    Compiled to cuda_tile.extract.
    """
    @noinline function extract(tile::Tile{T}, ::Val{Index}, ::Val{Shape}) where {T, Index, Shape}
        Tile{T, Shape}()
    end
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.extract), args)
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
    validate_tile_shape(output_shape, "extract")

    # Get element type
    elem_type = unwrap_type(source.jltype).parameters[1]

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

# TODO: cuda_tile.get_global

# cuda_tile.get_num_tile_blocks
@eval Intrinsics begin
    """
        get_num_tile_blocks(axis)::Int32

    Get the grid size along the given axis (0=x, 1=y, 2=z).
    Compiled to cuda_tile.get_num_tile_blocks.
    """
    @noinline get_num_tile_blocks(axis::Integer) = compilerbarrier(:const, zero(Int32))
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.get_num_tile_blocks), args)
    axis = @something get_constant(ctx, args[1]) error("get_num_tile_blocks() axis must be a compile-time constant")
    axis in (0, 1, 2) || error("get_num_tile_blocks() axis must be 0, 1, or 2, got $axis")

    res_type = tile_type!(ctx.tt, I32(ctx.tt), Int[])
    nb_x, nb_y, nb_z = encode_GetNumTileBlocksOp!(ctx.cb, res_type, res_type, res_type)

    CGVal((nb_x, nb_y, nb_z)[axis + 1], res_type, Int32)
end

# cuda_tile.get_tile_block_id
@eval Intrinsics begin
    """
        get_tile_block_id(axis)::Int32

    Get the block ID along the given axis (0=x, 1=y, 2=z).
    Compiled to cuda_tile.get_tile_block_id.
    """
    @noinline get_tile_block_id(axis::Integer) = compilerbarrier(:const, zero(Int32))
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.get_tile_block_id), args)
    axis = @something get_constant(ctx, args[1]) error("get_tile_block_id() axis must be a compile-time constant")
    axis in (0, 1, 2) || error("get_tile_block_id() axis must be 0, 1, or 2, got $axis")

    res_type = tile_type!(ctx.tt, I32(ctx.tt), Int[])
    bid_x, bid_y, bid_z = encode_GetTileBlockIdOp!(ctx.cb, res_type, res_type, res_type)
    result = (bid_x, bid_y, bid_z)[axis + 1]

    CGVal(result, res_type, Int32)
end

# TODO: cuda_tile.global

# cuda_tile.iota
@eval Intrinsics begin
    """
        iota(shape, T)

    Create a 1D tile with values [0, 1, 2, ..., shape[1]-1] (0-indexed).
    Compiled to cuda_tile.iota.
    """
    @noinline function iota(shape::NTuple{1, Int}, ::Type{T}) where {T}
        Tile{T, shape}()
    end
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.iota), args)
    cb = ctx.cb
    tt = ctx.tt

    # Extract shape
    shape = get_constant(ctx, args[1])
    shape isa Tuple || error("iota() shape must be a compile-time constant tuple")
    tile_shape = collect(Int, shape)
    validate_tile_shape(tile_shape, "arange")

    # Extract dtype from Type{T} argument
    elem_type = @something get_constant(ctx, args[2]) error("iota() requires a compile-time element type")

    dtype = julia_to_tile_dtype!(tt, elem_type)
    tile_type = tile_type!(tt, dtype, tile_shape)

    # Emit IotaOp
    result = encode_IotaOp!(cb, tile_type)

    CGVal(result, tile_type, Tile{elem_type, Tuple(tile_shape)}, tile_shape)
end

# cuda_tile.mmaf, cuda_tile.mmai
@eval Intrinsics begin
    """
        mma(a, b, acc)

    Matrix-multiply-accumulate: result = a @ b + acc.
    Compiled to cuda_tile.mmaf or cuda_tile.mmai.
    """
    @noinline function mma(a::Tile{T1}, b::Tile{T2}, acc::Tile{T3, SC}) where {T1, T2, T3, SC}
        Tile{T3, SC}()
    end
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.mma), args)
    cb = ctx.cb

    lhs = emit_value!(ctx, args[1])
    rhs = emit_value!(ctx, args[2])
    acc = emit_value!(ctx, args[3])

    (lhs === nothing || rhs === nothing || acc === nothing) && error("Cannot resolve operands for mma()")

    result = encode_MmaFOp!(cb, acc.type_id, lhs.v, rhs.v, acc.v)

    CGVal(result, acc.type_id, acc.jltype, acc.shape)
end

# TODO: cuda_tile.module

# cuda_tile.offset
@eval Intrinsics begin
    """
        offset(base, offsets)

    Compute base_ptr + offsets for each element of offsets tile (element-scaled).
    Returns a tile of pointers. Compiled to cuda_tile.offset.
    """
    @noinline function offset(base::Ptr{T}, offsets::Tile{I, S}) where {T, I <: Integer, S}
        Tile{Ptr{T}, S}()
    end
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.offset), args)
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

    # Get pointer element type from base pointer type (Ptr{T})
    base_ptr_type = unwrap_type(base_ptr_tv.jltype)
    ptr_elem_type = eltype(base_ptr_type)  # T from Ptr{T}
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

    result_jltype = Tile{Ptr{ptr_elem_type}, Tuple(tile_shape)}
    CGVal(pointers, ptr_tile_type, result_jltype, tile_shape)
end

# TODO: cudatile.pack

# cuda_tile.permute
@eval Intrinsics begin
    """
        permute(tile, perm_val)

    Permute tile dimensions according to 0-indexed permutation.
    Compiled to cuda_tile.permute.
    """
    @noinline function permute(tile::Tile{T, S}, ::Val{Perm}) where {T, S, Perm}
        # Compute permuted shape: for each position i in output, take S[Perm[i]+1]
        permuted_shape = ntuple(i -> S[Perm[i] + 1], length(Perm))
        Tile{T, permuted_shape}()
    end
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.permute), args)
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
    elem_type = unwrap_type(source.jltype).parameters[1]

    # Create output tile type
    dtype = julia_to_tile_dtype!(tt, elem_type)
    output_tile_type = tile_type!(tt, dtype, output_shape)

    # Emit PermuteOp
    result = encode_PermuteOp!(cb, output_tile_type, source.v, permutation)

    CGVal(result, output_tile_type, Tile{elem_type, Tuple(output_shape)}, output_shape)
end

# cuda_tile.transpose
@eval Intrinsics begin
    """
        transpose(tile)

    Transpose a 2D tile, swapping its dimensions.
    Compiled to cuda_tile.permute with perm=(1, 0).
    """
    @noinline function transpose(tile::Tile{T, S}) where {T, S}
        Tile{T, reverse(S)}()
    end
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.transpose), args)
    cb = ctx.cb
    tt = ctx.tt

    source = emit_value!(ctx, args[1])
    source === nothing && error("Cannot resolve operand for transpose()")

    input_shape = source.shape
    isempty(input_shape) && error("Cannot determine tile shape for transpose()")

    output_shape = reverse(input_shape)

    elem_type = unwrap_type(source.jltype).parameters[1]

    dtype = julia_to_tile_dtype!(tt, elem_type)
    output_tile_type = tile_type!(tt, dtype, output_shape)

    ndim = length(output_shape)
    permutation = collect(ndim-1:-1:0)

    result = encode_PermuteOp!(cb, output_tile_type, source.v, permutation)

    CGVal(result, output_tile_type, Tile{elem_type, Tuple(output_shape)}, output_shape)
end


# cuda_tile.reduce
@eval Intrinsics begin
    """
        reduce(tile, axis_val, f, identity)

    Reduction along 0-indexed axis with the given binary function and identity.
    f is any binary function (e.g., +, max, or a lambda).
    Compiled to cuda_tile.reduce.
    """
    @noinline function reduce(tile::Tile{T, S}, ::Val{axis}, f, identity) where {T, S, axis}
        # Julia semantics: reduced dimension becomes size 1 (not removed)
        reduced_shape = ntuple(i -> i == axis + 1 ? 1 : S[i], length(S))
        Tile{T, reduced_shape}()
    end
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.reduce), args)
    emit_reduce!(ctx, args)
end
function emit_reduce!(ctx::CGCtx, args)
    cb = ctx.cb
    tt = ctx.tt

    # Get input tile
    input_tv = emit_value!(ctx, args[1])
    input_tv === nothing && error("Cannot resolve input tile for reduction")

    # Get reduction axis
    axis = @something get_constant(ctx, args[2]) error("Reduction axis must be a compile-time constant")

    # Resolve function: get the actual Julia function object
    func = resolve_function(ctx, args[3])

    # Get identity value
    identity_val = @something get_constant(ctx, args[4]) error("Reduction identity must be a compile-time constant")

    # Get element type and shapes
    input_type = unwrap_type(input_tv.jltype)
    elem_type = input_type.parameters[1]
    input_shape = input_tv.shape
    isempty(input_shape) && error("Cannot reduce scalar tile")

    # ReduceOp removes the dimension; we'll reshape after to reintroduce it as size 1
    reduced_shape = Int[input_shape[i] for i in eachindex(input_shape) if i != axis + 1]

    dtype = julia_to_tile_dtype!(tt, elem_type)

    # Tile type for ReduceOp output (dimension removed)
    reduced_tile_type = tile_type!(tt, dtype, reduced_shape)

    # Scalar type for reduction body (0D tile)
    scalar_tile_type = tile_type!(tt, dtype, Int[])

    # Create identity value
    identity = make_identity_val(identity_val, dtype, elem_type)

    # Emit ReduceOp with compiled combiner body
    results = encode_ReduceOp!(cb, [reduced_tile_type], [input_tv.v],
                               axis, [identity], [scalar_tile_type]) do block_args
        emit_subprogram!(ctx, func,
                         [elem_type, elem_type],
                         block_args, [scalar_tile_type, scalar_tile_type])
    end

    # Julia semantics: reintroduce reduced dimension as size 1 via ReshapeOp
    output_shape = copy(input_shape)
    output_shape[axis + 1] = 1
    output_tile_type = tile_type!(tt, dtype, output_shape)
    result = encode_ReshapeOp!(cb, output_tile_type, results[1])

    CGVal(result, output_tile_type, Tile{elem_type, Tuple(output_shape)}, output_shape)
end

#=============================================================================#
# Reduce Identity Values via Dispatch
#=============================================================================#

"""
    to_uint128(value)

Convert an integer value to UInt128 for storage in IntegerIdentityVal.
For signed types, this returns the two's complement bit representation.
"""
to_uint128(value::T) where T <: Unsigned = UInt128(value)
to_uint128(value::T) where T <: Signed = UInt128(reinterpret(unsigned(T), value))

"""
    make_identity_val(val, dtype, elem_type) -> IdentityVal

Convert a Julia constant identity value to bytecode IdentityVal format.
"""
make_identity_val(val, dtype, ::Type{T}) where T <: AbstractFloat =
    FloatIdentityVal(Float64(T(val)), dtype, T)
make_identity_val(val, dtype, ::Type{T}) where T <: Integer =
    IntegerIdentityVal(to_uint128(T(val)), dtype, T)


# cuda_tile.reshape
@eval Intrinsics begin
    """
        reshape(tile, shape_val)

    Reshape a tile to a new shape (same total elements).
    Compiled to cuda_tile.reshape.
    """
    @noinline function reshape(tile::Tile{T}, ::Val{Shape}) where {T, Shape}
        Tile{T, Shape}()
    end
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.reshape), args)
    cb = ctx.cb
    tt = ctx.tt

    # Get source tile
    source = emit_value!(ctx, args[1])
    source === nothing && error("Cannot resolve source operand for reshape()")

    # Extract target shape from Val{Shape} argument
    target_shape_tuple = get_constant(ctx, args[2])
    target_shape_tuple isa Tuple || error("reshape() shape must be a compile-time constant tuple")
    target_shape = collect(Int, target_shape_tuple)
    validate_tile_shape(target_shape, "reshape")

    # Get element type and source shape
    source_type = unwrap_type(source.jltype)
    elem_type = source_type.parameters[1]
    source_shape = collect(Int, source_type.parameters[2])

    dtype = julia_to_tile_dtype!(tt, elem_type)

    # Tile IR's ReshapeOp uses row-major element ordering, but Julia uses column-major.
    # To achieve Julia's column-major reshape semantics, we need to:
    # 1. Permute source to row-major order (reverse dims) if ndim > 1
    # 2. Reshape with reversed target shape
    # 3. Permute result back to column-major order (reverse dims) if ndim > 1

    current_val = source.v
    current_shape = source_shape

    # Step 1: Permute source if >1 dimension (column-major → row-major)
    if length(current_shape) > 1
        perm = collect(length(current_shape)-1:-1:0)  # 0-indexed reverse
        permuted_shape = reverse(current_shape)
        perm_type_id = tile_type!(tt, dtype, permuted_shape)
        current_val = encode_PermuteOp!(cb, perm_type_id, current_val, perm)
        current_shape = permuted_shape
    end

    # Step 2: ReshapeOp with reversed target shape
    reversed_target = reverse(target_shape)
    reshape_type_id = tile_type!(tt, dtype, reversed_target)
    current_val = encode_ReshapeOp!(cb, reshape_type_id, current_val)
    current_shape = reversed_target

    # Step 3: Permute result back if >1 dimension (row-major → column-major)
    if length(target_shape) > 1
        perm = collect(length(target_shape)-1:-1:0)  # 0-indexed reverse
        result_type_id = tile_type!(tt, dtype, target_shape)
        current_val = encode_PermuteOp!(cb, result_type_id, current_val, perm)
    else
        result_type_id = tile_type!(tt, dtype, target_shape)
    end

    CGVal(current_val, result_type_id, Tile{elem_type, Tuple(target_shape)}, target_shape)
end

# cuda_tile.scan
@eval Intrinsics begin
    """
        scan(tile, axis_val, f, identity, reverse)

    Parallel prefix scan along specified dimension.
    f is any binary function (e.g., +, max, or a lambda).
    reverse=false for forward scan, true for reverse scan.
    Compiled to cuda_tile.scan.
    """
    @noinline function scan(tile::Tile{T, S}, ::Val{axis}, f, identity, reverse::Bool=false) where {T, S, axis}
        # Scan preserves shape - result has same dimensions as input
        Tile{T, S}()
    end
end

function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.scan), args)
    cb = ctx.cb
    tt = ctx.tt

    # Get input tile
    input_tv = emit_value!(ctx, args[1])
    input_tv === nothing && error("Cannot resolve input tile for scan")

    # Get scan axis
    axis = @something get_constant(ctx, args[2]) error("Scan axis must be a compile-time constant")

    # Resolve function: get the actual Julia function object
    func = resolve_function(ctx, args[3])

    # Get identity value
    identity_val = @something get_constant(ctx, args[4]) error("Scan identity must be a compile-time constant")

    # Get reverse flag (optional, defaults to false)
    reverse = false
    if length(args) >= 5
        reverse_val = get_constant(ctx, args[5])
        reverse = reverse_val === true
    end

    # Get element type and shapes
    input_type = unwrap_type(input_tv.jltype)
    elem_type = input_type <: Tile ? input_type.parameters[1] : input_type
    input_shape = input_tv.shape

    # For scan, output shape is same as input shape
    output_shape = copy(input_shape)

    dtype = julia_to_tile_dtype!(tt, elem_type)

    # Output tile type (same shape as input)
    output_tile_type = tile_type!(tt, dtype, output_shape)

    # Scalar type for scan body (0D tile)
    scalar_tile_type = tile_type!(tt, dtype, Int[])

    # Create identity value
    identity = make_identity_val(identity_val, dtype, elem_type)

    # Emit ScanOp with compiled combiner body
    results = encode_ScanOp!(cb, [output_tile_type], [input_tv.v], axis, reverse, [identity], [scalar_tile_type]) do block_args
        emit_subprogram!(ctx, func,
                         [elem_type, elem_type],
                         block_args, [scalar_tile_type, scalar_tile_type])
    end

    CGVal(results[1], output_tile_type, Tile{elem_type, Tuple(output_shape)}, output_shape)
end

# cuda_tile.select
@eval Intrinsics begin
    """
        select(cond, x, y)

    Element-wise conditional selection.
    Compiled to cuda_tile.select.
    """
    @noinline function select(cond::Tile{Bool, S}, x::Tile{T, S}, y::Tile{T, S}) where {T, S}
        Tile{T, S}()
    end
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.select), args)
    cb = ctx.cb

    cond_tv = emit_value!(ctx, args[1])
    x_tv = emit_value!(ctx, args[2])
    y_tv = emit_value!(ctx, args[3])

    (cond_tv === nothing || x_tv === nothing || y_tv === nothing) &&
        error("Cannot resolve operands for select()")

    result = encode_SelectOp!(cb, x_tv.type_id, cond_tv.v, x_tv.v, y_tv.v)

    CGVal(result, x_tv.type_id, x_tv.jltype, x_tv.shape)
end

# TODO: cuda_tile.unpack
