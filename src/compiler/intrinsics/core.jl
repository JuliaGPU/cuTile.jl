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
            throw(IRError("$context: tile dimension $i must be positive, got $dim"))
        end
        if !ispow2(dim)
            throw(IRError("$context: tile dimension $i must be a power of 2, got $dim"))
        end
    end
end

# cuda_tile.broadcast
@intrinsic broadcast(tile, shape)
function tfunc(𝕃, ::typeof(Intrinsics.broadcast), @nospecialize(tile), @nospecialize(shape_arg))
    tile_type = CC.widenconst(tile)
    tile_type <: Tile || return nothing
    shape_arg = shape_arg
    isa(shape_arg, CC.Const) || return nothing
    shape = shape_arg.val
    T = eltype(tile_type)
    return Tile{T, Tuple{shape...}}
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.broadcast), args)
    cb = ctx.cb
    tt = ctx.tt

    # Get source tile
    source = emit_value!(ctx, args[1])
    source === nothing && throw(IRError("Cannot resolve source operand for broadcast()"))

    # Get source element type
    source_type = CC.widenconst(source.jltype)
    source_elem = eltype(source_type)

    # Extract target shape (from Julia user code) and reverse to Tile IR order
    target_shape_tuple = @something get_constant(ctx, args[2]) throw(IRError("broadcast() shape must be a compile-time constant"))
    target_shape_tuple isa Tuple || throw(IRError("broadcast() shape must be a tuple, got $(typeof(target_shape_tuple))"))
    validate_tile_shape(collect(Int, target_shape_tuple), "broadcast")
    julia_shape = ColMajorShape(target_shape_tuple)
    target_shape = RowMajorShape(julia_shape)

    # If already the right shape, return unchanged
    if source.shape == target_shape
        return source
    end

    # Use the existing broadcast helper
    dtype = julia_to_tile_dtype!(tt, source_elem)
    result_v = broadcast_tile_to_shape!(cb, tt, source, target_shape, dtype)
    result_type_id = tile_type!(tt, dtype, target_shape)

    CGVal(result_v, result_type_id, Tile{source_elem, Tuple{target_shape_tuple...}}, target_shape)
end

"""
    broadcast_tile_to_shape!(cb, tt, tv::CGVal, target_shape::RowMajorShape, dtype::TypeId) -> Value

Broadcast a tile to a target shape by inserting ReshapeOp (for trailing 1s) and BroadcastOp.
Returns the value after broadcasting, or the original value if shapes already match.
"""
function broadcast_tile_to_shape!(cb::CodeBuilder, tt::TypeTable, tv::CGVal,
                                   target_shape::RowMajorShape, dtype::TypeId)
    src_shape = tv.shape

    # Already the right shape?
    if src_shape == target_shape
        return tv.v
    end

    current_val = tv.v
    current_shape = src_shape

    # Step 1: Add leading 1s via ReshapeOp if needed (dimension mismatch)
    # In Tile IR row-major order, Julia's trailing singleton padding becomes leading 1s.
    if length(current_shape) < length(target_shape)
        n_extra = length(target_shape) - length(current_shape)
        current_shape = RowMajorShape(vcat(fill(1, n_extra), collect(current_shape)))
        reshaped_type = tile_type!(tt, dtype, current_shape)
        current_val = encode_ReshapeOp!(cb, reshaped_type, current_val)
    end

    # Step 2: Broadcast dimensions that are 1 to target size
    if current_shape != target_shape
        broadcast_type = tile_type!(tt, dtype, target_shape)
        current_val = encode_BroadcastOp!(cb, broadcast_type, current_val)
    end

    current_val
end

# cuda_tile.cat
@intrinsic cat(tiles, axis)
function tfunc(𝕃, ::typeof(Intrinsics.cat), @nospecialize(tiles), @nospecialize(axis_arg))
    tuple_type = CC.widenconst(tiles)
    tuple_type isa DataType && tuple_type <: Tuple{Tile, Tile} || return nothing
    isa(axis_arg, CC.Const) || return nothing
    axis = axis_arg.val
    t1_type = tuple_type.parameters[1]
    t2_type = tuple_type.parameters[2]
    (t1_type <: Tile && t2_type <: Tile) || return nothing
    T = eltype(t1_type)
    s1 = size(t1_type)
    s2 = size(t2_type)
    isempty(s1) && return nothing
    n = length(s1)
    a = axis < 0 ? n + axis : axis
    result_shape = ntuple(i -> i == a + 1 ? s1[i] + s2[i] : s1[i], n)
    return Tile{T, Tuple{result_shape...}}
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.cat), args)
    cb = ctx.cb
    tt = ctx.tt

    # Emit tuple value to get CGVal with component refs in .tuple
    tuple_tv = emit_value!(ctx, args[1])
    tuple_tv === nothing && throw(IRError("cat() cannot resolve tuple argument"))

    # Extract component refs from .tuple field
    tuple_tv.tuple !== nothing || throw(IRError("cat() requires tuple with tracked components"))
    length(tuple_tv.tuple) == 2 || throw(IRError("cat() expects exactly 2 tiles, got $(length(tuple_tv.tuple))"))

    # Emit tiles from refs (looks up ctx.values, not stmts!)
    lhs = emit_value!(ctx, tuple_tv.tuple[1])
    rhs = emit_value!(ctx, tuple_tv.tuple[2])
    (lhs === nothing || rhs === nothing) && throw(IRError("Cannot resolve tile operands for cat()"))

    # Get axis
    axis_val = @something get_constant(ctx, args[2]) throw(IRError("cat() axis must be a compile-time constant"))
    axis_val isa Integer || throw(IRError("cat() axis must be an integer, got $(typeof(axis_val))"))

    # Handle negative axis and flip to Tile IR order
    lhs_shape = lhs.shape
    ndims = length(lhs_shape)
    julia_axis = axis_val < 0 ? ndims + axis_val : axis_val
    tileir_axis = ndims - 1 - julia_axis

    # Compute output shape - concatenate along the axis (in Tile IR order)
    rhs_shape = rhs.shape
    output_shape = copy(lhs_shape)
    output_shape[tileir_axis + 1] += rhs_shape[tileir_axis + 1]
    validate_tile_shape(collect(output_shape), "cat")

    # Get element type
    lhs_type = CC.widenconst(lhs.jltype)
    elem_type = eltype(lhs_type)

    # Create output tile type
    dtype = julia_to_tile_dtype!(tt, elem_type)
    output_tile_type = tile_type!(tt, dtype, output_shape)

    # Emit CatOp (Tile IR axis)
    result = encode_CatOp!(cb, output_tile_type, lhs.v, rhs.v, tileir_axis)

    julia_output = ColMajorShape(output_shape)
    CGVal(result, output_tile_type, Tile{elem_type, TupleType(julia_output)}, output_shape)
end

# cuda_tile.constant
@intrinsic constant(shape, value, T)
function tfunc(𝕃, ::typeof(Intrinsics.constant), @nospecialize(shape_arg), @nospecialize(value), @nospecialize(type_arg_lat))
    isa(shape_arg, CC.Const) || return nothing
    shape = shape_arg.val
    T = instanceof_tfunc(type_arg_lat)
    T === nothing && return nothing
    return Tile{T, Tuple{shape...}}
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.constant), args)
    cb = ctx.cb
    tt = ctx.tt

    # Extract shape (from Julia user code) and reverse to Tile IR order
    shape = @something get_constant(ctx, args[1]) throw(IRError("fill() shape must be a compile-time constant"))
    shape isa Tuple || throw(IRError("fill() shape must be a tuple, got $(typeof(shape))"))
    validate_tile_shape(collect(Int, shape), "fill")
    tile_shape = RowMajorShape(ColMajorShape(shape))

    # Extract dtype from Type{T} argument
    elem_type = @something get_constant(ctx, args[3]) throw(IRError("constant() requires a compile-time element type"))

    dtype = julia_to_tile_dtype!(tt, elem_type)
    tile_type = tile_type!(tt, dtype, tile_shape)

    tv = emit_value!(ctx, args[2])
    tv === nothing && throw(IRError("fill() value must be a constant or a runtime scalar"))
    if tv.constant !== nothing
        # Compile-time constant: use ConstantOp directly
        value_bytes = constant_to_bytes(something(tv.constant), elem_type)
        result = encode_ConstantOp!(cb, tile_type, value_bytes)
    else
        # Runtime value: broadcast 0D tile to the target shape
        result = broadcast_tile_to_shape!(cb, tt, tv, tile_shape, dtype)
    end

    CGVal(result, tile_type, Tile{elem_type, Tuple{shape...}}, tile_shape)
end

# TODO: cuda_tile.entry

# cuda_tile.extract
@intrinsic extract(tile, index, shape)
function tfunc(𝕃, ::typeof(Intrinsics.extract), @nospecialize(tile_lat), @nospecialize(index), @nospecialize(shape_arg))
    tile_type = CC.widenconst(tile_lat)
    tile_type <: Tile || return nothing
    isa(shape_arg, CC.Const) || return nothing
    shape = shape_arg.val
    T = eltype(tile_type)
    return Tile{T, Tuple{shape...}}
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.extract), args)
    cb = ctx.cb
    tt = ctx.tt

    # Get source tile
    source = emit_value!(ctx, args[1])
    source === nothing && throw(IRError("Cannot resolve source operand for extract()"))

    # Extract index (reverse for Tile IR order)
    index_tuple = @something get_constant(ctx, args[2]) throw(IRError("extract() index must be a compile-time constant"))
    index_tuple isa Tuple || throw(IRError("extract() index must be a tuple, got $(typeof(index_tuple))"))

    # Extract shape (reverse for Tile IR order)
    shape_tuple = @something get_constant(ctx, args[3]) throw(IRError("extract() shape must be a compile-time constant"))
    shape_tuple isa Tuple || throw(IRError("extract() shape must be a tuple, got $(typeof(shape_tuple))"))
    validate_tile_shape(collect(Int, shape_tuple), "extract")
    output_shape = RowMajorShape(ColMajorShape(shape_tuple))

    # Get element type
    elem_type = eltype(CC.widenconst(source.jltype))

    # Create output tile type
    dtype = julia_to_tile_dtype!(tt, elem_type)
    output_tile_type = tile_type!(tt, dtype, output_shape)

    # Create constant index values (0D i32 tiles), reversed for Tile IR order
    scalar_i32 = tile_type!(tt, I32(tt), ScalarShape())
    index_vals = Value[]
    for idx in reverse(index_tuple)
        idx_bytes = collect(reinterpret(UInt8, [Int32(idx)]))
        idx_val = encode_ConstantOp!(cb, scalar_i32, idx_bytes)
        push!(index_vals, idx_val)
    end

    # Emit ExtractOp
    result = encode_ExtractOp!(cb, output_tile_type, source.v, index_vals)

    CGVal(result, output_tile_type, Tile{elem_type, Tuple{shape_tuple...}}, output_shape)
end

# TODO: cuda_tile.get_global

# cuda_tile.get_num_tile_blocks
@intrinsic get_num_tile_blocks(axis)
tfunc(𝕃, ::typeof(Intrinsics.get_num_tile_blocks), @nospecialize(axis)) = Int32
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.get_num_tile_blocks), args)
    axis = @something get_constant(ctx, args[1]) throw(IRError("get_num_tile_blocks() axis must be a compile-time constant"))
    axis in (0, 1, 2) || throw(IRError("get_num_tile_blocks() axis must be 0, 1, or 2, got $axis"))

    res_type = tile_type!(ctx.tt, I32(ctx.tt), ScalarShape())
    nb_x, nb_y, nb_z = encode_GetNumTileBlocksOp!(ctx.cb, res_type, res_type, res_type)

    CGVal((nb_x, nb_y, nb_z)[axis + 1], res_type, Tile{Int32, Tuple{}})
end

# cuda_tile.get_tile_block_id
@intrinsic get_tile_block_id(axis)
tfunc(𝕃, ::typeof(Intrinsics.get_tile_block_id), @nospecialize(axis)) = Int32
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.get_tile_block_id), args)
    axis = @something get_constant(ctx, args[1]) throw(IRError("get_tile_block_id() axis must be a compile-time constant"))
    axis in (0, 1, 2) || throw(IRError("get_tile_block_id() axis must be 0, 1, or 2, got $axis"))

    res_type = tile_type!(ctx.tt, I32(ctx.tt), ScalarShape())
    bid_x, bid_y, bid_z = encode_GetTileBlockIdOp!(ctx.cb, res_type, res_type, res_type)
    result = (bid_x, bid_y, bid_z)[axis + 1]

    CGVal(result, res_type, Tile{Int32, Tuple{}})
end

# TODO: cuda_tile.global

# cuda_tile.iota
@intrinsic iota(shape, T)
function tfunc(𝕃, ::typeof(Intrinsics.iota), @nospecialize(shape_arg), @nospecialize(type_arg_lat))
    isa(shape_arg, CC.Const) || return nothing
    shape = shape_arg.val
    T = instanceof_tfunc(type_arg_lat)
    T === nothing && return nothing
    return Tile{T, Tuple{shape...}}
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.iota), args)
    cb = ctx.cb
    tt = ctx.tt

    # Extract shape (from Julia) and reverse to Tile IR order
    shape = @something get_constant(ctx, args[1]) throw(IRError("iota() shape must be a compile-time constant"))
    shape isa Tuple || throw(IRError("iota() shape must be a tuple, got $(typeof(shape))"))
    validate_tile_shape(collect(Int, shape), "arange")
    tile_shape = RowMajorShape(ColMajorShape(shape))

    # Extract dtype from Type{T} argument
    elem_type = @something get_constant(ctx, args[2]) throw(IRError("iota() requires a compile-time element type"))

    dtype = julia_to_tile_dtype!(tt, elem_type)
    tile_type = tile_type!(tt, dtype, tile_shape)

    # Emit IotaOp
    result = encode_IotaOp!(cb, tile_type)

    CGVal(result, tile_type, Tile{elem_type, Tuple{shape...}}, tile_shape)
end

# cuda_tile.mmaf, cuda_tile.mmai
@intrinsic mma(a::Tile, b::Tile, acc::Tile)
tfunc(𝕃, ::typeof(Intrinsics.mma), @nospecialize(a), @nospecialize(b), @nospecialize(acc)) = CC.widenconst(acc)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.mma), args)
    cb = ctx.cb

    lhs = emit_value!(ctx, args[1])
    rhs = emit_value!(ctx, args[2])
    acc = emit_value!(ctx, args[3])

    (lhs === nothing || rhs === nothing || acc === nothing) && throw(IRError("Cannot resolve operands for mma()"))

    result = encode_MmaFOp!(cb, acc.type_id, lhs.v, rhs.v, acc.v)

    CGVal(result, acc.type_id, acc.jltype, acc.shape)
end

# TODO: cuda_tile.module

# cuda_tile.offset
@intrinsic offset(base, offsets)
function tfunc(𝕃, ::typeof(Intrinsics.offset), @nospecialize(base), @nospecialize(offsets))
    base_type = CC.widenconst(base)
    base_type <: Ptr || return nothing
    offsets_type = CC.widenconst(offsets)
    offsets_type isa DataType && offsets_type <: Tile || return nothing
    T = eltype(base_type)
    S = offsets_type.parameters[2]
    return Tile{Ptr{T}, S}
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.offset), args)
    cb = ctx.cb
    tt = ctx.tt

    # Get base pointer (arg 1)
    base_ptr_tv = emit_value!(ctx, args[1])
    base_ptr_tv === nothing && throw(IRError("offset: cannot resolve base pointer"))
    base_ptr = base_ptr_tv.v

    # Get offsets tile (arg 2)
    offsets_tv = emit_value!(ctx, args[2])
    offsets_tv === nothing && throw(IRError("offset: cannot resolve offsets tile"))
    offsets = offsets_tv.v
    tile_shape = offsets_tv.shape

    # Get pointer element type from base pointer type (Ptr{T})
    base_ptr_type = CC.widenconst(base_ptr_tv.jltype)
    ptr_elem_type = eltype(base_ptr_type)  # T from Ptr{T}
    elem_dtype = julia_to_tile_dtype!(tt, ptr_elem_type)
    ptr_dtype = pointer_type!(tt, elem_dtype)
    ptr_tile_type = tile_type!(tt, ptr_dtype, tile_shape)

    # Broadcast base pointer to tile shape
    ndims = length(tile_shape)
    if ndims > 0
        ones_shape = RowMajorShape(fill(1, ndims))
        reshaped_ptr_type = tile_type!(tt, ptr_dtype, ones_shape)
        base_ptr_reshaped = encode_ReshapeOp!(cb, reshaped_ptr_type, base_ptr)
        base_ptr_tile = encode_BroadcastOp!(cb, ptr_tile_type, base_ptr_reshaped)
    else
        base_ptr_tile = base_ptr
    end

    # Compute offset pointers: base_ptr + offsets (element offset)
    pointers = encode_OffsetOp!(cb, ptr_tile_type, base_ptr_tile, offsets)

    julia_shape = ColMajorShape(tile_shape)
    result_jltype = Tile{Ptr{ptr_elem_type}, TupleType(julia_shape)}
    CGVal(pointers, ptr_tile_type, result_jltype, tile_shape)
end

# TODO: cudatile.pack

# cuda_tile.permute
@intrinsic permute(tile, perm)
function tfunc(𝕃, ::typeof(Intrinsics.permute), @nospecialize(tile_lat), @nospecialize(perm_arg))
    tile_type = CC.widenconst(tile_lat)
    tile_type <: Tile || return nothing
    isa(perm_arg, CC.Const) || return nothing
    perm = perm_arg.val
    s = size(tile_type)
    isempty(s) && return nothing
    T = eltype(tile_type)
    permuted_shape = ntuple(i -> s[perm[i] + 1], length(perm))
    return Tile{T, Tuple{permuted_shape...}}
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.permute), args)
    cb = ctx.cb
    tt = ctx.tt

    # Get source tile
    source = emit_value!(ctx, args[1])
    source === nothing && throw(IRError("Cannot resolve source operand for permute()"))

    input_shape = source.shape
    isempty(input_shape) && throw(IRError("Cannot determine tile shape for permute()"))

    # Extract permutation (0-indexed Julia order) and transform to Tile IR order
    perm_tuple = @something get_constant(ctx, args[2]) throw(IRError("permute() permutation must be a compile-time constant"))
    perm_tuple isa Tuple || throw(IRError("permute() permutation must be a tuple, got $(typeof(perm_tuple))"))

    julia_perm = collect(Int, perm_tuple)
    n = length(julia_perm)
    # Transform: q[i'] = n-1 - p[n-1-i'] (maps Julia perm to Tile IR perm)
    tileir_perm = [n - 1 - julia_perm[n - i] for i in 0:n-1]

    # Compute output shape based on Tile IR permutation (input_shape is already Tile IR order)
    output_shape = RowMajorShape([input_shape[q + 1] for q in tileir_perm])

    # Get element type
    elem_type = eltype(CC.widenconst(source.jltype))

    # Create output tile type
    dtype = julia_to_tile_dtype!(tt, elem_type)
    output_tile_type = tile_type!(tt, dtype, output_shape)

    # Emit PermuteOp with Tile IR permutation
    result = encode_PermuteOp!(cb, output_tile_type, source.v, tileir_perm)

    julia_output = ColMajorShape(output_shape)
    CGVal(result, output_tile_type, Tile{elem_type, TupleType(julia_output)}, output_shape)
end


# cuda_tile.reduce
@intrinsic reduce(tiles, axis, f, identities)
function tfunc(𝕃, ::typeof(Intrinsics.reduce), @nospecialize(tiles), @nospecialize(axis_arg), @nospecialize args...)
    tuple_type = CC.widenconst(tiles)
    tuple_type isa DataType && tuple_type <: Tuple || return nothing
    isa(axis_arg, CC.Const) || return nothing
    axis = axis_arg.val
    result_params = Any[]
    for p in tuple_type.parameters
        p isa DataType && p <: Tile || return nothing
        T = eltype(p)
        s = size(p)
        isempty(s) && return nothing
        reduced_shape = ntuple(i -> i == axis + 1 ? 1 : s[i], length(s))
        push!(result_params, Tile{T, Tuple{reduced_shape...}})
    end
    return Tuple{result_params...}
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.reduce), args)
    emit_reduce!(ctx, args)
end
function emit_reduce!(ctx::CGCtx, args)
    cb = ctx.cb
    tt = ctx.tt

    # Extract tile CGVals from the tuple argument
    first_tv = emit_value!(ctx, args[1])
    first_tv === nothing && throw(IRError("Cannot resolve input for reduction"))
    first_tv.tuple === nothing && throw(IRError("reduce() requires a tuple of tiles (got $(first_tv.jltype))"))
    tile_tvs = CGVal[let tv = emit_value!(ctx, ref)
        tv === nothing && throw(IRError("Cannot resolve tile operand in reduce"))
        tv
    end for ref in first_tv.tuple]
    N = length(tile_tvs)

    julia_axis = @something get_constant(ctx, args[2]) throw(IRError("reduce() axis must be a compile-time constant"))
    func = @something get_constant(ctx, args[3]) throw(IRError("reduce() combiner function must be a compile-time constant"))

    id_tv = emit_value!(ctx, args[4])
    id_tv === nothing && throw(IRError("Cannot resolve identity tuple for reduce"))
    id_tv.tuple !== nothing || throw(IRError("reduce() identities must be a tuple of compile-time constants"))
    identity_vals = Any[@something(get_constant(ctx, ref),
                                   throw(IRError("reduce() identity values must be compile-time constants")))
                        for ref in id_tv.tuple]

    # Get shapes from the first tile (already in Tile IR order)
    input_shape = tile_tvs[1].shape
    isempty(input_shape) && throw(IRError("Cannot reduce scalar tile"))

    # Flip axis from Julia 0-indexed to Tile IR order
    ndim = length(input_shape)
    axis = ndim - 1 - julia_axis

    # ReduceOp removes the dimension; we'll reshape after to reintroduce it as size 1
    reduced_shape = RowMajorShape([input_shape[i] for i in eachindex(input_shape) if i != axis + 1])

    # Build per-operand types and values
    elem_types = Type[]
    dtypes = TypeId[]
    reduced_tile_types = TypeId[]
    scalar_tile_types = TypeId[]
    operand_values = Value[]
    identities = IdentityVal[]

    for (k, tv) in enumerate(tile_tvs)
        etype = eltype(CC.widenconst(tv.jltype))
        push!(elem_types, etype)
        dtype = julia_to_tile_dtype!(tt, etype)
        push!(dtypes, dtype)
        push!(reduced_tile_types, tile_type!(tt, dtype, reduced_shape))
        push!(scalar_tile_types, tile_type!(tt, dtype, ScalarShape()))
        push!(operand_values, tv.v::Value)
        push!(identities, make_identity_val(identity_vals[k], dtype, etype))
    end

    # Body arg types: for each operand, (acc_type, elem_type) interleaved
    body_arg_types = Type[]
    body_type_ids = TypeId[]
    for k in 1:N
        push!(body_arg_types, elem_types[k])
        push!(body_arg_types, elem_types[k])
        push!(body_type_ids, scalar_tile_types[k])
        push!(body_type_ids, scalar_tile_types[k])
    end

    # Emit ReduceOp with compiled combiner body
    results = encode_ReduceOp!(cb, reduced_tile_types, operand_values,
                               axis, identities, scalar_tile_types) do block_args
        emit_subprogram!(ctx, func, body_arg_types, block_args, body_type_ids)
    end

    # Julia semantics: reintroduce reduced dimension as size 1 via ReshapeOp
    output_shape = copy(input_shape)
    output_shape[axis + 1] = 1

    julia_output = ColMajorShape(output_shape)
    reshaped_values = Value[]
    component_types = Type[]
    for (k, res) in enumerate(results)
        out_type = tile_type!(tt, dtypes[k], output_shape)
        reshaped_val = encode_ReshapeOp!(cb, out_type, res)
        push!(reshaped_values, reshaped_val)
        push!(component_types, Tile{elem_types[k], TupleType(julia_output)})
    end

    # Return multi-value CGVal (tuple)
    jltype = Tuple{component_types...}
    return CGVal(reshaped_values, jltype)
end

"""
    to_uint128(value)

Convert an integer value to UInt128 for storage in IntegerIdentityVal.
For signed types, this returns the two's complement bit representation.
"""
to_uint128(value::Bool) = UInt128(value)
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
@intrinsic reshape(tile, shape)
function tfunc(𝕃, ::typeof(Intrinsics.reshape), @nospecialize(tile_lat), @nospecialize(shape_arg))
    tile_type = CC.widenconst(tile_lat)
    tile_type <: Tile || return nothing
    isa(shape_arg, CC.Const) || return nothing
    shape = shape_arg.val
    T = eltype(tile_type)
    return Tile{T, Tuple{shape...}}
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.reshape), args)
    cb = ctx.cb
    tt = ctx.tt

    # Get source tile
    source = emit_value!(ctx, args[1])
    source === nothing && throw(IRError("Cannot resolve source operand for reshape()"))

    # Extract target shape (from Julia) and reverse to Tile IR order
    target_shape_tuple = @something get_constant(ctx, args[2]) throw(IRError("reshape() shape must be a compile-time constant"))
    target_shape_tuple isa Tuple || throw(IRError("reshape() shape must be a tuple, got $(typeof(target_shape_tuple))"))
    validate_tile_shape(collect(Int, target_shape_tuple), "reshape")
    target_shape = RowMajorShape(ColMajorShape(target_shape_tuple))

    # Get element type
    elem_type = eltype(CC.widenconst(source.jltype))
    dtype = julia_to_tile_dtype!(tt, elem_type)

    # Tile IR shapes are already in row-major order, so ReshapeOp's row-major element
    # ordering matches directly. No permutes needed!
    result_type_id = tile_type!(tt, dtype, target_shape)
    result = encode_ReshapeOp!(cb, result_type_id, source.v)

    CGVal(result, result_type_id, Tile{elem_type, Tuple{target_shape_tuple...}}, target_shape)
end

# cuda_tile.scan
@intrinsic scan(tiles, axis, f, identities, reverse=false)
function tfunc(𝕃, ::typeof(Intrinsics.scan), @nospecialize(tiles), @nospecialize args...)
    tuple_type = CC.widenconst(tiles)
    tuple_type isa DataType && tuple_type <: Tuple || return nothing
    result_params = Any[]
    for p in tuple_type.parameters
        p isa DataType && p <: Tile || return nothing
        push!(result_params, p)
    end
    return Tuple{result_params...}
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.scan), args)
    cb = ctx.cb
    tt = ctx.tt

    # Extract tile CGVals from the tuple argument
    first_tv = emit_value!(ctx, args[1])
    first_tv === nothing && throw(IRError("Cannot resolve input for scan"))
    first_tv.tuple === nothing && throw(IRError("scan() requires a tuple of tiles (got $(first_tv.jltype))"))
    tile_tvs = CGVal[let tv = emit_value!(ctx, ref)
        tv === nothing && throw(IRError("Cannot resolve tile operand in scan"))
        tv
    end for ref in first_tv.tuple]
    N = length(tile_tvs)

    julia_axis = @something get_constant(ctx, args[2]) throw(IRError("scan() axis must be a compile-time constant"))
    func = @something get_constant(ctx, args[3]) throw(IRError("scan() combiner function must be a compile-time constant"))

    id_tv = emit_value!(ctx, args[4])
    id_tv === nothing && throw(IRError("Cannot resolve identity tuple for scan"))
    id_tv.tuple !== nothing || throw(IRError("scan() identities must be a tuple of compile-time constants"))
    identity_vals = Any[@something(get_constant(ctx, ref),
                                   throw(IRError("scan() identity values must be compile-time constants")))
                        for ref in id_tv.tuple]

    reverse = false
    if length(args) >= 5
        reverse_val = @something get_constant(ctx, args[5]) false
        reverse = reverse_val === true
    end

    # Get shapes from the first tile (already in Tile IR order)
    input_shape = tile_tvs[1].shape
    isempty(input_shape) && throw(IRError("Cannot scan scalar tile"))

    # Flip axis from Julia 0-indexed to Tile IR order
    ndim = length(input_shape)
    axis = ndim - 1 - julia_axis

    # For scan, output shape is same as input shape
    output_shape = copy(input_shape)

    # Build per-operand types and values
    elem_types = Type[]
    dtypes = TypeId[]
    output_tile_types = TypeId[]
    scalar_tile_types = TypeId[]
    operand_values = Value[]
    identities = IdentityVal[]

    for (k, tv) in enumerate(tile_tvs)
        etype = eltype(CC.widenconst(tv.jltype))
        push!(elem_types, etype)
        dtype = julia_to_tile_dtype!(tt, etype)
        push!(dtypes, dtype)
        push!(output_tile_types, tile_type!(tt, dtype, output_shape))
        push!(scalar_tile_types, tile_type!(tt, dtype, ScalarShape()))
        push!(operand_values, tv.v::Value)
        push!(identities, make_identity_val(identity_vals[k], dtype, etype))
    end

    # Body arg types: for each operand, (acc_type, elem_type) interleaved
    body_arg_types = Type[]
    body_type_ids = TypeId[]
    for k in 1:N
        push!(body_arg_types, elem_types[k])
        push!(body_arg_types, elem_types[k])
        push!(body_type_ids, scalar_tile_types[k])
        push!(body_type_ids, scalar_tile_types[k])
    end

    # Emit ScanOp with compiled combiner body
    results = encode_ScanOp!(cb, output_tile_types, operand_values,
                             axis, reverse, identities, scalar_tile_types) do block_args
        emit_subprogram!(ctx, func, body_arg_types, block_args, body_type_ids)
    end

    # Return multi-value CGVal (tuple)
    julia_output = ColMajorShape(output_shape)
    component_types = Type[]
    for k in 1:N
        push!(component_types, Tile{elem_types[k], TupleType(julia_output)})
    end
    jltype = Tuple{component_types...}
    return CGVal(results, jltype)
end

# cuda_tile.select
@intrinsic select(cond::Bool, x::T, y::T) where {T}# = Core.ifelse(cond, x, y)
@intrinsic select(cond::Tile{Bool}, x::T, y::T) where {T}
function tfunc(𝕃, ::typeof(Intrinsics.select), @nospecialize(cond), @nospecialize(x), @nospecialize(y))
    if isa(cond, CC.Const)
        if cond.val === true
            return x
        elseif cond.val === false
            return y
        else
            return Union{}
        end
    end
    return CC.tmerge(𝕃, x, y)
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.select), args)
    cb = ctx.cb

    cond_tv = emit_value!(ctx, args[1])
    x_tv = emit_value!(ctx, args[2])
    y_tv = emit_value!(ctx, args[3])

    (cond_tv === nothing || x_tv === nothing || y_tv === nothing) &&
        throw(IRError("Cannot resolve operands for select()"))

    result = encode_SelectOp!(cb, x_tv.type_id, cond_tv.v, x_tv.v, y_tv.v)

    CGVal(result, x_tv.type_id, x_tv.jltype, x_tv.shape)
end

# cuda_tile.to_scalar / cuda_tile.from_scalar
# Interpretation-only intrinsics for Julia's broadcast overlay dispatch.
# to_scalar changes jltype from Tile{T,S} to T; from_scalar restores it.
# Both are eliminated by scalar_elim_pass! before codegen — no emit needed.
@intrinsic to_scalar(tile)
@intrinsic from_scalar(x, S)
function tfunc(𝕃, ::typeof(Intrinsics.from_scalar), @nospecialize(x), @nospecialize(S_lat))
    T = CC.widenconst(x)
    S = instanceof_tfunc(S_lat)
    S === nothing && return nothing
    return Tile{T, S}
end
function tfunc(𝕃, ::typeof(Intrinsics.to_scalar), @nospecialize(tile_lat))
    tile_type = CC.widenconst(tile_lat)
    tile_type <: Tile || return nothing
    return eltype(tile_type)
end

# TODO: cuda_tile.unpack
