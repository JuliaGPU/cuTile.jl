# atomics

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


## cuda_tile.atomic_cas_tko

@eval Intrinsics begin
    """
        atomic_cas(array, index, expected, desired, memory_order, memory_scope)

    Atomic compare-and-swap at 0-indexed position.
    Returns the original value.
    Compiled to cuda_tile.atomic_cas_tko.
    """
    @noinline function atomic_cas(array::TileArray{T, N}, index, expected, desired,
                                   memory_order::Int, memory_scope::Int) where {T, N}
        Base.donotdelete(array, index, expected, desired)
        Base.inferencebarrier(zero(T))::T
    end
end

function emit_intrinsic!(ctx::CodegenContext, ::typeof(Intrinsics.atomic_cas), args, @nospecialize(result_type))
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


## cuda_tile.atomic_rmw_tko

@eval Intrinsics begin
    """
        atomic_xchg(array, index, val, memory_order, memory_scope)

    Atomic exchange at 0-indexed position.
    Returns the original value.
    Compiled to cuda_tile.atomic_rmw_tko with XCHG.
    """
    @noinline function atomic_xchg(array::TileArray{T, N}, index, val,
                                    memory_order::Int, memory_scope::Int) where {T, N}
        Base.donotdelete(array, index, val)
        Base.inferencebarrier(zero(T))::T
    end

    """
        atomic_add(array, index, val, memory_order, memory_scope)

    Atomic addition at 0-indexed position.
    Returns the original value.
    Compiled to cuda_tile.atomic_rmw_tko with ADD.
    """
    @noinline function atomic_add(array::TileArray{T, N}, index, val,
                                   memory_order::Int, memory_scope::Int) where {T, N}
        Base.donotdelete(array, index, val)
        Base.inferencebarrier(zero(T))::T
    end
end

function emit_intrinsic!(ctx::CodegenContext, ::typeof(Intrinsics.atomic_xchg), args, @nospecialize(result_type))
    emit_atomic_rmw!(ctx, args, result_type, AtomicXCHG)
end

function emit_intrinsic!(ctx::CodegenContext, ::typeof(Intrinsics.atomic_add), args, @nospecialize(result_type))
    emit_atomic_rmw!(ctx, args, result_type, AtomicADD)
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
