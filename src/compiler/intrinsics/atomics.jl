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

# cuda_tile.atomic_cas_tko
@eval Intrinsics begin
    """
        atomic_cas(array, index, expected, desired, memory_order, memory_scope)

    Atomic compare-and-swap at 0-indexed position.
    Returns the original value.
    Compiled to cuda_tile.atomic_cas_tko.
    """
    @noinline function atomic_cas(array::TileArray{T, N}, index, expected, desired,
                                   memory_order::Int, memory_scope::Int) where {T, N}
        donotdelete(array, index, expected, desired)
        compilerbarrier(:const, zero(T))::T
    end
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.atomic_cas), args)
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

    # Get index and create pointer type
    index_tv = emit_value!(ctx, args[2])
    ptr_type = pointer_type!(tt, dtype)
    ptr_tile_type = tile_type!(tt, ptr_type, Int[])

    # Compute pointer using OffsetOp (handles any integer index type)
    pointers = encode_OffsetOp!(cb, ptr_tile_type, array_val, index_tv.v)

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

# cuda_tile.atomic_rmw_tko (shared helper for atomic RMW operations)
function emit_atomic_rmw!(ctx::CGCtx, args::AbstractVector, mode::AtomicRMWMode)
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

    # Get index and create pointer type
    index_tv = emit_value!(ctx, args[2])
    ptr_type = pointer_type!(tt, dtype)
    ptr_tile_type = tile_type!(tt, ptr_type, Int[])

    # Compute pointer using OffsetOp (handles any integer index type)
    pointers = encode_OffsetOp!(cb, ptr_tile_type, array_val, index_tv.v)

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

# cuda_tile.atomic_rmw_tko with XCHG
@eval Intrinsics begin
    """
        atomic_xchg(array, index, val, memory_order, memory_scope)

    Atomic exchange at 0-indexed position.
    Returns the original value.
    Compiled to cuda_tile.atomic_rmw_tko with XCHG.
    """
    @noinline function atomic_xchg(array::TileArray{T, N}, index, val,
                                    memory_order::Int, memory_scope::Int) where {T, N}
        donotdelete(array, index, val)
        compilerbarrier(:const, zero(T))
    end
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.atomic_xchg), args)
    emit_atomic_rmw!(ctx, args, AtomicXCHG)
end

# cuda_tile.atomic_rmw_tko with ADD
@eval Intrinsics begin
    """
        atomic_add(array, index, val, memory_order, memory_scope)

    Atomic addition at 0-indexed position.
    Returns the original value.
    Compiled to cuda_tile.atomic_rmw_tko with ADD.
    """
    @noinline function atomic_add(array::TileArray{T, N}, index, val,
                                   memory_order::Int, memory_scope::Int) where {T, N}
        donotdelete(array, index, val)
        compilerbarrier(:const, zero(T))
    end
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.atomic_add), args)
    emit_atomic_rmw!(ctx, args, AtomicADD)
end

# ============================================================================
# Tile-wise atomic operations
# These take pre-computed pointer tiles, value tiles, and masks.
# Used by the public API for tile-indexed atomic operations.
# ============================================================================

# cuda_tile.atomic_cas_tko with tile pointers
@eval Intrinsics begin
    """
        atomic_cas_tile(ptr_tile, expected, desired, mask, memory_order, memory_scope)

    Tile-wise atomic compare-and-swap.
    Operates on a tile of pointers with a tile of expected/desired values.
    Mask controls which elements are active (bounds checking).
    Returns a tile of original values.
    """
    @noinline function atomic_cas_tile(ptr_tile::Tile, expected::Tile{T, S},
                                        desired::Tile{T, S}, mask::Tile,
                                        memory_order::Int, memory_scope::Int) where {T, S}
        donotdelete(ptr_tile, expected, desired, mask)
        compilerbarrier(:const, expected)::Tile{T, S}
    end
end

function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.atomic_cas_tile), args)
    cb = ctx.cb
    tt = ctx.tt

    # args: (ptr_tile, expected, desired, mask, memory_order, memory_scope)
    ptr_tv = emit_value!(ctx, args[1])
    ptr_tv === nothing && error("atomic_cas_tile requires ptr_tile")
    expected_tv = emit_value!(ctx, args[2])
    expected_tv === nothing && error("atomic_cas_tile requires expected value")
    desired_tv = emit_value!(ctx, args[3])
    desired_tv === nothing && error("atomic_cas_tile requires desired value")
    mask_tv = emit_value!(ctx, args[4])
    mask_tv === nothing && error("atomic_cas_tile requires mask")

    # Get memory order and scope
    memory_order = @something get_constant(ctx, args[5]) error("atomic_cas_tile requires constant memory_order")
    memory_scope = @something get_constant(ctx, args[6]) error("atomic_cas_tile requires constant memory_scope")

    # Get shape and element type from expected tile
    shape = expected_tv.shape
    elem_type = expected_tv.jltype.parameters[1]  # T from Tile{T, S}

    # Create result type (tile with same shape as inputs)
    dtype = julia_to_tile_dtype!(tt, elem_type)
    result_tile_type = tile_type!(tt, dtype, collect(shape))
    token_type = Token(tt)

    # Emit atomic CAS with mask
    mem_ordering = memory_order_to_semantics(memory_order)
    mem_scope = memory_scope_to_scope(memory_scope)

    old_val, new_token = encode_AtomicCASPtrOp!(cb, result_tile_type, token_type,
                                                 ptr_tv.v, expected_tv.v, desired_tv.v;
                                                 mask=mask_tv.v,
                                                 token=ctx.token,
                                                 memory_ordering=mem_ordering,
                                                 memory_scope=mem_scope)
    ctx.token = new_token

    # Return Tile type with the same shape
    CGVal(old_val, result_tile_type, Tile{elem_type, Tuple(shape)}, collect(shape))
end

# Shared helper for tile-wise atomic RMW operations
function emit_atomic_rmw_tile!(ctx::CGCtx, args::AbstractVector, mode::AtomicRMWMode)
    cb = ctx.cb
    tt = ctx.tt

    # args: (ptr_tile, val, mask, memory_order, memory_scope)
    ptr_tv = emit_value!(ctx, args[1])
    ptr_tv === nothing && error("atomic RMW tile requires ptr_tile")
    val_tv = emit_value!(ctx, args[2])
    val_tv === nothing && error("atomic RMW tile requires value")
    mask_tv = emit_value!(ctx, args[3])
    mask_tv === nothing && error("atomic RMW tile requires mask")

    # Get memory order and scope
    memory_order = @something get_constant(ctx, args[4]) error("atomic RMW tile requires constant memory_order")
    memory_scope = @something get_constant(ctx, args[5]) error("atomic RMW tile requires constant memory_scope")

    # Get shape and element type from value tile
    shape = val_tv.shape
    elem_type = val_tv.jltype.parameters[1]  # T from Tile{T, S}

    # Create result type (tile with same shape as inputs)
    dtype = julia_to_tile_dtype!(tt, elem_type)
    result_tile_type = tile_type!(tt, dtype, collect(shape))
    token_type = Token(tt)

    # Use float add mode for floating point types
    actual_mode = mode
    if mode == AtomicADD && elem_type <: AbstractFloat
        actual_mode = AtomicADDF
    end

    # Emit atomic RMW with mask
    mem_ordering = memory_order_to_semantics(memory_order)
    mem_scope = memory_scope_to_scope(memory_scope)

    old_val, new_token = encode_AtomicRMWPtrOp!(cb, result_tile_type, token_type,
                                                 ptr_tv.v, val_tv.v, actual_mode;
                                                 mask=mask_tv.v,
                                                 token=ctx.token,
                                                 memory_ordering=mem_ordering,
                                                 memory_scope=mem_scope)
    ctx.token = new_token

    # Return Tile type with the same shape
    CGVal(old_val, result_tile_type, Tile{elem_type, Tuple(shape)}, collect(shape))
end

# cuda_tile.atomic_rmw_tko with XCHG (tile version)
@eval Intrinsics begin
    """
        atomic_xchg_tile(ptr_tile, val, mask, memory_order, memory_scope)

    Tile-wise atomic exchange.
    Operates on a tile of pointers with a tile of values.
    Mask controls which elements are active (bounds checking).
    Returns a tile of original values.
    """
    @noinline function atomic_xchg_tile(ptr_tile::Tile, val::Tile{T, S}, mask::Tile,
                                         memory_order::Int, memory_scope::Int) where {T, S}
        donotdelete(ptr_tile, val, mask)
        compilerbarrier(:const, val)::Tile{T, S}
    end
end

function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.atomic_xchg_tile), args)
    emit_atomic_rmw_tile!(ctx, args, AtomicXCHG)
end

# cuda_tile.atomic_rmw_tko with ADD (tile version)
@eval Intrinsics begin
    """
        atomic_add_tile(ptr_tile, val, mask, memory_order, memory_scope)

    Tile-wise atomic addition.
    Operates on a tile of pointers with a tile of values.
    Mask controls which elements are active (bounds checking).
    Returns a tile of original values.
    """
    @noinline function atomic_add_tile(ptr_tile::Tile, val::Tile{T, S}, mask::Tile,
                                        memory_order::Int, memory_scope::Int) where {T, S}
        donotdelete(ptr_tile, val, mask)
        compilerbarrier(:const, val)::Tile{T, S}
    end
end

function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.atomic_add_tile), args)
    emit_atomic_rmw_tile!(ctx, args, AtomicADD)
end
