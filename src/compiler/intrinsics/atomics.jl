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
@intrinsic atomic_cas(array, index, expected, desired,
                      memory_order, memory_scope)
tfunc(𝕃, ::typeof(Intrinsics.atomic_cas), @nospecialize(array), @nospecialize args...) = eltype(CC.widenconst(array))
efunc(::typeof(Intrinsics.atomic_cas), effects::CC.Effects) =
    CC.Effects(effects; effect_free=CC.ALWAYS_FALSE)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.atomic_cas), args)
    cb = ctx.cb
    tt = ctx.tt

    # args: (array, index, expected, desired, memory_order, memory_scope)
    array_arg = args[1]

    # Get array info
    arg_idx = extract_argument_index(array_arg)
    is_tilearray = arg_idx !== nothing && is_destructured_arg(ctx, arg_idx)

    if !is_tilearray
        throw(IRError("atomic_cas requires a TileArray argument"))
    end

    ptr_vals = get_arg_flat_values(ctx, arg_idx, :ptr)
    isempty(ptr_vals) && throw(IRError("Cannot get ptr from TileArray argument"))
    array_val = ptr_vals[1]
    tilearray_type = get_arg_type(ctx, arg_idx)
    elem_type = eltype(tilearray_type)

    # Get expected and desired values
    expected_tv = emit_value!(ctx, args[3])
    expected_tv === nothing && throw(IRError("atomic_cas requires expected value"))
    desired_tv = emit_value!(ctx, args[4])
    desired_tv === nothing && throw(IRError("atomic_cas requires desired value"))

    # Get memory order and scope from args
    memory_order = @something get_constant(ctx, args[5]) throw(IRError("atomic_cas requires constant memory_order"))
    memory_scope = @something get_constant(ctx, args[6]) throw(IRError("atomic_cas requires constant memory_scope"))

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
        throw(IRError("atomic operations require a TileArray argument"))
    end

    ptr_vals = get_arg_flat_values(ctx, arg_idx, :ptr)
    isempty(ptr_vals) && throw(IRError("Cannot get ptr from TileArray argument"))
    array_val = ptr_vals[1]
    tilearray_type = get_arg_type(ctx, arg_idx)
    elem_type = eltype(tilearray_type)

    # Get update value
    val_tv = emit_value!(ctx, args[3])
    val_tv === nothing && throw(IRError("atomic operation requires value"))

    # Get memory order and scope from args
    memory_order = @something get_constant(ctx, args[4]) throw(IRError("atomic operation requires constant memory_order"))
    memory_scope = @something get_constant(ctx, args[5]) throw(IRError("atomic operation requires constant memory_scope"))

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
@intrinsic atomic_xchg(array, index, val, memory_order, memory_scope)
tfunc(𝕃, ::typeof(Intrinsics.atomic_xchg), @nospecialize(array), @nospecialize args...) = eltype(CC.widenconst(array))
efunc(::typeof(Intrinsics.atomic_xchg), effects::CC.Effects) =
    CC.Effects(effects; effect_free=CC.ALWAYS_FALSE)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.atomic_xchg), args)
    emit_atomic_rmw!(ctx, args, AtomicXCHG)
end

# cuda_tile.atomic_rmw_tko with ADD
@intrinsic atomic_add(array, index, val,
                      memory_order, memory_scope)
tfunc(𝕃, ::typeof(Intrinsics.atomic_add), @nospecialize(array), @nospecialize args...) = eltype(CC.widenconst(array))
efunc(::typeof(Intrinsics.atomic_add), effects::CC.Effects) =
    CC.Effects(effects; effect_free=CC.ALWAYS_FALSE)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.atomic_add), args)
    emit_atomic_rmw!(ctx, args, AtomicADD)
end
