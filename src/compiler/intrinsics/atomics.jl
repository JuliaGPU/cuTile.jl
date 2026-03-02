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

"""
    atomic_tfunc(ptrs) -> Type

Shared tfunc for atomic operations (add, xchg, cas).
Returns raw T for 0D pointer tiles, Tile{T, S} for N-D.
"""
function atomic_tfunc(𝕃, @nospecialize(ptrs), @nospecialize args...)
    ptrs_type = CC.widenconst(ptrs)
    ptrs_type isa DataType && ptrs_type <: Tile || return nothing
    ptr_type = eltype(ptrs_type)
    ptr_type <: Ptr || return nothing
    T = eltype(ptr_type)
    S = ptrs_type.parameters[2]
    S === Tuple{} && return T
    return Tile{T, S}
end

# cuda_tile.atomic_cas_tko
@intrinsic atomic_cas(ptr_tile, expected, desired, mask, memory_order, memory_scope)
function tfunc(𝕃, ::typeof(Intrinsics.atomic_cas), @nospecialize(ptrs), @nospecialize args...)
    atomic_tfunc(𝕃, ptrs, args...)
end
efunc(::typeof(Intrinsics.atomic_cas), effects::CC.Effects) =
    CC.Effects(effects; effect_free=CC.ALWAYS_FALSE)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.atomic_cas), args)
    cb = ctx.cb
    tt = ctx.tt

    # args: (ptr_tile, expected, desired, mask, memory_order, memory_scope)
    ptr_tv = emit_value!(ctx, args[1])
    ptr_tv === nothing && throw(IRError("atomic CAS requires ptr_tile"))
    expected_tv = emit_value!(ctx, args[2])
    expected_tv === nothing && throw(IRError("atomic CAS requires expected value"))
    desired_tv = emit_value!(ctx, args[3])
    desired_tv === nothing && throw(IRError("atomic CAS requires desired value"))

    # Check if mask is provided (ghost Nothing = no mask)
    has_mask = get_constant(ctx, args[4]) !== nothing

    memory_order = @something get_constant(ctx, args[5]) throw(IRError("atomic CAS requires constant memory_order"))
    memory_scope = @something get_constant(ctx, args[6]) throw(IRError("atomic CAS requires constant memory_scope"))

    shape = ptr_tv.shape

    # Get element type from pointer tile: Tile{Ptr{T}, S} -> T
    ptrs_type = CC.widenconst(ptr_tv.jltype)
    ptr_type = eltype(ptrs_type)
    elem_type = eltype(ptr_type)

    dtype = julia_to_tile_dtype!(tt, elem_type)
    result_tile_type = tile_type!(tt, dtype, collect(shape))
    token_type = Token(tt)

    # Emit atomic CAS
    mem_ordering = memory_order_to_semantics(memory_order)
    mem_scope = memory_scope_to_scope(memory_scope)

    if has_mask
        mask_tv = emit_value!(ctx, args[4])
        mask_tv === nothing && throw(IRError("atomic CAS: cannot resolve mask"))
        old_val, new_token = encode_AtomicCASPtrOp!(cb, result_tile_type, token_type,
                                                     ptr_tv.v, expected_tv.v, desired_tv.v;
                                                     mask=mask_tv.v,
                                                     token=ctx.token,
                                                     memory_ordering=mem_ordering,
                                                     memory_scope=mem_scope)
    else
        old_val, new_token = encode_AtomicCASPtrOp!(cb, result_tile_type, token_type,
                                                     ptr_tv.v, expected_tv.v, desired_tv.v;
                                                     token=ctx.token,
                                                     memory_ordering=mem_ordering,
                                                     memory_scope=mem_scope)
    end
    ctx.token = new_token

    # Return type depends on shape: raw T for 0D, Tile{T, S} for N-D
    if isempty(shape)
        CGVal(old_val, result_tile_type, elem_type, Int[])
    else
        CGVal(old_val, result_tile_type, Tile{elem_type, Tuple{shape...}}, collect(shape))
    end
end

# cuda_tile.atomic_rmw_tko (shared helper for atomic RMW operations)
function emit_atomic_rmw!(ctx::CGCtx, args::AbstractVector, mode::AtomicRMWMode)
    cb = ctx.cb
    tt = ctx.tt

    # args: (ptr_tile, val, mask, memory_order, memory_scope)
    ptr_tv = emit_value!(ctx, args[1])
    ptr_tv === nothing && throw(IRError("atomic RMW requires ptr_tile"))
    val_tv = emit_value!(ctx, args[2])
    val_tv === nothing && throw(IRError("atomic RMW requires value"))

    # Check if mask is provided (ghost Nothing = no mask)
    has_mask = get_constant(ctx, args[3]) !== nothing

    # Get memory order and scope from args
    memory_order = @something get_constant(ctx, args[4]) throw(IRError("atomic RMW requires constant memory_order"))
    memory_scope = @something get_constant(ctx, args[5]) throw(IRError("atomic RMW requires constant memory_scope"))

    shape = ptr_tv.shape

    # Get element type from pointer tile: Tile{Ptr{T}, S} -> T
    ptrs_type = CC.widenconst(ptr_tv.jltype)
    ptr_type = eltype(ptrs_type)
    elem_type = eltype(ptr_type)

    # Create result type
    dtype = julia_to_tile_dtype!(tt, elem_type)
    result_tile_type = tile_type!(tt, dtype, collect(shape))
    token_type = Token(tt)

    # Use float add mode for floating point types
    actual_mode = mode
    if mode == AtomicADD && elem_type <: AbstractFloat
        actual_mode = AtomicADDF
    end

    # Emit atomic RMW
    mem_ordering = memory_order_to_semantics(memory_order)
    mem_scope = memory_scope_to_scope(memory_scope)

    if has_mask
        mask_tv = emit_value!(ctx, args[3])
        mask_tv === nothing && throw(IRError("atomic RMW: cannot resolve mask"))
        old_val, new_token = encode_AtomicRMWPtrOp!(cb, result_tile_type, token_type,
                                                     ptr_tv.v, val_tv.v, actual_mode;
                                                     mask=mask_tv.v,
                                                     token=ctx.token,
                                                     memory_ordering=mem_ordering,
                                                     memory_scope=mem_scope)
    else
        old_val, new_token = encode_AtomicRMWPtrOp!(cb, result_tile_type, token_type,
                                                     ptr_tv.v, val_tv.v, actual_mode;
                                                     token=ctx.token,
                                                     memory_ordering=mem_ordering,
                                                     memory_scope=mem_scope)
    end
    ctx.token = new_token

    # Return type depends on shape: raw T for 0D, Tile{T, S} for N-D
    if isempty(shape)
        CGVal(old_val, result_tile_type, elem_type, Int[])
    else
        CGVal(old_val, result_tile_type, Tile{elem_type, Tuple{shape...}}, collect(shape))
    end
end

# cuda_tile.atomic_rmw_tko with XCHG
@intrinsic atomic_xchg(ptr_tile, val, mask, memory_order, memory_scope)
tfunc(𝕃, ::typeof(Intrinsics.atomic_xchg), @nospecialize args...) = atomic_tfunc(𝕃, args...)
efunc(::typeof(Intrinsics.atomic_xchg), effects::CC.Effects) =
    CC.Effects(effects; effect_free=CC.ALWAYS_FALSE)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.atomic_xchg), args)
    emit_atomic_rmw!(ctx, args, AtomicXCHG)
end

# cuda_tile.atomic_rmw_tko with ADD
@intrinsic atomic_add(ptr_tile, val, mask, memory_order, memory_scope)
tfunc(𝕃, ::typeof(Intrinsics.atomic_add), @nospecialize args...) = atomic_tfunc(𝕃, args...)
efunc(::typeof(Intrinsics.atomic_add), effects::CC.Effects) =
    CC.Effects(effects; effect_free=CC.ALWAYS_FALSE)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.atomic_add), args)
    emit_atomic_rmw!(ctx, args, AtomicADD)
end
