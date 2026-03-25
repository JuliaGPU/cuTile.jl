# atomics


"""
    atomic_tfunc(ptrs) -> Type

Shared tfunc for atomic operations (add, xchg, cas).
Always returns Tile{T, S}, even for 0D (S = Tuple{}).
"""
function atomic_tfunc(𝕃, @nospecialize(ptrs), @nospecialize args...)
    ptrs_type = CC.widenconst(ptrs)
    ptrs_type isa DataType && ptrs_type <: Tile || return nothing
    ptr_type = eltype(ptrs_type)
    ptr_type <: Ptr || return nothing
    T = eltype(ptr_type)
    S = ptrs_type.parameters[2]
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

    mask_tv, has_mask = emit_optional_mask(ctx, args, 4)

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
    mem_ordering = convert_enum(MemoryOrderingSemantics, memory_order)
    mem_scope = convert_enum(MemoryScope, memory_scope)

    old_val, new_token = if has_mask
        encode_AtomicCASPtrOp!(cb, result_tile_type, token_type,
                               ptr_tv.v, expected_tv.v, desired_tv.v;
                               mask=mask_tv.v,
                               token=ctx.token,
                               memory_ordering=mem_ordering,
                               memory_scope=mem_scope)
    else
        encode_AtomicCASPtrOp!(cb, result_tile_type, token_type,
                               ptr_tv.v, expected_tv.v, desired_tv.v;
                               token=ctx.token,
                               memory_ordering=mem_ordering,
                               memory_scope=mem_scope)
    end
    ctx.token = new_token

    julia_shape = ColMajorShape(shape)
    CGVal(old_val, result_tile_type, Tile{elem_type, TupleType(julia_shape)}, shape)
end

# cuda_tile.atomic_rmw_tko (shared helper for atomic RMW operations)
function emit_atomic_rmw!(ctx::CGCtx, args::AbstractVector, mode::AtomicRMWMode.T)
    cb = ctx.cb
    tt = ctx.tt

    # args: (ptr_tile, val, mask, memory_order, memory_scope)
    ptr_tv = emit_value!(ctx, args[1])
    ptr_tv === nothing && throw(IRError("atomic RMW requires ptr_tile"))
    val_tv = emit_value!(ctx, args[2])
    val_tv === nothing && throw(IRError("atomic RMW requires value"))

    mask_tv, has_mask = emit_optional_mask(ctx, args, 3)

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
    if mode == AtomicRMWMode.ADD && elem_type <: AbstractFloat
        actual_mode = AtomicRMWMode.ADDF
    end

    # Emit atomic RMW
    mem_ordering = convert_enum(MemoryOrderingSemantics, memory_order)
    mem_scope = convert_enum(MemoryScope, memory_scope)

    old_val, new_token = if has_mask
        encode_AtomicRMWPtrOp!(cb, result_tile_type, token_type,
                                ptr_tv.v, val_tv.v, actual_mode;
                                mask=mask_tv.v,
                                token=ctx.token,
                                memory_ordering=mem_ordering,
                                memory_scope=mem_scope)
    else
        encode_AtomicRMWPtrOp!(cb, result_tile_type, token_type,
                                ptr_tv.v, val_tv.v, actual_mode;
                                token=ctx.token,
                                memory_ordering=mem_ordering,
                                memory_scope=mem_scope)
    end
    ctx.token = new_token

    julia_shape = ColMajorShape(shape)
    CGVal(old_val, result_tile_type, Tile{elem_type, TupleType(julia_shape)}, shape)
end

# cuda_tile.atomic_rmw_tko variants
for (op, mode) in ((:xchg, AtomicRMWMode.XCHG), (:add, AtomicRMWMode.ADD),
                    (:max, AtomicRMWMode.MAX),   (:min, AtomicRMWMode.MIN),
                    (:or,  AtomicRMWMode.OR),    (:and, AtomicRMWMode.AND),
                    (:xor, AtomicRMWMode.XOR))
    name = Symbol(:atomic_, op)
    @eval begin
        @intrinsic $name(ptr_tile, val, mask, memory_order, memory_scope)
        tfunc(𝕃, ::typeof(Intrinsics.$name), @nospecialize args...) = atomic_tfunc(𝕃, args...)
        efunc(::typeof(Intrinsics.$name), effects::CC.Effects) =
            CC.Effects(effects; effect_free=CC.ALWAYS_FALSE)
        function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.$name), args)
            emit_atomic_rmw!(ctx, args, $mode)
        end
    end
end
