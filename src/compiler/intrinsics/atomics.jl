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

"""
    Intrinsics.atomic_cas(ptr_tile::Tile{Ptr{T},S}, expected::Tile{T,S}, desired::Tile{T,S},
                          mask::Union{Tile{Bool,S},Nothing},
                          memory_order::MemoryOrderingSemantics.T,
                          memory_scope::MemoryScope.T) -> Tile{T,S}

Element-wise token-ordered atomic compare-and-swap on a tile of pointers;
lowers to `cuda_tile.atomic_cas_tko`. Returns the original values prior
to the swap.

`memory_order` and `memory_scope` are compile-time constants. When `mask`
is provided, masked-out elements are not modified and the corresponding
result entry is `expected[i]`. The token argument is appended by
`token_order_pass!` and is not part of the user-visible signature.
"""
@intrinsic atomic_cas(ptr_tile, expected, desired, mask, memory_order, memory_scope)
function tfunc(𝕃, ::typeof(Intrinsics.atomic_cas), @nospecialize(ptrs), @nospecialize args...)
    atomic_tfunc(𝕃, ptrs, args...)
end
efunc(::typeof(Intrinsics.atomic_cas), effects::CC.Effects) =
    CC.Effects(effects; effect_free=CC.ALWAYS_FALSE)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.atomic_cas), args)
    cb = ctx.cb
    tt = ctx.tt

    # Extract input token from last arg (added by token_order_pass!)
    input_token = extract_token_arg!(ctx, args)

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
    result_tile_type = tile_type!(tt, dtype, shape)
    token_type = Token(tt)

    # Emit atomic CAS
    mem_ordering = convert_enum(MemoryOrderingSemantics, memory_order)
    mem_scope = convert_enum(MemoryScope, memory_scope)

    old_val, new_token = if has_mask
        encode_AtomicCASPtrOp!(cb, result_tile_type, token_type,
                               ptr_tv.v, expected_tv.v, desired_tv.v;
                               mask=mask_tv.v,
                               token=input_token,
                               memory_ordering=mem_ordering,
                               memory_scope=mem_scope)
    else
        encode_AtomicCASPtrOp!(cb, result_tile_type, token_type,
                               ptr_tv.v, expected_tv.v, desired_tv.v;
                               token=input_token,
                               memory_ordering=mem_ordering,
                               memory_scope=mem_scope)
    end
    # Store result token for TokenResultNode
    ctx.result_tokens[ctx.current_ssa_idx] = new_token

    julia_shape = ColMajorShape(shape)
    CGVal(old_val, result_tile_type, Tile{elem_type, TupleType(julia_shape)}, shape)
end

# cuda_tile.atomic_rmw_tko (shared helper for atomic RMW operations)
function emit_atomic_rmw!(ctx::CGCtx, args::AbstractVector, mode::AtomicRMWMode.T)
    cb = ctx.cb
    tt = ctx.tt

    # Extract input token from last arg (added by token_order_pass!)
    input_token = extract_token_arg!(ctx, args)

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
    result_tile_type = tile_type!(tt, dtype, shape)
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
                                token=input_token,
                                memory_ordering=mem_ordering,
                                memory_scope=mem_scope)
    else
        encode_AtomicRMWPtrOp!(cb, result_tile_type, token_type,
                                ptr_tv.v, val_tv.v, actual_mode;
                                token=input_token,
                                memory_ordering=mem_ordering,
                                memory_scope=mem_scope)
    end
    # Store result token for TokenResultNode
    ctx.result_tokens[ctx.current_ssa_idx] = new_token

    julia_shape = ColMajorShape(shape)
    CGVal(old_val, result_tile_type, Tile{elem_type, TupleType(julia_shape)}, shape)
end

# cuda_tile.atomic_rmw_tko variants
for (op, mode, desc) in ((:xchg, AtomicRMWMode.XCHG, "exchange (`val`, returning the old value)"),
                         (:add,  AtomicRMWMode.ADD,  "integer addition (or floating-point addition for AbstractFloat element types, via `cuda_tile.atomic_rmw_tko`'s `addf` mode)"),
                         (:max,  AtomicRMWMode.MAX,  "signed maximum"),
                         (:min,  AtomicRMWMode.MIN,  "signed minimum"),
                         (:or,   AtomicRMWMode.OR,   "bitwise OR"),
                         (:and,  AtomicRMWMode.AND,  "bitwise AND"),
                         (:xor,  AtomicRMWMode.XOR,  "bitwise XOR"))
    name = Symbol(:atomic_, op)
    docstring = """
        Intrinsics.$name(ptr_tile::Tile{Ptr{T},S}, val::Tile{T,S},
                         mask::Union{Tile{Bool,S},Nothing},
                         memory_order::MemoryOrderingSemantics.T,
                         memory_scope::MemoryScope.T) -> Tile{T,S}

    Element-wise token-ordered atomic read-modify-write performing $desc;
    lowers to `cuda_tile.atomic_rmw_tko`. Returns the original values
    prior to the modification.

    `memory_order` and `memory_scope` are compile-time constants. When
    `mask` is provided, masked-out elements are not modified. The token
    argument is appended by `token_order_pass!` and is not part of the
    user-visible signature.
    """
    @eval begin
        @doc $docstring @intrinsic $name(ptr_tile, val, mask, memory_order, memory_scope)
        tfunc(𝕃, ::typeof(Intrinsics.$name), @nospecialize args...) = atomic_tfunc(𝕃, args...)
        efunc(::typeof(Intrinsics.$name), effects::CC.Effects) =
            CC.Effects(effects; effect_free=CC.ALWAYS_FALSE)
        function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.$name), args)
            emit_atomic_rmw!(ctx, args, $mode)
        end
    end
end
