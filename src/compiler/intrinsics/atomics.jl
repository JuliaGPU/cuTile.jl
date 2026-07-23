# atomics

# `MemoryOrder.Weak` exists for non-atomic loads/stores; atomics reject it.
function check_atomic_memory_order(memory_order, op_name::String)
    if memory_order === MemoryOrder.Weak
        throw(IRError("$op_name: memory_order=MemoryOrder.Weak is not supported on " *
                      "atomic operations; pick Relaxed, Acquire, Release, or AcqRel"))
    end
    return memory_order
end


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
    check_atomic_memory_order(memory_order, "atomic_cas")

    shape = ptr_tv.shape

    # Get element type from pointer tile: Tile{Ptr{T}, S} -> T
    ptrs_type = CC.widenconst(ptr_tv.jltype)
    ptr_type = eltype(ptrs_type)
    elem_type = eltype(ptr_type)

    dtype = lookup_dtype!(tt, elem_type)
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

function select_rmw_mode(base_mode::AtomicRMWMode.T, @nospecialize(elem_type))
    if elem_type <: AbstractFloat
        base_mode == AtomicRMWMode.ADD ? AtomicRMWMode.ADDF : base_mode
    elseif elem_type <: Unsigned
        base_mode == AtomicRMWMode.MAX ? AtomicRMWMode.UMAX :
        base_mode == AtomicRMWMode.MIN ? AtomicRMWMode.UMIN : base_mode
    else
        base_mode
    end
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
    check_atomic_memory_order(memory_order, "atomic RMW")

    shape = ptr_tv.shape

    # Get element type from pointer tile: Tile{Ptr{T}, S} -> T
    ptrs_type = CC.widenconst(ptr_tv.jltype)
    ptr_type = eltype(ptrs_type)
    elem_type = eltype(ptr_type)

    # Create result type
    dtype = lookup_dtype!(tt, elem_type)
    result_tile_type = tile_type!(tt, dtype, shape)
    token_type = Token(tt)

    actual_mode = select_rmw_mode(mode, elem_type)
    check_atomic_bf16_support(cb, actual_mode, elem_type)

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
                         (:max,  AtomicRMWMode.MAX,  "maximum (signed for `Signed`, unsigned via `umax` for `Unsigned`)"),
                         (:min,  AtomicRMWMode.MIN,  "minimum (signed for `Signed`, unsigned via `umin` for `Unsigned`)"),
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

# cuda_tile.atomic_red_view_tko
const RED_VIEW_INT_DTYPES = (Int32, Int64, UInt32, UInt64)
const RED_VIEW_ADD_DTYPES = (RED_VIEW_INT_DTYPES..., Float16, BFloat16, Float32, Float64)

function check_atomic_bf16_support(cb::CodeBuilder, mode::AtomicRMWMode.T, @nospecialize(elem_type))
    if mode == AtomicRMWMode.ADDF && elem_type === BFloat16 && cb.version < v"13.3"
        throw(IRError("atomic add on BFloat16 requires Tile IR bytecode ≥ 13.3, got v$(cb.version)"))
    end
end

function emit_atomic_red_view!(ctx::CGCtx, args::AbstractVector,
                               base_mode::AtomicRMWMode.T, name::String)
    cb = ctx.cb
    tt = ctx.tt

    cb.version >= v"13.3" ||
        throw(IRError("$name requires Tile IR bytecode ≥ 13.3 (tileiras too old), got v$(cb.version)"))

    input_token = extract_token_arg!(ctx, args)

    view_arg = emit_value!(ctx, args[1])
    view_arg === nothing && throw(IRError("$name requires a view argument"))
    view_arg.v === nothing && throw(IRError("$name requires a materialized view"))
    view_arg.constant === nothing && throw(IRError("$name: view missing ndim info"))
    ndim = something(view_arg.constant)

    value_tv = emit_value!(ctx, args[2])
    value_tv === nothing && throw(IRError("$name requires a value"))
    elem_type = eltype(CC.widenconst(value_tv.jltype))

    supported = base_mode == AtomicRMWMode.ADD ? RED_VIEW_ADD_DTYPES : RED_VIEW_INT_DTYPES
    elem_type in supported ||
        throw(IRError("$name: unsupported element type $elem_type"))

    mode = select_rmw_mode(base_mode, elem_type)
    check_atomic_bf16_support(cb, mode, elem_type)

    index_tvs = resolve_tuple(ctx, args[3], "$name indices")
    index_vals = Value[tv.v for tv in index_tvs]
    index_jl_types = Type[tv.jltype for tv in index_tvs]
    unique_types = unique(index_jl_types)
    length(unique_types) <= 1 || throw(IRError("All index types must match, got: $unique_types"))
    isempty(unique_types) && ndim > 0 && throw(IRError("$name: indices required for $(ndim)D view"))
    index_jl_type = isempty(unique_types) ? Int32 : unique_types[1]
    index_type = tile_type_for_julia!(ctx, index_jl_type)
    index_vals = pad_indices(ctx, index_vals, ndim, index_type, index_jl_type)
    reverse!(index_vals)

    token_type = Token(tt)
    result_token = encode_AtomicRedViewTkoOp!(cb, token_type, view_arg.v, index_vals, value_tv.v;
                                              token=input_token,
                                              memory_ordering=MemoryOrderingSemantics.Relaxed,
                                              memory_scope=MemoryScope.Device,
                                              mode=mode)
    ctx.result_tokens[ctx.current_ssa_idx] = result_token
    return nothing
end

for (op, mode) in ((:add, AtomicRMWMode.ADD), (:max, AtomicRMWMode.MAX), (:min, AtomicRMWMode.MIN),
                   (:or, AtomicRMWMode.OR), (:and, AtomicRMWMode.AND), (:xor, AtomicRMWMode.XOR))
    name = Symbol(:atomic_red_view_, op)
    docstring = """
        Intrinsics.$name(view, value::Tile, indices::NTuple) -> Nothing

    Apply an atomic $op reduction to a tile of `view`. Uses relaxed,
    device-wide ordering and returns `nothing`.
    """
    @eval begin
        @doc $docstring @intrinsic $name(view, value, indices)
        tfunc(𝕃, ::typeof(Intrinsics.$name), @nospecialize args...) = Nothing
        efunc(::typeof(Intrinsics.$name), effects::CC.Effects) =
            CC.Effects(effects; effect_free=CC.ALWAYS_FALSE)
        emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.$name), args) =
            emit_atomic_red_view!(ctx, args, $mode, $(string(name)))
    end
end
