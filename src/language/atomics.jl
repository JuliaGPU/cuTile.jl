# Atomic Operations for Tiles
#
# Provides atomic compare-and-swap, exchange, and add operations for TileArrays.

public atomic_cas, atomic_xchg, atomic_add, atomic_max, atomic_min, atomic_or, atomic_and, atomic_xor
public atomic_store_add, atomic_store_max, atomic_store_min,
       atomic_store_or, atomic_store_and, atomic_store_xor

"""
Memory ordering for atomic operations.
Use these constants with atomic_cas, atomic_xchg, etc.
"""
@enumx MemoryOrder begin
    Weak = 0
    Relaxed = 1
    Acquire = 2
    Release = 3
    AcqRel = 4
end

"""
Memory scope for atomic operations.
"""
@enumx MemScope begin
    Block = 0
    Device = 1
    System = 2
end

# ============================================================================
# Pointer/mask helpers
#
# Both scalar and tile-indexed paths compute (ptr_tile, mask, shape) here,
# then pass to a single set of intrinsics.
# ============================================================================

# Scalar index -> 0D pointer tile, no mask
@inline function _atomic_ptr_and_mask(array::TileArray{T}, index::Integer; check_bounds::Bool=true) where {T}
    idx_0 = Tile(Int32(index - One()))
    ptr_tile = Intrinsics.offset(Tile(array.ptr), idx_0)
    (ptr_tile, nothing, ())
end

# N-D tile indices -> N-D pointer tile with optional bounds mask
@inline function _atomic_ptr_and_mask(array::TileArray{T, N},
                                       indices::NTuple{N, Tile{<:Integer}};
                                       check_bounds::Bool=true) where {T, N}
    indices_0 = ntuple(Val(N)) do d
        indices[d] .- one(eltype(indices[d]))
    end

    S = reduce(broadcast_shape, ntuple(d -> size(indices[d]), Val(N)))

    indices_i32 = ntuple(Val(N)) do d
        convert(Tile{Int32}, broadcast_to(indices_0[d], S))
    end

    linear_idx = reduce(.+, ntuple(Val(N)) do d
        indices_i32[d] .* broadcast_to(Tile(array.strides[d]), S)
    end)

    ptr_tile = Intrinsics.offset(Tile(array.ptr), linear_idx)

    mask = if check_bounds
        zero_bc = broadcast_to(Tile(Int32(0)), S)
        reduce(.&, ntuple(Val(N)) do d
            (indices_i32[d] .>= zero_bc) .& (indices_i32[d] .< broadcast_to(Tile(size(array, d)), S))
        end)
    else
        nothing
    end

    (ptr_tile, mask, S)
end

# 1D convenience: single Tile -> 1-tuple
@inline function _atomic_ptr_and_mask(array::TileArray{T, 1}, indices::Tile{<:Integer};
                                       check_bounds::Bool=true) where {T}
    _atomic_ptr_and_mask(array, (indices,); check_bounds)
end

# ============================================================================
# Atomic CAS
# ============================================================================

"""
    atomic_cas(array::TileArray, index, expected, desired; memory_order, memory_scope) -> T

Atomic compare-and-swap. Atomically compares the value at `index` with `expected`,
and if equal, replaces it with `desired`. Returns the original value.
Index is 1-indexed.

# Example
```julia
# Spin-lock acquisition
while ct.atomic_cas(locks, idx, Int32(0), Int32(1); memory_order=ct.MemoryOrder.Acquire) == Int32(1)
    # spin
end
```
"""
@inline function atomic_cas(array::TileArray{T}, indices,
                            expected::TileOrScalar{T}, desired::TileOrScalar{T};
                            check_bounds::Bool=true,
                            memory_order::MemoryOrder.T=MemoryOrder.AcqRel,
                            memory_scope::MemScope.T=MemScope.Device) where {T}
    ptr_tile, mask, S = _atomic_ptr_and_mask(array, indices; check_bounds)
    expected_bc = S === () ? Tile(expected) : broadcast_to(Tile(expected), S)
    desired_bc = S === () ? Tile(desired) : broadcast_to(Tile(desired), S)
    result = Intrinsics.atomic_cas(ptr_tile, expected_bc, desired_bc, mask,
                                   memory_order, memory_scope)
    S === () ? Intrinsics.to_scalar(result) : result
end

@inline function atomic_cas(array::TileArray{T}, indices,
                            expected::TileOrScalar, desired::TileOrScalar;
                            check_bounds::Bool=true,
                            memory_order::MemoryOrder.T=MemoryOrder.AcqRel,
                            memory_scope::MemScope.T=MemScope.Device) where {T}
    atomic_cas(array, indices, T(expected), T(desired); check_bounds, memory_order, memory_scope)
end

# ============================================================================
# Atomic RMW operations (atomic_add, atomic_xchg)
# ============================================================================

"""
    atomic_add(array::TileArray, index, val; memory_order, memory_scope) -> T

Atomic addition. Atomically adds `val` to the value at `index` and returns
the original value. Index is 1-indexed.

# Example
```julia
old_val = ct.atomic_add(counters, idx, Int32(1))
```
"""
function atomic_add end

"""
    atomic_xchg(array::TileArray, index, val; memory_order, memory_scope) -> T

Atomic exchange. Atomically replaces the value at `index` with `val` and returns
the original value. Index is 1-indexed.

# Example
```julia
# Spin-lock release
ct.atomic_xchg(locks, idx, Int32(0); memory_order=ct.MemoryOrder.Release)
```
"""
function atomic_xchg end

"""
    atomic_max(array::TileArray, index, val; memory_order, memory_scope) -> T

Atomic maximum. Atomically replaces the value at `index` with `max(old, val)`
and returns the original value. Index is 1-indexed. The comparison is signed
for `Signed` element types and unsigned for `Unsigned` ones.
"""
function atomic_max end

"""
    atomic_min(array::TileArray, index, val; memory_order, memory_scope) -> T

Atomic minimum. Atomically replaces the value at `index` with `min(old, val)`
and returns the original value. Index is 1-indexed. The comparison is signed
for `Signed` element types and unsigned for `Unsigned` ones.
"""
function atomic_min end

"""
    atomic_or(array::TileArray, index, val; memory_order, memory_scope) -> T

Atomic bitwise OR. Atomically replaces the value at `index` with `old | val`
and returns the original value. Index is 1-indexed.
"""
function atomic_or end

"""
    atomic_and(array::TileArray, index, val; memory_order, memory_scope) -> T

Atomic bitwise AND. Atomically replaces the value at `index` with `old & val`
and returns the original value. Index is 1-indexed.
"""
function atomic_and end

"""
    atomic_xor(array::TileArray, index, val; memory_order, memory_scope) -> T

Atomic bitwise XOR. Atomically replaces the value at `index` with `old ⊻ val`
and returns the original value. Index is 1-indexed.
"""
function atomic_xor end

for op in (:add, :xchg, :max, :min, :or, :and, :xor)
    fname = Symbol(:atomic_, op)
    intrinsic = Symbol(:atomic_, op)
    bitwise = op in (:or, :and, :xor)

    @eval @inline function $fname(array::TileArray{T}, indices, val::TileOrScalar{T};
                                   check_bounds::Bool=true,
                                   memory_order::MemoryOrder.T=MemoryOrder.AcqRel,
                                   memory_scope::MemScope.T=MemScope.Device) where {T}
        ptr_tile, mask, S = _atomic_ptr_and_mask(array, indices; check_bounds)
        val_bc = S === () ? Tile(val) : broadcast_to(Tile(val), S)
        result = Intrinsics.$intrinsic(ptr_tile, val_bc, mask, memory_order, memory_scope)
        S === () ? Intrinsics.to_scalar(result) : result
    end

    if bitwise
        # Bitwise atomics require the update to exactly match the array element
        # type — no implicit conversion (mirrors cuTile Python's
        # `_cast_rmw_update_dtype`). add/max/min/xchg convert below.
        @eval @inline function $fname(array::TileArray{T}, indices, val::TileOrScalar;
                                       kwargs...) where {T}
            throw(ArgumentError($("$fname requires the update value to exactly match " *
                "the array element type; bitwise atomics do not convert implicitly")))
        end
    else
        @eval @inline function $fname(array::TileArray{T}, indices, val::TileOrScalar;
                                       check_bounds::Bool=true,
                                       memory_order::MemoryOrder.T=MemoryOrder.AcqRel,
                                       memory_scope::MemScope.T=MemScope.Device) where {T}
            $fname(array, indices, T(val); check_bounds, memory_order, memory_scope)
        end
    end
end

# ============================================================================
# View-based atomic reductions (atomic_store_add/max/min/or/and/xor)
#
# Reduce a whole tile into a view and return `nothing`, without reading back
# the old values; these lower to `cuda_tile.atomic_red_view_tko` (relaxed,
# device). Unlike the pointer-style `atomic_*`, they take no
# `memory_order`/`memory_scope`/`mask`/`check_bounds` keywords. Out-of-bounds
# handling comes from the view: partial tiles are clipped, and a tile that lies
# entirely out of bounds is undefined. Requires Tile IR bytecode ≥ 13.3.
# ============================================================================

"""
    atomic_store_add(dst, index, update) -> Nothing

Atomically reduce `update` into the tile of `dst` selected by `index`, without
reading back the old values. `dst` is a `TileArray` (the update tile's shape
selects the partition) or a `TiledView` from [`eachtile`](@ref). `update`
broadcasts to the view's tile shape. The reduction is relaxed/device-scoped.

Also available: `atomic_store_max`, `atomic_store_min`, `atomic_store_or`,
`atomic_store_and`, `atomic_store_xor`. The bitwise variants require `update`
to exactly match the destination element type; `add`/`max`/`min` convert.

Requires Tile IR bytecode ≥ 13.3.
"""
function atomic_store_add end

# Coerce an update tile to `Target` element type: arithmetic ops convert,
# bitwise ops require an exact match (mirrors Python's `_cast_rmw_update_dtype`).
@inline _red_update(::Type{Target}, tile::Tile{Target}, bitwise::Bool) where {Target} = tile
@inline function _red_update(::Type{Target}, tile::Tile, bitwise::Bool) where {Target}
    bitwise && throw(ArgumentError("bitwise atomic reduction requires the update tile element type to exactly match the destination; no implicit conversion"))
    convert(Tile{Target}, tile)
end

for op in (:add, :max, :min, :or, :and, :xor)
    fname = Symbol(:atomic_store_, op)
    intrinsic = Symbol(:atomic_red_view_, op)
    bitwise = op in (:or, :and, :xor)

    # TileArray: the update tile's shape drives the partition view.
    @eval @inline function $fname(arr::TileArray{T}, index, tile::Tile) where {T}
        update = _red_update(T, tile, $bitwise)
        reshaped = _reshape_to_rank(update, Val(ndims(arr)))
        tv = Intrinsics.make_tensor_view(typeof(arr), arr.ptr, arr.sizes, arr.strides)
        pv = Intrinsics.make_partition_view(tv, size(reshaped), PaddingMode.Undetermined, nothing)
        Intrinsics.$intrinsic(pv, reshaped, promote(index...) .- One())
        return nothing
    end
    @eval @inline $fname(arr::TileArray{T}, index::Integer, tile::Tile) where {T} =
        $fname(arr, (index,), tile)
    # Scalar update → 1-element tile. Arithmetic converts to the array element
    # type; bitwise requires an exact match. (A single `Number` method — a
    # diagonal `val::T` method would be shadowed by this one.)
    if bitwise
        @eval @inline function $fname(arr::TileArray{T}, index, val::Number) where {T}
            val isa T || throw(ArgumentError($("$fname requires the update value to exactly " *
                "match the destination element type; bitwise reductions do not convert implicitly")))
            shape = ntuple(_ -> 1, Val(ndims(arr)))
            $fname(arr, index, reshape(Intrinsics.from_scalar(val, Tuple{}), shape))
        end
    else
        @eval @inline function $fname(arr::TileArray{T}, index, val::Number) where {T}
            shape = ntuple(_ -> 1, Val(ndims(arr)))
            $fname(arr, index, reshape(Intrinsics.from_scalar(convert(T, val), Tuple{}), shape))
        end
    end

    # TiledView: update broadcasts to the view's tile shape (eachtile windows).
    @eval @inline function $fname(tiles::TiledView{A}, index::NTuple{N, <:Integer},
                                  tile::Tile) where {A, N}
        N == ndims(tiles) || throw(ArgumentError("eachtile: expected $(ndims(tiles)) tile indices, got $N"))
        update = _red_update(eltype(A), tile, $bitwise)
        view = make_tile_view(tiles)
        Intrinsics.$intrinsic(view, broadcast_to(update, tiled_view_shape(tiles)),
                              promote(index...) .- One())
        return nothing
    end
    @eval @inline $fname(tiles::TiledView, index::Integer, tile::Tile) =
        $fname(tiles, (index,), tile)
end

# ============================================================================
# `@atomic` — Base-style atomic operators
#
# Statement forms (`A[i] += v`, `A[i] = op(A[i], v)`) return `nothing`. Value
# forms (`A[i] + v`, `max(A[i], v)`) follow Base and return `old => new`.
#
# The two forms use different default orderings (see the macro docstring):
# statement forms default to Relaxed, so a tile or view target lowers to
# `atomic_red_view_tko`; value forms default to AcqRel, matching the
# `ct.atomic_*` functions.
# ============================================================================

public @atomic

const _TileIndex = Union{Integer, Tuple{Vararg{Integer}}}
const _GatherIndex = Union{Tile, Tuple{Vararg{Tile}}}

# op function → (view-reduction fn, fetching fn, update transform). `-` negates
# the update and adds (Tile IR has no atomic subtract).
_red_op(::typeof(+))   = (atomic_store_add, atomic_add, identity)
_red_op(::typeof(-))   = (atomic_store_add, atomic_add, -)
_red_op(::typeof(max)) = (atomic_store_max, atomic_max, identity)
_red_op(::typeof(min)) = (atomic_store_min, atomic_min, identity)
_red_op(::typeof(&))   = (atomic_store_and, atomic_and, identity)
_red_op(::typeof(|))   = (atomic_store_or,  atomic_or,  identity)
_red_op(::typeof(xor)) = (atomic_store_xor, atomic_xor, identity)

# Statement-form lowering. The macro picks the relaxed or ordered path at
# expansion time (it knows the ordering literally); the choice between a view
# reduction and a fetching atomic below is by target type only, so no branch on
# the ordering value is emitted inside the kernel.

# Relaxed: a tile or view target reduces into the view; a gather target fetches.
@inline function atomic_reduce_relaxed!(target::Union{TileArray, TiledView},
                                        index::_TileIndex, op::F, val) where {F}
    store_fn, _, transform = _red_op(op)
    store_fn(target, index, transform(val))
    return nothing
end
@inline function atomic_reduce_relaxed!(target::TileArray, index::_GatherIndex,
                                        op::F, val) where {F}
    _, fetch_fn, transform = _red_op(op)
    fetch_fn(target, index, transform(val); memory_order=MemoryOrder.Relaxed)
    return nothing
end

# Stronger ordering: read the old value and discard it (a TiledView cannot).
@inline function atomic_reduce_ordered!(target::TileArray, index, op::F, val, order) where {F}
    _, fetch_fn, transform = _red_op(op)
    fetch_fn(target, index, transform(val); memory_order=order)
    return nothing
end
@inline atomic_reduce_ordered!(::TiledView, index, op, val, order) =
    throw(ArgumentError("@atomic on a TiledView must use the default (relaxed) ordering: a view reduction does not return the old value"))

# Value-form lowering: fetch the old value, recompute `new` elementwise, and
# return the `old => new` pair (matching Base).
@inline _atomic_update(::Type{T}, op, val::Tile) where {T} = convert(Tile{T}, val)
@inline _atomic_update(::Type{T}, op, val) where {T} = convert(T, val)
@inline _atomic_update(::Type, ::typeof(&), val) = val
@inline _atomic_update(::Type, ::typeof(|), val) = val
@inline _atomic_update(::Type, ::typeof(xor), val) = val

@inline function atomic_fetch(target::TileArray{T}, index, op::F, val, order) where {T, F}
    _, fetch_fn, transform = _red_op(op)
    update = _atomic_update(T, op, val)
    old = fetch_fn(target, index, transform(update); memory_order=order)
    return old => op(old, update)
end

# --- macro parsing (runs at expansion time) ---

const _OPASSIGN_SYM = Dict(Symbol("+=") => :+, Symbol("-=") => :-, Symbol("&=") => :&,
                           Symbol("|=") => :|, Symbol("⊻=") => :⊻)

# Map an operator symbol to its function value, rejecting unsupported ops.
function _atomic_op_fn(sym)
    sym === :+ && return Base.:+
    sym === :- && return Base.:-
    sym === :max && return Base.max
    sym === :min && return Base.min
    sym === :& && return Base.:&
    sym === :| && return Base.:|
    (sym === :⊻ || sym === :xor) && return Base.xor
    error("@atomic: unsupported operator `$sym`; expected +, -, max, min, &, |, or ⊻")
end

# Map an ordering symbol to a MemoryOrder value; `nothing` when none was given.
function _atomic_order(order_arg)
    order_arg === nothing && return nothing
    sym = order_arg isa QuoteNode ? order_arg.value : order_arg
    sym isa Symbol || error("@atomic: ordering must be a symbol, e.g. :monotonic")
    sym === :monotonic && return MemoryOrder.Relaxed
    sym === :acquire && return MemoryOrder.Acquire
    sym === :release && return MemoryOrder.Release
    sym === :acquire_release && return MemoryOrder.AcqRel
    sym === :sequentially_consistent &&
        error("@atomic: :sequentially_consistent has no Tile IR equivalent")
    error("@atomic: unknown ordering :$sym")
end

# Build the statement-form call. Statement default is Relaxed; the relaxed vs
# ordered choice is made here so the ordering is never compared inside the kernel.
function _atomic_statement(target, idx, op, val, order)
    if order === nothing || order === MemoryOrder.Relaxed
        :($atomic_reduce_relaxed!($target, $idx, $op, $val))
    else
        :($atomic_reduce_ordered!($target, $idx, $op, $val, $order))
    end
end

# `A[i]` / `A[i,j]` → (target, index-expr). Errors on any other LHS.
function _atomic_target(ref)
    ref isa Expr && ref.head === :ref ||
        error("@atomic: expected an indexed reference like A[i] on the left, got `$ref`")
    inds = ref.args[2:end]
    idx = length(inds) == 1 ? inds[1] : Expr(:tuple, inds...)
    return ref.args[1], idx
end

function _atomic_macro(order_arg, ex)
    order = _atomic_order(order_arg)
    head = ex isa Expr ? ex.head : nothing

    # Statement form: A[i] op= v
    if haskey(_OPASSIGN_SYM, head)
        target, idx = _atomic_target(ex.args[1])
        op = _atomic_op_fn(_OPASSIGN_SYM[head])
        return esc(_atomic_statement(target, idx, op, ex.args[2], order))
    end

    # Statement form: A[i] = op(A[i], v)
    if head === :(=)
        lhs, rhs = ex.args
        (rhs isa Expr && rhs.head === :call && length(rhs.args) == 3) ||
            error("@atomic: plain `A[i] = v` is unsupported; write `A[i] op= v` or `A[i] = op(A[i], v)`")
        target, idx = _atomic_target(lhs)
        op = _atomic_op_fn(rhs.args[1])
        a, b = rhs.args[2], rhs.args[3]
        val = a == lhs ? b : b == lhs ? a :
              error("@atomic: the right-hand `op(...)` must reference the left-hand `$lhs`")
        return esc(_atomic_statement(target, idx, op, val, order))
    end

    # Value form: A[i] + v  or  op(A[i], v)  → old => new
    if head === :call && length(ex.args) == 3
        op = _atomic_op_fn(ex.args[1])
        target, idx = _atomic_target(ex.args[2])
        ord = order === nothing ? MemoryOrder.AcqRel : order
        return esc(:($atomic_fetch($target, $idx, $op, $(ex.args[3]), $ord)))
    end

    error("@atomic: unsupported expression `$ex`")
end

"""
    @atomic [order] expr

Atomic reductions over a `TileArray`, `TiledView`, or gather target.

Statement forms return `nothing`:

    @atomic A[i] += v          # also -=, &=, |=, ⊻=
    @atomic A[i] = max(A[i], v)  # op ∈ + - max min & | ⊻; RHS must reference A[i]

Value forms follow Base and return `old => new`:

    pair = @atomic A[i] + v        # pair.first == old, pair.second == new

An optional leading ordering symbol maps to Tile IR memory orderings:
`:monotonic`→Relaxed, `:acquire`, `:release`, `:acquire_release`→AcqRel.
`:sequentially_consistent` is rejected (no Tile IR equivalent).

The two forms use different default orderings. **Statement forms default to
`:monotonic` (Relaxed)**, so a tile or view target reduces into the view via
`atomic_red_view_tko`. **Value forms default to `:acquire_release`**, matching
`ct.atomic_*`. Under a stronger ordering a statement instead reads the old
value with `atomic_*` and discards it.
"""
macro atomic(ex)
    _atomic_macro(nothing, ex)
end
macro atomic(order, ex)
    _atomic_macro(order, ex)
end
