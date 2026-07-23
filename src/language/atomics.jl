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
        # Bitwise atomics do not convert the update.
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

# View-based atomic reductions

"""
    atomic_store_add(dst, index, update) -> Nothing

Reduce `update` into a tile of `dst` without returning its previous value.
`dst` may be a `TileArray` or a `TiledView` from [`eachtile`](@ref). Updates
broadcast to the tile shape. The operation uses relaxed, device-wide ordering.

Also available: `atomic_store_max`, `atomic_store_min`, `atomic_store_or`,
`atomic_store_and`, and `atomic_store_xor`. Bitwise updates must have the
destination element type.

Requires Tile IR bytecode ≥ 13.3.
"""
function atomic_store_add end

@inline atomic_red_update(::Type{Target}, tile::Tile{Target}, bitwise::Bool) where {Target} = tile
@inline function atomic_red_update(::Type{Target}, tile::Tile, bitwise::Bool) where {Target}
    bitwise && throw(ArgumentError("bitwise atomic reduction requires the update tile element type to exactly match the destination; no implicit conversion"))
    convert(Tile{Target}, tile)
end

# Tile IR atomic views cannot carry padding.
@inline function make_atomic_tile_view(
        tiles::TiledView{A, RequestedShape, Shape, Shape}) where {A, RequestedShape, Shape}
    parent = tiles.parent
    tv = Intrinsics.make_tensor_view(typeof(parent), parent.ptr, parent.sizes, parent.strides)
    Intrinsics.make_partition_view(
        tv, tiled_view_shape(tiles), PaddingMode.Undetermined, tiled_view_order(tiles))
end
@inline function make_atomic_tile_view(
        tiles::TiledView{A, RequestedShape, Shape, Step}) where {A, RequestedShape, Shape, Step}
    parent = tiles.parent
    tv = Intrinsics.make_tensor_view(typeof(parent), parent.ptr, parent.sizes, parent.strides)
    Intrinsics.make_strided_view(
        tv, tiled_view_shape(tiles), tiled_view_step(tiles),
        PaddingMode.Undetermined, tiled_view_order(tiles))
end

for op in (:add, :max, :min, :or, :and, :xor)
    fname = Symbol(:atomic_store_, op)
    intrinsic = Symbol(:atomic_red_view_, op)
    bitwise = op in (:or, :and, :xor)

    @eval @inline function $fname(arr::TileArray{T}, index, tile::Tile) where {T}
        update = atomic_red_update(T, tile, $bitwise)
        reshaped = _reshape_to_rank(update, Val(ndims(arr)))
        tv = Intrinsics.make_tensor_view(typeof(arr), arr.ptr, arr.sizes, arr.strides)
        pv = Intrinsics.make_partition_view(tv, size(reshaped), PaddingMode.Undetermined, nothing)
        Intrinsics.$intrinsic(pv, reshaped, promote(index...) .- One())
        return nothing
    end
    @eval @inline $fname(arr::TileArray{T}, index::Integer, tile::Tile) where {T} =
        $fname(arr, (index,), tile)
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

    @eval @inline function $fname(tiles::TiledView{A}, index::NTuple{N, <:Integer},
                                  tile::Tile) where {A, N}
        N == ndims(tiles) || throw(ArgumentError("eachtile: expected $(ndims(tiles)) tile indices, got $N"))
        update = atomic_red_update(eltype(A), tile, $bitwise)
        view = make_atomic_tile_view(tiles)
        Intrinsics.$intrinsic(view, broadcast_to(update, tiled_view_shape(tiles)),
                              promote(index...) .- One())
        return nothing
    end
    @eval @inline $fname(tiles::TiledView, index::Integer, tile::Tile) =
        $fname(tiles, (index,), tile)
end

# `@atomic`

public @atomic

const AtomicTileIndex = Union{Integer, Tuple{Vararg{Integer}}}
const AtomicGatherIndex = Union{Tile, Tuple{Vararg{Tile}}}

atomic_red_op(::typeof(+))   = (atomic_store_add, atomic_add, identity)
atomic_red_op(::typeof(-))   = (atomic_store_add, atomic_add, -)
atomic_red_op(::typeof(max)) = (atomic_store_max, atomic_max, identity)
atomic_red_op(::typeof(min)) = (atomic_store_min, atomic_min, identity)
atomic_red_op(::typeof(&))   = (atomic_store_and, atomic_and, identity)
atomic_red_op(::typeof(|))   = (atomic_store_or,  atomic_or,  identity)
atomic_red_op(::typeof(xor)) = (atomic_store_xor, atomic_xor, identity)

@inline function atomic_reduce_relaxed!(target::Union{TileArray, TiledView},
                                        index::AtomicTileIndex, op::F, val) where {F}
    store_fn, _, transform = atomic_red_op(op)
    store_fn(target, index, transform(val))
    return nothing
end
@inline function atomic_reduce_relaxed!(target::TileArray, index::AtomicGatherIndex,
                                        op::F, val) where {F}
    _, fetch_fn, transform = atomic_red_op(op)
    fetch_fn(target, index, transform(val); memory_order=MemoryOrder.Relaxed)
    return nothing
end

@inline function atomic_reduce_ordered!(target::TileArray, index, op::F, val, order) where {F}
    _, fetch_fn, transform = atomic_red_op(op)
    fetch_fn(target, index, transform(val); memory_order=order)
    return nothing
end
@inline atomic_reduce_ordered!(::TiledView, index, op, val, order) =
    throw(ArgumentError("@atomic on TiledView only supports relaxed ordering"))

@inline atomic_update(::Type{T}, op, val::Tile) where {T} = convert(Tile{T}, val)
@inline atomic_update(::Type{T}, op, val) where {T} = convert(T, val)
@inline atomic_update(::Type, ::typeof(&), val) = val
@inline atomic_update(::Type, ::typeof(|), val) = val
@inline atomic_update(::Type, ::typeof(xor), val) = val

@inline function atomic_fetch(target::TileArray{T}, index, op::F, val, order) where {T, F}
    _, fetch_fn, transform = atomic_red_op(op)
    update = atomic_update(T, op, val)
    old = fetch_fn(target, index, transform(update); memory_order=order)
    return old => op(old, update)
end

const ATOMIC_OPASSIGN = Dict(Symbol("+=") => :+, Symbol("-=") => :-, Symbol("&=") => :&,
                             Symbol("|=") => :|, Symbol("⊻=") => :⊻)

function atomic_op(sym)
    sym === :+ && return Base.:+
    sym === :- && return Base.:-
    sym === :max && return Base.max
    sym === :min && return Base.min
    sym === :& && return Base.:&
    sym === :| && return Base.:|
    (sym === :⊻ || sym === :xor) && return Base.xor
    error("@atomic: unsupported operator `$sym`; expected +, -, max, min, &, |, or ⊻")
end

function atomic_order(order_arg)
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

function atomic_statement(target, idx, op, val, order)
    if order === nothing || order === MemoryOrder.Relaxed
        :($atomic_reduce_relaxed!($target, $idx, $op, $val))
    else
        :($atomic_reduce_ordered!($target, $idx, $op, $val, $order))
    end
end

function atomic_target(ref)
    ref isa Expr && ref.head === :ref ||
        error("@atomic: expected an indexed reference like A[i] on the left, got `$ref`")
    inds = ref.args[2:end]
    idx = length(inds) == 1 ? inds[1] : Expr(:tuple, inds...)
    return ref.args[1], idx
end

function atomic_macro(order_arg, ex)
    order = atomic_order(order_arg)
    head = ex isa Expr ? ex.head : nothing

    if haskey(ATOMIC_OPASSIGN, head)
        target, idx = atomic_target(ex.args[1])
        op = atomic_op(ATOMIC_OPASSIGN[head])
        return esc(atomic_statement(target, idx, op, ex.args[2], order))
    end

    if head === :(=)
        lhs, rhs = ex.args
        (rhs isa Expr && rhs.head === :call && length(rhs.args) == 3) ||
            error("@atomic: plain `A[i] = v` is unsupported; write `A[i] op= v` or `A[i] = op(A[i], v)`")
        target, idx = atomic_target(lhs)
        op = atomic_op(rhs.args[1])
        a, b = rhs.args[2], rhs.args[3]
        if a != lhs
            b == lhs &&
                error("@atomic: the first argument to `op(...)` must be the left-hand `$lhs`")
            error("@atomic: the right-hand `op(...)` must reference the left-hand `$lhs`")
        end
        val = b
        return esc(atomic_statement(target, idx, op, val, order))
    end

    if head === :call && length(ex.args) == 3
        op = atomic_op(ex.args[1])
        target, idx = atomic_target(ex.args[2])
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
`:monotonic` maps to Relaxed; `:acquire`, `:release`, and
`:acquire_release` map directly.
`:sequentially_consistent` is rejected (no Tile IR equivalent).

Statement forms default to `:monotonic`; value forms default to
`:acquire_release`. Stronger statement orderings use `atomic_*` and discard
the result.
"""
macro atomic(ex)
    atomic_macro(nothing, ex)
end
macro atomic(order, ex)
    atomic_macro(order, ex)
end
