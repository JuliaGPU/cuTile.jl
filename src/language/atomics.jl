# Atomic Operations for Tiles
#
# Provides atomic compare-and-swap, exchange, and add operations for TileArrays.

public atomic_cas, atomic_xchg, atomic_add, atomic_max, atomic_min, atomic_or, atomic_and, atomic_xor

"""
Memory ordering for atomic operations.
Use these constants with atomic_cas, atomic_xchg, etc.
"""
module MemoryOrder
    const Weak = 0
    const Relaxed = 1
    const Acquire = 2
    const Release = 3
    const AcqRel = 4
end

"""
Memory scope for atomic operations.
"""
module MemScope
    const Block = 0
    const Device = 1
    const System = 2
end

# ============================================================================
# Pointer/mask helpers
#
# Both scalar and tile-indexed paths compute (ptr_tile, mask, shape) here,
# then pass to a single set of intrinsics.
# ============================================================================

# Scalar index -> 0D pointer tile, no mask
@inline function _atomic_ptr_and_mask(array::TileArray{T}, index::Integer) where {T}
    idx_0 = Tile(Int32(index - One()))
    ptr_tile = Intrinsics.offset(array.ptr, idx_0)
    (ptr_tile, nothing, ())
end

# N-D tile indices -> N-D pointer tile with bounds mask
@inline function _atomic_ptr_and_mask(array::TileArray{T, N},
                                       indices::NTuple{N, Tile{<:Integer}}) where {T, N}
    # Convert each index to 0-indexed
    indices_0 = ntuple(Val(N)) do d
        indices[d] .- one(eltype(indices[d]))
    end

    # Broadcast all index tiles to a common shape
    S = reduce(broadcast_shape, ntuple(d -> size(indices[d]), Val(N)))

    # Broadcast and convert to Int32
    indices_i32 = ntuple(Val(N)) do d
        convert(Tile{Int32}, broadcast_to(indices_0[d], S))
    end

    # Linear index: sum(idx[d] * stride[d])
    linear_idx = reduce(.+, ntuple(Val(N)) do d
        indices_i32[d] .* broadcast_to(Tile(array.strides[d]), S)
    end)

    ptr_tile = Intrinsics.offset(array.ptr, linear_idx)

    # Bounds mask: 0 <= idx[d] < size[d] for all d
    zero_bc = broadcast_to(Tile(Int32(0)), S)
    mask = reduce(.&, ntuple(Val(N)) do d
        (indices_i32[d] .>= zero_bc) .& (indices_i32[d] .< broadcast_to(Tile(size(array, d)), S))
    end)

    (ptr_tile, mask, S)
end

# 1D convenience: single Tile -> 1-tuple
@inline function _atomic_ptr_and_mask(array::TileArray{T, 1}, indices::Tile{<:Integer}) where {T}
    _atomic_ptr_and_mask(array, (indices,))
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
                            memory_order::Int=MemoryOrder.AcqRel,
                            memory_scope::Int=MemScope.Device) where {T}
    ptr_tile, mask, S = _atomic_ptr_and_mask(array, indices)
    expected_bc = S === () ? Tile(expected) : broadcast_to(Tile(expected), S)
    desired_bc = S === () ? Tile(desired) : broadcast_to(Tile(desired), S)
    result = Intrinsics.atomic_cas(ptr_tile, expected_bc, desired_bc, mask,
                                   memory_order, memory_scope)
    S === () ? Intrinsics.to_scalar(result) : result
end

# Convert mismatched scalar/tile types to match array element type
@inline function atomic_cas(array::TileArray{T}, indices,
                            expected::TileOrScalar, desired::TileOrScalar;
                            memory_order::Int=MemoryOrder.AcqRel,
                            memory_scope::Int=MemScope.Device) where {T}
    atomic_cas(array, indices, T(expected), T(desired); memory_order, memory_scope)
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
and returns the original value. Index is 1-indexed.
"""
function atomic_max end

"""
    atomic_min(array::TileArray, index, val; memory_order, memory_scope) -> T

Atomic minimum. Atomically replaces the value at `index` with `min(old, val)`
and returns the original value. Index is 1-indexed.
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

    @eval @inline function $fname(array::TileArray{T}, indices, val::TileOrScalar{T};
                                   memory_order::Int=MemoryOrder.AcqRel,
                                   memory_scope::Int=MemScope.Device) where {T}
        ptr_tile, mask, S = _atomic_ptr_and_mask(array, indices)
        val_bc = S === () ? Tile(val) : broadcast_to(Tile(val), S)
        result = Intrinsics.$intrinsic(ptr_tile, val_bc, mask, memory_order, memory_scope)
        S === () ? Intrinsics.to_scalar(result) : result
    end

    # Convert mismatched scalar/tile types to match array element type
    @eval @inline function $fname(array::TileArray{T}, indices, val::TileOrScalar;
                                   memory_order::Int=MemoryOrder.AcqRel,
                                   memory_scope::Int=MemScope.Device) where {T}
        $fname(array, indices, T(val); memory_order, memory_scope)
    end
end
