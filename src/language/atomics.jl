# Atomic Operations for Tiles
#
# Provides atomic compare-and-swap, exchange, and add operations for TileArrays.

public atomic_cas, atomic_xchg, atomic_add

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

for op in (:add, :xchg)
    fname = Symbol(:atomic_, op)
    intrinsic = Symbol(:atomic_, op)

    # Scalar index, scalar val
    @eval @inline function $fname(array::TileArray{T}, index, val::T;
                                   memory_order::Int=MemoryOrder.AcqRel,
                                   memory_scope::Int=MemScope.Device) where {T}
        ptr_tile, mask, _ = _atomic_ptr_and_mask(array, index)
        Intrinsics.$intrinsic(ptr_tile, Tile(val), mask, memory_order, memory_scope)
    end

    # N-D tile indices, scalar val
    @eval @inline function $fname(array::TileArray{T, N},
                                   indices::NTuple{N, Tile{<:Integer}}, val::T;
                                   memory_order::Int=MemoryOrder.AcqRel,
                                   memory_scope::Int=MemScope.Device) where {T, N}
        ptr_tile, mask, S = _atomic_ptr_and_mask(array, indices)
        val_tile = broadcast_to(Tile(val), S)
        Intrinsics.$intrinsic(ptr_tile, val_tile, mask, memory_order, memory_scope)
    end

    # N-D tile indices, tile val
    @eval @inline function $fname(array::TileArray{T, N},
                                   indices::NTuple{N, Tile{<:Integer}}, val::Tile{T};
                                   memory_order::Int=MemoryOrder.AcqRel,
                                   memory_scope::Int=MemScope.Device) where {T, N}
        ptr_tile, mask, S = _atomic_ptr_and_mask(array, indices)
        val_bc = broadcast_to(val, S)
        Intrinsics.$intrinsic(ptr_tile, val_bc, mask, memory_order, memory_scope)
    end

    # 1D convenience: single Tile index
    @eval @inline function $fname(array::TileArray{T, 1}, indices::Tile{<:Integer}, val::T;
                                   memory_order::Int=MemoryOrder.AcqRel,
                                   memory_scope::Int=MemScope.Device) where {T}
        $fname(array, (indices,), val; memory_order, memory_scope)
    end

    @eval @inline function $fname(array::TileArray{T, 1}, indices::Tile{<:Integer}, val::Tile{T};
                                   memory_order::Int=MemoryOrder.AcqRel,
                                   memory_scope::Int=MemScope.Device) where {T}
        $fname(array, (indices,), val; memory_order, memory_scope)
    end
end

# ============================================================================
# Atomic CAS operations
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
# Scalar index
@inline function atomic_cas(array::TileArray{T}, index, expected::T, desired::T;
                            memory_order::Int=MemoryOrder.AcqRel,
                            memory_scope::Int=MemScope.Device) where {T}
    ptr_tile, mask, _ = _atomic_ptr_and_mask(array, index)
    Intrinsics.atomic_cas(ptr_tile, Tile(expected), Tile(desired), mask,
                           memory_order, memory_scope)
end

# N-D tile indices, scalar expected/desired
@inline function atomic_cas(array::TileArray{T, N},
                            indices::NTuple{N, Tile{<:Integer}},
                            expected::T, desired::T;
                            memory_order::Int=MemoryOrder.AcqRel,
                            memory_scope::Int=MemScope.Device) where {T, N}
    ptr_tile, mask, S = _atomic_ptr_and_mask(array, indices)
    expected_tile = broadcast_to(Tile(expected), S)
    desired_tile = broadcast_to(Tile(desired), S)
    Intrinsics.atomic_cas(ptr_tile, expected_tile, desired_tile, mask,
                           memory_order, memory_scope)
end

# N-D tile indices, tile expected/desired
@inline function atomic_cas(array::TileArray{T, N},
                            indices::NTuple{N, Tile{<:Integer}},
                            expected::Tile{T}, desired::Tile{T};
                            memory_order::Int=MemoryOrder.AcqRel,
                            memory_scope::Int=MemScope.Device) where {T, N}
    ptr_tile, mask, S = _atomic_ptr_and_mask(array, indices)
    expected_bc = broadcast_to(expected, S)
    desired_bc = broadcast_to(desired, S)
    Intrinsics.atomic_cas(ptr_tile, expected_bc, desired_bc, mask,
                           memory_order, memory_scope)
end

# 1D convenience: single Tile index
@inline function atomic_cas(array::TileArray{T, 1}, indices::Tile{<:Integer},
                            expected::T, desired::T;
                            memory_order::Int=MemoryOrder.AcqRel,
                            memory_scope::Int=MemScope.Device) where {T}
    atomic_cas(array, (indices,), expected, desired; memory_order, memory_scope)
end

@inline function atomic_cas(array::TileArray{T, 1}, indices::Tile{<:Integer},
                            expected::Tile{T}, desired::Tile{T};
                            memory_order::Int=MemoryOrder.AcqRel,
                            memory_scope::Int=MemScope.Device) where {T}
    atomic_cas(array, (indices,), expected, desired; memory_order, memory_scope)
end
