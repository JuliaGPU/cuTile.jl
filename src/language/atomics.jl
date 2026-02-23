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
@inline function atomic_cas(array::TileArray{T}, index, expected::T, desired::T;
                            memory_order::Int=MemoryOrder.AcqRel,
                            memory_scope::Int=MemScope.Device) where {T}
    Intrinsics.atomic_cas(array, index - One(), expected, desired, memory_order, memory_scope)
end

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
@inline function atomic_xchg(array::TileArray{T}, index, val::T;
                             memory_order::Int=MemoryOrder.AcqRel,
                             memory_scope::Int=MemScope.Device) where {T}
    Intrinsics.atomic_xchg(array, index - One(), val, memory_order, memory_scope)
end

"""
    atomic_add(array::TileArray, index, val; memory_order, memory_scope) -> T

Atomic addition. Atomically adds `val` to the value at `index` and returns
the original value. Index is 1-indexed.

# Example
```julia
old_val = ct.atomic_add(counters, idx, Int32(1))
```
"""
@inline function atomic_add(array::TileArray{T}, index, val::T;
                            memory_order::Int=MemoryOrder.AcqRel,
                            memory_scope::Int=MemScope.Device) where {T}
    Intrinsics.atomic_add(array, index - One(), val, memory_order, memory_scope)
end

# ============================================================================
# Tile-indexed atomic operations (scatter-gather style indexing)
# These accept Tile indices to perform atomic operations on multiple elements.
# ============================================================================

# --- Pointer/mask helpers (same pattern as gather/scatter in operations.jl) ---

@inline function _atomic_ptrs_mask(array::TileArray{T, 1}, indices::Tile{I}) where {T, I <: Integer}
    indices_0 = indices .- one(I)
    indices_i32 = convert(Tile{Int32}, indices_0)
    ptr_tile = Intrinsics.offset(array.ptr, indices_i32)
    zero_0d = Tile(Int32(0))
    size_0d = Tile(size(array, 1))
    mask = (indices_i32 .>= zero_0d) .& (indices_i32 .< size_0d)
    (ptr_tile, mask, size(indices))
end

@inline function _atomic_ptrs_mask(array::TileArray{T, 2},
                                    indices::Tuple{Tile{I0}, Tile{I1}}) where {T, I0 <: Integer, I1 <: Integer}
    idx0_0 = indices[1] .- one(I0)
    idx1_0 = indices[2] .- one(I1)

    S = broadcast_shape(size(indices[1]), size(indices[2]))
    idx0_bc = broadcast_to(idx0_0, S)
    idx1_bc = broadcast_to(idx1_0, S)

    idx0_i32 = convert(Tile{Int32}, idx0_bc)
    idx1_i32 = convert(Tile{Int32}, idx1_bc)

    stride0_0d = Tile(array.strides[1])
    stride1_0d = Tile(array.strides[2])
    stride0 = broadcast_to(stride0_0d, S)
    stride1 = broadcast_to(stride1_0d, S)

    linear_idx = idx0_i32 .* stride0 + idx1_i32 .* stride1
    ptr_tile = Intrinsics.offset(array.ptr, linear_idx)

    zero_0d = Tile(Int32(0))
    zero_bc = broadcast_to(zero_0d, S)
    size0_bc = broadcast_to(Tile(size(array, 1)), S)
    size1_bc = broadcast_to(Tile(size(array, 2)), S)

    mask0 = (idx0_i32 .>= zero_bc) .& (idx0_i32 .< size0_bc)
    mask1 = (idx1_i32 .>= zero_bc) .& (idx1_i32 .< size1_bc)
    mask = mask0 .& mask1

    (ptr_tile, mask, S)
end

# --- RMW operations (atomic_add, atomic_xchg) ---

const _ATOMIC_RMW_OPS = (
    (:add,  :atomic_add_tile),
    (:xchg, :atomic_xchg_tile),
)

for (op, intrinsic) in _ATOMIC_RMW_OPS
    fname = Symbol(:atomic_, op)

    # 1D with scalar value
    @eval @inline function $fname(array::TileArray{T, 1}, indices::Tile{I}, val::T;
                                   memory_order::Int=MemoryOrder.AcqRel,
                                   memory_scope::Int=MemScope.Device) where {T, I <: Integer}
        ptr_tile, mask, S = _atomic_ptrs_mask(array, indices)
        val_tile = broadcast_to(Tile(val), S)
        Intrinsics.$intrinsic(ptr_tile, val_tile, mask, memory_order, memory_scope)
    end

    # 1D with tile value
    @eval @inline function $fname(array::TileArray{T, 1}, indices::Tile{I}, val::Tile{T};
                                   memory_order::Int=MemoryOrder.AcqRel,
                                   memory_scope::Int=MemScope.Device) where {T, I <: Integer}
        ptr_tile, mask, _ = _atomic_ptrs_mask(array, indices)
        Intrinsics.$intrinsic(ptr_tile, val, mask, memory_order, memory_scope)
    end

    # 2D with scalar value
    @eval @inline function $fname(array::TileArray{T, 2},
                                   indices::Tuple{Tile{I0}, Tile{I1}}, val::T;
                                   memory_order::Int=MemoryOrder.AcqRel,
                                   memory_scope::Int=MemScope.Device) where {T, I0 <: Integer, I1 <: Integer}
        ptr_tile, mask, S = _atomic_ptrs_mask(array, indices)
        val_tile = broadcast_to(Tile(val), S)
        Intrinsics.$intrinsic(ptr_tile, val_tile, mask, memory_order, memory_scope)
    end

    # 2D with tile value
    @eval @inline function $fname(array::TileArray{T, 2},
                                   indices::Tuple{Tile{I0}, Tile{I1}}, val::Tile{T};
                                   memory_order::Int=MemoryOrder.AcqRel,
                                   memory_scope::Int=MemScope.Device) where {T, I0 <: Integer, I1 <: Integer}
        ptr_tile, mask, S = _atomic_ptrs_mask(array, indices)
        val_bc = broadcast_to(val, S)
        Intrinsics.$intrinsic(ptr_tile, val_bc, mask, memory_order, memory_scope)
    end
end

# --- CAS operations (separate due to different signature) ---

# 1D with scalar expected/desired
@inline function atomic_cas(array::TileArray{T, 1}, indices::Tile{I},
                            expected::T, desired::T;
                            memory_order::Int=MemoryOrder.AcqRel,
                            memory_scope::Int=MemScope.Device) where {T, I <: Integer}
    ptr_tile, mask, S = _atomic_ptrs_mask(array, indices)
    expected_tile = broadcast_to(Tile(expected), S)
    desired_tile = broadcast_to(Tile(desired), S)
    Intrinsics.atomic_cas_tile(ptr_tile, expected_tile, desired_tile, mask,
                               memory_order, memory_scope)
end

# 1D with tile expected/desired
@inline function atomic_cas(array::TileArray{T, 1}, indices::Tile{I},
                            expected::Tile{T}, desired::Tile{T};
                            memory_order::Int=MemoryOrder.AcqRel,
                            memory_scope::Int=MemScope.Device) where {T, I <: Integer}
    ptr_tile, mask, _ = _atomic_ptrs_mask(array, indices)
    Intrinsics.atomic_cas_tile(ptr_tile, expected, desired, mask,
                               memory_order, memory_scope)
end

# 2D with scalar expected/desired
@inline function atomic_cas(array::TileArray{T, 2},
                            indices::Tuple{Tile{I0}, Tile{I1}},
                            expected::T, desired::T;
                            memory_order::Int=MemoryOrder.AcqRel,
                            memory_scope::Int=MemScope.Device) where {T, I0 <: Integer, I1 <: Integer}
    ptr_tile, mask, S = _atomic_ptrs_mask(array, indices)
    expected_tile = broadcast_to(Tile(expected), S)
    desired_tile = broadcast_to(Tile(desired), S)
    Intrinsics.atomic_cas_tile(ptr_tile, expected_tile, desired_tile, mask,
                               memory_order, memory_scope)
end

# 2D with tile expected/desired
@inline function atomic_cas(array::TileArray{T, 2},
                            indices::Tuple{Tile{I0}, Tile{I1}},
                            expected::Tile{T}, desired::Tile{T};
                            memory_order::Int=MemoryOrder.AcqRel,
                            memory_scope::Int=MemScope.Device) where {T, I0 <: Integer, I1 <: Integer}
    ptr_tile, mask, S = _atomic_ptrs_mask(array, indices)
    expected_bc = broadcast_to(expected, S)
    desired_bc = broadcast_to(desired, S)
    Intrinsics.atomic_cas_tile(ptr_tile, expected_bc, desired_bc, mask,
                               memory_order, memory_scope)
end
