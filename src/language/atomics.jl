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
@inline function atomic_cas(array::TileArray{T, N}, index, expected::T, desired::T;
                            memory_order::Int=MemoryOrder.AcqRel,
                            memory_scope::Int=MemScope.Device) where {T, N}
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
@inline function atomic_xchg(array::TileArray{T, N}, index, val::T;
                             memory_order::Int=MemoryOrder.AcqRel,
                             memory_scope::Int=MemScope.Device) where {T, N}
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
@inline function atomic_add(array::TileArray{T, N}, index, val::T;
                            memory_order::Int=MemoryOrder.AcqRel,
                            memory_scope::Int=MemScope.Device) where {T, N}
    Intrinsics.atomic_add(array, index - One(), val, memory_order, memory_scope)
end
