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
# Tile-indexed atomic operations
# These accept Tile indices to perform atomic operations on multiple elements.
# ============================================================================

# --- Pointer/mask helper (N-dimensional) ---

@inline function _atomic_ptrs_mask(array::TileArray{T, N},
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
@inline function _atomic_ptrs_mask(array::TileArray{T, 1}, indices::Tile{<:Integer}) where {T}
    _atomic_ptrs_mask(array, (indices,))
end

# --- RMW operations (atomic_add, atomic_xchg) ---

const _ATOMIC_RMW_OPS = (
    (:add,  :atomic_add_tile),
    (:xchg, :atomic_xchg_tile),
)

for (op, intrinsic) in _ATOMIC_RMW_OPS
    fname = Symbol(:atomic_, op)

    # N-D with scalar value
    @eval @inline function $fname(array::TileArray{T, N},
                                   indices::NTuple{N, Tile{<:Integer}}, val::T;
                                   memory_order::Int=MemoryOrder.AcqRel,
                                   memory_scope::Int=MemScope.Device) where {T, N}
        ptr_tile, mask, S = _atomic_ptrs_mask(array, indices)
        val_tile = broadcast_to(Tile(val), S)
        Intrinsics.$intrinsic(ptr_tile, val_tile, mask, memory_order, memory_scope)
    end

    # N-D with tile value
    @eval @inline function $fname(array::TileArray{T, N},
                                   indices::NTuple{N, Tile{<:Integer}}, val::Tile{T};
                                   memory_order::Int=MemoryOrder.AcqRel,
                                   memory_scope::Int=MemScope.Device) where {T, N}
        ptr_tile, mask, S = _atomic_ptrs_mask(array, indices)
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

# --- CAS operations (separate due to different signature) ---

# N-D with scalar expected/desired
@inline function atomic_cas(array::TileArray{T, N},
                            indices::NTuple{N, Tile{<:Integer}},
                            expected::T, desired::T;
                            memory_order::Int=MemoryOrder.AcqRel,
                            memory_scope::Int=MemScope.Device) where {T, N}
    ptr_tile, mask, S = _atomic_ptrs_mask(array, indices)
    expected_tile = broadcast_to(Tile(expected), S)
    desired_tile = broadcast_to(Tile(desired), S)
    Intrinsics.atomic_cas_tile(ptr_tile, expected_tile, desired_tile, mask,
                               memory_order, memory_scope)
end

# N-D with tile expected/desired
@inline function atomic_cas(array::TileArray{T, N},
                            indices::NTuple{N, Tile{<:Integer}},
                            expected::Tile{T}, desired::Tile{T};
                            memory_order::Int=MemoryOrder.AcqRel,
                            memory_scope::Int=MemScope.Device) where {T, N}
    ptr_tile, mask, S = _atomic_ptrs_mask(array, indices)
    expected_bc = broadcast_to(expected, S)
    desired_bc = broadcast_to(desired, S)
    Intrinsics.atomic_cas_tile(ptr_tile, expected_bc, desired_bc, mask,
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

# ============================================================================
# Tile-space atomic operations
# These accept tile-space integer indices (like store) to atomically operate
# on contiguous tile-shaped blocks of an array.
# ============================================================================

# --- Pointer/mask helper for tile-space indexing ---

@inline function _tile_space_ptrs_mask(array::TileArray{T, N},
                                        index::NTuple{N, Integer},
                                        ::Val{Shape}) where {T, N, Shape}
    # Build per-dimension element index tiles (1-indexed)
    # For dim d: arange [1..Shape[d]], reshaped for N-D broadcasting, plus base offset
    idx_tiles = ntuple(Val(N)) do d
        bcast_shape = ntuple(i -> i == d ? Shape[d] : 1, Val(N))
        base = Int32((index[d] - 1) * Shape[d])
        reshape(arange((Shape[d],), Int32), bcast_shape) .+ Tile(base)
    end

    # 0-indexed linear offset: sum((idx[d] - 1) * stride[d])
    linear_idx = reduce(.+, ntuple(Val(N)) do d
        (idx_tiles[d] .- Tile(Int32(1))) .* Tile(array.strides[d])
    end)

    ptr_tile = Intrinsics.offset(array.ptr, linear_idx)

    # Bounds mask: 1 <= idx[d] <= size(array, d) for all d
    mask = reduce(.&, ntuple(Val(N)) do d
        (idx_tiles[d] .>= Tile(Int32(1))) .& (idx_tiles[d] .<= Tile(size(array, d)))
    end)

    (ptr_tile, mask)
end

# --- Tile-space atomic_add ---

# N-D tuple index + tile value (like store)
@inline function atomic_add(array::TileArray{T, N},
                            index::NTuple{N, Integer}, tile::Tile{T};
                            memory_order::Int=MemoryOrder.AcqRel,
                            memory_scope::Int=MemScope.Device) where {T, N}
    reshaped = _reshape_to_rank(tile, Val(N))
    ptr_tile, mask = _tile_space_ptrs_mask(array, index, Val(size(reshaped)))
    Intrinsics.atomic_add_tile(ptr_tile, reshaped, mask, memory_order, memory_scope)
end

# 1D convenience (scalar index)
@inline function atomic_add(array::TileArray{T, 1},
                            index::Integer, tile::Tile{T};
                            memory_order::Int=MemoryOrder.AcqRel,
                            memory_scope::Int=MemScope.Device) where {T}
    atomic_add(array, (index,), tile; memory_order, memory_scope)
end
