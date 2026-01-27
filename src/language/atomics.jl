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

# ============================================================================
# Tile-wise atomic operations
# These accept Tile indices to perform atomic operations on multiple elements.
# ============================================================================

# Operation registry: (name, intrinsic) pairs for RMW operations
# To add a new operation: 1) add entry here, 2) add intrinsic in compiler/intrinsics/atomics.jl
const ATOMIC_RMW_OPS = [
    (:add,  :atomic_add_tile),
    (:xchg, :atomic_xchg_tile),
]

# ============================================================================
# Pointer/Mask Helpers
# ============================================================================

"""
Compute pointer tile and bounds mask for 1D tile-wise atomic operations.
Returns (ptr_tile, mask, output_shape).
"""
@inline function _atomic_ptr_mask_1d(array::TileArray{T, 1}, indices::Tile{I, S}) where {T, I <: Integer, S}
    indices_0 = indices .- One()
    indices_i32 = astype(indices_0, Int32)
    ptr_tile = Intrinsics.offset(array.ptr, indices_i32)
    mask = (indices_i32 .>= Tile(Int32(0))) .& (indices_i32 .< Tile(array.sizes[1]))
    (ptr_tile, mask, S)
end

"""
Compute pointer tile and bounds mask for 2D tile-wise atomic operations.
Returns (ptr_tile, mask, output_shape).
"""
@inline function _atomic_ptr_mask_2d(array::TileArray{T, 2},
                                      indices::Tuple{Tile{I0, S0}, Tile{I1, S1}}) where {T, I0 <: Integer, I1 <: Integer, S0, S1}
    idx0_0 = indices[1] .- One()
    idx1_0 = indices[2] .- One()
    S = broadcast_shape(S0, S1)
    idx0_i32 = astype(broadcast_to(idx0_0, S), Int32)
    idx1_i32 = astype(broadcast_to(idx1_0, S), Int32)
    linear_idx = idx0_i32 .* Tile(array.strides[1]) .+ idx1_i32 .* Tile(array.strides[2])
    ptr_tile = Intrinsics.offset(array.ptr, linear_idx)
    mask = (idx0_i32 .>= Tile(Int32(0))) .& (idx0_i32 .< Tile(array.sizes[1])) .&
           (idx1_i32 .>= Tile(Int32(0))) .& (idx1_i32 .< Tile(array.sizes[2]))
    (ptr_tile, mask, S)
end

"""
Compute pointer tile and bounds mask for 2D tile-wise operations with value broadcasting.
Returns (ptr_tile, mask, output_shape, idx0_i32, idx1_i32) for value broadcasting.
"""
@inline function _atomic_ptr_mask_2d_bc(array::TileArray{T, 2},
                                         indices::Tuple{Tile{I0, S0}, Tile{I1, S1}},
                                         Sval) where {T, I0 <: Integer, I1 <: Integer, S0, S1}
    idx0_0 = indices[1] .- One()
    idx1_0 = indices[2] .- One()
    S = broadcast_shape(broadcast_shape(S0, S1), Sval)
    idx0_i32 = astype(broadcast_to(idx0_0, S), Int32)
    idx1_i32 = astype(broadcast_to(idx1_0, S), Int32)
    linear_idx = idx0_i32 .* Tile(array.strides[1]) .+ idx1_i32 .* Tile(array.strides[2])
    ptr_tile = Intrinsics.offset(array.ptr, linear_idx)
    mask = (idx0_i32 .>= Tile(Int32(0))) .& (idx0_i32 .< Tile(array.sizes[1])) .&
           (idx1_i32 .>= Tile(Int32(0))) .& (idx1_i32 .< Tile(array.sizes[2]))
    (ptr_tile, mask, S)
end

"""
Compute pointer tile and bounds mask for N-dimensional tile-level atomic operations.
`index` is an N-tuple of tile-space indices (1-indexed).
`Shape` is the tile shape.
Returns (ptr_tile, mask).
"""
@inline function _tile_level_atomic_args(array::TileArray{T, N}, index::NTuple{N, Integer},
                                          ::Val{Shape}) where {T, N, Shape}
    # Create 1-indexed element index tiles for each dimension
    # For dim d: arange [1..Shape[d]], reshaped for broadcasting, plus base offset
    idx_tiles = ntuple(N) do d
        bcast_shape = ntuple(i -> i == d ? Shape[d] : 1, N)
        base = Int32((index[d] - 1) * Shape[d])
        reshape(arange((Shape[d],), Int32), bcast_shape) .+ Tile(base)
    end

    # Compute 0-indexed linear offset: sum((idx[d] - 1) * stride[d])
    linear_idx = reduce(.+, ntuple(N) do d
        (idx_tiles[d] .- Tile(Int32(1))) .* Tile(array.strides[d])
    end)

    ptr_tile = Intrinsics.offset(array.ptr, linear_idx)

    # Bounds mask: 1 <= idx[d] <= sizes[d] for all d
    mask = reduce(.&, ntuple(N) do d
        (idx_tiles[d] .>= Tile(Int32(1))) .& (idx_tiles[d] .<= Tile(array.sizes[d]))
    end)

    (ptr_tile, mask)
end

# ============================================================================
# Generated RMW Operations (add, xchg, ...)
# ============================================================================

for (op, intrinsic) in ATOMIC_RMW_OPS
    fname = Symbol("atomic_", op)
    doc_op = string(op)

    # 1D tile-wise with scalar value
    @eval begin
        @doc """
            $($fname)(array::TileArray{T, 1}, indices::Tile, val::T; memory_order, memory_scope) -> Tile{T, S}

        Tile-wise atomic $($doc_op) on a 1D array.
        Indices are 1-indexed. Out-of-bounds indices are masked.
        """
        @inline function $fname(array::TileArray{T, 1}, indices::Tile{I, S}, val::T;
                                memory_order::Int=MemoryOrder.AcqRel,
                                memory_scope::Int=MemScope.Device) where {T, I <: Integer, S}
            ptr_tile, mask, _ = _atomic_ptr_mask_1d(array, indices)
            val_tile = broadcast_to(Tile(val), S)
            Intrinsics.$intrinsic(ptr_tile, val_tile, mask, memory_order, memory_scope)
        end
    end

    # 1D tile-wise with tile value
    @eval begin
        @doc """
            $($fname)(array::TileArray{T, 1}, indices::Tile, val::Tile{T, S}; ...) -> Tile{T, S}

        Tile-wise atomic $($doc_op) with a tile of values.
        """
        @inline function $fname(array::TileArray{T, 1}, indices::Tile{I, S}, val::Tile{T, S};
                                memory_order::Int=MemoryOrder.AcqRel,
                                memory_scope::Int=MemScope.Device) where {T, I <: Integer, S}
            ptr_tile, mask, _ = _atomic_ptr_mask_1d(array, indices)
            Intrinsics.$intrinsic(ptr_tile, val, mask, memory_order, memory_scope)
        end
    end

    # 2D tile-wise with scalar value
    @eval begin
        @doc """
            $($fname)(array::TileArray{T, 2}, indices::Tuple{Tile, Tile}, val::T; ...) -> Tile{T, S}

        Tile-wise atomic $($doc_op) on a 2D array.
        """
        @inline function $fname(array::TileArray{T, 2},
                                indices::Tuple{Tile{I0, S0}, Tile{I1, S1}}, val::T;
                                memory_order::Int=MemoryOrder.AcqRel,
                                memory_scope::Int=MemScope.Device) where {T, I0 <: Integer, I1 <: Integer, S0, S1}
            ptr_tile, mask, S = _atomic_ptr_mask_2d(array, indices)
            val_tile = broadcast_to(Tile(val), S)
            Intrinsics.$intrinsic(ptr_tile, val_tile, mask, memory_order, memory_scope)
        end
    end

    # 2D tile-wise with tile value
    @eval begin
        @doc """
            $($fname)(array::TileArray{T, 2}, indices::Tuple{Tile, Tile}, val::Tile; ...) -> Tile{T, S}

        Tile-wise atomic $($doc_op) on a 2D array with a tile of values.
        """
        @inline function $fname(array::TileArray{T, 2},
                                indices::Tuple{Tile{I0, S0}, Tile{I1, S1}}, val::Tile{T, Stile};
                                memory_order::Int=MemoryOrder.AcqRel,
                                memory_scope::Int=MemScope.Device) where {T, I0 <: Integer, I1 <: Integer, S0, S1, Stile}
            ptr_tile, mask, S = _atomic_ptr_mask_2d_bc(array, indices, Stile)
            val_bc = broadcast_to(val, S)
            Intrinsics.$intrinsic(ptr_tile, val_bc, mask, memory_order, memory_scope)
        end
    end

    # Tile-level N-D (tuple of integer indices, tile value)
    @eval begin
        @doc """
            $($fname)(array::TileArray{T, N}, index, tile::Tile{T, Shape}; ...) -> Tile{T, Shape}

        Atomic $($doc_op) at tile-level index (like `store`).
        Index can be an Integer (1D) or NTuple{N, Integer} (N-D).
        """
        @inline function $fname(array::TileArray{T, N}, index::NTuple{N, Integer}, tile::Tile{T, Shape};
                                memory_order::Int=MemoryOrder.AcqRel,
                                memory_scope::Int=MemScope.Device) where {T, N, Shape}
            ptr_tile, mask = _tile_level_atomic_args(array, index, Val(Shape))
            Intrinsics.$intrinsic(ptr_tile, tile, mask, memory_order, memory_scope)
        end
    end

    # Tile-level 1D convenience (integer index -> 1-tuple)
    @eval begin
        @inline function $fname(array::TileArray{T, 1}, index::Integer, tile::Tile{T, Shape};
                                memory_order::Int=MemoryOrder.AcqRel,
                                memory_scope::Int=MemScope.Device) where {T, Shape}
            $fname(array, (index,), tile; memory_order, memory_scope)
        end
    end
end

# ============================================================================
# CAS Operations (separate - has expected + desired args)
# ============================================================================

"""
    atomic_cas(array::TileArray{T, 1}, indices::Tile, expected, desired; memory_order, memory_scope) -> Tile{T, S}

Tile-wise atomic compare-and-swap on a 1D array.
Indices are 1-indexed. Out-of-bounds indices are masked.
"""
@inline function atomic_cas(array::TileArray{T, 1}, indices::Tile{I, S},
                            expected::T, desired::T;
                            memory_order::Int=MemoryOrder.AcqRel,
                            memory_scope::Int=MemScope.Device) where {T, I <: Integer, S}
    ptr_tile, mask, _ = _atomic_ptr_mask_1d(array, indices)
    expected_tile = broadcast_to(Tile(expected), S)
    desired_tile = broadcast_to(Tile(desired), S)
    Intrinsics.atomic_cas_tile(ptr_tile, expected_tile, desired_tile, mask,
                               memory_order, memory_scope)
end

"""
    atomic_cas(array::TileArray{T, 1}, indices::Tile, expected::Tile, desired::Tile; ...) -> Tile{T, S}

Tile-wise atomic compare-and-swap with tiles of expected/desired values.
"""
@inline function atomic_cas(array::TileArray{T, 1}, indices::Tile{I, S},
                            expected::Tile{T, S}, desired::Tile{T, S};
                            memory_order::Int=MemoryOrder.AcqRel,
                            memory_scope::Int=MemScope.Device) where {T, I <: Integer, S}
    ptr_tile, mask, _ = _atomic_ptr_mask_1d(array, indices)
    Intrinsics.atomic_cas_tile(ptr_tile, expected, desired, mask,
                               memory_order, memory_scope)
end

"""
    atomic_cas(array::TileArray{T, 2}, indices::Tuple{Tile, Tile}, expected, desired; ...) -> Tile{T, S}

Tile-wise atomic compare-and-swap on a 2D array.
"""
@inline function atomic_cas(array::TileArray{T, 2},
                            indices::Tuple{Tile{I0, S0}, Tile{I1, S1}},
                            expected::T, desired::T;
                            memory_order::Int=MemoryOrder.AcqRel,
                            memory_scope::Int=MemScope.Device) where {T, I0 <: Integer, I1 <: Integer, S0, S1}
    ptr_tile, mask, S = _atomic_ptr_mask_2d(array, indices)
    expected_tile = broadcast_to(Tile(expected), S)
    desired_tile = broadcast_to(Tile(desired), S)
    Intrinsics.atomic_cas_tile(ptr_tile, expected_tile, desired_tile, mask,
                               memory_order, memory_scope)
end

"""
    atomic_cas(array::TileArray{T, 2}, indices::Tuple{Tile, Tile}, expected::Tile, desired::Tile; ...) -> Tile{T, S}

Tile-wise atomic compare-and-swap on a 2D array with tiles of values.
"""
@inline function atomic_cas(array::TileArray{T, 2},
                            indices::Tuple{Tile{I0, S0}, Tile{I1, S1}},
                            expected::Tile{T, Se}, desired::Tile{T, Sd};
                            memory_order::Int=MemoryOrder.AcqRel,
                            memory_scope::Int=MemScope.Device) where {T, I0 <: Integer, I1 <: Integer, S0, S1, Se, Sd}
    S = broadcast_shape(broadcast_shape(broadcast_shape(S0, S1), Se), Sd)
    ptr_tile, mask, _ = _atomic_ptr_mask_2d_bc(array, indices, Se)
    expected_bc = broadcast_to(expected, S)
    desired_bc = broadcast_to(desired, S)
    Intrinsics.atomic_cas_tile(ptr_tile, expected_bc, desired_bc, mask,
                               memory_order, memory_scope)
end

"""
    atomic_cas(array::TileArray{T, N}, index, expected::Tile, desired::Tile; ...) -> Tile{T, Shape}

Atomic compare-and-swap at tile-level index (like `store`).
Index can be an Integer (1D) or NTuple{N, Integer} (N-D).
"""
@inline function atomic_cas(array::TileArray{T, N}, index::NTuple{N, Integer},
                            expected::Tile{T, Shape}, desired::Tile{T, Shape};
                            memory_order::Int=MemoryOrder.AcqRel,
                            memory_scope::Int=MemScope.Device) where {T, N, Shape}
    ptr_tile, mask = _tile_level_atomic_args(array, index, Val(Shape))
    Intrinsics.atomic_cas_tile(ptr_tile, expected, desired, mask,
                               memory_order, memory_scope)
end

# 1D convenience (integer index -> 1-tuple)
@inline function atomic_cas(array::TileArray{T, 1}, index::Integer,
                            expected::Tile{T, Shape}, desired::Tile{T, Shape};
                            memory_order::Int=MemoryOrder.AcqRel,
                            memory_scope::Int=MemScope.Device) where {T, Shape}
    atomic_cas(array, (index,), expected, desired; memory_order, memory_scope)
end
