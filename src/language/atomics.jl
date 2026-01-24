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

"""
    atomic_cas(array::TileArray{T, 1}, indices::Tile, expected, desired; memory_order, memory_scope) -> Tile{T, S}

Tile-wise atomic compare-and-swap on a 1D array.
Atomically compares values at `indices` with `expected`, and if equal, replaces with `desired`.
Returns a tile of original values. Indices are 1-indexed.
Out-of-bounds indices are masked (no operation performed).

# Example
```julia
indices = ct.arange((16,), Int)
old_vals = ct.atomic_cas(arr, indices, Int32(0), Int32(1))
```
"""
@inline function atomic_cas(array::TileArray{T, 1}, indices::Tile{I, S},
                            expected::T, desired::T;
                            memory_order::Int=MemoryOrder.AcqRel,
                            memory_scope::Int=MemScope.Device) where {T, I <: Integer, S}
    # Convert to 0-indexed
    indices_0 = indices .- One()

    # Convert to Int32 for consistency with array.sizes
    indices_i32 = astype(indices_0, Int32)

    # Compute pointer tile
    ptr_tile = Intrinsics.offset(array.ptr, indices_i32)

    # Bounds mask: 0 <= indices_i32 < size
    zero_0d = Tile(Int32(0))
    size_0d = Tile(array.sizes[1])  # Already Int32
    ge_zero = indices_i32 .>= zero_0d
    lt_size = indices_i32 .< size_0d
    mask = ge_zero .& lt_size

    # Broadcast values to tile shape
    expected_tile = broadcast_to(Tile(expected), S)
    desired_tile = broadcast_to(Tile(desired), S)

    Intrinsics.atomic_cas_tile(ptr_tile, expected_tile, desired_tile, mask,
                               memory_order, memory_scope)
end

"""
    atomic_cas(array::TileArray{T, 2}, indices::Tuple{Tile, Tile}, expected, desired; ...) -> Tile{T, S}

Tile-wise atomic compare-and-swap on a 2D array.
Index tiles are broadcast to a common shape.
"""
@inline function atomic_cas(array::TileArray{T, 2},
                            indices::Tuple{Tile{I0, S0}, Tile{I1, S1}},
                            expected::T, desired::T;
                            memory_order::Int=MemoryOrder.AcqRel,
                            memory_scope::Int=MemScope.Device) where {T, I0 <: Integer, I1 <: Integer, S0, S1}
    idx0_0 = indices[1] .- One()
    idx1_0 = indices[2] .- One()

    S = broadcast_shape(S0, S1)
    idx0_i32 = astype(broadcast_to(idx0_0, S), Int32)
    idx1_i32 = astype(broadcast_to(idx1_0, S), Int32)

    linear_idx = idx0_i32 .* Tile(array.strides[1]) .+ idx1_i32 .* Tile(array.strides[2])
    ptr_tile = Intrinsics.offset(array.ptr, linear_idx)

    mask = (idx0_i32 .>= Tile(Int32(0))) .& (idx0_i32 .< Tile(array.sizes[1])) .&
           (idx1_i32 .>= Tile(Int32(0))) .& (idx1_i32 .< Tile(array.sizes[2]))

    expected_tile = broadcast_to(Tile(expected), S)
    desired_tile = broadcast_to(Tile(desired), S)

    Intrinsics.atomic_cas_tile(ptr_tile, expected_tile, desired_tile, mask,
                               memory_order, memory_scope)
end

"""
    atomic_xchg(array::TileArray{T, 1}, indices::Tile, val; memory_order, memory_scope) -> Tile{T, S}

Tile-wise atomic exchange on a 1D array.
Atomically replaces values at `indices` with `val` and returns original values.
Indices are 1-indexed. Out-of-bounds indices are masked.

# Example
```julia
indices = ct.arange((16,), Int)
old_vals = ct.atomic_xchg(arr, indices, Int32(42))
```
"""
@inline function atomic_xchg(array::TileArray{T, 1}, indices::Tile{I, S}, val::T;
                             memory_order::Int=MemoryOrder.AcqRel,
                             memory_scope::Int=MemScope.Device) where {T, I <: Integer, S}
    # Convert to 0-indexed
    indices_0 = indices .- One()
    indices_i32 = astype(indices_0, Int32)

    # Compute pointer tile
    ptr_tile = Intrinsics.offset(array.ptr, indices_i32)

    # Bounds mask
    zero_0d = Tile(Int32(0))
    size_0d = Tile(array.sizes[1])
    ge_zero = indices_i32 .>= zero_0d
    lt_size = indices_i32 .< size_0d
    mask = ge_zero .& lt_size

    # Broadcast value to tile shape
    val_tile = broadcast_to(Tile(val), S)

    Intrinsics.atomic_xchg_tile(ptr_tile, val_tile, mask, memory_order, memory_scope)
end

"""
    atomic_xchg(array::TileArray{T, 2}, indices::Tuple{Tile, Tile}, val; ...) -> Tile{T, S}

Tile-wise atomic exchange on a 2D array.
"""
@inline function atomic_xchg(array::TileArray{T, 2},
                             indices::Tuple{Tile{I0, S0}, Tile{I1, S1}}, val::T;
                             memory_order::Int=MemoryOrder.AcqRel,
                             memory_scope::Int=MemScope.Device) where {T, I0 <: Integer, I1 <: Integer, S0, S1}
    idx0_0 = indices[1] .- One()
    idx1_0 = indices[2] .- One()

    S = broadcast_shape(S0, S1)
    idx0_i32 = astype(broadcast_to(idx0_0, S), Int32)
    idx1_i32 = astype(broadcast_to(idx1_0, S), Int32)

    linear_idx = idx0_i32 .* Tile(array.strides[1]) .+ idx1_i32 .* Tile(array.strides[2])
    ptr_tile = Intrinsics.offset(array.ptr, linear_idx)

    mask = (idx0_i32 .>= Tile(Int32(0))) .& (idx0_i32 .< Tile(array.sizes[1])) .&
           (idx1_i32 .>= Tile(Int32(0))) .& (idx1_i32 .< Tile(array.sizes[2]))

    val_tile = broadcast_to(Tile(val), S)

    Intrinsics.atomic_xchg_tile(ptr_tile, val_tile, mask, memory_order, memory_scope)
end

"""
    atomic_add(array::TileArray{T, 1}, indices::Tile, val; memory_order, memory_scope) -> Tile{T, S}

Tile-wise atomic addition on a 1D array.
Atomically adds `val` to values at `indices` and returns original values.
Indices are 1-indexed. Out-of-bounds indices are masked.

# Example
```julia
indices = ct.arange((16,), Int)
old_vals = ct.atomic_add(counters, indices, Int32(1))
```
"""
@inline function atomic_add(array::TileArray{T, 1}, indices::Tile{I, S}, val::T;
                            memory_order::Int=MemoryOrder.AcqRel,
                            memory_scope::Int=MemScope.Device) where {T, I <: Integer, S}
    # Convert to 0-indexed
    indices_0 = indices .- One()
    indices_i32 = astype(indices_0, Int32)

    # Compute pointer tile
    ptr_tile = Intrinsics.offset(array.ptr, indices_i32)

    # Bounds mask
    zero_0d = Tile(Int32(0))
    size_0d = Tile(array.sizes[1])
    ge_zero = indices_i32 .>= zero_0d
    lt_size = indices_i32 .< size_0d
    mask = ge_zero .& lt_size

    # Broadcast value to tile shape
    val_tile = broadcast_to(Tile(val), S)

    Intrinsics.atomic_add_tile(ptr_tile, val_tile, mask, memory_order, memory_scope)
end

"""
    atomic_add(array::TileArray{T, 2}, indices::Tuple{Tile, Tile}, val; ...) -> Tile{T, S}

Tile-wise atomic addition on a 2D array.
"""
@inline function atomic_add(array::TileArray{T, 2},
                            indices::Tuple{Tile{I0, S0}, Tile{I1, S1}}, val::T;
                            memory_order::Int=MemoryOrder.AcqRel,
                            memory_scope::Int=MemScope.Device) where {T, I0 <: Integer, I1 <: Integer, S0, S1}
    idx0_0 = indices[1] .- One()
    idx1_0 = indices[2] .- One()

    S = broadcast_shape(S0, S1)
    idx0_i32 = astype(broadcast_to(idx0_0, S), Int32)
    idx1_i32 = astype(broadcast_to(idx1_0, S), Int32)

    linear_idx = idx0_i32 .* Tile(array.strides[1]) .+ idx1_i32 .* Tile(array.strides[2])
    ptr_tile = Intrinsics.offset(array.ptr, linear_idx)

    mask = (idx0_i32 .>= Tile(Int32(0))) .& (idx0_i32 .< Tile(array.sizes[1])) .&
           (idx1_i32 .>= Tile(Int32(0))) .& (idx1_i32 .< Tile(array.sizes[2]))

    val_tile = broadcast_to(Tile(val), S)

    Intrinsics.atomic_add_tile(ptr_tile, val_tile, mask, memory_order, memory_scope)
end

# ============================================================================
# Tile-wise atomic operations with Tile values
# These accept both Tile indices AND Tile values (like scatter does).
# ============================================================================

"""
    atomic_add(array::TileArray{T, 1}, indices::Tile, val::Tile; ...) -> Tile{T, S}

Tile-wise atomic addition with a tile of values.
Each element of `val` is atomically added to the corresponding index position.
"""
@inline function atomic_add(array::TileArray{T, 1}, indices::Tile{I, S},
                            val::Tile{T, S};
                            memory_order::Int=MemoryOrder.AcqRel,
                            memory_scope::Int=MemScope.Device) where {T, I <: Integer, S}
    # Convert to 0-indexed
    indices_0 = indices .- One()
    indices_i32 = astype(indices_0, Int32)

    # Compute pointer tile
    ptr_tile = Intrinsics.offset(array.ptr, indices_i32)

    # Bounds mask
    zero_0d = Tile(Int32(0))
    size_0d = Tile(array.sizes[1])
    ge_zero = indices_i32 .>= zero_0d
    lt_size = indices_i32 .< size_0d
    mask = ge_zero .& lt_size

    Intrinsics.atomic_add_tile(ptr_tile, val, mask, memory_order, memory_scope)
end

"""
    atomic_xchg(array::TileArray{T, 1}, indices::Tile, val::Tile; ...) -> Tile{T, S}

Tile-wise atomic exchange with a tile of values.
"""
@inline function atomic_xchg(array::TileArray{T, 1}, indices::Tile{I, S},
                             val::Tile{T, S};
                             memory_order::Int=MemoryOrder.AcqRel,
                             memory_scope::Int=MemScope.Device) where {T, I <: Integer, S}
    indices_0 = indices .- One()
    indices_i32 = astype(indices_0, Int32)
    ptr_tile = Intrinsics.offset(array.ptr, indices_i32)

    zero_0d = Tile(Int32(0))
    size_0d = Tile(array.sizes[1])
    ge_zero = indices_i32 .>= zero_0d
    lt_size = indices_i32 .< size_0d
    mask = ge_zero .& lt_size

    Intrinsics.atomic_xchg_tile(ptr_tile, val, mask, memory_order, memory_scope)
end

"""
    atomic_cas(array::TileArray{T, 1}, indices::Tile, expected::Tile, desired::Tile; ...) -> Tile{T, S}

Tile-wise atomic compare-and-swap with tiles of expected/desired values.
"""
@inline function atomic_cas(array::TileArray{T, 1}, indices::Tile{I, S},
                            expected::Tile{T, S}, desired::Tile{T, S};
                            memory_order::Int=MemoryOrder.AcqRel,
                            memory_scope::Int=MemScope.Device) where {T, I <: Integer, S}
    indices_0 = indices .- One()
    indices_i32 = astype(indices_0, Int32)
    ptr_tile = Intrinsics.offset(array.ptr, indices_i32)

    zero_0d = Tile(Int32(0))
    size_0d = Tile(array.sizes[1])
    ge_zero = indices_i32 .>= zero_0d
    lt_size = indices_i32 .< size_0d
    mask = ge_zero .& lt_size

    Intrinsics.atomic_cas_tile(ptr_tile, expected, desired, mask,
                               memory_order, memory_scope)
end

"""
    atomic_add(array::TileArray{T, 2}, indices::Tuple{Tile, Tile}, val::Tile; ...) -> Tile{T, S}

Tile-wise atomic addition on a 2D array with a tile of values.
"""
@inline function atomic_add(array::TileArray{T, 2},
                            indices::Tuple{Tile{I0, S0}, Tile{I1, S1}},
                            val::Tile{T, Stile};
                            memory_order::Int=MemoryOrder.AcqRel,
                            memory_scope::Int=MemScope.Device) where {T, I0 <: Integer, I1 <: Integer, S0, S1, Stile}
    idx0_0 = indices[1] .- One()
    idx1_0 = indices[2] .- One()

    # Broadcast indices and value to common shape
    S = broadcast_shape(broadcast_shape(S0, S1), Stile)
    idx0_i32 = astype(broadcast_to(idx0_0, S), Int32)
    idx1_i32 = astype(broadcast_to(idx1_0, S), Int32)
    val_bc = broadcast_to(val, S)

    # Linear index and pointer - scalars broadcast automatically
    linear_idx = idx0_i32 .* Tile(array.strides[1]) .+ idx1_i32 .* Tile(array.strides[2])
    ptr_tile = Intrinsics.offset(array.ptr, linear_idx)

    # Bounds mask - scalars broadcast automatically
    mask = (idx0_i32 .>= Tile(Int32(0))) .& (idx0_i32 .< Tile(array.sizes[1])) .&
           (idx1_i32 .>= Tile(Int32(0))) .& (idx1_i32 .< Tile(array.sizes[2]))

    Intrinsics.atomic_add_tile(ptr_tile, val_bc, mask, memory_order, memory_scope)
end

"""
    atomic_xchg(array::TileArray{T, 2}, indices::Tuple{Tile, Tile}, val::Tile; ...) -> Tile{T, S}

Tile-wise atomic exchange on a 2D array with a tile of values.
"""
@inline function atomic_xchg(array::TileArray{T, 2},
                             indices::Tuple{Tile{I0, S0}, Tile{I1, S1}},
                             val::Tile{T, Stile};
                             memory_order::Int=MemoryOrder.AcqRel,
                             memory_scope::Int=MemScope.Device) where {T, I0 <: Integer, I1 <: Integer, S0, S1, Stile}
    idx0_0 = indices[1] .- One()
    idx1_0 = indices[2] .- One()

    S = broadcast_shape(broadcast_shape(S0, S1), Stile)
    idx0_i32 = astype(broadcast_to(idx0_0, S), Int32)
    idx1_i32 = astype(broadcast_to(idx1_0, S), Int32)
    val_bc = broadcast_to(val, S)

    linear_idx = idx0_i32 .* Tile(array.strides[1]) .+ idx1_i32 .* Tile(array.strides[2])
    ptr_tile = Intrinsics.offset(array.ptr, linear_idx)

    mask = (idx0_i32 .>= Tile(Int32(0))) .& (idx0_i32 .< Tile(array.sizes[1])) .&
           (idx1_i32 .>= Tile(Int32(0))) .& (idx1_i32 .< Tile(array.sizes[2]))

    Intrinsics.atomic_xchg_tile(ptr_tile, val_bc, mask, memory_order, memory_scope)
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
    idx0_0 = indices[1] .- One()
    idx1_0 = indices[2] .- One()

    S = broadcast_shape(broadcast_shape(broadcast_shape(S0, S1), Se), Sd)
    idx0_i32 = astype(broadcast_to(idx0_0, S), Int32)
    idx1_i32 = astype(broadcast_to(idx1_0, S), Int32)
    expected_bc = broadcast_to(expected, S)
    desired_bc = broadcast_to(desired, S)

    linear_idx = idx0_i32 .* Tile(array.strides[1]) .+ idx1_i32 .* Tile(array.strides[2])
    ptr_tile = Intrinsics.offset(array.ptr, linear_idx)

    mask = (idx0_i32 .>= Tile(Int32(0))) .& (idx0_i32 .< Tile(array.sizes[1])) .&
           (idx1_i32 .>= Tile(Int32(0))) .& (idx1_i32 .< Tile(array.sizes[2]))

    Intrinsics.atomic_cas_tile(ptr_tile, expected_bc, desired_bc, mask,
                               memory_order, memory_scope)
end

# ============================================================================
# Tile-level atomic operations (like store)
# These accept integer tile-space indices and atomically operate on a tile block.
# ============================================================================

"""
    atomic_add(array::TileArray{T, 1}, index::Integer, tile::Tile{T, Shape}; ...) -> Tile{T, Shape}

Atomic addition at tile-level index (like `store`).
Atomically adds each element of `tile` to the corresponding array position.
The tile-space index `index` is 1-indexed.

# Example
```julia
# Atomically add a tile of values starting at tile position 2
tile = ct.load(other_arr, 1)
ct.atomic_add(arr, 2, tile)  # Adds to elements 17:32 for tile size 16
```
"""
@inline function atomic_add(array::TileArray{T, 1}, index::Integer, tile::Tile{T, Shape};
                            memory_order::Int=MemoryOrder.AcqRel,
                            memory_scope::Int=MemScope.Device) where {T, Shape}
    tile_size = Int32(Shape[1])
    # Compute 1-indexed element indices
    # For tile index i, elements are at positions: (i-1)*tile_size + 1, ..., (i-1)*tile_size + tile_size
    # arange returns 1-indexed values [1, 2, ..., tile_size]
    base = Tile(Int32((index - 1) * tile_size))
    element_indices = arange(Shape, Int32) .+ base

    atomic_add(array, element_indices, tile; memory_order, memory_scope)
end

"""
    atomic_add(array::TileArray{T, 2}, index::NTuple{2, Integer}, tile::Tile{T, Shape}; ...) -> Tile{T, Shape}

Atomic addition at tile-level 2D index (like `store`).
The tile-space indices are 1-indexed.

# Example
```julia
# Atomically add a 16x16 tile at tile position (2, 3)
ct.atomic_add(arr, (2, 3), tile)
```
"""
@inline function atomic_add(array::TileArray{T, 2}, index::NTuple{2, Integer},
                            tile::Tile{T, Shape};
                            memory_order::Int=MemoryOrder.AcqRel,
                            memory_scope::Int=MemScope.Device) where {T, Shape}
    i, j = index
    rows, cols = Int32(Shape[1]), Int32(Shape[2])

    # arange returns 1-indexed [1..size], reshape for broadcasting, add base offset
    row_indices = reshape(arange((Shape[1],), Int32), (Shape[1], 1)) .+ Tile(Int32((i - 1) * rows))
    col_indices = reshape(arange((Shape[2],), Int32), (1, Shape[2])) .+ Tile(Int32((j - 1) * cols))

    atomic_add(array, (row_indices, col_indices), tile; memory_order, memory_scope)
end

"""
    atomic_xchg(array::TileArray{T, 1}, index::Integer, tile::Tile{T, Shape}; ...) -> Tile{T, Shape}

Atomic exchange at tile-level index (like `store`).
"""
@inline function atomic_xchg(array::TileArray{T, 1}, index::Integer, tile::Tile{T, Shape};
                             memory_order::Int=MemoryOrder.AcqRel,
                             memory_scope::Int=MemScope.Device) where {T, Shape}
    tile_size = Int32(Shape[1])
    # arange returns 1-indexed values [1, 2, ..., tile_size]
    base = Tile(Int32((index - 1) * tile_size))
    element_indices = arange(Shape, Int32) .+ base

    atomic_xchg(array, element_indices, tile; memory_order, memory_scope)
end

"""
    atomic_xchg(array::TileArray{T, 2}, index::NTuple{2, Integer}, tile::Tile{T, Shape}; ...) -> Tile{T, Shape}

Atomic exchange at tile-level 2D index (like `store`).
"""
@inline function atomic_xchg(array::TileArray{T, 2}, index::NTuple{2, Integer},
                             tile::Tile{T, Shape};
                             memory_order::Int=MemoryOrder.AcqRel,
                             memory_scope::Int=MemScope.Device) where {T, Shape}
    i, j = index
    rows, cols = Int32(Shape[1]), Int32(Shape[2])

    row_indices = reshape(arange((Shape[1],), Int32), (Shape[1], 1)) .+ Tile(Int32((i - 1) * rows))
    col_indices = reshape(arange((Shape[2],), Int32), (1, Shape[2])) .+ Tile(Int32((j - 1) * cols))

    atomic_xchg(array, (row_indices, col_indices), tile; memory_order, memory_scope)
end

"""
    atomic_cas(array::TileArray{T, 1}, index::Integer, expected::Tile{T, Shape}, desired::Tile{T, Shape}; ...) -> Tile{T, Shape}

Atomic compare-and-swap at tile-level index (like `store`).
"""
@inline function atomic_cas(array::TileArray{T, 1}, index::Integer,
                            expected::Tile{T, Shape}, desired::Tile{T, Shape};
                            memory_order::Int=MemoryOrder.AcqRel,
                            memory_scope::Int=MemScope.Device) where {T, Shape}
    tile_size = Int32(Shape[1])
    # arange returns 1-indexed values [1, 2, ..., tile_size]
    base = Tile(Int32((index - 1) * tile_size))
    element_indices = arange(Shape, Int32) .+ base

    atomic_cas(array, element_indices, expected, desired; memory_order, memory_scope)
end

"""
    atomic_cas(array::TileArray{T, 2}, index::NTuple{2, Integer}, expected::Tile{T, Shape}, desired::Tile{T, Shape}; ...) -> Tile{T, Shape}

Atomic compare-and-swap at tile-level 2D index (like `store`).
"""
@inline function atomic_cas(array::TileArray{T, 2}, index::NTuple{2, Integer},
                            expected::Tile{T, Shape}, desired::Tile{T, Shape};
                            memory_order::Int=MemoryOrder.AcqRel,
                            memory_scope::Int=MemScope.Device) where {T, Shape}
    i, j = index
    rows, cols = Int32(Shape[1]), Int32(Shape[2])

    row_indices = reshape(arange((Shape[1],), Int32), (Shape[1], 1)) .+ Tile(Int32((i - 1) * rows))
    col_indices = reshape(arange((Shape[2],), Int32), (1, Shape[2])) .+ Tile(Int32((j - 1) * cols))

    atomic_cas(array, (row_indices, col_indices), expected, desired; memory_order, memory_scope)
end
