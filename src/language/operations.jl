# cuTile DSL operations

# Helper to extract compile-time shape from various tuple types
@inline _extract_shape(s::NTuple{N, Int}) where N = s
@inline _extract_shape(s::Tuple{Constant{Int, V}, Vararg{Constant{Int}}}) where V =
    (V, _extract_shape(Base.tail(s))...)
@inline _extract_shape(::Tuple{}) = ()

#=============================================================================
 Load/Store
=============================================================================#

public bid, num_blocks, num_tiles, load, store, gather, scatter

"""
Padding mode for load operations.
Use these constants with ct.load to specify out-of-bounds behavior.
"""
module PaddingMode
    const Undetermined = 0
    const Zero = 1
    const NegZero = 2
    const Nan = 3
    const PosInf = 4
    const NegInf = 5
end

"""
    bid(axis) -> Int32

Get the block ID along the given axis (1=x, 2=y, 3=z).
Returns 1-indexed block ID.
"""
@inline bid(axis::Integer) = Intrinsics.get_tile_block_id(axis - One()) + One()

"""
    num_blocks(axis) -> Int32

Get the grid size along the given axis (1=x, 2=y, 3=z).
"""
@inline num_blocks(axis::Integer) = Intrinsics.get_num_tile_blocks(axis - One())

"""
    num_tiles(arr::TileArray, axis::Integer, shape::NTuple{M, Int}) -> Int32

Get the number of tiles along a specific axis of an array, given the tile shape.
Axis is 1-indexed. Equivalent to cld(arr.sizes[axis], shape[axis]).

# Example
```julia
# For a 1024x768 matrix with 32x32 tiles:
# num_tiles(arr, 1, (32, 32)) returns cld(1024, 32) = 32
# num_tiles(arr, 2, (32, 32)) returns cld(768, 32) = 24
```
"""
@inline function num_tiles(arr::TileArray{T, N}, axis::Integer, shape::NTuple{<:Any, Int}) where {T, N}
    tv = Intrinsics.make_tensor_view(arr)
    pv = Intrinsics.make_partition_view(tv, Val(shape), PaddingMode.Undetermined)
    Intrinsics.get_index_space_shape(pv, axis - One())  # convert to 0-indexed
end

"""
    load(arr::TileArray, index, shape; padding_mode=PaddingMode.Undetermined, latency=nothing, allow_tma=true) -> Tile

Load a tile from a TileArray at the given index with the specified shape.
Index is 1-indexed. Shape must be compile-time constant.

# Padding Modes
- `PaddingMode.Undetermined`: Unspecified behavior for OOB access
- `PaddingMode.Zero`: Return zero for OOB elements
- `PaddingMode.NegZero`: Return negative zero for OOB elements
- `PaddingMode.Nan`: Return NaN for OOB elements
- `PaddingMode.PosInf`: Return positive infinity for OOB elements
- `PaddingMode.NegInf`: Return negative infinity for OOB elements

# Optimization Hints
- `latency`: Optional latency hint (1-10), or nothing for compiler default
- `allow_tma`: Whether TMA (Tensor Memory Accelerator) is allowed (default: true)

# Example
```julia
tile = ct.load(arr, (bid,), (TILE_N[],); padding_mode=ct.PaddingMode.Zero, latency=3)
```
"""
@inline function load(arr::TileArray{T, N}, index, shape::NTuple{<:Any, Int};
                      padding_mode::Int=PaddingMode.Undetermined,
                      latency::Union{Int, Nothing}=nothing,
                      allow_tma::Bool=true) where {T, N}
    tv = Intrinsics.make_tensor_view(arr)
    pv = Intrinsics.make_partition_view(tv, Val(shape), padding_mode)
    Intrinsics.load_partition_view(pv, latency, allow_tma, (promote(index...) .- One())...)
end

@inline function load(arr::TileArray{T, N}, index::Integer, shape::NTuple{<:Any, Int};
                      padding_mode::Int=PaddingMode.Undetermined,
                      latency::Union{Int, Nothing}=nothing,
                      allow_tma::Bool=true) where {T, N}
    tv = Intrinsics.make_tensor_view(arr)
    pv = Intrinsics.make_partition_view(tv, Val(shape), padding_mode)
    Intrinsics.load_partition_view(pv, latency, allow_tma, index - One())
end

# Load with Constant shape tuple
@inline function load(arr::TileArray{T, N}, index, shape::Tuple{Vararg{Constant{Int}}};
                      padding_mode::Int=PaddingMode.Undetermined,
                      latency::Union{Int, Nothing}=nothing,
                      allow_tma::Bool=true) where {T, N}
    shape_val = _extract_shape(shape)
    tv = Intrinsics.make_tensor_view(arr)
    pv = Intrinsics.make_partition_view(tv, Val(shape_val), padding_mode)
    Intrinsics.load_partition_view(pv, latency, allow_tma, (promote(index...) .- One())...)
end

# Keyword argument version
@inline function load(arr::TileArray{T, N}; index, shape,
                      padding_mode::Int=PaddingMode.Undetermined,
                      latency::Union{Int, Nothing}=nothing,
                      allow_tma::Bool=true) where {T, N}
    shape_val = _extract_shape(shape)
    tv = Intrinsics.make_tensor_view(arr)
    pv = Intrinsics.make_partition_view(tv, Val(shape_val), padding_mode)
    Intrinsics.load_partition_view(pv, latency, allow_tma, (promote(index...) .- One())...)
end

# Auto-reshape tile to match target array rank for store.
@inline function _reshape_for_store(tile::Tile{T,Shape}, ::Val{N}) where {T, Shape, N}
    length(Shape) <= N && return tile
    new_shape = _store_shape(Val(Shape), Val(N))
    reshape(tile, new_shape)
end

@generated function _store_shape(::Val{Shape}, ::Val{N}) where {Shape, N}
    M = length(Shape)
    to_drop = M - N
    singletons = [i for i in 1:M if Shape[i] == 1]
    if length(singletons) < to_drop
        error("cannot squeeze shape $Shape to rank $N: only $(length(singletons)) singleton dims but need to drop $to_drop")
    end
    drop_set = Set(singletons[1:to_drop])
    kept = tuple((Shape[i] for i in 1:M if !(i in drop_set))...)
    # Partition views require at least 1D
    if isempty(kept)
        kept = (1,)
    end
    return :($kept)
end

"""
    store(arr::TileArray, index, tile::Tile; latency=nothing, allow_tma=true) -> Tile

Store a tile to a TileArray at the given index. Index is 1-indexed.
Returns the stored tile (enables chaining and helps constant folding).

# Optimization Hints
- `latency`: Optional latency hint (1-10), or nothing for compiler default
- `allow_tma`: Whether TMA (Tensor Memory Accelerator) is allowed (default: true)
"""
@inline function store(arr::TileArray{T,N}, index, tile::Tile{T, Shape};
                       latency::Union{Int, Nothing}=nothing,
                       allow_tma::Bool=true) where {T, N, Shape}
    reshaped = _reshape_for_store(tile, Val(N))
    _store_reshaped(arr, reshaped, latency, allow_tma, (promote(index...) .- One())...)
    return tile  # XXX: enables constant folding; remove when possible (see "constant folding" test)
end

@inline function store(arr::TileArray{T,N}, index::Integer, tile::Tile{T, Shape};
                       latency::Union{Int, Nothing}=nothing,
                       allow_tma::Bool=true) where {T, N, Shape}
    reshaped = _reshape_for_store(tile, Val(N))
    _store_reshaped(arr, reshaped, latency, allow_tma, index - One())
    return tile  # XXX: enables constant folding; remove when possible (see "constant folding" test)
end

@inline function _store_reshaped(arr::TileArray{T}, tile::Tile{T, SShape},
                                 latency, allow_tma, indices...) where {T, SShape}
    tv = Intrinsics.make_tensor_view(arr)
    pv = Intrinsics.make_partition_view(tv, Val(SShape), PaddingMode.Undetermined)
    Intrinsics.store_partition_view(pv, tile, latency, allow_tma, indices...)
end

# Keyword argument version - dispatch to positional version
@inline function store(arr::TileArray{T}; index, tile::Tile{T, Shape},
                       latency::Union{Int, Nothing}=nothing,
                       allow_tma::Bool=true) where {T, Shape}
    store(arr, index, tile; latency, allow_tma)
end

"""
    gather(array::TileArray{T, 1}, indices::Tile{I, S}; latency=nothing) -> Tile{T, S}

Gather elements from a 1D array using index tile.
Indices are 1-indexed. Out-of-bounds indices return zero.

# Optimization Hints
- `latency`: Optional latency hint (1-10), or nothing for compiler default

# Example
```julia
base = (bid - 1) * TILE
indices = base .+ ct.arange((TILE,), Int32)
tile = ct.gather(arr, indices; latency=3)
```
"""
@inline function gather(array::TileArray{T, 1}, indices::Tile{I, S};
                        latency::Union{Int, Nothing}=nothing) where {T, I <: Integer, S}
    # Convert to 0-indexed
    indices_0 = indices .- one(I)

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

    # Padding for OOB (zero)
    padding = broadcast_to(Tile(zero(T)), S)

    Intrinsics.load_ptr_tko(ptr_tile, latency, mask, padding)
end

"""
    gather(array::TileArray{T, 2}, indices::Tuple{Tile, Tile}; latency=nothing) -> Tile{T, S}

Gather elements from a 2D array using a tuple of index tiles.
Indices are 1-indexed. Index tiles are broadcast to a common shape.

# Optimization Hints
- `latency`: Optional latency hint (1-10), or nothing for compiler default
"""
@inline function gather(array::TileArray{T, 2}, indices::Tuple{Tile{I0, S0}, Tile{I1, S1}};
                        latency::Union{Int, Nothing}=nothing) where {T, I0 <: Integer, I1 <: Integer, S0, S1}
    # Convert to 0-indexed
    idx0_0 = indices[1] .- one(I0)
    idx1_0 = indices[2] .- one(I1)

    # Broadcast indices to common shape
    S = broadcast_shape(S0, S1)
    idx0_bc = broadcast_to(idx0_0, S)
    idx1_bc = broadcast_to(idx1_0, S)

    # Convert to Int32 for linear index computation
    idx0_i32 = astype(idx0_bc, Int32)
    idx1_i32 = astype(idx1_bc, Int32)

    # Get strides and broadcast to tile shape
    stride0_0d = Tile(array.strides[1])
    stride1_0d = Tile(array.strides[2])
    stride0 = broadcast_to(stride0_0d, S)
    stride1 = broadcast_to(stride1_0d, S)

    # Compute linear index = idx0 * stride0 + idx1 * stride1
    linear_idx = idx0_i32 .* stride0 + idx1_i32 .* stride1

    # Compute pointer tile
    ptr_tile = Intrinsics.offset(array.ptr, linear_idx)

    # 2D bounds mask: 0 <= idx0 < size0 && 0 <= idx1 < size1
    zero_0d = Tile(Int32(0))
    zero_bc = broadcast_to(zero_0d, S)
    size0_bc = broadcast_to(Tile(array.sizes[1]), S)
    size1_bc = broadcast_to(Tile(array.sizes[2]), S)

    mask0 = (idx0_i32 .>= zero_bc) .& (idx0_i32 .< size0_bc)
    mask1 = (idx1_i32 .>= zero_bc) .& (idx1_i32 .< size1_bc)
    mask = mask0 .& mask1

    # Padding for OOB (zero)
    padding = broadcast_to(Tile(zero(T)), S)

    Intrinsics.load_ptr_tko(ptr_tile, latency, mask, padding)
end

"""
    scatter(array::TileArray{T, 1}, indices::Tile{I, S}, tile::Tile{T, S}; latency=nothing) -> Nothing

Scatter elements to a 1D array at index tile positions.
Indices are 1-indexed. Out-of-bounds indices are ignored.

# Optimization Hints
- `latency`: Optional latency hint (1-10), or nothing for compiler default

# Example
```julia
base = (bid - 1) * TILE
indices = base .+ ct.arange((TILE,), Int32)
ct.scatter(arr, indices, result_tile; latency=3)
```
"""
@inline function scatter(array::TileArray{T, 1}, indices::Tile{I, S}, tile::Tile{T, S};
                         latency::Union{Int, Nothing}=nothing) where {T, I <: Integer, S}
    # Convert to 0-indexed
    indices_0 = indices .- one(I)

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

    Intrinsics.store_ptr_tko(ptr_tile, tile, latency, mask)
end

"""
    scatter(array::TileArray{T, 2}, indices::Tuple{Tile, Tile}, tile::Tile; latency=nothing) -> Nothing

Scatter elements to a 2D array at index tile positions.
Indices are 1-indexed. Index tiles and value tile must broadcast to same shape.

# Optimization Hints
- `latency`: Optional latency hint (1-10), or nothing for compiler default
"""
@inline function scatter(array::TileArray{T, 2}, indices::Tuple{Tile{I0, S0}, Tile{I1, S1}}, tile::Tile{T, Stile};
                         latency::Union{Int, Nothing}=nothing) where {T, I0 <: Integer, I1 <: Integer, S0, S1, Stile}
    # Convert to 0-indexed
    idx0_0 = indices[1] .- one(I0)
    idx1_0 = indices[2] .- one(I1)

    # Broadcast indices to common shape (include value tile shape)
    S = broadcast_shape(broadcast_shape(S0, S1), Stile)
    idx0_bc = broadcast_to(idx0_0, S)
    idx1_bc = broadcast_to(idx1_0, S)
    tile_bc = broadcast_to(tile, S)

    # Convert to Int32 for linear index computation
    idx0_i32 = astype(idx0_bc, Int32)
    idx1_i32 = astype(idx1_bc, Int32)

    # Get strides and broadcast to tile shape
    stride0_0d = Tile(array.strides[1])
    stride1_0d = Tile(array.strides[2])
    stride0 = broadcast_to(stride0_0d, S)
    stride1 = broadcast_to(stride1_0d, S)

    # Compute linear index = idx0 * stride0 + idx1 * stride1
    linear_idx = idx0_i32 .* stride0 + idx1_i32 .* stride1

    # Compute pointer tile
    ptr_tile = Intrinsics.offset(array.ptr, linear_idx)

    # 2D bounds mask: 0 <= idx0 < size0 && 0 <= idx1 < size1
    zero_0d = Tile(Int32(0))
    zero_bc = broadcast_to(zero_0d, S)
    size0_bc = broadcast_to(Tile(array.sizes[1]), S)
    size1_bc = broadcast_to(Tile(array.sizes[2]), S)

    mask0 = (idx0_i32 .>= zero_bc) .& (idx0_i32 .< size0_bc)
    mask1 = (idx1_i32 .>= zero_bc) .& (idx1_i32 .< size1_bc)
    mask = mask0 .& mask1

    Intrinsics.store_ptr_tko(ptr_tile, tile_bc, latency, mask)
end

#=============================================================================
 Factory
=============================================================================#

public arange, full, zeros

"""
    arange(shape::NTuple{1, Int}, dtype::Type{T}) -> Tile{T, shape}

Create a 1D tile with values [1, 2, 3, ..., shape[1]] (1-indexed).

# Example
```julia
indices = ct.arange((16,), Int32)  # Creates Tile with [1, 2, 3, ..., 16]
```
"""
@inline arange(shape::NTuple{1, Int}, ::Type{T}) where {T} =
    Intrinsics.iota(shape, T) .+ one(T)

# Helper for integer constant shape
@inline arange(shape::Tuple{Constant{Int, V}}, ::Type{T}) where {V, T} =
    arange((V,), T)

"""
    full(shape::NTuple{N, Int}, value, dtype::Type{T}) -> Tile{T, shape}

Create a tile filled with a constant value.

# Example
```julia
ones_tile = ct.full((32, 32), 1.0f0, Float32)
```
"""
@inline full(shape::NTuple{N, Int}, value, ::Type{T}) where {N, T} =
    Intrinsics.constant(shape, value, T)

"""
    zeros(shape::NTuple{N, Int}, dtype::Type{T}) -> Tile{T, shape}

Create a tile filled with zeros.

# Example
```julia
zeros_tile = ct.zeros((32, 32), Float32)
```
"""
@inline zeros(shape::NTuple{N, Int}, ::Type{T}) where {N, T} =
    full(shape, zero(T), T)

#=============================================================================
 Shape & DType
=============================================================================#

public cat, broadcast_to, reshape, permute, transpose, astype

"""
    cat(tiles::Tuple{Tile, Tile}, axis::Int) -> Tile

Concatenate two tiles along the specified axis (1-indexed).
Supports negative axis (e.g., -1 for last dimension).

# Example
```julia
tile_a = ct.load(arr_a, (1,), (4, 8))  # Shape (4, 8)
tile_b = ct.load(arr_b, (1,), (4, 8))  # Shape (4, 8)
# Concatenate along axis 1: (4, 8) + (4, 8) -> (8, 8)
combined = ct.cat((tile_a, tile_b), 1)
# Concatenate along axis -1 (last): (4, 8) + (4, 8) -> (4, 16)
combined_last = ct.cat((tile_a, tile_b), -1)
```
"""
@inline function cat(tiles::Tuple{Tile{T, S1}, Tile{T, S2}}, axis::Int) where {T, S1, S2}
    axis0 = axis < 0 ? axis : axis - 1
    Intrinsics.cat(tiles, Val(axis0))
end
@inline function cat(tiles::Tuple{Tile{T, S1}, Tile{T, S2}}, ::Val{Axis}) where {T, S1, S2, Axis}
    axis0 = Axis < 0 ? Axis : Axis - 1
    Intrinsics.cat(tiles, Val(axis0))
end

"""
    broadcast_to(tile::Tile{T, S}, shape::NTuple{N, Int}) -> Tile{T, shape}

Explicitly broadcast a tile to a target shape.

# Example
```julia
row = ct.load(arr, (1, 1), (1, 128))  # Shape (1, 128)
expanded = ct.broadcast_to(row, (64, 128))  # Shape (64, 128)
```
"""
@inline broadcast_to(tile::Tile{T}, shape::NTuple{<:Any, Int}) where {T} =
    Intrinsics.broadcast(tile, Val(shape))

"""
    reshape(tile::Tile{T, S}, shape::NTuple{N, Int}) -> Tile{T, shape}

Reshape a tile to a new shape. The total number of elements must remain the same.

# Example
```julia
tile = ct.load(arr, (1, 1), (4, 8))  # Shape (4, 8), 32 elements
reshaped = ct.reshape(tile, (2, 16))  # Shape (2, 16), still 32 elements
```
"""
@inline reshape(tile::Tile{T}, shape::NTuple{<:Any, Int}) where {T} =
    Intrinsics.reshape(tile, Val(shape))

"""
    permute(tile::Tile{T, S}, perm::NTuple{N, Int}) -> Tile{T, permuted_shape}

Permute the dimensions of a tile according to the given permutation.
The permutation uses 1-indexed axes (Julia convention).

# Example
```julia
tile = ct.load(arr, (1, 1, 1), (2, 3, 4))  # Shape (2, 3, 4)
# Permute axes: new_axis_1 = old_axis_3, new_axis_2 = old_axis_1, new_axis_3 = old_axis_2
permuted = ct.permute(tile, (3, 1, 2))  # Shape (4, 2, 3)
```
"""
@inline permute(tile::Tile{T}, perm::NTuple{<:Any, Int}) where {T} =
    Intrinsics.permute(tile, Val(map(p -> p - 1, perm)))
@inline permute(tile::Tile{T}, ::Val{Perm}) where {T, Perm} =
    Intrinsics.permute(tile, Val(map(p -> p - 1, Perm)))

"""
    transpose(tile::Tile{T, (M, N)}) -> Tile{T, (N, M)}

Transpose a 2D tile, swapping its dimensions.
"""
@inline transpose(tile::Tile{T}) where {T} =
    Intrinsics.transpose(tile)

"""
    astype(tile::Tile{T1, Shape}, ::Type{T2}) -> Tile{T2, Shape}

Convert a tile's element type from T1 to T2.

# Example
```julia
acc = ct.full((64, 64), 0.0f0, Float32)
result = ct.astype(acc, ct.TFloat32)  # Convert to TF32 for tensor cores
```
"""
@inline astype(tile::Tile{T1, Shape}, ::Type{T2}) where {T1, Shape, T2} =
    Intrinsics.astype(tile, T2)

# Julia-style convert syntax
@inline Base.convert(::Type{Tile{T2}}, tile::Tile{T1, Shape}) where {T1, T2, Shape} =
    astype(tile, T2)

#=============================================================================
 Reduction
=============================================================================#

"""
    reduce(f, tile::Tile{T,S}; dims::Integer, init) -> Tile{T, reduced_shape}

Reduce a tile along the specified dimension using binary function `f` with
identity element `init`. The `dims` axis is 1-indexed. The reduced dimension
becomes size 1 (Julia semantics).

Supported functions: `+`, `*`, `max`, `min`.

# Example
```julia
sums = reduce(+, tile; dims=2, init=zero(Float32))
```
"""
@inline function Base.reduce(f, tile::Tile{T,S}; dims::Integer, init) where {T<:Number, S}
    Intrinsics.reduce(tile, Val(dims - 1), f, T(init))
end

"""
    sum(tile::Tile{T,S}; dims) -> Tile{T, reduced_shape}

Sum reduction along the specified axis/axes (1-indexed).
Reduced dimensions become size 1.
"""
@inline Base.sum(tile::Tile{T,S}; dims) where {T<:Number, S} =
    _tile_reduce(+, tile, dims, zero(T))

"""
    prod(tile::Tile{T,S}; dims) -> Tile{T, reduced_shape}

Product reduction along the specified axis/axes (1-indexed).
Reduced dimensions become size 1.
"""
@inline Base.prod(tile::Tile{T,S}; dims) where {T<:Number, S} =
    _tile_reduce(*, tile, dims, one(T))

"""
    maximum(tile::Tile{T,S}; dims) -> Tile{T, reduced_shape}

Maximum reduction along the specified axis/axes (1-indexed).
Reduced dimensions become size 1.
"""
@inline Base.maximum(tile::Tile{T,S}; dims) where {T<:Number, S} =
    _tile_reduce(max, tile, dims, typemin(T))

"""
    minimum(tile::Tile{T,S}; dims) -> Tile{T, reduced_shape}

Minimum reduction along the specified axis/axes (1-indexed).
Reduced dimensions become size 1.
"""
@inline Base.minimum(tile::Tile{T,S}; dims) where {T<:Number, S} =
    _tile_reduce(min, tile, dims, typemax(T))

# Single-dim dispatch
@inline _tile_reduce(f, tile::Tile{T}, dims::Integer, init) where {T} =
    Intrinsics.reduce(tile, Val(dims - 1), f, T(init))

# Multi-dim reduction: chain single-dim reductions.
# Since reduced dims become size 1 (not removed), axis indices stay valid.

# Multi-dim reduction: chain single-dim reductions.
# Since reduced dims become size 1 (not removed), axis indices stay valid.
@inline function _reduce_multi(f, tile, dims::Tuple{Int, Vararg}, init)
    tile = Intrinsics.reduce(tile, Val(first(dims) - 1), f, init)
    _reduce_multi(f, tile, Base.tail(dims), init)
end
@inline function _reduce_multi(f, tile, dims::Tuple{Int}, init)
    Intrinsics.reduce(tile, Val(first(dims) - 1), f, init)
end

"""
    dropdims(tile::Tile{T,S}; dims) -> Tile{T, squeezed_shape}

Remove singleton dimensions from a tile. The specified dimensions must have
size 1. This is the inverse of the dimension-preserving behavior of `sum`,
`prod`, `maximum`, and `minimum`.

# Example
```julia
sums = sum(tile; dims=2)           # (64, 1)
squeezed = dropdims(sums; dims=2)  # (64,)
```
"""
@inline Base.dropdims(tile::Tile{T,S}; dims::Integer) where {T, S} =
    _dropdims(tile, Val(dims))

@inline function _dropdims(tile::Tile{T,S}, ::Val{D}) where {T, S, D}
    new_shape = ntuple(i -> i < D ? S[i] : S[i + 1], Val(length(S) - 1))
    reshape(tile, new_shape)
end

#=============================================================================
 Scan (Prefix Sum) Operations
=============================================================================#

"""
    accumulate(f, tile::Tile{T,S}; dims::Integer, init, rev::Bool=false) -> Tile{T, S}

Scan (prefix sum) along the specified dimension using binary function `f`.
The `dims` axis is 1-indexed.

Supported functions: `+`, `*`, `max`, `min`.
"""
@inline function Base.accumulate(f, tile::Tile{T,S}; dims::Integer,
                                 init, rev::Bool=false) where {T<:Number, S}
    Intrinsics.scan(tile, Val(dims - 1), f, T(init), rev)
end

"""
    cumsum(tile::Tile{T,S}; dims::Integer, rev::Bool=false) -> Tile{T, S}

Cumulative sum along the specified axis (1-indexed).
"""
@inline Base.cumsum(tile::Tile{T,S}; dims::Integer,
                    rev::Bool=false) where {T<:Number, S} =
    accumulate(+, tile; dims, init=zero(T), rev)

"""
    cumprod(tile::Tile{T,S}; dims::Integer, rev::Bool=false) -> Tile{T, S}

Cumulative product along the specified axis (1-indexed).
"""
@inline Base.cumprod(tile::Tile{T,S}; dims::Integer,
                     rev::Bool=false) where {T<:Number, S} =
    accumulate(*, tile; dims, init=one(T), rev)

#=============================================================================
 Matrix multiplication
=============================================================================#

# Matrix multiply-accumulate: muladd(a, b, acc) = a * b + acc
@inline Base.muladd(a::Tile{T1, SA}, b::Tile{T2, SB}, acc::Tile{T3, SC}) where {T1, T2, T3, SA, SB, SC} =
    Intrinsics.mma(a, b, acc)

# Matrix multiplication (A * B like Julia arrays)
@inline function Base.:(*)(a::Tile{T1, SA}, b::Tile{T2, SB}) where {T1, T2, SA, SB}
    _matmul(a, b, Val(length(SA)))
end

# 2D matmul: (M, K) × (K, N) → (M, N)
@inline function _matmul(a::Tile{T1, SA}, b::Tile{T2, SB}, ::Val{2}) where {T1, T2, SA, SB}
    M = SA[1]
    N = SB[2]
    acc = zeros((M, N), T1)
    muladd(a, b, acc)
end

# 3D batched matmul: (B, M, K) × (B, K, N) → (B, M, N)
@inline function _matmul(a::Tile{T1, SA}, b::Tile{T2, SB}, ::Val{3}) where {T1, T2, SA, SB}
    B = max(SA[1], SB[1])  # Broadcast batch dimension
    M = SA[2]
    N = SB[3]
    acc = zeros((B, M, N), T1)
    muladd(a, b, acc)
end

#=============================================================================
 Selection
=============================================================================#

public where, extract

"""
    where(cond::Tile{Bool}, x, y) -> Tile

Element-wise conditional selection: returns x where cond is true, y otherwise.
Similar to numpy.where() or torch.where(). Supports broadcasting and scalar arguments.

# Example
```julia
mask = tile_a .> tile_b  # Boolean tile
result = ct.where(mask, tile_a, tile_b)  # Element-wise max
result = ct.where(mask, tile_a, 0.0f0)  # Zero out where mask is false
```
"""
@inline function where(cond::Tile{Bool, S1}, x::Tile{T, S2}, y::Tile{T, S3}) where {T, S1, S2, S3}
    S = broadcast_shape(broadcast_shape(S1, S2), S3)
    Intrinsics.select(broadcast_to(cond, S), broadcast_to(x, S), broadcast_to(y, S))
end

# Scalar y
@inline function where(cond::Tile{Bool, S1}, x::Tile{T, S2}, y::Number) where {T, S1, S2}
    S = broadcast_shape(S1, S2)
    Intrinsics.select(broadcast_to(cond, S), broadcast_to(x, S), broadcast_to(Tile(T(y)), S))
end

# Scalar x
@inline function where(cond::Tile{Bool, S1}, x::Number, y::Tile{T, S2}) where {T, S1, S2}
    S = broadcast_shape(S1, S2)
    Intrinsics.select(broadcast_to(cond, S), broadcast_to(Tile(T(x)), S), broadcast_to(y, S))
end

# Both scalars
@inline function where(cond::Tile{Bool, S}, x::Number, y::Number) where {S}
    T = promote_type(typeof(x), typeof(y))
    Intrinsics.select(cond, broadcast_to(Tile(T(x)), S), broadcast_to(Tile(T(y)), S))
end

"""
    extract(tile::Tile{T, S}, index::NTuple{N, Int}, shape::NTuple{N, Int}) -> Tile{T, shape}

Extract a sub-tile from a tile at the given slice indices.

**IMPORTANT:** The `index` parameter specifies SLICE INDICES, not element offsets!

For each dimension, the source tile is divided into `S[i] ÷ shape[i]` non-overlapping slices.
The `index[i]` selects which slice to extract (1-indexed).

# Example: Extracting quadrants from an 8×8 tile
```julia
tile = ct.load(arr, (1, 1), (8, 8))
# 8÷4 = 2 slices per dimension, so valid indices are {1, 2} × {1, 2}
tl = ct.extract(tile, (1, 1), (4, 4))  # Top-left (rows 1-4, cols 1-4)
bl = ct.extract(tile, (2, 1), (4, 4))  # Bottom-left (rows 5-8, cols 1-4)
tr = ct.extract(tile, (1, 2), (4, 4))  # Top-right (rows 1-4, cols 5-8)
br = ct.extract(tile, (2, 2), (4, 4))  # Bottom-right (rows 5-8, cols 5-8)
```
"""
@inline extract(tile::Tile{T}, index::NTuple{<:Any, Int}, shape::NTuple{<:Any, Int}) where {T} =
    Intrinsics.extract(tile, Val(map(i -> i - 1, index)), Val(shape))
@inline extract(tile::Tile{T}, ::Val{Index}, ::Val{Shape}) where {T, Index, Shape} =
    Intrinsics.extract(tile, Val(map(i -> i - 1, Index)), Val(Shape))
