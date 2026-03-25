# cuTile DSL operations


# Helpers to deinterleave (acc1, elem1, acc2, elem2, ...) into separate tuples
@inline _deinterleave_accs(a, e, rest...) = (a, _deinterleave_accs(rest...)...)
@inline _deinterleave_accs() = ()
@inline _deinterleave_elems(a, e, rest...) = (e, _deinterleave_elems(rest...)...)
@inline _deinterleave_elems() = ()

#=============================================================================
 Load/Store
=============================================================================#

public bid, num_blocks, num_tiles, load, store, gather, scatter, Rounding

"""
Padding mode for load operations.
Use these constants with ct.load to specify out-of-bounds behavior.
"""
@enumx PaddingMode begin
    Undetermined = 0
    Zero = 1
    NegZero = 2
    Nan = 3
    PosInf = 4
    NegInf = 5
end

"""
Rounding mode for floating-point operations.
Use with reduction and scan kwargs (e.g., `sum(tile; dims, rounding_mode=ct.Rounding.Zero)`).
"""
@enumx Rounding begin
    NearestEven = 0
    Zero = 1
    NegInf = 2
    PosInf = 3
    Approx = 4
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
Axis is 1-indexed. Equivalent to cld(size(arr, axis), shape[axis]).

# Example
```julia
# For a 1024x768 matrix with 32x32 tiles:
# num_tiles(arr, 1, (32, 32)) returns cld(1024, 32) = 32
# num_tiles(arr, 2, (32, 32)) returns cld(768, 32) = 24
```
"""
@inline function num_tiles(arr::TileArray, axis::Integer, shape::NTuple{<:Any, Int})
    tv = Intrinsics.make_tensor_view(arr)
    pv = Intrinsics.make_partition_view(tv, shape, PaddingMode.Undetermined, nothing)
    Intrinsics.get_index_space_shape(pv, axis - One())  # convert to 0-indexed
end

# Match a shape tuple to a target rank N by padding trailing 1s or squeezing trailing singletons.
@generated function _match_shape(::Val{Shape}, ::Val{N}) where {Shape, N}
    M = length(Shape)
    M == N && return :($Shape)
    if M < N
        padded = (Shape..., ntuple(_ -> 1, N - M)...)
        return :($padded)
    end
    trailing = M - something(findlast(!=(1), Shape), 0)
    trailing >= M - N || error("cannot squeeze shape $Shape to rank $N: ",
                               "need to drop $(M-N) trailing singletons but only found $trailing")
    kept = Shape[1:N]
    return :($kept)
end

# Reshape a tile to match target rank N, preserving data layout.
@inline function _reshape_to_rank(tile::Tile, ::Val{N}) where {N}
    new_shape = _match_shape(Val(size(tile)), Val(N))
    reshape(tile, new_shape)
end


"""
    load(arr::TileArray, index, shape; order=nothing, padding_mode=PaddingMode.Undetermined, latency=nothing, allow_tma=nothing) -> Tile

Load a tile from a TileArray at the given index with the specified shape.
Index is 1-indexed. Shape must be compile-time constant.

# Dimension Ordering
- `order`: Optional tuple specifying the logical-to-physical dimension mapping (1-indexed).
  For example, `order=(2, 1)` indicates dimension 2 is contiguous in memory,
  enabling coalesced loads from transposed/permuted arrays.
  Default: `nothing` → identity `(1, 2, ..., N)`.

# Padding Modes
- `PaddingMode.Undetermined`: Unspecified behavior for OOB access
- `PaddingMode.Zero`: Return zero for OOB elements
- `PaddingMode.NegZero`: Return negative zero for OOB elements
- `PaddingMode.Nan`: Return NaN for OOB elements
- `PaddingMode.PosInf`: Return positive infinity for OOB elements
- `PaddingMode.NegInf`: Return negative infinity for OOB elements

# Optimization Hints
- `latency`: Optional latency hint (1-10), or nothing for compiler default
- `allow_tma`: Whether TMA (Tensor Memory Accelerator) is allowed (default: nothing, compiler decides)

# Example
```julia
tile = ct.load(arr, (bid,), (TILE_N[],); padding_mode=ct.PaddingMode.Zero, latency=3)

# Load from a transposed array with coalesced access
tile = ct.load(arr, (bidx, bidy), (TM, TN); order=(2, 1))
```
"""
@inline function load(arr::TileArray, index, shape::NTuple{<:Any, Int};
                      order::Union{NTuple{<:Any, Int}, Nothing}=nothing,
                      padding_mode::PaddingMode.T=PaddingMode.Undetermined,
                      latency::Union{Int, Nothing}=nothing,
                      allow_tma::Union{Bool, Nothing}=nothing)
    matched = _match_shape(Val(shape), Val(ndims(arr)))
    tv = Intrinsics.make_tensor_view(arr)
    pv = Intrinsics.make_partition_view(tv, matched, padding_mode, order)
    tile = Intrinsics.load_partition_view(pv, latency, allow_tma, promote(index...) .- One())
    reshape(tile, shape)
end

# Scalar index → wrap in tuple
@inline function load(arr::TileArray, index::Integer, shape::NTuple{<:Any, Int}; kwargs...)
    load(arr, (index,), shape; kwargs...)
end

# Scalar indexing: arr[i, j, ...] → scalar T
@overlay function Base.getindex(arr::TileArray{T, N}, indices::Vararg{Integer, N}) where {T, N}
    tv = Intrinsics.make_tensor_view(arr)
    shape = ntuple(_ -> 1, Val(N))
    pv = Intrinsics.make_partition_view(tv, shape, PaddingMode.Undetermined, nothing)
    tile = Intrinsics.load_partition_view(pv, nothing, nothing, promote(indices...) .- One())
    Intrinsics.to_scalar(reshape(tile, ()))
end

# Scalar indexing: tile[i, j, ...] → scalar T
@inline function Base.getindex(tile::Tile, indices::Vararg{Int})
    shape = ntuple(_ -> 1, Val(length(indices)))
    subtile = extract(tile, indices, shape)
    Intrinsics.to_scalar(reshape(subtile, ()))
end

# Functional setindex: Base.setindex(tile, val, i, j, ...) → new Tile with element replaced
@inline function Base.setindex(tile::Tile, val, indices::Vararg{Int})
    T = eltype(tile)
    S = size(tile)
    flat_len = prod(S)
    linear = _linear_index(S, indices)
    flat = reshape(tile, (flat_len,))
    idx = Intrinsics.iota((flat_len,), Int32)
    mask = idx .== Int32(linear)
    val_tile = broadcast_to(Tile(T(val)), (flat_len,))
    new_flat = where(mask, val_tile, flat)
    reshape(new_flat, S)
end

# 0-indexed column-major linear index from 1-indexed indices
@inline _linear_index(::Tuple{}, ::Tuple{}) = 0
@inline _linear_index(S::NTuple{N, Int}, indices::NTuple{N, Int}) where {N} =
    (indices[1] - 1) + S[1] * _linear_index(Base.tail(S), Base.tail(indices))

# Keyword argument version → extract and delegate
@inline function load(arr::TileArray; index, shape, kwargs...)
    load(arr, index, shape; kwargs...)
end

"""
    store(arr::TileArray, index, tile::Tile; order=nothing, latency=nothing, allow_tma=nothing) -> Tile

Store a tile to a TileArray at the given index. Index is 1-indexed.
Returns the stored tile (enables chaining and helps constant folding).

# Dimension Ordering
- `order`: Optional tuple specifying the logical-to-physical dimension mapping (1-indexed).
  Must match the `order` used in the corresponding `load` for permuted arrays.
  Default: `nothing` → identity `(1, 2, ..., N)`.

# Optimization Hints
- `latency`: Optional latency hint (1-10), or nothing for compiler default
- `allow_tma`: Whether TMA (Tensor Memory Accelerator) is allowed (default: nothing, compiler decides)
"""
@inline function store(arr::TileArray{T}, index, tile::Tile{T};
                       order::Union{NTuple{<:Any, Int}, Nothing}=nothing,
                       latency::Union{Int, Nothing}=nothing,
                       allow_tma::Union{Bool, Nothing}=nothing) where {T}
    reshaped = _reshape_to_rank(tile, Val(ndims(arr)))
    _store_reshaped(arr, reshaped, order, latency, allow_tma, promote(index...) .- One())
    return tile  # XXX: enables constant folding; remove when possible (see "constant folding" test)
end

# Scalar index → wrap in tuple
@inline function store(arr::TileArray{T}, index::Integer, tile::Tile{T}; kwargs...) where {T}
    store(arr, (index,), tile; kwargs...)
end

# Scalar value → wrap in 1-element tile and store
@inline function store(arr::TileArray{T}, index, val::T; kwargs...) where {T}
    shape = ntuple(_ -> 1, Val(ndims(arr)))
    tile = reshape(Intrinsics.from_scalar(val, Tuple{}), shape)
    store(arr, index, tile; kwargs...)
end
@inline function store(arr::TileArray{T}, index::Integer, val::T; kwargs...) where {T}
    store(arr, (index,), val; kwargs...)
end

@inline function _store_reshaped(arr::TileArray{T}, tile::Tile{T},
                                 order, latency, allow_tma, indices::NTuple{<:Any, <:Integer}) where {T}
    tv = Intrinsics.make_tensor_view(arr)
    pv = Intrinsics.make_partition_view(tv, size(tile), PaddingMode.Undetermined, order)
    Intrinsics.store_partition_view(pv, tile, latency, allow_tma, indices)
end

# Keyword argument version - dispatch to positional version
@inline function store(arr::TileArray{T}; index, tile::Tile{T},
                       order::Union{NTuple{<:Any, Int}, Nothing}=nothing,
                       latency::Union{Int, Nothing}=nothing,
                       allow_tma::Union{Bool, Nothing}=nothing) where {T}
    store(arr, index, tile; order, latency, allow_tma)
end

# Scalar store: arr[i, j, ...] = val
# NOTE: Cannot use @overlay (which adds @assume_effects :foldable) because
# setindex! is a side-effecting operation returning nothing — the compiler
# would DCE the entire call as a pure function with unused result.
Base.Experimental.@consistent_overlay cuTileMethodTable function Base.setindex!(arr::TileArray{T, N}, val::T, indices::Vararg{Integer, N}) where {T, N}
    shape = ntuple(_ -> 1, Val(N))
    tile = reshape(Intrinsics.from_scalar(val, Tuple{}), shape)
    store(arr, indices, tile)
    return
end

# Combine two masks with AND (dispatch-based to avoid Union types).
@inline _combine_masks(a::Tile, b::Tile) = a .& b
@inline _combine_masks(a::Tile, ::Nothing) = a
@inline _combine_masks(::Nothing, b::Tile) = b
@inline _combine_masks(::Nothing, ::Nothing) = nothing

# 1D bounds mask: 0 <= index < size
@inline function _bounds_mask_1d(indices_i32, array)
    (indices_i32 .>= Tile(Int32(0))) .& (indices_i32 .< Tile(size(array, 1)))
end

# 2D bounds mask: 0 <= idx0 < size0 && 0 <= idx1 < size1
@inline function _bounds_mask_2d(idx0_i32, idx1_i32, array, S)
    zero_0d = Tile(Int32(0))
    zero_bc = broadcast_to(zero_0d, S)
    size0_bc = broadcast_to(Tile(size(array, 1)), S)
    size1_bc = broadcast_to(Tile(size(array, 2)), S)
    mask0 = (idx0_i32 .>= zero_bc) .& (idx0_i32 .< size0_bc)
    mask1 = (idx1_i32 .>= zero_bc) .& (idx1_i32 .< size1_bc)
    mask0 .& mask1
end

# Padding tile for gather: zero(T) when no custom padding requested.
@inline _pad_value(::Nothing, ::Type{T}, S) where {T} = broadcast_to(Tile(zero(T)), S)
@inline _pad_value(val, ::Type{T}, S) where {T} = broadcast_to(Tile(T(val)), S)

# Gather load: dispatch on mask type. No mask → maskless load (fast path).
@inline function _gather_load(ptr_tile, latency, final_mask::Tile, padding_value, ::Type{T}, S) where {T}
    padding = _pad_value(padding_value, T, S)
    Intrinsics.load_ptr_tko(ptr_tile, latency, final_mask, padding)
end
@inline function _gather_load(ptr_tile, latency, ::Nothing, padding_value, ::Type{T}, S) where {T}
    Intrinsics.load_ptr_tko(ptr_tile, latency)
end

# Scatter store: dispatch on mask type. No mask → maskless store (fast path).
@inline function _scatter_store(ptr_tile, tile, latency, final_mask::Tile)
    Intrinsics.store_ptr_tko(ptr_tile, tile, latency, final_mask)
end
@inline function _scatter_store(ptr_tile, tile, latency, ::Nothing)
    Intrinsics.store_ptr_tko(ptr_tile, tile, latency)
end

"""
    gather(array::TileArray{T, 1}, indices::Tile{I, S}; kwargs...) -> Tile{T, S}

Gather elements from a 1D array using index tile.
Indices are 1-indexed. Out-of-bounds indices return `padding_value` (default: zero).

# Keyword Arguments
- `mask`: Optional `Tile{Bool}` — additional mask AND'd with automatic bounds check
- `padding_value`: Value for masked-out elements (default: `zero(T)`)
- `check_bounds::Bool`: Compute automatic bounds mask (default: `true`). Set to `false`
  when indices are known to be in-bounds to skip the comparisons.
- `latency`: Optional latency hint (1-10), or nothing for compiler default

# Example
```julia
base = (bid - 1) * TILE
indices = base .+ ct.arange(TILE)
tile = ct.gather(arr, indices; mask=valid_mask, padding_value=-1.0f0)
```
"""
@inline function gather(array::TileArray{T, 1}, indices::Tile{I};
                        mask=nothing,
                        padding_value=nothing,
                        check_bounds::Bool=true,
                        latency::Union{Int, Nothing}=nothing) where {T, I <: Integer}
    indices_0 = indices .- one(I)
    indices_i32 = convert(Tile{Int32}, indices_0)
    ptr_tile = Intrinsics.offset(array.ptr, indices_i32)

    bounds_mask = check_bounds ? _bounds_mask_1d(indices_i32, array) : nothing
    final_mask = _combine_masks(bounds_mask, mask)
    _gather_load(ptr_tile, latency, final_mask, padding_value, T, size(indices))
end

"""
    gather(array::TileArray{T, 2}, indices::Tuple{Tile, Tile}; kwargs...) -> Tile{T, S}

Gather elements from a 2D array using a tuple of index tiles.
Indices are 1-indexed. Index tiles are broadcast to a common shape.

# Keyword Arguments
- `mask`: Optional `Tile{Bool}` — additional mask AND'd with automatic bounds check
- `padding_value`: Value for masked-out elements (default: `zero(T)`)
- `check_bounds::Bool`: Compute automatic bounds mask (default: `true`). Set to `false`
  when indices are known to be in-bounds to skip the comparisons.
- `latency`: Optional latency hint (1-10), or nothing for compiler default
"""
@inline function gather(array::TileArray{T, 2}, indices::Tuple{Tile{I0}, Tile{I1}};
                        mask=nothing,
                        padding_value=nothing,
                        check_bounds::Bool=true,
                        latency::Union{Int, Nothing}=nothing) where {T, I0 <: Integer, I1 <: Integer}
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

    bounds_mask = check_bounds ? _bounds_mask_2d(idx0_i32, idx1_i32, array, S) : nothing
    final_mask = _combine_masks(bounds_mask, mask)
    _gather_load(ptr_tile, latency, final_mask, padding_value, T, S)
end

"""
    scatter(array::TileArray{T, 1}, indices::Tile{I, S}, tile::Tile{T, S}; kwargs...) -> Nothing

Scatter elements to a 1D array at index tile positions.
Indices are 1-indexed. Out-of-bounds indices are ignored.

# Keyword Arguments
- `mask`: Optional `Tile{Bool}` — additional mask AND'd with automatic bounds check
- `check_bounds::Bool`: Compute automatic bounds mask (default: `true`). Set to `false`
  when indices are known to be in-bounds to skip the comparisons.
- `latency`: Optional latency hint (1-10), or nothing for compiler default

# Example
```julia
base = (bid - 1) * TILE
indices = base .+ ct.arange(TILE)
ct.scatter(arr, indices, result_tile; mask=valid_mask)
```
"""
@inline function scatter(array::TileArray{T, 1}, indices::Tile{I, S}, tile::Tile{T, S};
                         mask=nothing,
                         check_bounds::Bool=true,
                         latency::Union{Int, Nothing}=nothing) where {T, I <: Integer, S}
    indices_0 = indices .- one(I)
    indices_i32 = convert(Tile{Int32}, indices_0)
    ptr_tile = Intrinsics.offset(array.ptr, indices_i32)

    bounds_mask = check_bounds ? _bounds_mask_1d(indices_i32, array) : nothing
    final_mask = _combine_masks(bounds_mask, mask)
    _scatter_store(ptr_tile, tile, latency, final_mask)
end

"""
    scatter(array::TileArray{T, 2}, indices::Tuple{Tile, Tile}, tile::Tile; kwargs...) -> Nothing

Scatter elements to a 2D array at index tile positions.
Indices are 1-indexed. Index tiles and value tile must broadcast to same shape.

# Keyword Arguments
- `mask`: Optional `Tile{Bool}` — additional mask AND'd with automatic bounds check
- `check_bounds::Bool`: Compute automatic bounds mask (default: `true`). Set to `false`
  when indices are known to be in-bounds to skip the comparisons.
- `latency`: Optional latency hint (1-10), or nothing for compiler default
"""
@inline function scatter(array::TileArray{T, 2}, indices::Tuple{Tile{I0}, Tile{I1}}, tile::Tile{T};
                         mask=nothing,
                         check_bounds::Bool=true,
                         latency::Union{Int, Nothing}=nothing) where {T, I0 <: Integer, I1 <: Integer}
    # Convert to 0-indexed
    idx0_0 = indices[1] .- one(I0)
    idx1_0 = indices[2] .- one(I1)

    # Broadcast indices to common shape
    S = broadcast_shape(broadcast_shape(size(indices[1]), size(indices[2])), size(tile))
    idx0_bc = broadcast_to(idx0_0, S)
    idx1_bc = broadcast_to(idx1_0, S)
    tile_bc = broadcast_to(tile, S)

    # Convert to Int32 for linear index computation
    idx0_i32 = convert(Tile{Int32}, idx0_bc)
    idx1_i32 = convert(Tile{Int32}, idx1_bc)

    # Get strides and broadcast to tile shape
    stride0_0d = Tile(array.strides[1])
    stride1_0d = Tile(array.strides[2])
    stride0 = broadcast_to(stride0_0d, S)
    stride1 = broadcast_to(stride1_0d, S)

    # Compute linear index = idx0 * stride0 + idx1 * stride1
    linear_idx = idx0_i32 .* stride0 + idx1_i32 .* stride1
    ptr_tile = Intrinsics.offset(array.ptr, linear_idx)

    bounds_mask = check_bounds ? _bounds_mask_2d(idx0_i32, idx1_i32, array, S) : nothing
    final_mask = _combine_masks(bounds_mask, mask)
    _scatter_store(ptr_tile, tile_bc, latency, final_mask)
end

#=============================================================================
 Factory
=============================================================================#

public arange

"""
    arange(shape; dtype=Int32) -> Tile{dtype, shape}
    arange(n; dtype=Int32) -> Tile{dtype, (n,)}

Create a 1D tile with values [1, 2, 3, ..., n] (1-indexed).

# Example
```julia
indices = ct.arange(16)              # Int32 [1, 2, ..., 16]
indices = ct.arange(16; dtype=Int64) # Int64 [1, 2, ..., 16]
```
"""
@inline arange(shape::NTuple{1, Int}; dtype::Type{T}=Int32) where {T} =
    Intrinsics.iota(shape, T) .+ one(T)
@inline arange(n::Int; dtype::Type{T}=Int32) where {T} = arange((n,); dtype)

# Internal: create a tile filled with a constant value.
# Used by Base.fill/zeros/ones overlays (see overlays.jl).
@inline _full(value, ::Type{T}, shape::NTuple{N, Int}) where {N, T} =
    Intrinsics.constant(shape, Tile(T(value)), T)
@inline _full(value::Tile, ::Type, shape::NTuple{N, Int}) where {N} =
    Intrinsics.constant(shape, value, eltype(value))

#=============================================================================
 Shape & DType
=============================================================================#

public cat, broadcast_to

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
    Intrinsics.cat(tiles, axis0)
end
@inline function cat(tiles::Tuple{Tile{T, S1}, Tile{T, S2}}, ::Val{Axis}) where {T, S1, S2, Axis}
    axis0 = Axis < 0 ? Axis : Axis - 1
    Intrinsics.cat(tiles, axis0)
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
    Intrinsics.broadcast(tile, shape)

"""
    reshape(tile::Tile{T, S}, shape::NTuple{N, Int}) -> Tile{T, shape}

Reshape a tile to a new shape. The total number of elements must remain the same.

# Example
```julia
tile = ct.load(arr, (1, 1), (4, 8))  # Shape (4, 8), 32 elements
reshaped = reshape(tile, (2, 16))  # Shape (2, 16), still 32 elements
```
"""
@inline function Base.reshape(tile::Tile{T}, shape::NTuple{<:Any, Int}) where {T}
    size(tile) === shape && return tile
    Intrinsics.reshape(tile, shape)
end
@inline Base.reshape(tile::Tile{T}, dims::Int...) where {T} = reshape(tile, dims)

"""
    permutedims(tile::Tile{T, S}, perm) -> Tile{T, permuted_shape}

Permute the dimensions of a tile according to the given permutation.
The permutation uses 1-indexed axes (Julia convention).

# Example
```julia
tile = ct.load(arr, (1, 1, 1), (2, 3, 4))  # Shape (2, 3, 4)
permuted = permutedims(tile, (3, 1, 2))    # Shape (4, 2, 3)
```
"""
@inline Base.permutedims(tile::Tile{T}, perm::NTuple{<:Any, Int}) where {T} =
    Intrinsics.permute(tile, map(p -> p - 1, perm))
@inline Base.permutedims(tile::Tile{T}, ::Val{Perm}) where {T, Perm} =
    Intrinsics.permute(tile, map(p -> p - 1, Perm))

"""
    permutedims(tile::Tile{T, (M, N)}) -> Tile{T, (N, M)}

Permute a 2D tile, swapping its dimensions. Defaults to permutation `(2, 1)`.

Equivalent to `transpose`.

---

    permutedims(tile::Tile{T, (N,)}) -> Tile{T, (1, N)}

Reshape a 1D tile into a `1 × N` row tile.

Equivalent to `transpose`.
"""
@generated function Base.permutedims(tile::T) where {T <: Tile}
    n = ndims(T)
    first_dim = n >= 1 ? size(T, 1) : nothing

    if n == 2
        return :(Intrinsics.permute(tile, (1, 0)))
    elseif n == 1
        return :(Intrinsics.reshape(tile, (1, $first_dim)))
    else
        return :(throw(ArgumentError("permutedims(tile) only works for 1D or 2D tiles")))
    end
end

"""
    transpose(tile::Tile{T, (M, N)}) -> Tile{T, (N, M)}

Transpose a 2D tile, swapping its dimensions.

---

    transpose(tile::Tile{T, (N,)}) -> Tile{T, (1, N)}

Reshape a 1D tile into a `1 × N` row tile.

Equivalent to single-arg `permutedims`.
"""
@generated function Base.transpose(tile::T) where {T <: Tile}
    n = ndims(T)
    first_dim = n >= 1 ? size(T, 1) : nothing

    if n == 2
        return :(Intrinsics.permute(tile, (1, 0)))
    elseif n == 1
        return :(Intrinsics.reshape(tile, (1, $first_dim)))
    else
        return :(throw(ArgumentError("transpose(tile) only works for 1D or 2D tiles")))
    end
end

@inline Base.convert(::Type{Tile{T}}, tile::Tile{T}) where {T} = tile
@inline Base.convert(::Type{Tile{T2}}, tile::Tile{T1, Shape}) where {T1, T2, Shape} =
    map(T2, tile)

#=============================================================================
 Reduction
=============================================================================#

"""
    map(f, a::Tile{<:Any,S}, rest::Tile{<:Any,S}...) -> Tile

Apply function `f` element-wise across one or more same-shaped tiles.
The function `f` must be a zero-size callable (singleton or capture-free lambda).
All tiles must have the same shape `S` — use broadcasting (`.+` etc.) or explicit
`broadcast_to` for shape-mismatched operands.

# Examples
```julia
result = map(abs, tile)           # Element-wise absolute value
result = map(x -> x * x, tile)   # Element-wise square
result = map(+, a, b)            # Element-wise addition (same shape required)
```
"""
@inline function Base.map(f, a::Tile{<:Any,S}, rest::Tile{<:Any,S}...) where {S}
    Intrinsics.from_scalar(f(Intrinsics.to_scalar(a), map(Intrinsics.to_scalar, rest)...), S)
end

"""
    mapreduce(identity, f, tile::Tile{T,S}; dims, init) -> Tile{T, reduced_shape}
    mapreduce(f, op, tile::Tile{T,S}; dims, init) -> Tile{T, reduced_shape}
    mapreduce(identity, f, tile1, tile2, ...; dims, init) -> Tuple{Tile...}

Reduce one or more tiles along `dims` using binary function `f`.

The single-tile form reduces a single tile along `dims` with identity element
`init`. The multi-tile form reduces several same-shaped tiles simultaneously:
`f` receives two tuples `(accumulators...)` and `(elements...)` and must return
a tuple of updated accumulators. Each tile requires a corresponding entry in the
`init` tuple.

When a non-identity map function is provided, it is applied element-wise via
`map` before reduction. The map function must be type-preserving.

# Examples
```julia
sums = mapreduce(identity, +, tile; dims=2, init=zero(Float32))
sum_of_squares = mapreduce(x -> x * x, +, tile; dims=2, init=zero(Float32))
sum_of_abs = mapreduce(abs, +, tile; dims=2, init=zero(Float32))

# Simultaneous reduction of two tiles
vals, idxs = mapreduce(identity, reducer, vals_tile, idx_tile;
                       dims=1, init=(typemin(Float32), Int32(0)))
```
"""
@inline function Base.mapreduce(::typeof(identity), f, tile::Tile{T,S};
                                dims::Integer, init) where {T<:Number, S}
    Intrinsics.reduce((tile,), dims - 1, f, (T(init),))[1]
end

@inline function Base.mapreduce(f, op, tile::Tile{T,S};
                                dims::Integer, init) where {T<:Number, S}
    reduce(op, map(f, tile); dims, init)
end

@inline function Base.mapreduce(::typeof(identity), f,
                                tile1::Tile{<:Any,S}, tile2::Tile{<:Any,S},
                                tiles::Tile{<:Any,S}...;
                                dims::Integer,
                                init::Tuple{Any, Any, Vararg{Any}}) where {S}
    all_tiles = (tile1, tile2, tiles...)
    function _combiner(args...)
        f(_deinterleave_accs(args...), _deinterleave_elems(args...))
    end
    Intrinsics.reduce(all_tiles, dims - 1, _combiner, init)
end

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
    mapreduce(identity, f, tile; dims, init)
end

# Callable operators with rounding/ftz encoded in type parameters,
# because emit_subprogram! does not support closures with captures.
struct AddF{RM, FTZ} end
struct MulF{RM, FTZ} end
struct MaxF{FTZ} end
struct MinF{FTZ} end
(::AddF{RM, FTZ})(a, b) where {RM, FTZ} = Intrinsics.addf(a, b, RM, FTZ)
(::MulF{RM, FTZ})(a, b) where {RM, FTZ} = Intrinsics.mulf(a, b, RM, FTZ)
(::MaxF{FTZ})(a, b) where {FTZ} = Intrinsics.maxf(a, b, FTZ)
(::MinF{FTZ})(a, b) where {FTZ} = Intrinsics.minf(a, b, FTZ)

for (f, op, custom_op, init_expr, has_rounding) in [
    (:sum,     :(+),   :AddF,  :(zero(T)),    true),
    (:prod,    :(*),   :MulF,  :(one(T)),     true),
    (:maximum, :(max), :MaxF,  :(typemin(T)), false),
    (:minimum, :(min), :MinF,  :(typemax(T)), false),
]
    # Integer: no rounding/ftz kwargs
    @eval @inline function Base.$f(tile::Tile{T,S}; dims) where {T<:Integer, S}
        reduce($op, tile; dims, init=$init_expr)
    end

    # Float: with rounding/ftz kwargs (sum/prod get rounding_mode; max/min don't)
    if has_rounding
        @eval @inline function Base.$f(tile::Tile{T,S}; dims,
                                        rounding_mode=nothing, flush_to_zero=false) where {T<:AbstractFloat, S}
            op = rounding_mode === nothing && !flush_to_zero ? $op : $custom_op{rounding_mode, flush_to_zero}()
            reduce(op, tile; dims, init=$init_expr)
        end
    else
        @eval @inline function Base.$f(tile::Tile{T,S}; dims,
                                        flush_to_zero=false) where {T<:AbstractFloat, S}
            op = !flush_to_zero ? $op : $custom_op{flush_to_zero}()
            reduce(op, tile; dims, init=$init_expr)
        end
    end
end

# sum/prod/max/min without dims — reduce all dimensions, return scalar T
# Recursive: reduce dim 1, dropdims, recurse. Base case: 0D tile → scalar.
for f in (:sum, :prod, :maximum, :minimum)
    # Multiple definitions for specificity reasons
    for T in (:Integer, :AbstractFloat)
        @eval @inline Base.$f(tile::Tile{T,Tuple{}}) where {T<:$T} =
            Intrinsics.to_scalar(tile)

        @eval @inline Base.$f(tile::Tile{T,S}) where {T<:$T, S<:Tuple{Any,Vararg}} =
            $f(dropdims($f(tile; dims=1); dims=1))
    end
end

"""
    any(tile::Tile{Bool,S}; dims) -> Tile{Bool, reduced_shape}

Logical OR reduction along the specified axis (1-indexed).
Reduced dimensions become size 1.
"""
@inline Base.any(tile::Tile{Bool,S}; dims::Integer) where {S} =
    reduce(|, tile; dims, init=false)

"""
    all(tile::Tile{Bool,S}; dims) -> Tile{Bool, reduced_shape}

Logical AND reduction along the specified axis (1-indexed).
Reduced dimensions become size 1.
"""
@inline Base.all(tile::Tile{Bool,S}; dims::Integer) where {S} =
    reduce(&, tile; dims, init=true)

"""
    count(tile::Tile{Bool,S}; dims) -> Tile{Int32, reduced_shape}

Count true elements along `dims` (1-indexed). Apply predicates via
broadcasting before calling count:

# Example
```julia
n_positive = count(tile .> 0.0f0; dims=1)
```
"""
@inline function Base.count(tile::Tile{Bool,S}; dims::Integer) where {S}
    sum(convert(Tile{Int32}, tile); dims)
end

# any/all without dims — return scalar Bool
for f in (:any, :all)
    @eval @inline Base.$f(tile::Tile{Bool,Tuple{}}) =
        Intrinsics.to_scalar(tile)

    @eval @inline Base.$f(tile::Tile{Bool,S}) where {S<:Tuple{Any,Vararg}} =
        $f(dropdims($f(tile; dims=1); dims=1))
end

# count without dims — return scalar Int32
# count reduces to Int32 (not Bool), so after first dim use sum for remaining.
@inline Base.count(tile::Tile{Bool,Tuple{}}) =
    Intrinsics.to_scalar(tile)
@inline Base.count(tile::Tile{Bool,S}) where {S<:Tuple{Any,Vararg}} =
    sum(dropdims(count(tile; dims=1); dims=1))

"""
    argmax(tile::Tile{T,S}; dims) -> Tile{Int32, reduced_shape}

Return 1-indexed positions of maximum values along `dims`.
Ties are broken by smallest index.

# Example
```julia
indices = argmax(tile; dims=2)  # Column indices of max per row
```
"""
@inline function Base.argmax(tile::Tile{T}; dims::Integer) where {T<:Number}
    n = size(tile, dims)
    indices = reshape(Intrinsics.iota((n,), Int32),
                      ntuple(i -> i == dims ? n : 1, ndims(tile)))
    indices = broadcast_to(indices, size(tile))

    function reducer(accs, elems)
        val_acc, idx_acc = accs
        val_elem, idx_elem = elems
        strict = val_acc > val_elem
        eq = val_acc == val_elem
        cond = strict | (eq & (idx_acc < idx_elem))
        (ifelse(cond, val_acc, val_elem),
         ifelse(cond, idx_acc, idx_elem))
    end
    _, idx = mapreduce(identity, reducer, tile, indices; dims, init=(typemin(T), Int32(0)))
    idx .+ one(Int32)
end

"""
    argmin(tile::Tile{T,S}; dims) -> Tile{Int32, reduced_shape}

Return 1-indexed positions of minimum values along `dims`.
Ties are broken by smallest index.

# Example
```julia
indices = argmin(tile; dims=2)  # Column indices of min per row
```
"""
@inline function Base.argmin(tile::Tile{T}; dims::Integer) where {T<:Number}
    n = size(tile, dims)
    indices = reshape(Intrinsics.iota((n,), Int32),
                      ntuple(i -> i == dims ? n : 1, ndims(tile)))
    indices = broadcast_to(indices, size(tile))

    function reducer(accs, elems)
        val_acc, idx_acc = accs
        val_elem, idx_elem = elems
        strict = val_elem > val_acc
        eq = val_acc == val_elem
        cond = strict | (eq & (idx_acc < idx_elem))
        (ifelse(cond, val_acc, val_elem),
         ifelse(cond, idx_acc, idx_elem))
    end
    _, idx = mapreduce(identity, reducer, tile, indices; dims, init=(typemax(T), Int32(0)))
    idx .+ one(Int32)
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
@inline Base.dropdims(tile::Tile; dims::Integer) =
    _dropdims(tile, Val(dims))

@inline function _dropdims(tile::Tile, ::Val{D}) where {D}
    new_shape = ntuple(i -> i < D ? size(tile, i) : size(tile, i + 1), Val(ndims(tile) - 1))
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
    Intrinsics.scan((tile,), dims - 1, f, (T(init),), rev)[1]
end

for (f, op, custom_op, init_expr) in [
    (:cumsum,  :(+), :AddF, :(zero(T))),
    (:cumprod, :(*), :MulF, :(one(T))),
]
    # Integer: no rounding/ftz kwargs
    @eval @inline function Base.$f(tile::Tile{T,S}; dims::Integer,
                                    rev::Bool=false) where {T<:Integer, S}
        accumulate($op, tile; dims, init=$init_expr, rev)
    end

    # Float: with rounding/ftz kwargs
    @eval @inline function Base.$f(tile::Tile{T,S}; dims::Integer,
                                    rev::Bool=false, rounding_mode=nothing,
                                    flush_to_zero=false) where {T<:AbstractFloat, S}
        op = rounding_mode === nothing && !flush_to_zero ? $op : $custom_op{rounding_mode, flush_to_zero}()
        accumulate(op, tile; dims, init=$init_expr, rev)
    end
end

#=============================================================================
 Matrix multiplication
=============================================================================#

# Matrix multiply-accumulate: muladd(a, b, acc) = a * b + acc
# Handles 1D promotion, type promotion, and batched dims (≥3D).
# Note: SA, SB, SC type parameters required to avoid ambiguity with scalar methods during codegen
@inline function Base.muladd(a::Tile{T1, SA}, b::Tile{T2, SB}, acc::Tile{T3, SC}) where {T1, T2, T3, SA, SB, SC}
    _muladd(a, b, acc, Val(ndims(a)), Val(ndims(b)))
end

# 2D × 2D: MmaFOp with swapped operands for row-major Tile IR
# Julia (M,K)*(K,N) → TileIR (K,M)*(N,K) → mmaf(b,a,acc) → TileIR (N,M) → Julia (M,N)
@inline function _muladd(a::Tile, b::Tile, acc::Tile, ::Val{2}, ::Val{2})
    Intrinsics.mma(b, a, acc)
end

# Vec-mat (1D × 2D): reshape (M,) → (M, 1), MmaFOp, acc is already (M, N)
@inline function _muladd(a::Tile, b::Tile, acc::Tile, ::Val{1}, ::Val{2})
    a2d = reshape(a, (size(a, 1), 1))
    _muladd(a2d, b, acc, Val(2), Val(2))
end

# Mat-vec (2D × 1D): reshape b (K,) → (K, 1), acc (M,) → (M, 1), MmaFOp, squeeze back
@inline function _muladd(a::Tile, b::Tile, acc::Tile, ::Val{2}, ::Val{1})
    M, K = size(a, 1), size(b, 1)
    b2d = reshape(b, (K, 1))
    acc2d = reshape(acc, (M, 1))
    result = _muladd(a, b2d, acc2d, Val(2), Val(2))
    reshape(result, (M,))
end

# Vec-vec (1D × 1D): not supported
@generated function _muladd(::Tile, ::Tile, ::Tile, ::Val{1}, ::Val{1})
    return :(throw(ArgumentError("Vector-vector multiply-accumulate is not supported.")))
end

# Batched mat-vec / vec-mat (≥3D × 1D or 1D × ≥3D): not supported, unsqueeze manually
@generated function _muladd(::Tile, ::Tile, ::Tile, ::Val{1}, ::Val{NB}) where {NB}
    NB >= 3 || return :(throw(ArgumentError("unreachable")))
    return :(throw(ArgumentError("Batched vec-mat is not supported. Reshape the 1D operand to 2D first.")))
end
@generated function _muladd(::Tile, ::Tile, ::Tile, ::Val{NA}, ::Val{1}) where {NA}
    NA >= 3 || return :(throw(ArgumentError("unreachable")))
    return :(throw(ArgumentError("Batched mat-vec is not supported. Reshape the 1D operand to 2D first.")))
end

# Batched matmul (≥3D × ≥3D): trailing batch dims with broadcast
# Julia convention: first two dims are matrix (M,K)/(K,N), trailing dims are batch.
# With row-major Tile IR shapes, Julia (M,K,B) → TileIR (B,K,M), so:
#   1. Broadcast batch dims to a common shape
#   2. Flatten batch dims into one via reshape (no permute needed!)
#   3. MmaFOp with swapped operands: mmaf(b, a, acc)
#   4. Unflatten batch dims via reshape
@generated function _muladd(a::Tile{T1, SA}, b::Tile{T2, SB}, acc::Tile{T3, SC},
                            ::Val{NA}, ::Val{NB}) where {T1, T2, T3, SA, SB, SC, NA, NB}
    sa = Tuple(SA.parameters)
    sb = Tuple(SB.parameters)

    # Matrix dims are first two; batch dims are trailing
    M = sa[1]; K = sa[2]; N = sb[2]
    a_batch = sa[3:end]
    b_batch = sb[3:end]

    # Broadcast batch dims (pad shorter with trailing 1s, then broadcast)
    n_batch = max(length(a_batch), length(b_batch))
    a_batch_padded = (a_batch..., ntuple(Returns(1), n_batch - length(a_batch))...)
    b_batch_padded = (b_batch..., ntuple(Returns(1), n_batch - length(b_batch))...)
    batch_shape = map(max, a_batch_padded, b_batch_padded)
    B_flat = prod(batch_shape)

    quote
        # Reshape + broadcast to align batch dims (still trailing)
        a_bc = broadcast_to(reshape(a, $((M, K, a_batch_padded...))), $((M, K, batch_shape...)))
        b_bc = broadcast_to(reshape(b, $((K, N, b_batch_padded...))), $((K, N, batch_shape...)))
        acc_bc = broadcast_to(acc, $((M, N, batch_shape...)))
        # Flatten batch dims to one — no permute needed since row-major Tile IR
        # already has batch as the leading (slowest) dimension
        a_3d = reshape(a_bc, $((M, K, B_flat)))
        b_3d = reshape(b_bc, $((K, N, B_flat)))
        acc_3d = reshape(acc_bc, $((M, N, B_flat)))
        # MmaFOp with swapped operands for row-major convention
        result_3d = Intrinsics.mma(b_3d, a_3d, acc_3d)
        # Unflatten batch dims
        reshape(result_3d, $((M, N, batch_shape...)))
    end
end

# Matrix multiplication: A * B = muladd(A, B, zeros)
# Note: SA, SB type parameters required to avoid ambiguity with scalar*tile methods during codegen
@inline function Base.:(*)(a::Tile{T1, SA}, b::Tile{T2, SB}) where {T1, T2, SA, SB}
    _matmul(a, b, Val(ndims(a)), Val(ndims(b)))
end

# 2D × 2D → (M, N)
@inline function _matmul(a::Tile{T1}, b::Tile, ::Val{2}, ::Val{2}) where {T1}
    acc = zeros(T1, (size(a, 1), size(b, 2)))
    muladd(a, b, acc)
end

# Vec-mat (1D × 2D) → (M, N)
@inline function _matmul(a::Tile{T1}, b::Tile, ::Val{1}, ::Val{2}) where {T1}
    acc = zeros(T1, (size(a, 1), size(b, 2)))
    muladd(a, b, acc)
end

# Mat-vec (2D × 1D) → (M,)
@inline function _matmul(a::Tile{T1}, b::Tile, ::Val{2}, ::Val{1}) where {T1}
    acc = zeros(T1, (size(a, 1),))
    muladd(a, b, acc)
end

# Vec-vec (1D × 1D): not supported
@generated function _matmul(::Tile, ::Tile, ::Val{1}, ::Val{1})
    return :(throw(ArgumentError("Vector-vector multiplication is not supported. Use dot(a, b) for inner products, or reshape explicitly.")))
end

# Batched (≥3D × ≥3D) → (M, N, batch...)
@generated function _matmul(a::Tile{T1, SA}, b::Tile{T2, SB},
                            ::Val{NA}, ::Val{NB}) where {T1, T2, SA, SB, NA, NB}
    sa = Tuple(SA.parameters)
    sb = Tuple(SB.parameters)
    a_batch = sa[3:end]
    b_batch = sb[3:end]
    n_batch = max(length(a_batch), length(b_batch))
    a_batch_padded = (a_batch..., ntuple(_ -> 1, n_batch - length(a_batch))...)
    b_batch_padded = (b_batch..., ntuple(_ -> 1, n_batch - length(b_batch))...)
    batch_shape = map(max, a_batch_padded, b_batch_padded)
    M = sa[1]; N = sb[2]
    out_shape = (M, N, batch_shape...)
    quote
        acc = zeros(T1, $out_shape)
        muladd(a, b, acc)
    end
end

# Batched × 1D: not supported — unsqueeze the 1D operand manually
@generated function _matmul(::Tile, ::Tile, ::Val{NA}, ::Val{1}) where {NA}
    NA >= 3 || return :(throw(ArgumentError("unreachable")))
    return :(throw(ArgumentError("Batched mat-vec is not supported. Reshape the 1D operand to 2D first.")))
end
@generated function _matmul(::Tile, ::Tile, ::Val{1}, ::Val{NB}) where {NB}
    NB >= 3 || return :(throw(ArgumentError("unreachable")))
    return :(throw(ArgumentError("Batched vec-mat is not supported. Reshape the 1D operand to 2D first.")))
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
where(cond, x, y) = ifelse.(cond, x, y)

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
    Intrinsics.extract(tile, map(i -> i - 1, index), shape)
@inline extract(tile::Tile{T}, ::Val{Index}, ::Val{Shape}) where {T, Index, Shape} =
    Intrinsics.extract(tile, map(i -> i - 1, Index), Shape)

#=============================================================================
 Assert
=============================================================================#

public @assert

"""
    @assert cond [message]

Assert that `cond` is true, aborting the kernel with `message` on failure.
If no message is given, the stringified condition is used.

Works like `Base.@assert` but compiles to a Tile IR assert op.
Failed assertions are **fatal** — they crash the kernel and corrupt the
CUDA context (not a catchable exception).

# Examples
```julia
ct.@assert bid > Int32(0)
ct.@assert bid > Int32(0) "bid must be positive"
```
"""
macro assert(cond)
    msg = string(cond)
    :($(Intrinsics.assert)($(esc(cond)), $msg))
end
macro assert(cond, msg)
    :($(Intrinsics.assert)($(esc(cond)), $(esc(msg))))
end

#=============================================================================
 @compiler_options macro
=============================================================================#

const _COMPILER_OPTION_NAMES = Set([:num_ctas, :occupancy, :opt_level])

"""
    @compiler_options key=val...

Specify per-architecture optimization hints inside a kernel function body.
Hints are embedded as `:meta` nodes and resolved at compile time based on
the target `sm_arch`.

Supported options: `num_ctas`, `occupancy`, `opt_level`.

Values can be plain scalars or `ByTarget(...)` for per-architecture dispatch.

# Examples
```julia
function my_kernel(A, B)
    ct.@compiler_options num_ctas=2
    ...
end

function my_kernel(A, B)
    ct.@compiler_options num_ctas=ByTarget(v"10.0" => 2, v"12.0" => 4) occupancy=8
    ...
end
```
"""
macro compiler_options(args...)
    isempty(args) && error("@compiler_options requires at least one key=val pair")

    # Validate and collect hints
    hints = Pair{Symbol, Any}[]
    for h in args
        h isa Expr && h.head === :(=) || error("@compiler_options: expected key=val, got $h")
        key = h.args[1]::Symbol
        key in _COMPILER_OPTION_NAMES || error("@compiler_options: unknown option '$key'; expected one of $(_COMPILER_OPTION_NAMES)")
        push!(hints, key => h.args[2])
    end

    # Evaluate hint values at macro expansion time and embed directly in :meta nodes.
    # Core.eval is needed because :meta nodes are not processed by lowering — they
    # pass through as-is, so their arguments must be concrete values in the AST.
    metas = Expr[]
    for (key, val) in hints
        evaluated = Core.eval(__module__, val)
        # Validate concrete values (including inside ByTarget)
        if evaluated isa ByTarget
            for v in values(evaluated.targets)
                validate_hint(key, v)
            end
            evaluated.default !== nothing && validate_hint(key, something(evaluated.default))
        else
            validate_hint(key, evaluated)
        end
        push!(metas, Expr(:meta, :cuTile, key, evaluated))
    end

    return Expr(:block, metas...)
end
