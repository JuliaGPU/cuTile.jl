#=============================================================================
 Tile Shape Broadcasting
=============================================================================#

"""
    broadcast_shape(s1::Tuple, s2::Tuple) -> Tuple

Compute the broadcast shape from two tile shapes using NumPy-style broadcasting rules.
- Shapes are compared from right to left (trailing dimensions)
- Dimensions are compatible if they're equal or one of them is 1
- Missing dimensions are treated as 1

This is a pure function that Julia's const-prop can evaluate at compile time.

# Examples
```julia
broadcast_shape((128,), (1, 128))   # => (1, 128)
broadcast_shape((1,), (128,))       # => (128,)
broadcast_shape((4, 1), (1, 8))     # => (4, 8)
broadcast_shape((16, 32), (16, 32)) # => (16, 32)
```
"""
@inline function broadcast_shape(s1::Tuple, s2::Tuple)
    max_ndim = max(length(s1), length(s2))
    ntuple(max_ndim) do i
        # Index from the right (trailing dimensions)
        idx1 = length(s1) - max_ndim + i
        idx2 = length(s2) - max_ndim + i
        d1 = idx1 > 0 ? s1[idx1] : 1
        d2 = idx2 > 0 ? s2[idx2] : 1
        # Check compatibility
        (d1 == d2 || d1 == 1 || d2 == 1) || error("Shapes $s1 and $s2 are not broadcastable")
        max(d1, d2)
    end
end

# Special case: same shape returns same shape (fast path)
@inline broadcast_shape(s1::S, s2::S) where {S <: Tuple} = s1

#=============================================================================
 Tile Arithmetic
=============================================================================#

# These are stub implementations that the compiler intercepts.
# They return a new Tile with the broadcast result shape, enabling proper type inference.

@noinline function tile_add(a::Tile{T, S1}, b::Tile{T, S2})::Tile{T, broadcast_shape(S1, S2)} where {T, S1, S2}
    Base.donotdelete(a, b)
    Tile{T, broadcast_shape(S1, S2)}()
end

@noinline function tile_sub(a::Tile{T, S1}, b::Tile{T, S2})::Tile{T, broadcast_shape(S1, S2)} where {T, S1, S2}
    Base.donotdelete(a, b)
    Tile{T, broadcast_shape(S1, S2)}()
end

@noinline function tile_mul(a::Tile{T, S1}, b::Tile{T, S2})::Tile{T, broadcast_shape(S1, S2)} where {T, S1, S2}
    Base.donotdelete(a, b)
    Tile{T, broadcast_shape(S1, S2)}()
end

# Operator overloads dispatch to the intrinsic functions
Base.:(+)(a::Tile{T, S1}, b::Tile{T, S2}) where {T, S1, S2} = tile_add(a, b)
Base.:(-)(a::Tile{T, S1}, b::Tile{T, S2}) where {T, S1, S2} = tile_sub(a, b)
Base.:(*)(a::Tile{T, S1}, b::Tile{T, S2}) where {T, S1, S2} = tile_mul(a, b)

#=============================================================================
 Tile Shape Operations
=============================================================================#

public transpose

"""
    transpose(tile::Tile{T, (M, N)}) -> Tile{T, (N, M)}

Transpose a 2D tile, swapping its dimensions.
"""
@noinline function transpose(tile::Tile{T, S})::Tile{T, reverse(S)} where {T, S}
    Tile{T, reverse(S)}()
end

#=============================================================================
 GPU Intrinsics (stub implementations for host-side)
=============================================================================#

public bid, num_blocks, load, store

# Note: These stubs are markers for the compiler to recognize.
# The actual implementations are handled by the compiler which emits Tile IR ops.
#
# We use Base.compilerbarrier(:const, ...) to prevent constant folding while
# allowing other optimizations. This lets us use optimized IR.
# If these reach runtime (not compiled), they error with a clear message.

"""
    bid(axis) -> Int32

Get the block ID along the given axis (0=x, 1=y, 2=z).
In kernel code, this is compiled to GetTileBlockIdOp.
"""
@noinline bid(axis::Integer)::Int32 = Base.inferencebarrier(zero(Int32))

"""
    num_blocks(axis) -> Int32

Get the grid size along the given axis (0=x, 1=y, 2=z).
In kernel code, this is compiled to GetNumTileBlocksOp.
"""
@noinline num_blocks(axis::Integer)::Int32 = Base.inferencebarrier(zero(Int32))

"""
    load(ptr, index, shape::NTuple{N, Int}) -> Tile{T, shape}

Load a tile from a pointer at the given index with the specified shape.
In kernel code, this is compiled to LoadViewTkoOp.

Returns a `Tile{T, Shape}` where T is the pointer element type and Shape
is the compile-time constant shape tuple.
"""
# Internal function with shape as type parameter for proper type inference
@noinline function _load(ptr::Ptr{T}, index, ::Val{shape}) where {T, shape}
    Tile{T, shape}()
end
# Public API - inline wrapper that captures shape as type parameter
@inline load(ptr::Ptr{T}, index, shape::NTuple{N, Int}) where {T, N} = _load(ptr, index, Val(shape))

"""
    store(ptr, index, tile::Tile) -> Nothing

Store a tile to a pointer at the given index.
In kernel code, this is compiled to StoreViewTkoOp.
"""
@noinline function store(ptr::Ptr{T}, index, tile::Tile{T})::Nothing where T
    # donotdelete prevents the optimizer from eliminating this call
    # even though it has no observable effects in Julia
    Base.donotdelete(ptr, index, tile)
    nothing
end

# TileArray overloads - these are intercepted by the compiler
# The compiler extracts ptr/sizes/strides from the destructured TileArray

"""
    load(arr::TileArray, index, shape) -> Tile

Load a tile from a TileArray at the given index with the specified shape.
The TileArray's sizes and strides are used to construct the TensorView.
"""
# Internal function with shape as type parameter for proper type inference
@noinline function _load(arr::TileArray{T, N}, index, ::Val{shape}) where {T, N, shape}
    Base.donotdelete(arr, index)
    Tile{T, shape}()
end
# Public API - inline wrapper that captures shape as type parameter
@inline load(arr::TileArray{T, N}, index, shape::NTuple{M, Int}) where {T, N, M} = _load(arr, index, Val(shape))

# Load with Constant shape tuple (1D) - extracts value from Constant type parameter
@inline function load(arr::TileArray{T, N}, index, shape::Tuple{Constant{Int, V}}) where {T, N, V}
    _load(arr, index, Val((V,)))
end

# Load with Constant shape tuple (2D)
@inline function load(arr::TileArray{T, N}, index, shape::Tuple{Constant{Int, V1}, Constant{Int, V2}}) where {T, N, V1, V2}
    _load(arr, index, Val((V1, V2)))
end

# Load with Constant shape tuple (3D)
@inline function load(arr::TileArray{T, N}, index, shape::Tuple{Constant{Int, V1}, Constant{Int, V2}, Constant{Int, V3}}) where {T, N, V1, V2, V3}
    _load(arr, index, Val((V1, V2, V3)))
end

# Keyword argument version for ct.load(arr; index=..., shape=...)
@inline function load(arr::TileArray{T, N}; index, shape) where {T, N}
    shape_val = _extract_shape(shape)
    _load(arr, index, Val(shape_val))
end

# Helper to extract compile-time shape from various tuple types
@inline _extract_shape(s::NTuple{N, Int}) where N = s
@inline _extract_shape(s::Tuple{Constant{Int, V}}) where V = (V,)
@inline _extract_shape(s::Tuple{Constant{Int, V1}, Constant{Int, V2}}) where {V1, V2} = (V1, V2)
@inline _extract_shape(s::Tuple{Constant{Int, V1}, Constant{Int, V2}, Constant{Int, V3}}) where {V1, V2, V3} = (V1, V2, V3)

"""
    store(arr::TileArray, index, tile::Tile) -> Nothing

Store a tile to a TileArray at the given index.
"""
@noinline function store(arr::TileArray{T, N}, index, tile::Tile{T})::Nothing where {T, N}
    Base.donotdelete(arr, index, tile)
    nothing
end

# Keyword argument version for ct.store(arr; index=..., tile=...)
@noinline function store(arr::TileArray{T, N}; index, tile::Tile{T})::Nothing where {T, N}
    Base.donotdelete(arr, index, tile)
    nothing
end

#=============================================================================
 Matrix Multiply-Accumulate
=============================================================================#

public mma

"""
    mma(a::Tile{T1, (M, K)}, b::Tile{T2, (K, N)}, acc::Tile{T3, (M, N)}) -> Tile{T3, (M, N)}

Perform matrix-multiply-accumulate: result = a @ b + acc.
Uses tensor cores when available.

The input tiles must have compatible shapes:
- a: (M, K)
- b: (K, N)
- acc: (M, N)
- result: (M, N)
"""
@noinline function mma(a::Tile{T1, SA}, b::Tile{T2, SB}, acc::Tile{T3, SC}) where {T1, T2, T3, SA, SB, SC}
    Base.donotdelete(a, b, acc)
    Tile{T3, SC}()
end

#=============================================================================
 Tile Construction
=============================================================================#

public full, astype

"""
    full(shape::NTuple{N, Int}, value, dtype::Type{T}) -> Tile{T, shape}

Create a tile filled with a constant value.

# Example
```julia
zeros_tile = ct.full((32, 32), 0, Float32)  # 32x32 tile of zeros
```
"""
@noinline function full(shape::NTuple{N, Int}, value, ::Type{T})::Tile{T, shape} where {N, T}
    Base.donotdelete(value)  # shape and T are type parameters, can't be deleted
    Tile{T, shape}()
end

"""
    convert(Tile{T2}, tile::Tile{T1, Shape}) -> Tile{T2, Shape}
    astype(tile::Tile{T1, Shape}, ::Type{T2}) -> Tile{T2, Shape}

Convert a tile's element type from T1 to T2.

# Example
```julia
acc = ct.full((64, 64), 0.0f0, Float32)
result = convert(ct.Tile{ct.TFloat32}, acc)  # Convert to TF32 for tensor cores
result = convert(ct.Tile{Float16}, acc)      # Convert to Float16
```
"""
@noinline function astype(tile::Tile{T1, Shape}, ::Type{T2})::Tile{T2, Shape} where {T1, Shape, T2}
    Base.donotdelete(tile)
    Tile{T2, Shape}()
end

# Julia-style convert syntax builds on astype
Base.convert(::Type{Tile{T2}}, tile::Tile{T1, Shape}) where {T1, T2, Shape} = astype(tile, T2)

#=============================================================================
 Array Dimension Operations
=============================================================================#

public num_tiles

"""
    num_tiles(arr::TileArray{T, N}, axis::Integer, shape::NTuple{M, Int}) -> Int32

Get the number of tiles along a specific axis of an array, given the tile shape.
This is equivalent to cdiv(arr.sizes[axis+1], shape[axis+1]).

# Arguments
- `arr`: The array to query
- `axis`: The axis (0-indexed) to count tiles along
- `shape`: The tile shape used for partitioning

# Example
```julia
# For a 1024x768 matrix with 32x32 tiles:
# num_tiles(arr, 0, (32, 32)) returns cdiv(1024, 32) = 32
# num_tiles(arr, 1, (32, 32)) returns cdiv(768, 32) = 24
```
"""
@noinline function num_tiles(arr::TileArray{T, N}, axis::Integer, shape::NTuple{M, Int})::Int32 where {T, N, M}
    Base.inferencebarrier(zero(Int32))
end

#=============================================================================
 Integer Arithmetic Operations
=============================================================================#

public cdiv, floordiv

"""
    cdiv(a::Integer, b::Integer) -> Int32

Ceiling division: ⌈a/b⌉ = (a + b - 1) ÷ b

This is useful for computing grid dimensions from array sizes and tile sizes.
"""
@noinline cdiv(a::Integer, b::Integer)::Int32 = Base.inferencebarrier(zero(Int32))

"""
    floordiv(a::Integer, b::Integer) -> Int32

Floor division: ⌊a/b⌋

This is equivalent to `a ÷ b` but provided for consistency with the cuTile API.
"""
@noinline floordiv(a::Integer, b::Integer)::Int32 = Base.inferencebarrier(zero(Int32))

"""
    rem(a::Integer, b::Integer) -> Int32

Remainder operation: a % b (C-style, result has same sign as dividend)
"""
@noinline Base.rem(a::Int32, b::Int32)::Int32 = Base.inferencebarrier(zero(Int32))

"""
    min(a::Integer, b::Integer) -> Int32

Minimum of two integers.
"""
@noinline Base.min(a::Int32, b::Int32)::Int32 = Base.inferencebarrier(zero(Int32))

#=============================================================================
 Floating-Point Arithmetic
=============================================================================#

public tile_div

@noinline function tile_div(a::Tile{T, S1}, b::Tile{T, S2})::Tile{T, broadcast_shape(S1, S2)} where {T <: AbstractFloat, S1, S2}
    Base.donotdelete(a, b)
    Tile{T, broadcast_shape(S1, S2)}()
end

# Division operator for tiles
Base.:(/)(a::Tile{T, S1}, b::Tile{T, S2}) where {T <: AbstractFloat, S1, S2} = tile_div(a, b)

# Scalar-tile division (tile / scalar - broadcast scalar to tile)
@noinline function tile_div_scalar(a::Tile{T, S}, b::T)::Tile{T, S} where {T <: AbstractFloat, S}
    Base.donotdelete(a, b)
    Tile{T, S}()
end
Base.:(/)(a::Tile{T, S}, b::T) where {T <: AbstractFloat, S} = tile_div_scalar(a, b)

# Also support division by integers (common for mean calculation)
@noinline function tile_div_int(a::Tile{T, S}, b::Integer)::Tile{T, S} where {T <: AbstractFloat, S}
    Base.donotdelete(a, b)
    Tile{T, S}()
end
Base.:(/)(a::Tile{T, S}, b::Integer) where {T <: AbstractFloat, S} = tile_div_int(a, b)

# Reciprocal division (scalar / tile)
@noinline function scalar_div_tile(a::T, b::Tile{T, S})::Tile{T, S} where {T <: AbstractFloat, S}
    Base.donotdelete(a, b)
    Tile{T, S}()
end
Base.:(/)(a::T, b::Tile{T, S}) where {T <: AbstractFloat, S} = scalar_div_tile(a, b)

# Scalar-tile addition (tile + scalar)
@noinline function tile_add_scalar(a::Tile{T, S}, b::T)::Tile{T, S} where {T <: AbstractFloat, S}
    Base.donotdelete(a, b)
    Tile{T, S}()
end
Base.:(+)(a::Tile{T, S}, b::T) where {T <: AbstractFloat, S} = tile_add_scalar(a, b)
Base.:(+)(a::T, b::Tile{T, S}) where {T <: AbstractFloat, S} = tile_add_scalar(b, a)

# Scalar-tile subtraction (tile - scalar)
@noinline function tile_sub_scalar(a::Tile{T, S}, b::T)::Tile{T, S} where {T <: AbstractFloat, S}
    Base.donotdelete(a, b)
    Tile{T, S}()
end
Base.:(-)(a::Tile{T, S}, b::T) where {T <: AbstractFloat, S} = tile_sub_scalar(a, b)

# Scalar - tile
@noinline function scalar_sub_tile(a::T, b::Tile{T, S})::Tile{T, S} where {T <: AbstractFloat, S}
    Base.donotdelete(a, b)
    Tile{T, S}()
end
Base.:(-)(a::T, b::Tile{T, S}) where {T <: AbstractFloat, S} = scalar_sub_tile(a, b)

# Scalar-tile multiplication (tile * scalar)
@noinline function tile_mul_scalar(a::Tile{T, S}, b::T)::Tile{T, S} where {T <: AbstractFloat, S}
    Base.donotdelete(a, b)
    Tile{T, S}()
end
Base.:(*)(a::Tile{T, S}, b::T) where {T <: AbstractFloat, S} = tile_mul_scalar(a, b)
Base.:(*)(a::T, b::Tile{T, S}) where {T <: AbstractFloat, S} = tile_mul_scalar(b, a)

#=============================================================================
 Math Operations
=============================================================================#

public sqrt, rsqrt

"""
    sqrt(tile::Tile{T, S}) -> Tile{T, S}

Compute element-wise square root of a tile.
"""
@noinline function Base.sqrt(tile::Tile{T, S})::Tile{T, S} where {T <: AbstractFloat, S}
    Base.donotdelete(tile)
    Tile{T, S}()
end

"""
    rsqrt(tile::Tile{T, S}) -> Tile{T, S}

Compute element-wise reciprocal square root (1/sqrt(x)) of a tile.
"""
@noinline function rsqrt(tile::Tile{T, S})::Tile{T, S} where {T <: AbstractFloat, S}
    Base.donotdelete(tile)
    Tile{T, S}()
end

#=============================================================================
 Tile Factory Operations
=============================================================================#

public arange

"""
    arange(shape::NTuple{1, Int}, dtype::Type{T}) -> Tile{T, shape}

Create a 1D tile with values [0, 1, 2, ..., shape[1]-1].
Similar to Python's ct.arange() or np.arange().

# Example
```julia
indices = ct.arange((16,), Int32)  # Creates Tile with [0, 1, 2, ..., 15]
```
"""
@noinline function arange(shape::NTuple{1, Int}, ::Type{T})::Tile{T, shape} where {T}
    Tile{T, shape}()
end

# Helper for integer constant shape
@inline arange(shape::Tuple{Constant{Int, V}}, ::Type{T}) where {V, T} = arange((V,), T)

#=============================================================================
 Reduction Operations
=============================================================================#

public reduce_sum, reduce_max

"""
    reduce_sum(tile::Tile{T, S}, axis::Integer) -> Tile{T, reduced_shape}

Sum reduction along the specified axis.
Returns a tile with the specified dimension removed.

# Arguments
- `tile`: Input tile to reduce
- `axis`: Axis to reduce along (0-indexed)

# Example
```julia
# For a (128, 64) tile, reducing along axis 1:
sums = ct.reduce_sum(tile, 1)  # Returns (128,) tile
```
"""
@noinline function reduce_sum(tile::Tile{T, S}, axis::Integer) where {T <: AbstractFloat, S}
    # Compute the reduced shape by removing the reduced dimension
    reduced_shape = ntuple(i -> S[i < axis + 1 ? i : i + 1], length(S) - 1)
    Base.donotdelete(tile)
    Tile{T, reduced_shape}()
end

"""
    reduce_max(tile::Tile{T, S}, axis::Integer) -> Tile{T, reduced_shape}

Maximum reduction along the specified axis.

# Arguments
- `tile`: Input tile to reduce
- `axis`: Axis to reduce along (0-indexed)
"""
@noinline function reduce_max(tile::Tile{T, S}, axis::Integer) where {T <: AbstractFloat, S}
    reduced_shape = ntuple(i -> S[i < axis + 1 ? i : i + 1], length(S) - 1)
    Base.donotdelete(tile)
    Tile{T, reduced_shape}()
end

#=============================================================================
 Conditional Selection
=============================================================================#

public where

"""
    where(cond::Tile{Bool, S}, x::Tile{T, S}, y::Tile{T, S}) -> Tile{T, S}

Element-wise conditional selection: returns x where cond is true, y otherwise.
Similar to numpy.where() or torch.where().

# Example
```julia
mask = tile_a .> tile_b  # Boolean tile
result = ct.where(mask, tile_a, tile_b)  # Element-wise max
```
"""
@noinline function where(cond::Tile{Bool, S}, x::Tile{T, S}, y::Tile{T, S})::Tile{T, S} where {T, S}
    Base.donotdelete(cond, x, y)
    Tile{T, S}()
end

#=============================================================================
 Comparison Operations (returning Boolean tiles)
=============================================================================#

# Element-wise comparisons that return Boolean tiles
@noinline function tile_gt(a::Tile{T, S}, b::Tile{T, S})::Tile{Bool, S} where {T <: AbstractFloat, S}
    Base.donotdelete(a, b)
    Tile{Bool, S}()
end

@noinline function tile_lt(a::Tile{T, S}, b::Tile{T, S})::Tile{Bool, S} where {T <: AbstractFloat, S}
    Base.donotdelete(a, b)
    Tile{Bool, S}()
end

@noinline function tile_ge(a::Tile{T, S}, b::Tile{T, S})::Tile{Bool, S} where {T <: AbstractFloat, S}
    Base.donotdelete(a, b)
    Tile{Bool, S}()
end

@noinline function tile_le(a::Tile{T, S}, b::Tile{T, S})::Tile{Bool, S} where {T <: AbstractFloat, S}
    Base.donotdelete(a, b)
    Tile{Bool, S}()
end

# Operator overloads for tile comparisons
Base.:(>)(a::Tile{T, S}, b::Tile{T, S}) where {T <: AbstractFloat, S} = tile_gt(a, b)
Base.:(<)(a::Tile{T, S}, b::Tile{T, S}) where {T <: AbstractFloat, S} = tile_lt(a, b)
Base.:(>=)(a::Tile{T, S}, b::Tile{T, S}) where {T <: AbstractFloat, S} = tile_ge(a, b)
Base.:(<=)(a::Tile{T, S}, b::Tile{T, S}) where {T <: AbstractFloat, S} = tile_le(a, b)
