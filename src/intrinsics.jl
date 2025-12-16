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

#=============================================================================
 Tile Arithmetic (element-wise operations)
=============================================================================#

# Tile arithmetic intrinsics - same shape version is the intrinsic (@noinline),
# different shape version broadcasts and recurses (@inline).
# Julia's dispatch prefers the more specific same-shape method when shapes match.

# Same-shape intrinsics - these are what the compiler intercepts
@noinline function tile_add(a::Tile{T, S}, b::Tile{T, S}) where {T, S}
    Base.donotdelete(a, b)
    Tile{T, S}()
end

@noinline function tile_sub(a::Tile{T, S}, b::Tile{T, S}) where {T, S}
    Base.donotdelete(a, b)
    Tile{T, S}()
end

@noinline function tile_mul(a::Tile{T, S}, b::Tile{T, S}) where {T, S}
    Base.donotdelete(a, b)
    Tile{T, S}()
end

# Broadcasting versions - different shapes, broadcast then recurse
@inline function tile_add(a::Tile{T, S1}, b::Tile{T, S2}) where {T, S1, S2}
    S = broadcast_shape(S1, S2)
    tile_add(broadcast_to(a, S), broadcast_to(b, S))
end

@inline function tile_sub(a::Tile{T, S1}, b::Tile{T, S2}) where {T, S1, S2}
    S = broadcast_shape(S1, S2)
    tile_sub(broadcast_to(a, S), broadcast_to(b, S))
end

@inline function tile_mul(a::Tile{T, S1}, b::Tile{T, S2}) where {T, S1, S2}
    S = broadcast_shape(S1, S2)
    tile_mul(broadcast_to(a, S), broadcast_to(b, S))
end

# Scalar variants convert to 0D tile and delegate to tile-tile
# broadcast_shape(S, ()) returns S, so the scalar gets broadcast to tile shape
@inline tile_add(a::Tile{T, S}, b::T) where {T <: AbstractFloat, S} = tile_add(a, Tile(b))
@inline tile_add(a::T, b::Tile{T, S}) where {T <: AbstractFloat, S} = tile_add(Tile(a), b)
@inline tile_sub(a::Tile{T, S}, b::T) where {T <: AbstractFloat, S} = tile_sub(a, Tile(b))
@inline tile_sub(a::T, b::Tile{T, S}) where {T <: AbstractFloat, S} = tile_sub(Tile(a), b)
@inline tile_mul(a::Tile{T, S}, b::T) where {T <: AbstractFloat, S} = tile_mul(a, Tile(b))
@inline tile_mul(a::T, b::Tile{T, S}) where {T <: AbstractFloat, S} = tile_mul(Tile(a), b)

# Operator overloads dispatch to the intrinsic functions (same shape required)
# @inline ensures these inline so codegen sees tile_add etc. instead of Base.:(+)
@inline Base.:(+)(a::Tile{T, S}, b::Tile{T, S}) where {T, S} = tile_add(a, b)
@inline Base.:(-)(a::Tile{T, S}, b::Tile{T, S}) where {T, S} = tile_sub(a, b)
@inline Base.:(*)(a::Tile{T, S}, b::Tile{T, S}) where {T, S} = tile_mul(a, b)

# Scalar-tile operators
@inline Base.:(+)(a::Tile{T, S}, b::T) where {T <: AbstractFloat, S} = tile_add(a, b)
@inline Base.:(+)(a::T, b::Tile{T, S}) where {T <: AbstractFloat, S} = tile_add(a, b)
@inline Base.:(-)(a::Tile{T, S}, b::T) where {T <: AbstractFloat, S} = tile_sub(a, b)
@inline Base.:(-)(a::T, b::Tile{T, S}) where {T <: AbstractFloat, S} = tile_sub(a, b)
@inline Base.:(*)(a::Tile{T, S}, b::T) where {T <: AbstractFloat, S} = tile_mul(a, b)
@inline Base.:(*)(a::T, b::Tile{T, S}) where {T <: AbstractFloat, S} = tile_mul(a, b)

#=============================================================================
 Tile Broadcasting (different shapes allowed via .+, .-, .*, ./)
=============================================================================#

public broadcast_to

"""
    broadcast_to(tile::Tile{T, S}, shape::NTuple{N, Int}) -> Tile{T, shape}

Explicitly broadcast a tile to a target shape.
The source shape must be broadcastable to the target shape.

# Example
```julia
row = ct.load(arr, (0, 0), (1, 128))  # Shape (1, 128)
expanded = ct.broadcast_to(row, (64, 128))  # Shape (64, 128)
```
"""
# Use Val{Shape} so Julia can infer the exact return type
@noinline function broadcast_to(tile::Tile{T, S}, ::Val{Shape}) where {T, S, Shape}
    Base.donotdelete(tile)
    Tile{T, Shape}()
end

# Convenience overload - inline wrapper that converts tuple to Val
@inline broadcast_to(tile::Tile{T, S}, shape::NTuple{N, Int}) where {T, S, N} = broadcast_to(tile, Val(shape))

# Hook into Julia's broadcasting system
# Define a custom BroadcastStyle for Tiles
import Base.Broadcast: BroadcastStyle, Broadcasted, broadcastable

struct TileStyle <: BroadcastStyle end
Base.Broadcast.BroadcastStyle(::Type{<:Tile}) = TileStyle()

# When combining TileStyle with itself, return TileStyle
Base.Broadcast.BroadcastStyle(::TileStyle, ::TileStyle) = TileStyle()

# When combining TileStyle with scalars, TileStyle wins
Base.Broadcast.BroadcastStyle(::TileStyle, ::Base.Broadcast.DefaultArrayStyle{0}) = TileStyle()

# Tiles are already broadcastable - return as-is
Base.Broadcast.broadcastable(t::Tile) = t

# Intercept broadcasted calls for Tile types
# a .+ b becomes broadcasted(+, a, b) which we intercept here
# These call the unified intrinsics which handle broadcasting internally
# @inline ensures these inline so codegen sees tile_add etc.
@inline Base.Broadcast.broadcasted(::TileStyle, ::typeof(+), a::Tile{T, S1}, b::Tile{T, S2}) where {T, S1, S2} =
    tile_add(a, b)
@inline Base.Broadcast.broadcasted(::TileStyle, ::typeof(-), a::Tile{T, S1}, b::Tile{T, S2}) where {T, S1, S2} =
    tile_sub(a, b)
@inline Base.Broadcast.broadcasted(::TileStyle, ::typeof(*), a::Tile{T, S1}, b::Tile{T, S2}) where {T, S1, S2} =
    tile_mul(a, b)
@inline Base.Broadcast.broadcasted(::TileStyle, ::typeof(/), a::Tile{T, S1}, b::Tile{T, S2}) where {T, S1, S2} =
    tile_div(a, b)

#=============================================================================
 Tile Shape Operations
=============================================================================#

public transpose

"""
    transpose(tile::Tile{T, (M, N)}) -> Tile{T, (N, M)}

Transpose a 2D tile, swapping its dimensions.
"""
@noinline function transpose(tile::Tile{T, S}) where {T, S}
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
@noinline function store(ptr::Ptr{T}, index, tile::Tile{T}) where T
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
@noinline function store(arr::TileArray{T, N}, index, tile::Tile{T}) where {T, N}
    Base.donotdelete(arr, index, tile)
    nothing
end

# Keyword argument version for ct.store(arr; index=..., tile=...)
@noinline function store(arr::TileArray{T, N}; index, tile::Tile{T}) where {T, N}
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
@noinline function full(shape::NTuple{N, Int}, value, ::Type{T}) where {N, T}
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
@noinline function astype(tile::Tile{T1, Shape}, ::Type{T2}) where {T1, Shape, T2}
    Base.donotdelete(tile)
    Tile{T2, Shape}()
end

# Julia-style convert syntax builds on astype
@inline Base.convert(::Type{Tile{T2}}, tile::Tile{T1, Shape}) where {T1, T2, Shape} = astype(tile, T2)

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
# Return type annotation needed here because inferencebarrier returns Any
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
 Floating-Point Division
=============================================================================#

public tile_div

# Same-shape division intrinsic
@noinline function tile_div(a::Tile{T, S}, b::Tile{T, S}) where {T <: AbstractFloat, S}
    Base.donotdelete(a, b)
    Tile{T, S}()
end

# Broadcasting version - different shapes, broadcast then recurse
@inline function tile_div(a::Tile{T, S1}, b::Tile{T, S2}) where {T <: AbstractFloat, S1, S2}
    S = broadcast_shape(S1, S2)
    tile_div(broadcast_to(a, S), broadcast_to(b, S))
end

# Scalar variants convert to 0D tile and delegate to tile-tile
@inline tile_div(a::Tile{T, S}, b::T) where {T <: AbstractFloat, S} = tile_div(a, Tile(b))
@inline tile_div(a::T, b::Tile{T, S}) where {T <: AbstractFloat, S} = tile_div(Tile(a), b)

# tile / integer: convert integer to tile's element type, then to 0D tile
@inline tile_div(a::Tile{T, S}, b::Integer) where {T <: AbstractFloat, S} = tile_div(a, Tile(T(b)))

# Division operator for tiles (same shape required)
# @inline ensures these inline so codegen sees tile_div instead of Base.:(/)
@inline Base.:(/)(a::Tile{T, S}, b::Tile{T, S}) where {T <: AbstractFloat, S} = tile_div(a, b)

# Scalar-tile division operators
@inline Base.:(/)(a::Tile{T, S}, b::T) where {T <: AbstractFloat, S} = tile_div(a, b)
@inline Base.:(/)(a::T, b::Tile{T, S}) where {T <: AbstractFloat, S} = tile_div(a, b)
@inline Base.:(/)(a::Tile{T, S}, b::Integer) where {T <: AbstractFloat, S} = tile_div(a, b)

#=============================================================================
 Math Operations
=============================================================================#

public sqrt, rsqrt

"""
    sqrt(tile::Tile{T, S}) -> Tile{T, S}

Compute element-wise square root of a tile.
"""
@noinline function Base.sqrt(tile::Tile{T, S}) where {T <: AbstractFloat, S}
    Base.donotdelete(tile)
    Tile{T, S}()
end

"""
    rsqrt(tile::Tile{T, S}) -> Tile{T, S}

Compute element-wise reciprocal square root (1/sqrt(x)) of a tile.
"""
@noinline function rsqrt(tile::Tile{T, S}) where {T <: AbstractFloat, S}
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
@noinline function arange(shape::NTuple{1, Int}, ::Type{T}) where {T}
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
@noinline function where(cond::Tile{Bool, S}, x::Tile{T, S}, y::Tile{T, S}) where {T, S}
    Base.donotdelete(cond, x, y)
    Tile{T, S}()
end

#=============================================================================
 Comparison Operations (returning Boolean tiles)
=============================================================================#

# Element-wise comparisons - same shape intrinsics (work for any element type T)
# These are what the compiler intercepts
@noinline function tile_lt(a::Tile{T, S}, b::Tile{T, S}) where {T, S}
    Base.donotdelete(a, b)
    Tile{Bool, S}()
end

@noinline function tile_gt(a::Tile{T, S}, b::Tile{T, S}) where {T, S}
    Base.donotdelete(a, b)
    Tile{Bool, S}()
end

@noinline function tile_le(a::Tile{T, S}, b::Tile{T, S}) where {T, S}
    Base.donotdelete(a, b)
    Tile{Bool, S}()
end

@noinline function tile_ge(a::Tile{T, S}, b::Tile{T, S}) where {T, S}
    Base.donotdelete(a, b)
    Tile{Bool, S}()
end

@noinline function tile_eq(a::Tile{T, S}, b::Tile{T, S}) where {T, S}
    Base.donotdelete(a, b)
    Tile{Bool, S}()
end

@noinline function tile_ne(a::Tile{T, S}, b::Tile{T, S}) where {T, S}
    Base.donotdelete(a, b)
    Tile{Bool, S}()
end

# Broadcasting versions - different shapes, broadcast then recurse
@inline function tile_lt(a::Tile{T, S1}, b::Tile{T, S2}) where {T, S1, S2}
    S = broadcast_shape(S1, S2)
    tile_lt(broadcast_to(a, S), broadcast_to(b, S))
end

@inline function tile_gt(a::Tile{T, S1}, b::Tile{T, S2}) where {T, S1, S2}
    S = broadcast_shape(S1, S2)
    tile_gt(broadcast_to(a, S), broadcast_to(b, S))
end

@inline function tile_le(a::Tile{T, S1}, b::Tile{T, S2}) where {T, S1, S2}
    S = broadcast_shape(S1, S2)
    tile_le(broadcast_to(a, S), broadcast_to(b, S))
end

@inline function tile_ge(a::Tile{T, S1}, b::Tile{T, S2}) where {T, S1, S2}
    S = broadcast_shape(S1, S2)
    tile_ge(broadcast_to(a, S), broadcast_to(b, S))
end

@inline function tile_eq(a::Tile{T, S1}, b::Tile{T, S2}) where {T, S1, S2}
    S = broadcast_shape(S1, S2)
    tile_eq(broadcast_to(a, S), broadcast_to(b, S))
end

@inline function tile_ne(a::Tile{T, S1}, b::Tile{T, S2}) where {T, S1, S2}
    S = broadcast_shape(S1, S2)
    tile_ne(broadcast_to(a, S), broadcast_to(b, S))
end

# Broadcast hooks for comparison operators (tile .< tile, etc.)
# Tile-Tile comparisons
@inline Base.Broadcast.broadcasted(::TileStyle, ::typeof(<), a::Tile{T,S1}, b::Tile{T,S2}) where {T,S1,S2} =
    tile_lt(a, b)
@inline Base.Broadcast.broadcasted(::TileStyle, ::typeof(>), a::Tile{T,S1}, b::Tile{T,S2}) where {T,S1,S2} =
    tile_gt(a, b)
@inline Base.Broadcast.broadcasted(::TileStyle, ::typeof(<=), a::Tile{T,S1}, b::Tile{T,S2}) where {T,S1,S2} =
    tile_le(a, b)
@inline Base.Broadcast.broadcasted(::TileStyle, ::typeof(>=), a::Tile{T,S1}, b::Tile{T,S2}) where {T,S1,S2} =
    tile_ge(a, b)
@inline Base.Broadcast.broadcasted(::TileStyle, ::typeof(==), a::Tile{T,S1}, b::Tile{T,S2}) where {T,S1,S2} =
    tile_eq(a, b)
@inline Base.Broadcast.broadcasted(::TileStyle, ::typeof(!=), a::Tile{T,S1}, b::Tile{T,S2}) where {T,S1,S2} =
    tile_ne(a, b)

# Tile-Scalar comparisons (convert scalar to 0D tile, then broadcast)
@inline Base.Broadcast.broadcasted(::TileStyle, ::typeof(<), a::Tile{T,S}, b::Number) where {T,S} =
    tile_lt(a, Tile(T(b)))
@inline Base.Broadcast.broadcasted(::TileStyle, ::typeof(<), a::Number, b::Tile{T,S}) where {T,S} =
    tile_lt(Tile(T(a)), b)
@inline Base.Broadcast.broadcasted(::TileStyle, ::typeof(>), a::Tile{T,S}, b::Number) where {T,S} =
    tile_gt(a, Tile(T(b)))
@inline Base.Broadcast.broadcasted(::TileStyle, ::typeof(>), a::Number, b::Tile{T,S}) where {T,S} =
    tile_gt(Tile(T(a)), b)
@inline Base.Broadcast.broadcasted(::TileStyle, ::typeof(<=), a::Tile{T,S}, b::Number) where {T,S} =
    tile_le(a, Tile(T(b)))
@inline Base.Broadcast.broadcasted(::TileStyle, ::typeof(<=), a::Number, b::Tile{T,S}) where {T,S} =
    tile_le(Tile(T(a)), b)
@inline Base.Broadcast.broadcasted(::TileStyle, ::typeof(>=), a::Tile{T,S}, b::Number) where {T,S} =
    tile_ge(a, Tile(T(b)))
@inline Base.Broadcast.broadcasted(::TileStyle, ::typeof(>=), a::Number, b::Tile{T,S}) where {T,S} =
    tile_ge(Tile(T(a)), b)
@inline Base.Broadcast.broadcasted(::TileStyle, ::typeof(==), a::Tile{T,S}, b::Number) where {T,S} =
    tile_eq(a, Tile(T(b)))
@inline Base.Broadcast.broadcasted(::TileStyle, ::typeof(==), a::Number, b::Tile{T,S}) where {T,S} =
    tile_eq(Tile(T(a)), b)
@inline Base.Broadcast.broadcasted(::TileStyle, ::typeof(!=), a::Tile{T,S}, b::Number) where {T,S} =
    tile_ne(a, Tile(T(b)))
@inline Base.Broadcast.broadcasted(::TileStyle, ::typeof(!=), a::Number, b::Tile{T,S}) where {T,S} =
    tile_ne(Tile(T(a)), b)
