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

# Power operation (float only)
@noinline function tile_pow(a::Tile{T, S}, b::Tile{T, S}) where {T <: AbstractFloat, S}
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

@inline function tile_pow(a::Tile{T, S1}, b::Tile{T, S2}) where {T <: AbstractFloat, S1, S2}
    S = broadcast_shape(S1, S2)
    tile_pow(broadcast_to(a, S), broadcast_to(b, S))
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
@inline Base.Broadcast.broadcasted(::TileStyle, ::typeof(^), a::Tile{T, S1}, b::Tile{T, S2}) where {T <: AbstractFloat, S1, S2} =
    tile_pow(a, b)

# Tile-Scalar arithmetic (tile .+ scalar, scalar .+ tile, etc.)
@inline Base.Broadcast.broadcasted(::TileStyle, ::typeof(+), a::Tile{T,S}, b::Number) where {T,S} =
    tile_add(a, Tile(T(b)))
@inline Base.Broadcast.broadcasted(::TileStyle, ::typeof(+), a::Number, b::Tile{T,S}) where {T,S} =
    tile_add(Tile(T(a)), b)
@inline Base.Broadcast.broadcasted(::TileStyle, ::typeof(-), a::Tile{T,S}, b::Number) where {T,S} =
    tile_sub(a, Tile(T(b)))
@inline Base.Broadcast.broadcasted(::TileStyle, ::typeof(-), a::Number, b::Tile{T,S}) where {T,S} =
    tile_sub(Tile(T(a)), b)
@inline Base.Broadcast.broadcasted(::TileStyle, ::typeof(*), a::Tile{T,S}, b::Number) where {T,S} =
    tile_mul(a, Tile(T(b)))
@inline Base.Broadcast.broadcasted(::TileStyle, ::typeof(*), a::Number, b::Tile{T,S}) where {T,S} =
    tile_mul(Tile(T(a)), b)
@inline Base.Broadcast.broadcasted(::TileStyle, ::typeof(/), a::Tile{T,S}, b::Number) where {T,S} =
    tile_div(a, Tile(T(b)))
@inline Base.Broadcast.broadcasted(::TileStyle, ::typeof(/), a::Number, b::Tile{T,S}) where {T,S} =
    tile_div(Tile(T(a)), b)

# Tile-Scalar power (tile .^ scalar, scalar .^ tile)
@inline Base.Broadcast.broadcasted(::TileStyle, ::typeof(^), a::Tile{T,S}, b::Number) where {T <: AbstractFloat, S} =
    tile_pow(a, Tile(T(b)))
@inline Base.Broadcast.broadcasted(::TileStyle, ::typeof(^), a::Number, b::Tile{T,S}) where {T <: AbstractFloat, S} =
    tile_pow(Tile(T(a)), b)

#=============================================================================
 Tile Shape Operations
=============================================================================#

public transpose, reshape, permute

"""
    transpose(tile::Tile{T, (M, N)}) -> Tile{T, (N, M)}

Transpose a 2D tile, swapping its dimensions.
"""
@noinline function transpose(tile::Tile{T, S}) where {T, S}
    Tile{T, reverse(S)}()
end

"""
    reshape(tile::Tile{T, S}, shape::NTuple{N, Int}) -> Tile{T, shape}

Reshape a tile to a new shape. The total number of elements must remain the same.

# Example
```julia
tile = ct.load(arr, (0, 0), (4, 8))  # Shape (4, 8), 32 elements
reshaped = ct.reshape(tile, (2, 16))  # Shape (2, 16), still 32 elements
```
"""
@noinline function reshape(tile::Tile{T, S}, ::Val{Shape}) where {T, S, Shape}
    Base.donotdelete(tile)
    Tile{T, Shape}()
end

# Convenience overload - inline wrapper that converts tuple to Val
@inline reshape(tile::Tile{T, S}, shape::NTuple{N, Int}) where {T, S, N} = reshape(tile, Val(shape))

"""
    permute(tile::Tile{T, S}, perm::NTuple{N, Int}) -> Tile{T, permuted_shape}

Permute the dimensions of a tile according to the given permutation.
The permutation uses 0-indexed axes (matching cuTile Python convention).

# Example
```julia
tile = ct.load(arr, (0, 0, 0), (2, 3, 4))  # Shape (2, 3, 4)
# Permute axes: new_axis_0 = old_axis_2, new_axis_1 = old_axis_0, new_axis_2 = old_axis_1
permuted = ct.permute(tile, (2, 0, 1))  # Shape (4, 2, 3)
```
"""
@noinline function permute(tile::Tile{T, S}, ::Val{Perm}) where {T, S, Perm}
    Base.donotdelete(tile)
    # Compute permuted shape: for each position i in output, take S[Perm[i]+1]
    # (Perm is 0-indexed, S is 1-indexed tuple access)
    permuted_shape = ntuple(i -> S[Perm[i] + 1], length(Perm))
    Tile{T, permuted_shape}()
end

# Convenience overload - inline wrapper that converts tuple to Val
@inline permute(tile::Tile{T, S}, perm::NTuple{N, Int}) where {T, S, N} = permute(tile, Val(perm))

public extract

"""
    extract(tile::Tile{T, S}, index::NTuple{N, Int}, shape::NTuple{N, Int}) -> Tile{T, shape}

Extract a slice from a tile starting at the given index with the specified shape.
Both index and shape are 0-indexed tuples matching cuTile Python convention.

# Example
```julia
# Given a tile with shape (batch, N, 2) where 2 represents real/imag components
tile = ct.load(arr, (0, 0, 0), (BS, N, 2))
# Extract just the real component (index 0 in last dimension)
real_part = ct.extract(tile, (0, 0, 0), (BS, N, 1))  # Shape (BS, N, 1)
# Extract just the imaginary component (index 1 in last dimension)
imag_part = ct.extract(tile, (0, 0, 1), (BS, N, 1))  # Shape (BS, N, 1)
```
"""
@noinline function extract(tile::Tile{T, S}, ::Val{Index}, ::Val{Shape}) where {T, S, Index, Shape}
    Base.donotdelete(tile)
    Tile{T, Shape}()
end

# Convenience overload - inline wrapper that converts tuples to Val
@inline extract(tile::Tile{T, S}, index::NTuple{N, Int}, shape::NTuple{M, Int}) where {T, S, N, M} =
    extract(tile, Val(index), Val(shape))

public cat

"""
    cat(tiles::Tuple{Tile, Tile}, axis::Int) -> Tile

Concatenate two tiles along the specified axis.
Supports negative axis (e.g., -1 for last dimension).

# Example
```julia
tile_a = ct.load(arr_a, (0,), (4, 8))  # Shape (4, 8)
tile_b = ct.load(arr_b, (0,), (4, 8))  # Shape (4, 8)
# Concatenate along axis 0: (4, 8) + (4, 8) -> (8, 8)
combined = ct.cat((tile_a, tile_b), 0)
# Concatenate along axis -1 (last): (4, 8) + (4, 8) -> (4, 16)
combined_last = ct.cat((tile_a, tile_b), -1)
```
"""
@noinline function cat(tiles::Tuple{Tile{T, S1}, Tile{T, S2}}, ::Val{Axis}) where {T, S1, S2, Axis}
    Base.donotdelete(tiles)
    # Compute output shape: dimensions must match except at the concatenation axis
    # Handle negative axis
    ndims = length(S1)
    axis = Axis < 0 ? ndims + Axis : Axis
    # Result shape: sum the sizes along axis, keep others same
    result_shape = ntuple(ndims) do i
        if i == axis + 1  # 0-indexed axis, 1-indexed tuple access
            S1[i] + S2[i]
        else
            S1[i]  # S1[i] should equal S2[i] for valid concatenation
        end
    end
    Tile{T, result_shape}()
end

# Convenience overload - inline wrapper that converts axis to Val
@inline cat(tiles::Tuple{Tile{T, S1}, Tile{T, S2}}, axis::Int) where {T, S1, S2} =
    cat(tiles, Val(axis))

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
# Note: For Ptr, we also use variadic indices for consistency
@noinline function _load(ptr::Ptr{T}, ::Val{shape}, indices...) where {T, shape}
    Tile{T, shape}()
end
# Public API - inline wrapper that captures shape and splats indices
@inline load(ptr::Ptr{T}, index, shape::NTuple{N, Int}) where {T, N} = _load(ptr, Val(shape), index...)
@inline load(ptr::Ptr{T}, index::Integer, shape::NTuple{N, Int}) where {T, N} = _load(ptr, Val(shape), index)

"""
    store(ptr, index, tile::Tile) -> Nothing

Store a tile to a pointer at the given index.
In kernel code, this is compiled to StoreViewTkoOp.
"""
# Internal function with variadic indices for SROA
@noinline function _store(ptr::Ptr{T}, tile::Tile{T}, indices...) where T
    Base.donotdelete(ptr, tile, indices...)
    nothing
end
# Public API - inline wrapper that splats indices
@inline store(ptr::Ptr{T}, index, tile::Tile{T}) where T = _store(ptr, tile, index...)
@inline store(ptr::Ptr{T}, index::Integer, tile::Tile{T}) where T = _store(ptr, tile, index)

# TileArray overloads - these are intercepted by the compiler
# The compiler extracts ptr/sizes/strides from the destructured TileArray

"""
    load(arr::TileArray, index, shape; padding_mode=PaddingMode.Undetermined) -> Tile

Load a tile from a TileArray at the given index with the specified shape.
The TileArray's sizes and strides are used to construct the TensorView.

# Arguments
- `arr`: The TileArray to load from
- `index`: The tile index (0-indexed)
- `shape`: The tile shape (must be compile-time constants)
- `padding_mode`: Behavior for out-of-bounds loads (default: Undetermined)

# Padding Modes
- `PaddingMode.Undetermined`: Unspecified behavior for OOB access
- `PaddingMode.Zero`: Return zero for OOB elements
- `PaddingMode.NegZero`: Return negative zero for OOB elements
- `PaddingMode.Nan`: Return NaN for OOB elements
- `PaddingMode.PosInf`: Return positive infinity for OOB elements
- `PaddingMode.NegInf`: Return negative infinity for OOB elements

# Example
```julia
tile = ct.load(arr, (bid,), (TILE_N[],); padding_mode=ct.PaddingMode.Zero)
```
"""
# Internal function with shape as type parameter for proper type inference
# Indices are variadic at the end so Julia can SROA the tuple
@noinline function _load(arr::TileArray{T, N}, ::Val{shape}, padding_mode::Int, indices...) where {T, N, shape}
    Base.donotdelete(arr, indices..., padding_mode)
    Tile{T, shape}()
end
# Public API - inline wrapper that captures shape and splats indices
@inline function load(arr::TileArray{T, N}, index, shape::NTuple{M, Int};
                      padding_mode::Int=PaddingMode.Undetermined) where {T, N, M}
    _load(arr, Val(shape), padding_mode, index...)
end

# Single index (scalar) - no splatting needed
@inline function load(arr::TileArray{T, N}, index::Integer, shape::NTuple{M, Int};
                      padding_mode::Int=PaddingMode.Undetermined) where {T, N, M}
    _load(arr, Val(shape), padding_mode, index)
end

# Load with Constant shape tuple (1D) - extracts value from Constant type parameter
@inline function load(arr::TileArray{T, N}, index, shape::Tuple{Constant{Int, V}};
                      padding_mode::Int=PaddingMode.Undetermined) where {T, N, V}
    _load(arr, Val((V,)), padding_mode, index...)
end

# Load with Constant shape tuple (2D)
@inline function load(arr::TileArray{T, N}, index, shape::Tuple{Constant{Int, V1}, Constant{Int, V2}};
                      padding_mode::Int=PaddingMode.Undetermined) where {T, N, V1, V2}
    _load(arr, Val((V1, V2)), padding_mode, index...)
end

# Load with Constant shape tuple (3D)
@inline function load(arr::TileArray{T, N}, index, shape::Tuple{Constant{Int, V1}, Constant{Int, V2}, Constant{Int, V3}};
                      padding_mode::Int=PaddingMode.Undetermined) where {T, N, V1, V2, V3}
    _load(arr, Val((V1, V2, V3)), padding_mode, index...)
end

# Keyword argument version for ct.load(arr; index=..., shape=..., padding_mode=...)
@inline function load(arr::TileArray{T, N}; index, shape,
                      padding_mode::Int=PaddingMode.Undetermined) where {T, N}
    shape_val = _extract_shape(shape)
    _load(arr, Val(shape_val), padding_mode, index...)
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
# Internal function with variadic indices at the end for SROA
@noinline function _store(arr::TileArray{T, N}, tile::Tile{T}, indices...) where {T, N}
    Base.donotdelete(arr, tile, indices...)
    nothing
end

# Public API - inline wrapper that splats indices
@inline function store(arr::TileArray{T, N}, index, tile::Tile{T}) where {T, N}
    _store(arr, tile, index...)
end

# Single index (scalar) - no splatting needed
@inline function store(arr::TileArray{T, N}, index::Integer, tile::Tile{T}) where {T, N}
    _store(arr, tile, index)
end

# Keyword argument version for ct.store(arr; index=..., tile=...)
@inline function store(arr::TileArray{T, N}; index, tile::Tile{T}) where {T, N}
    _store(arr, tile, index...)
end

#=============================================================================
 Matrix Multiply-Accumulate
=============================================================================#

public mma, matmul

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

"""
    matmul(a::Tile{T1, (M, K)}, b::Tile{T2, (K, N)}) -> Tile{T1, (M, N)}

Perform matrix multiplication: result = a @ b.
Equivalent to `mma(a, b, zeros((M, N), T1))`.

# Example
```julia
c = ct.matmul(a, b)  # c = a @ b
```
"""
@inline function matmul(a::Tile{T1, SA}, b::Tile{T2, SB}) where {T1, T2, SA, SB}
    M = SA[1]
    N = SB[2]
    acc = zeros((M, N), T1)
    mma(a, b, acc)
end

#=============================================================================
 Tile Construction
=============================================================================#

public full, zeros, astype

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
    zeros(shape::NTuple{N, Int}, dtype::Type{T}) -> Tile{T, shape}

Create a tile filled with zeros. Equivalent to `full(shape, zero(T), T)`.

# Example
```julia
zeros_tile = ct.zeros((32, 32), Float32)  # 32x32 tile of zeros
```
"""
@inline zeros(shape::NTuple{N, Int}, ::Type{T}) where {N, T} = full(shape, zero(T), T)

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
- `axis`: Axis to reduce along (0-indexed). Must be a compile-time constant.

# Example
```julia
# For a (128, 64) tile, reducing along axis 1:
sums = ct.reduce_sum(tile, 1)  # Returns (128,) tile
```
"""
@inline function reduce_sum(tile::Tile{T, S}, axis::Integer) where {T <: AbstractFloat, S}
    # Forward to Val-based version for type stability
    reduce_sum(tile, Val(axis))
end

@noinline function reduce_sum(tile::Tile{T, S}, ::Val{axis}) where {T <: AbstractFloat, S, axis}
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
- `axis`: Axis to reduce along (0-indexed). Must be a compile-time constant.
"""
@inline function reduce_max(tile::Tile{T, S}, axis::Integer) where {T <: AbstractFloat, S}
    reduce_max(tile, Val(axis))
end

@noinline function reduce_max(tile::Tile{T, S}, ::Val{axis}) where {T <: AbstractFloat, S, axis}
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

#=============================================================================
 Padding Mode (for load operations)
=============================================================================#

"""
Padding mode for load operations.
Use these constants with ct.load to specify out-of-bounds behavior.

# Example
```julia
tile = ct.load(arr, (bid,), (TILE_N[],); padding_mode=ct.PaddingMode.Zero)
```
"""
module PaddingMode
    const Undetermined = 0
    const Zero = 1
    const NegZero = 2
    const Nan = 3
    const PosInf = 4
    const NegInf = 5
end

#=============================================================================
 Memory Ordering (for atomic operations)
=============================================================================#

# Memory ordering constants for atomic operations
# These are simple integer constants that get converted to bytecode enums in codegen

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

#=============================================================================
 Atomic Operations
=============================================================================#

public atomic_cas, atomic_xchg, atomic_add

# Inner stub - @noinline, positional-only, appears in IR for codegen
@noinline function _atomic_cas(array::TileArray{T, N}, index, expected, desired,
                               memory_order::Int, memory_scope::Int) where {T, N}
    Base.donotdelete(array, index, expected, desired)
    # Return scalar (not Tile) so comparisons work in control flow (e.g., spinloops)
    # Use inferencebarrier to prevent Julia from constant-folding the return value
    Base.inferencebarrier(zero(T))::T
end

"""
    atomic_cas(array::TileArray, index, expected, desired; memory_order, memory_scope) -> T

Atomic compare-and-swap. Atomically compares the value at `index` with `expected`,
and if equal, replaces it with `desired`. Returns the original value as a scalar.

Used for implementing locks and lock-free data structures. Returns a scalar (not a Tile)
so that comparisons work naturally in control flow conditions like spinloops.

# Example
```julia
# Spin-lock acquisition
while ct.atomic_cas(locks, idx, Int32(0), Int32(1); memory_order=ct.MemoryOrder.Acquire) == Int32(1)
    # spin
end
```
"""
@inline function atomic_cas(array::TileArray{T, N}, index, expected, desired;
                            memory_order::Int=MemoryOrder.AcqRel,
                            memory_scope::Int=MemScope.Device) where {T, N}
    _atomic_cas(array, index, expected, desired, memory_order, memory_scope)::T
end

# Inner stub - @noinline, positional-only, appears in IR for codegen
@noinline function _atomic_xchg(array::TileArray{T, N}, index, val,
                                memory_order::Int, memory_scope::Int) where {T, N}
    Base.donotdelete(array, index, val)
    Base.inferencebarrier(zero(T))::T
end

"""
    atomic_xchg(array::TileArray, index, val; memory_order, memory_scope) -> T

Atomic exchange. Atomically replaces the value at `index` with `val` and returns
the original value.

Used for implementing locks (release) and other synchronization primitives.

# Example
```julia
# Spin-lock release
ct.atomic_xchg(locks, idx, Int32(0); memory_order=ct.MemoryOrder.Release)
```
"""
@inline function atomic_xchg(array::TileArray{T, N}, index, val;
                             memory_order::Int=MemoryOrder.AcqRel,
                             memory_scope::Int=MemScope.Device) where {T, N}
    _atomic_xchg(array, index, val, memory_order, memory_scope)::T
end

# Inner stub - @noinline, positional-only, appears in IR for codegen
@noinline function _atomic_add(array::TileArray{T, N}, index, val,
                               memory_order::Int, memory_scope::Int) where {T, N}
    Base.donotdelete(array, index, val)
    Base.inferencebarrier(zero(T))::T
end

"""
    atomic_add(array::TileArray, index, val; memory_order, memory_scope) -> T

Atomic addition. Atomically adds `val` to the value at `index` and returns
the original value.

# Example
```julia
old_val = ct.atomic_add(counters, idx, Int32(1))
```
"""
@inline function atomic_add(array::TileArray{T, N}, index, val;
                            memory_order::Int=MemoryOrder.AcqRel,
                            memory_scope::Int=MemScope.Device) where {T, N}
    _atomic_add(array, index, val, memory_order, memory_scope)::T
end
