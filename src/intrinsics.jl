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
The permutation uses 1-indexed axes (Julia convention).

# Example
```julia
tile = ct.load(arr, (1, 1, 1), (2, 3, 4))  # Shape (2, 3, 4)
# Permute axes: new_axis_1 = old_axis_3, new_axis_2 = old_axis_1, new_axis_3 = old_axis_2
permuted = ct.permute(tile, (3, 1, 2))  # Shape (4, 2, 3)
```
"""
@noinline function _permute(tile::Tile{T, S}, ::Val{Perm}) where {T, S, Perm}
    Base.donotdelete(tile)
    # Compute permuted shape: for each position i in output, take S[Perm[i]+1]
    # (Perm is 0-indexed, S is 1-indexed tuple access)
    permuted_shape = ntuple(i -> S[Perm[i] + 1], length(Perm))
    Tile{T, permuted_shape}()
end

@inline permute(tile::Tile{T, S}, ::Val{Perm}) where {T, S, Perm} =
    _permute(tile, Val(map(p -> p - 1, Perm)))
@inline permute(tile::Tile{T, S}, perm::NTuple{N, Int}) where {T, S, N} =
    _permute(tile, Val(map(p -> p - 1, perm)))

public extract

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

# Example: Separating real/imag components (FFT pattern)
```julia
# Shape (BS, N, 2) where last dim is real/imag
tile = ct.load(arr, (1, 1, 1), (BS, N, 2))
# 2÷1 = 2 slices in last dim, valid indices are {1, 2}
real_part = ct.extract(tile, (1, 1, 1), (BS, N, 1))  # Slice 1 = real
imag_part = ct.extract(tile, (1, 1, 2), (BS, N, 1))  # Slice 2 = imag
```
"""
@noinline function _extract(tile::Tile{T, S}, ::Val{Index}, ::Val{Shape}) where {T, S, Index, Shape}
    Base.donotdelete(tile)
    Tile{T, Shape}()
end

@inline extract(tile::Tile{T, S}, ::Val{Index}, ::Val{Shape}) where {T, S, Index, Shape} =
    _extract(tile, Val(map(i -> i - 1, Index)), Val(Shape))
@inline extract(tile::Tile{T, S}, index::NTuple{N, Int}, shape::NTuple{M, Int}) where {T, S, N, M} =
    _extract(tile, Val(map(i -> i - 1, index)), Val(shape))

public cat

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
@noinline function _cat(tiles::Tuple{Tile{T, S1}, Tile{T, S2}}, ::Val{Axis}) where {T, S1, S2, Axis}
    Base.donotdelete(tiles)
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

@inline function cat(tiles::Tuple{Tile{T, S1}, Tile{T, S2}}, ::Val{Axis}) where {T, S1, S2, Axis}
    axis0 = Axis < 0 ? Axis : Axis - 1
    _cat(tiles, Val(axis0))
end
@inline function cat(tiles::Tuple{Tile{T, S1}, Tile{T, S2}}, axis::Int) where {T, S1, S2}
    axis0 = axis < 0 ? axis : axis - 1
    _cat(tiles, Val(axis0))
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

Get the block ID along the given axis (1=x, 2=y, 3=z).
In kernel code, this is compiled to GetTileBlockIdOp.
"""
@noinline _bid(axis::Integer)::Int32 = Base.inferencebarrier(zero(Int32))
@inline bid(axis::Integer)::Int32 = _bid(axis - one(axis)) + Int32(1)

"""
    num_blocks(axis) -> Int32

Get the grid size along the given axis (1=x, 2=y, 3=z).
In kernel code, this is compiled to GetNumTileBlocksOp.
"""
@noinline _num_blocks(axis::Integer)::Int32 = Base.inferencebarrier(zero(Int32))
@inline num_blocks(axis::Integer)::Int32 = _num_blocks(axis - one(axis))

# Helper: subtract 1 from each element of a tuple, preserving element types
# Uses map instead of ntuple to support heterogeneous tuples (e.g., Tuple{Int32, Int64})
@inline _sub1(index::Tuple) = map(i -> i - one(i), index)

"""
    load(ptr, index, shape::NTuple{N, Int}) -> Tile{T, shape}

Load a tile from a pointer at the given index with the specified shape.
In kernel code, this is compiled to LoadViewTkoOp.

Returns a `Tile{T, Shape}` where T is the pointer element type and Shape
is the compile-time constant shape tuple.
"""
@noinline function _load(ptr::Ptr{T}, ::Val{shape}, indices...) where {T, shape}
    Tile{T, shape}()
end
@inline load(ptr::Ptr{T}, index, shape::NTuple{N, Int}) where {T, N} = _load(ptr, Val(shape), _sub1(index)...)
@inline load(ptr::Ptr{T}, index::Integer, shape::NTuple{N, Int}) where {T, N} = _load(ptr, Val(shape), index - one(index))

"""
    store(ptr, index, tile::Tile) -> Nothing

Store a tile to a pointer at the given index.
In kernel code, this is compiled to StoreViewTkoOp.
"""
@noinline function _store(ptr::Ptr{T}, tile::Tile{T}, indices...) where T
    Base.donotdelete(ptr, tile, indices...)
    nothing
end
@inline store(ptr::Ptr{T}, index, tile::Tile{T}) where T = _store(ptr, tile, _sub1(index)...)
@inline store(ptr::Ptr{T}, index::Integer, tile::Tile{T}) where T = _store(ptr, tile, index - one(index))

# TileArray overloads - these are intercepted by the compiler
# The compiler extracts ptr/sizes/strides from the destructured TileArray

"""
    load(arr::TileArray, index, shape; padding_mode=PaddingMode.Undetermined) -> Tile

Load a tile from a TileArray at the given index with the specified shape.
The TileArray's sizes and strides are used to construct the TensorView.

# Arguments
- `arr`: The TileArray to load from
- `index`: The tile index
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
@noinline function _load(arr::TileArray{T, N}, ::Val{shape}, padding_mode::Int, indices...) where {T, N, shape}
    Base.donotdelete(arr, indices..., padding_mode)
    Tile{T, shape}()
end
@inline function load(arr::TileArray{T, N}, index, shape::NTuple{M, Int};
                      padding_mode::Int=PaddingMode.Undetermined) where {T, N, M}
    _load(arr, Val(shape), padding_mode, _sub1(index)...)
end

# Single index (scalar) - no splatting needed
@inline function load(arr::TileArray{T, N}, index::Integer, shape::NTuple{M, Int};
                      padding_mode::Int=PaddingMode.Undetermined) where {T, N, M}
    _load(arr, Val(shape), padding_mode, index - one(index))
end

# Load with Constant shape tuple (1D) - extracts value from Constant type parameter
@inline function load(arr::TileArray{T, N}, index, shape::Tuple{Constant{Int, V}};
                      padding_mode::Int=PaddingMode.Undetermined) where {T, N, V}
    _load(arr, Val((V,)), padding_mode, _sub1(index)...)
end

# Load with Constant shape tuple (2D)
@inline function load(arr::TileArray{T, N}, index, shape::Tuple{Constant{Int, V1}, Constant{Int, V2}};
                      padding_mode::Int=PaddingMode.Undetermined) where {T, N, V1, V2}
    _load(arr, Val((V1, V2)), padding_mode, _sub1(index)...)
end

# Load with Constant shape tuple (3D)
@inline function load(arr::TileArray{T, N}, index, shape::Tuple{Constant{Int, V1}, Constant{Int, V2}, Constant{Int, V3}};
                      padding_mode::Int=PaddingMode.Undetermined) where {T, N, V1, V2, V3}
    _load(arr, Val((V1, V2, V3)), padding_mode, _sub1(index)...)
end

# Keyword argument version for ct.load(arr; index=..., shape=..., padding_mode=...)
@inline function load(arr::TileArray{T, N}; index, shape,
                      padding_mode::Int=PaddingMode.Undetermined) where {T, N}
    shape_val = _extract_shape(shape)
    _load(arr, Val(shape_val), padding_mode, _sub1(index)...)
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
@noinline function _store(arr::TileArray{T, N}, tile::Tile{T}, indices...) where {T, N}
    Base.donotdelete(arr, tile, indices...)
    nothing
end
@inline function store(arr::TileArray{T, N}, index, tile::Tile{T}) where {T, N}
    _store(arr, tile, _sub1(index)...)
end

# Single index (scalar) - no splatting needed
@inline function store(arr::TileArray{T, N}, index::Integer, tile::Tile{T}) where {T, N}
    _store(arr, tile, index - one(index))
end

# Keyword argument version for ct.store(arr; index=..., tile=...)
@inline function store(arr::TileArray{T, N}; index, tile::Tile{T}) where {T, N}
    _store(arr, tile, _sub1(index)...)
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
    matmul(a::Tile{T1, S}, b::Tile{T2, S}) -> Tile{T1, S}

Perform matrix multiplication: result = a @ b.
Equivalent to `mma(a, b, zeros(output_shape, T1))`.

Supports both 2D and 3D (batched) inputs:
- 2D: a:(M, K) × b:(K, N) → (M, N)
- 3D: a:(B, M, K) × b:(B, K, N) → (B, M, N)  (or b:(1, K, N) for broadcasting)

# Example
```julia
c = ct.matmul(a, b)  # c = a @ b
```
"""
@inline function matmul(a::Tile{T1, SA}, b::Tile{T2, SB}) where {T1, T2, SA, SB}
    _matmul(a, b, Val(length(SA)))
end

# 2D matmul: (M, K) × (K, N) → (M, N)
@inline function _matmul(a::Tile{T1, SA}, b::Tile{T2, SB}, ::Val{2}) where {T1, T2, SA, SB}
    M = SA[1]
    N = SB[2]
    acc = zeros((M, N), T1)
    mma(a, b, acc)
end

# 3D batched matmul: (B, M, K) × (B, K, N) → (B, M, N)
# Also supports broadcasting: (1, M, K) × (B, K, N) or (B, M, K) × (1, K, N)
@inline function _matmul(a::Tile{T1, SA}, b::Tile{T2, SB}, ::Val{3}) where {T1, T2, SA, SB}
    B = max(SA[1], SB[1])  # Broadcast batch dimension
    M = SA[2]
    N = SB[3]
    acc = zeros((B, M, N), T1)
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
This is equivalent to cdiv(arr.sizes[axis], shape[axis]).

# Arguments
- `arr`: The array to query
- `axis`: The axis (1-indexed) to count tiles along
- `shape`: The tile shape used for partitioning

# Example
```julia
# For a 1024x768 matrix with 32x32 tiles:
# num_tiles(arr, 1, (32, 32)) returns cdiv(1024, 32) = 32
# num_tiles(arr, 2, (32, 32)) returns cdiv(768, 32) = 24
```
"""
# Return type annotation needed here because inferencebarrier returns Any
@noinline function _num_tiles(arr::TileArray{T, N}, axis::Integer, shape::NTuple{M, Int})::Int32 where {T, N, M}
    Base.inferencebarrier(zero(Int32))
end
@inline function num_tiles(arr::TileArray{T, N}, axis::Integer, shape::NTuple{M, Int})::Int32 where {T, N, M}
    _num_tiles(arr, axis - 1, shape)
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

Create a 1D tile with values [1, 2, 3, ..., shape[1]] (1-indexed).

# Example
```julia
indices = ct.arange((16,), Int32)  # Creates Tile with [1, 2, 3, ..., 16]
```
"""
@noinline function _arange(shape::NTuple{1, Int}, ::Type{T}) where {T}
    Tile{T, shape}()
end
@inline arange(shape::NTuple{1, Int}, ::Type{T}) where {T} = _arange(shape, T) .+ one(T)

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
- `axis`: Axis to reduce along. Must be a compile-time constant.

# Example
```julia
# For a (128, 64) tile, reducing along axis 2:
sums = ct.reduce_sum(tile, 2)  # Returns (128,) tile
```
"""
@noinline function _reduce_sum(tile::Tile{T, S}, ::Val{axis}) where {T <: AbstractFloat, S, axis}
    reduced_shape = ntuple(i -> S[i < axis + 1 ? i : i + 1], length(S) - 1)
    Base.donotdelete(tile)
    Tile{T, reduced_shape}()
end
@inline function reduce_sum(tile::Tile{T, S}, axis::Integer) where {T <: AbstractFloat, S}
    _reduce_sum(tile, Val(axis - 1))
end
@inline function reduce_sum(tile::Tile{T, S}, ::Val{axis}) where {T <: AbstractFloat, S, axis}
    _reduce_sum(tile, Val(axis - 1))
end

"""
    reduce_max(tile::Tile{T, S}, axis::Integer) -> Tile{T, reduced_shape}

Maximum reduction along the specified axis.

# Arguments
- `tile`: Input tile to reduce
- `axis`: Axis to reduce along. Must be a compile-time constant.
"""
@noinline function _reduce_max(tile::Tile{T, S}, ::Val{axis}) where {T <: AbstractFloat, S, axis}
    reduced_shape = ntuple(i -> S[i < axis + 1 ? i : i + 1], length(S) - 1)
    Base.donotdelete(tile)
    Tile{T, reduced_shape}()
end
@inline function reduce_max(tile::Tile{T, S}, axis::Integer) where {T <: AbstractFloat, S}
    _reduce_max(tile, Val(axis - 1))
end
@inline function reduce_max(tile::Tile{T, S}, ::Val{axis}) where {T <: AbstractFloat, S, axis}
    _reduce_max(tile, Val(axis - 1))
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
    _atomic_cas(array, index - one(index), expected, desired, memory_order, memory_scope)::T
end

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
    _atomic_xchg(array, index - one(index), val, memory_order, memory_scope)::T
end

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
    _atomic_add(array, index - one(index), val, memory_order, memory_scope)::T
end

#=============================================================================
 Gather and Scatter Operations
=============================================================================#

public gather, scatter

"""
    gather(array::TileArray{T, 1}, indices::Tile{I, S}) -> Tile{T, S}

Gather elements from a 1D array using index tile.
Returns a tile with the same shape as indices and element type of array.

Out-of-bounds indices are handled with zero padding (elements outside bounds return zero).

# Example
```julia
base = (bid - 1) * TILE
indices = base .+ ct.arange((TILE,), Int32)
tile = ct.gather(arr, indices)
```
"""
@noinline function _gather(array::TileArray{T, 1}, indices::Tile{I, S}) where {T, I <: Integer, S}
    Base.donotdelete(array, indices)
    Tile{T, S}()
end
@inline function gather(array::TileArray{T, 1}, indices::Tile{I, S}) where {T, I <: Integer, S}
    _gather(array, indices .- one(I))
end

"""
    gather(array::TileArray{T, 2}, indices::Tuple{Tile, Tile}) -> Tile{T, S}

Gather elements from a 2D array using a tuple of index tiles.
The index tiles are broadcast to a common shape, which becomes the output shape.

# Example
```julia
x = (bid_x - 1) * TILE_X .+ ct.arange((TILE_X,), Int32)
y = (bid_y - 1) * TILE_Y .+ ct.arange((TILE_Y,), Int32)
x = ct.reshape(x, (TILE_X, 1))
y = ct.reshape(y, (1, TILE_Y))
tile = ct.gather(arr, (x, y))
```
"""
@noinline function _gather(array::TileArray{T, 2}, idx0::Tile{I0, S0}, idx1::Tile{I1, S1}) where {T, I0 <: Integer, I1 <: Integer, S0, S1}
    S = broadcast_shape(S0, S1)
    Base.donotdelete(array, idx0, idx1)
    Tile{T, S}()
end
@inline function gather(array::TileArray{T, 2}, indices::Tuple{Tile{I0, S0}, Tile{I1, S1}}) where {T, I0 <: Integer, I1 <: Integer, S0, S1}
    _gather(array, indices[1] .- one(I0), indices[2] .- one(I1))
end

"""
    scatter(array::TileArray{T, 1}, indices::Tile{I, S}, tile::Tile{T, S}) -> Nothing

Scatter elements to a 1D array at indices specified by index tile.
Out-of-bounds indices are ignored (no write for elements outside bounds).

# Example
```julia
base = (bid - 1) * TILE
indices = base .+ ct.arange((TILE,), Int32)
ct.scatter(arr, indices, result_tile)
```
"""
@noinline function _scatter(array::TileArray{T, 1}, indices::Tile{I, S}, tile::Tile{T, S}) where {T, I <: Integer, S}
    Base.donotdelete(array, indices, tile)
    nothing
end
@inline function scatter(array::TileArray{T, 1}, indices::Tile{I, S}, tile::Tile{T, S}) where {T, I <: Integer, S}
    _scatter(array, indices .- one(I), tile)
end

"""
    scatter(array::TileArray{T, 2}, indices::Tuple{Tile, Tile}, tile::Tile) -> Nothing

Scatter elements to a 2D array at indices specified by tuple of index tiles.
The index tiles and value tile must broadcast to the same shape.

# Example
```julia
x = ct.reshape((bid_x - 1) * TILE_X .+ ct.arange((TILE_X,), Int32), (TILE_X, 1))
y = ct.reshape((bid_y - 1) * TILE_Y .+ ct.arange((TILE_Y,), Int32), (1, TILE_Y))
ct.scatter(arr, (x, y), result_tile)
```
"""
@noinline function _scatter(array::TileArray{T, 2}, idx0::Tile{I0, S0}, idx1::Tile{I1, S1}, tile::Tile{T, Stile}) where {T, I0 <: Integer, I1 <: Integer, S0, S1, Stile}
    S = broadcast_shape(S0, S1)
    S == Stile || error("Tile shape $Stile doesn't match broadcast shape $S of indices")
    Base.donotdelete(array, idx0, idx1, tile)
    nothing
end
@inline function scatter(array::TileArray{T, 2}, indices::Tuple{Tile{I0, S0}, Tile{I1, S1}}, tile::Tile{T, Stile}) where {T, I0 <: Integer, I1 <: Integer, S0, S1, Stile}
    _scatter(array, indices[1] .- one(I0), indices[2] .- one(I1), tile)
end
