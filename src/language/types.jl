public TileArray, Tile, Constant, TFloat32, similar_type,
       ScalarInt, ScalarFloat, TileInt, TileFloat, ScalarOrTileInt, ScalarOrTileFloat

"""
    ArraySpec{N}

Specialization hints for N-dimensional array arguments. Encoded as a type
parameter to enable kernel specialization based on array properties.

# Fields
- `alignment::Int`: Base pointer alignment in bytes (0 = unknown)
- `contiguous::Bool`: Whether stride[1] == 1 (contiguous in first dimension)
- `stride_div_by::NTuple{N,Int}`: Per-dimension stride divisibility (0 = unknown)
- `shape_div_by::NTuple{N,Int}`: Per-dimension shape divisibility (0 = unknown)

Common alignment values:
- 0: Unknown/unaligned
- 16: 16-byte aligned (enables basic vectorization)
- 128: 128-byte aligned (optimal for TMA on Blackwell)

Divisibility values enable optimizations:
- stride_div_by[i] = 4 means stride[i] is divisible by 4 (enables vectorized access)
- shape_div_by[i] = 16 means shape[i] is divisible by 16 (no tile boundary handling needed)
"""
struct ArraySpec{N}
    alignment::Int
    contiguous::Bool
    stride_div_by::NTuple{N, Int}
    shape_div_by::NTuple{N, Int}
end

# Convenience constructors
function ArraySpec(alignment::Int, contiguous::Bool)
    # 0-dimensional fallback (scalar pointers)
    ArraySpec{0}(alignment, contiguous, (), ())
end

function ArraySpec{N}(alignment::Int, contiguous::Bool) where N
    # N-dimensional with no divisibility info
    ArraySpec{N}(alignment, contiguous, ntuple(_ -> 0, N), ntuple(_ -> 0, N))
end

"""
    compute_alignment(ptr_int)

Compute largest power-of-2 alignment of a pointer address (up to 128 bytes).
Returns 0 for null pointers.
"""
function compute_alignment(ptr_int::Int)
    ptr_int == 0 && return 0
    for align in (128, 64, 32, 16, 8, 4, 2, 1)
        if ptr_int % align == 0
            return align
        end
    end
    return 0
end

"""
    compute_divisibility(value, max_divisor=16)

Compute largest power-of-2 that divides `value` (up to max_divisor).
Returns 0 if value is 0 or not divisible by any power of 2.
"""
function compute_divisibility(value::Integer, max_divisor::Int=16)
    value == 0 && return 0
    divisor = 1
    while divisor <= max_divisor && value % (divisor * 2) == 0
        divisor *= 2
    end
    return divisor >= 2 ? divisor : 0  # Only return if at least divisible by 2
end

"""
    compute_array_spec(ptr, sizes, strides, elem_size)

Compute ArraySpec from array properties.

# Arguments
- `ptr`: Base pointer
- `sizes`: Array dimensions
- `strides`: Stride in each dimension (in elements)
- `elem_size`: Size of element type in bytes

# Returns
ArraySpec{N} with:
- `alignment`: Pointer alignment in bytes
- `contiguous`: Whether stride[1] == 1
- `stride_div_by`: Per-dimension stride divisibility (enables vectorized access)
- `shape_div_by`: Per-dimension shape divisibility (eliminates boundary checks)
"""
function compute_array_spec(ptr::Ptr{T}, sizes::NTuple{N, Int32}, strides::NTuple{N, Int32}) where {T, N}
    elem_size = sizeof(T)

    # Pointer alignment
    alignment = compute_alignment(Int(ptr))

    # Contiguity (first dimension)
    contiguous = N > 0 && strides[1] == 1

    # Per-dimension stride divisibility
    # For stride to enable 16-byte vectorization, stride * elem_size must be divisible by 16
    # E.g., for Float32 (4 bytes): stride must be divisible by 4 to get 16-byte alignment
    stride_div_by = ntuple(N) do i
        stride_bytes = strides[i] * elem_size
        # Check if stride in bytes is 16-byte divisible
        if stride_bytes % 16 == 0
            # Return divisibility in elements (not bytes)
            return 16 ÷ elem_size
        end
        return 0
    end

    # Per-dimension shape divisibility (for tile boundary optimization)
    shape_div_by = ntuple(N) do i
        compute_divisibility(sizes[i], 16)
    end

    ArraySpec{N}(alignment, contiguous, stride_div_by, shape_div_by)
end


"""
    TileArray{T, N, S}

Represents an N-dimensional array argument to a kernel with element type `T`
and specialization `S::ArraySpec`.

Unlike raw pointers, TileArray carries size and stride information that is
passed to the kernel as runtime parameters, enabling dynamic array sizes.

The specialization parameter `S` drives kernel compilation - different
specializations (e.g., aligned vs unaligned) produce different cubins.

# Fields
- `ptr::Ptr{T}`: Base pointer to array data
- `sizes::NTuple{N, Int32}`: Size in each dimension
- `strides::NTuple{N, Int32}`: Stride in each dimension (in elements)
"""
struct TileArray{T, N, S}
    ptr::Ptr{T}
    sizes::NTuple{N, Int32}
    strides::NTuple{N, Int32}
end

# Type accessors for TileArray
Base.eltype(::Type{<:TileArray{T}}) where {T} = T
Base.eltype(::TileArray{T}) where {T} = T
Base.ndims(::Type{<:TileArray{<:Any, N}}) where {N} = N
Base.ndims(::TileArray{<:Any, N}) where {N} = N
# Return the ArraySpec value (third type parameter) if present
function array_spec(@nospecialize(T::Type{<:TileArray}))
    T isa DataType || return nothing  # UnionAll types don't have full parameters
    length(T.parameters) >= 3 || return nothing
    S = T.parameters[3]
    S isa ArraySpec && return S
    nothing
end
array_spec(arr::TileArray) = array_spec(typeof(arr))

"""
    TileArray(ptr, sizes, strides)

Create a TileArray from a pointer, sizes, and strides.
Computes the ArraySpec automatically based on alignment, contiguity, and divisibility.
"""
function TileArray(ptr::Ptr{T}, sizes::NTuple{N, Int32}, strides::NTuple{N, Int32}) where {T, N}
    spec = compute_array_spec(ptr, sizes, strides)
    TileArray{T, N, spec}(ptr, sizes, strides)
end

"""
    TileArray(arr)

Create a TileArray from a CUDA array (CuArray or similar).
Automatically extracts pointer, sizes, strides, and computes ArraySpec.

This method works with any array type that supports:
- `pointer(arr)` - returns device pointer
- `size(arr)` - returns array dimensions
- `strides(arr)` - returns array strides
"""
function TileArray(arr::AbstractArray{T, N}) where {T, N}
    # Use reinterpret to handle both Ptr and CuPtr (device pointers)
    ptr = reinterpret(Ptr{T}, pointer(arr))
    sizes = NTuple{N, Int32}(Int32.(size(arr)))
    strides_val = NTuple{N, Int32}(Int32.(strides(arr)))
    TileArray(ptr, sizes, strides_val)
end

function TileArray(arr::PermutedDimsArray{T, N}) where {T, N}
    ptr = reinterpret(Ptr{T}, pointer(parent(arr)))
    sizes = NTuple{N, Int32}(Int32.(size(arr)))
    strides_val = NTuple{N, Int32}(Int32.(strides(arr)))
    TileArray(ptr, sizes, strides_val)
end


"""
    Tile{T, Shape}

Represents a tile of data with element type `T` and static shape `Shape`.
Shape is a tuple of integers representing the tile dimensions.

This is a compile-time abstraction - at runtime in kernel code, tiles are
represented as Tile IR values. The struct exists to enable proper type
inference and operator dispatch.

Note: This is a mutable struct (despite having no fields) to prevent Julia's
optimizer from treating it as a singleton. Each Tile instance represents a
distinct Tile IR value, and we need SSA references to be preserved rather
than being replaced with constant QuoteNodes.
"""
mutable struct Tile{T, Shape}
    # Inner constructor that's never actually called at runtime
    function Tile{T, Shape}() where {T, Shape}
        new{T, Shape}()
    end
end

"""
    Tile(val::T) -> Tile{T, Tuple{}}

Create a 0-dimensional (scalar) tile from a scalar value.
This is used internally to convert scalars to tiles for broadcasting.

In kernel code, this is compiled to a ConstantOp.
"""
@noinline function Tile(val::T) where {T <: Number}
    Base.donotdelete(val)
    Tile{T, Tuple{}}()
end

#=============================================================================
 View Types
=============================================================================#

"""
    TensorView{T, N}

Opaque type representing a TensorView in Tile IR.
Created by `make_tensor_view` from a TileArray.

Uses mutable struct to prevent constant folding by Julia.
"""
mutable struct TensorView{T, N} end

Base.eltype(::Type{<:TensorView{T}}) where {T} = T
Base.eltype(::TensorView{T}) where {T} = T
Base.ndims(::Type{<:TensorView{<:Any, N}}) where {N} = N
Base.ndims(::TensorView{<:Any, N}) where {N} = N

"""
    PartitionView{T, N, Shape}

Opaque type representing a PartitionView in Tile IR.
Created by `make_partition_view` from a TensorView with a tile shape.

Uses mutable struct to prevent constant folding by Julia.
"""
mutable struct PartitionView{T, N, Shape} end

Base.eltype(::Type{<:PartitionView{T}}) where {T} = T
Base.eltype(::PartitionView{T}) where {T} = T
Base.ndims(::Type{<:PartitionView{<:Any, N}}) where {N} = N
Base.ndims(::PartitionView{<:Any, N}) where {N} = N
# Fix: return VALUES not type (Shape is Tuple{64, 32}, return (64, 32))
Base.size(::Type{<:PartitionView{<:Any, <:Any, Shape}}) where {Shape} = Tuple(Shape.parameters)
Base.size(::Type{<:PartitionView{<:Any, <:Any, Shape}}, d::Integer) where {Shape} = Shape.parameters[d]
Base.size(pv::PartitionView) = size(typeof(pv))
Base.size(pv::PartitionView, d::Integer) = size(typeof(pv), d)

"""
    Constant{T, V}

Compile-time constant with element type `T` and value `V`.
This is a ghost type (zero-size) - the value is encoded in the type parameter
and extracted at compile time.

Use `c[]` to access the constant value in kernel code.

# Example
```julia
function kernel(a::Ptr{T}, tile::Constant{Int}) where {T}
    data = ct.load(a, ct.bid(0), (tile[],))  # tile[] extracts the value
end

# Compile with specific constant value
argtypes = Tuple{Ptr{Float32}, Constant{Int, 16}}
```
"""
struct Constant{T, V} end

# Convenience constructor that infers type from value
Constant(val::T) where {T} = Constant{T, val}()

# Extract constant value - @inline ensures this folds to a constant in IR
@inline Base.getindex(::Constant{T, V}) where {T, V} = V

# Type accessors
Base.eltype(::Type{Tile{T, Shape}}) where {T, Shape} = T
Base.eltype(::Tile{T, Shape}) where {T, Shape} = T
# Shape is always a tuple TYPE (e.g., Tuple{16, 32}). Convert to value for user convenience.
Base.size(::Type{Tile{T, Shape}}) where {T, Shape} = Tuple(Shape.parameters)
Base.size(::Type{Tile{T, Shape}}, d::Integer) where {T, Shape} = Shape.parameters[d]
Base.ndims(::Type{Tile{T, Shape}}) where {T, Shape} = length(Shape.parameters)
Base.size(::Tile{T, Shape}) where {T, Shape} = size(Tile{T, Shape})
Base.size(t::Tile, d::Integer) = size(typeof(t), d)
Base.ndims(::Tile{T, Shape}) where {T, Shape} = ndims(Tile{T, Shape})
Base.length(::Type{Tile{T, Shape}}) where {T, Shape} = prod(Tuple(Shape.parameters))
Base.length(t::Tile) = length(typeof(t))

# Reconstruct Tile type with different element type and/or shape
similar_type(::Type{Tile{T, Shape}}, ::Type{U}) where {T, Shape, U} = Tile{U, Shape}
similar_type(::Type{Tile{T, Shape}}, ::Type{U}, new_shape::Tuple) where {T, Shape, U} =
    Tile{U, Tuple{new_shape...}}
similar_type(::Type, ::Type{T}) where {T} = T  # fallback for non-Tile types


"""
    TFloat32 <: AbstractFloat

Tensor Float 32 - a 32-bit floating-point type optimized for tensor core operations.
Has the same range as Float32 (8 exponent bits) but reduced precision (10 mantissa bits).

Convert Float32 tiles to TFloat32 for tensor core acceleration:
```julia
a = ct.load(A, (bid_m, k), (tm, tk))
a_tf32 = convert(ct.Tile{ct.TFloat32}, a)
```

Note: This is a compile-time only type for Tile IR code generation.
"""
primitive type TFloat32 <: AbstractFloat 32 end


#=============================================================================
 Type Unions

 These unions define the supported element types for Tile IR operations,
 matching the types in the Tile IR spec (sections 8.7 and 8.8).
=============================================================================#

"""Scalar integer types supported by Tile IR (i8, i16, i32, i64)."""
const ScalarInt = Union{Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64}

"""Scalar floating-point types supported by Tile IR (f16, bf16, tf32, f32, f64)."""
const ScalarFloat = Union{Float16, BFloat16, Float32, Float64, TFloat32}

"""Integer tile types."""
const TileInt{S} = Tile{T, S} where {T <: ScalarInt}

"""Floating-point tile types."""
const TileFloat{S} = Tile{T, S} where {T <: ScalarFloat}

"""Integer values (scalar or tile)."""
const ScalarOrTileInt = Union{ScalarInt, TileInt}

"""Floating-point values (scalar or tile)."""
const ScalarOrTileFloat = Union{ScalarFloat, TileFloat}


#=============================================================================
 One Singleton

 Singleton that adopts the type of the other operand in arithmetic.
 `x - One()` returns the same type as `x`, enabling clean index normalization
 with `promote(index...) .- One()`.
=============================================================================#

"""
    One()

Singleton that adopts the type of the other operand in arithmetic.
`x - One()` returns the same type as `x`.

Used internally for 1-to-0 index conversion that preserves types after promotion.
"""
struct One end
@inline Base.:-(x::T, ::One) where {T<:Integer} = x - one(T)
@inline Base.:+(x::T, ::One) where {T<:Integer} = x + one(T)
@inline Base.:-(::One, x::T) where {T<:Integer} = one(T) - x
@inline Base.:+(::One, x::T) where {T<:Integer} = one(T) + x
@inline Base.Broadcast.broadcastable(o::One) = Ref(o)
