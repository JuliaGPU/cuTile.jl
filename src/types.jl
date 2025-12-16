public TileArray, Tile, Constant, TFloat32

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

# Backwards compatibility with boolean aligned
ArraySpec(aligned::Bool, contiguous::Bool) = ArraySpec(aligned ? 16 : 0, contiguous)

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

# Backwards compatible version without sizes (no shape_div_by info)
function compute_array_spec(ptr::Ptr{T}, strides::NTuple{N, Int32}) where {T, N}
    compute_array_spec(ptr, ntuple(_ -> Int32(0), N), strides)
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
Base.eltype(::Type{TileArray{T, N, S}}) where {T, N, S} = T
Base.eltype(::Type{TileArray{T, N}}) where {T, N} = T
Base.eltype(::Type{TileArray{T}}) where {T} = T
Base.eltype(::TileArray{T, N, S}) where {T, N, S} = T
Base.ndims(::Type{TileArray{T, N, S}}) where {T, N, S} = N
Base.ndims(::TileArray{T, N, S}) where {T, N, S} = N

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
    Tile(val::T) -> Tile{T, ()}

Create a 0-dimensional (scalar) tile from a scalar value.
This is used internally to convert scalars to tiles for broadcasting.

In kernel code, this is compiled to a ConstantOp.
"""
@noinline function Tile(val::T)::Tile{T, ()} where {T <: Number}
    Base.donotdelete(val)
    Tile{T, ()}()
end


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
tile_shape(::Type{Tile{T, Shape}}) where {T, Shape} = Shape
tile_shape(::Tile{T, Shape}) where {T, Shape} = Shape


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
