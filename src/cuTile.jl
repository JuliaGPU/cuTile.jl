module cuTile

using CUDA: CuModule, CuFunction, cudacall, device, capability
using Core: MethodInstance, CodeInfo, SSAValue, Argument, SlotNumber,
            GotoNode, GotoIfNot, ReturnNode, PhiNode, PiNode, QuoteNode, GlobalRef
using Core.Compiler
const CC = Core.Compiler

# Bytecode infrastructure
include("bytecode/basic.jl")
include("bytecode/types.jl")
include("bytecode/writer.jl")
include("bytecode/encodings.jl")

# Public API
export emit_tileir, compile, launch
export Tile, Constant, TileArray, ArraySpec, flatten
export mma, full, num_tiles, cdiv
export code_structured, validate_structured_control_flow, UnstructuredControlFlowError

# Compilation cache - stores CuFunction directly to avoid re-loading CuModule
const _compilation_cache = Dict{Any, Any}()  # (f, argtypes, sm_arch, opt_level) => CuFunction

#=============================================================================
 API Types
=============================================================================#

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
    flatten(x)

Flatten a value into a tuple of its leaf fields for kernel launch.
Scalars return themselves wrapped in a tuple. Structs like TileArray
return their fields in order.

This is used by the launch helper to splat arguments to cudacall.
"""
flatten(x) = (x,)
flatten(arr::TileArray{T, N}) where {T, N} = (arr.ptr, arr.sizes..., arr.strides...)
flatten(::Constant) = ()  # Ghost types are not passed to cudacall

"""
    to_tile_arg(x)

Convert launch arguments to their kernel argument form.
AbstractArrays (like CuArray) are converted to TileArray for metadata.
Other values pass through unchanged.
"""
to_tile_arg(x) = x
to_tile_arg(arr::AbstractArray) = TileArray(arr)

#=============================================================================
 Tile Arithmetic
=============================================================================#

# These are stub implementations that the compiler intercepts.
# They return a new Tile with the same shape, enabling proper type inference.

@noinline function tile_add(a::Tile{T, S}, b::Tile{T, S})::Tile{T, S} where {T, S}
    Base.donotdelete(a, b)
    Tile{T, S}()
end

@noinline function tile_sub(a::Tile{T, S}, b::Tile{T, S})::Tile{T, S} where {T, S}
    Base.donotdelete(a, b)
    Tile{T, S}()
end

@noinline function tile_mul(a::Tile{T, S}, b::Tile{T, S})::Tile{T, S} where {T, S}
    Base.donotdelete(a, b)
    Tile{T, S}()
end

# Operator overloads dispatch to the intrinsic functions
Base.:(+)(a::Tile{T, S}, b::Tile{T, S}) where {T, S} = tile_add(a, b)
Base.:(-)(a::Tile{T, S}, b::Tile{T, S}) where {T, S} = tile_sub(a, b)
Base.:(*)(a::Tile{T, S}, b::Tile{T, S}) where {T, S} = tile_mul(a, b)

#=============================================================================
 Tile Shape Operations
=============================================================================#

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

#=============================================================================
 Array Dimension Operations
=============================================================================#

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
    mod(a::Integer, b::Integer) -> Int32

Modulo operation: a % b (C-style, result has same sign as dividend)
"""
@noinline Base.mod(a::Int32, b::Int32)::Int32 = Base.inferencebarrier(zero(Int32))

#=============================================================================
 Compiler Infrastructure (must be after stub definitions for multiple dispatch)
=============================================================================#

include("compiler/ir.jl")
include("compiler/restructuring.jl")
include("compiler/interpreter.jl")
include("compiler/target.jl")
include("compiler/lowering.jl")
include("compiler/codegen.jl")

#=============================================================================
 Compilation and Launch Functions
=============================================================================#

"""
    default_sm_arch() -> String

Get the SM architecture string for the current CUDA device.
Returns e.g. "sm_120" for compute capability 12.0.
"""
function default_sm_arch()
    cap = capability(device())
    "sm_$(cap.major)$(cap.minor)"
end

"""
    compile(f, argtypes; name=nothing, sm_arch=default_sm_arch(), opt_level=3) -> Vector{UInt8}

Compile a Julia kernel function to CUBIN.
"""
function compile(@nospecialize(f), @nospecialize(argtypes);
                 name::Union{String, Nothing}=nothing,
                 sm_arch::String=default_sm_arch(),
                 opt_level::Int=3)
    tile_bytecode = emit_tileir(f, argtypes; name)

    input_path = tempname() * ".tile"
    output_path = tempname() * ".cubin"

    try
        write(input_path, tile_bytecode)
        run(`tileiras $input_path -o $output_path --gpu-name $sm_arch -O$opt_level`)
        return read(output_path)
    finally
        rm(input_path, force=true)
        rm(output_path, force=true)
    end
end

"""
    launch(f, grid, args...; name=nothing, sm_arch=default_sm_arch(), opt_level=3)

Compile and launch a kernel function with the given grid size and arguments.

Arguments are automatically flattened using `flatten()` - TileArray arguments
are expanded to their constituent ptr, sizes, and strides parameters.

# Arguments
- `f`: The kernel function to launch
- `grid`: Grid size as Int or NTuple for multi-dimensional grids
- `args...`: Kernel arguments (may include TileArray)
- `name`: Optional kernel name for debugging
- `sm_arch`: Target GPU architecture (default: current device's capability)
- `opt_level`: Optimization level 0-3 (default: 3)

# Example
```julia
using CUDA, cuTile

a = CUDA.zeros(Float32, 1024)
b = CUDA.ones(Float32, 1024)
c = CUDA.zeros(Float32, 1024)

function vadd_kernel(a::cuTile.TileArray{Float32,1}, b::cuTile.TileArray{Float32,1},
                     c::cuTile.TileArray{Float32,1})
    pid = cuTile.bid(0)
    ta = cuTile.load(a, (pid,), (16,))
    tb = cuTile.load(b, (pid,), (16,))
    cuTile.store(c, (pid,), ta + tb)
    return nothing
end

# CuArrays are automatically converted to TileArray
cuTile.launch(vadd_kernel, 64, a, b, c)
```
"""
function launch(@nospecialize(f), grid, args...;
                name::Union{String, Nothing}=nothing,
                sm_arch::String=default_sm_arch(),
                opt_level::Int=3)
    # Convert CuArray -> TileArray (and other conversions)
    tile_args = map(to_tile_arg, args)

    # Compute argument types from the converted arguments
    argtypes = Tuple{map(typeof, tile_args)...}

    # Determine kernel name
    kernel_name = name !== nothing ? name : string(nameof(f))

    # Check compilation cache - returns CuFunction directly
    cache_key = (f, argtypes, sm_arch, opt_level)
    cufunc = get!(_compilation_cache, cache_key) do
        cubin = compile(f, argtypes; name, sm_arch, opt_level)
        cumod = CuModule(cubin)
        CuFunction(cumod, kernel_name)
    end

    # Flatten arguments for cudacall - Constant returns () so ghost types disappear
    flat_args = Tuple(Iterators.flatten(map(flatten, tile_args)))
    flat_types = Tuple{map(typeof, flat_args)...}

    # Get grid dimensions
    grid_dims = grid isa Integer ? (grid,) : grid

    # Validate grid dimensions - Tile IR has a 24-bit limit
    max_grid_dim = (1 << 24) - 1  # 16,777,215
    for (i, dim) in enumerate(grid_dims)
        if dim > max_grid_dim
            error("Grid[$i] exceeds 24-bit limit: max=$max_grid_dim, got=$dim. " *
                  "Use multiple kernel launches for larger workloads.")
        end
    end

    # Launch with cached CuFunction
    # Note: threads=1 allows the driver to use the cubin's EIATTR_REQNTID metadata
    # which specifies the actual thread count (typically 128 for Tile kernels)
    cudacall(cufunc, flat_types, flat_args...; blocks=grid_dims, threads=1)

    return nothing
end

end # module cuTile
