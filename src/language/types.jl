public AbstractTileArray, TileArray, Tile, Constant, TFloat32, similar_type,
       ScalarInt, ScalarFloat, IntTile, FloatTile, TileOrInt, TileOrFloat,
       TileOrScalar

"""
    AbstractTileArray{T, N}

Supertype for N-dimensional kernel array arguments with element type `T`.
"""
abstract type AbstractTileArray{T, N} end

Base.eltype(::Type{<:AbstractTileArray{T}}) where T = T
Base.ndims(::Type{<:AbstractTileArray{<:Any,N}}) where N = N
Base.eltype(arr::AbstractTileArray) = eltype(typeof(arr))
Base.ndims(arr::AbstractTileArray) = ndims(typeof(arr))


"""
    ArraySpec{N}

Specialization hints for N-dimensional array arguments. Encoded as a type
parameter to enable kernel specialization based on array properties.

# Fields
- `alignment::Int`: Base pointer alignment in bytes (0 = unknown)
- `contiguous::Bool`: Whether stride[1] == 1 (contiguous in first dimension)
- `stride_div_by::NTuple{N,Int}`: Per-dimension stride divisibility (0 = unknown)
- `shape_div_by::NTuple{N,Int}`: Per-dimension shape divisibility (0 = unknown)
- `may_alias_internally::Bool`: Whether two distinct in-bounds indices may
  refer to the same memory location (e.g. a zero or repeated stride).
  `false` asserts the layout is internally non-overlapping, which enables
  optimizations such as loop-parallel stores; `compute_array_spec` derives
  it exactly from the runtime sizes/strides.

Common alignment values:
- 0: Unknown/unaligned
- 16: 16-byte aligned (enables basic vectorization)
- 128: 128-byte aligned (optimal for TMA on Blackwell)

Divisibility values enable optimizations:
- stride_div_by[i] = 4 means stride[i] is divisible by 4 (enables vectorized access)
- shape_div_by[i] = 16 means shape[i] is divisible by 16 (no tile boundary handling needed)
"""
struct ArraySpec{N, Alignment, Contiguous, StrideDivBy, ShapeDivBy, MayAliasInternally}
    # Validate invariants once per concrete spec type (this struct is a
    # singleton, so the inner constructor runs on every instantiation but
    # the result is then cached as a type parameter). Catches synthetic
    # specs that combine `contiguous=true` with a `stride_div_by[1]` that
    # contradicts `stride[1] == 1` — `1 % d == 0` only for `d ∈ {0, 1}`.
    function ArraySpec{N, Alignment, Contiguous, StrideDivBy, ShapeDivBy, MayAliasInternally}() where
            {N, Alignment, Contiguous, StrideDivBy, ShapeDivBy, MayAliasInternally}
        if Contiguous && N >= 1
            sdb1 = StrideDivBy[1]
            (sdb1 == 0 || sdb1 == 1) || throw(ArgumentError(
                "ArraySpec: contiguous=true requires stride_div_by[1] ∈ {0, 1} " *
                "(stride[1]=1, and 1 % d == 0 only for d ∈ {0, 1}); got $sdb1"))
        end
        MayAliasInternally isa Bool || throw(ArgumentError(
            "ArraySpec: MayAliasInternally must be a Bool; got $MayAliasInternally"))
        new{N, Alignment, Contiguous, StrideDivBy, ShapeDivBy, MayAliasInternally}()
    end
end

# Constructors
function ArraySpec{N}(alignment::Int, contiguous::Bool,
                      stride_div_by::NTuple{N,Int}, shape_div_by::NTuple{N,Int},
                      may_alias_internally::Bool=false) where N
    ArraySpec{N, alignment, contiguous, stride_div_by, shape_div_by, may_alias_internally}()
end

function ArraySpec(alignment::Int, contiguous::Bool)
    # 0-dimensional fallback (scalar pointers)
    ArraySpec{0, alignment, contiguous, (), (), false}()
end

function ArraySpec{N}(alignment::Int, contiguous::Bool) where N
    # N-dimensional with no divisibility info
    ArraySpec{N}(alignment, contiguous, ntuple(_ -> 0, N), ntuple(_ -> 0, N))
end

# Property access — preserves existing dot-syntax (spec.alignment, etc.)
function Base.getproperty(spec::ArraySpec{N, Alignment, Contiguous, StrideDivBy, ShapeDivBy, MayAliasInternally},
                          s::Symbol) where {N, Alignment, Contiguous, StrideDivBy, ShapeDivBy, MayAliasInternally}
    s === :alignment            && return Alignment
    s === :contiguous           && return Contiguous
    s === :stride_div_by        && return StrideDivBy
    s === :shape_div_by         && return ShapeDivBy
    s === :may_alias_internally && return MayAliasInternally
    getfield(spec, s)
end

Base.propertynames(::ArraySpec) = (:alignment, :contiguous, :stride_div_by, :shape_div_by,
                                   :may_alias_internally)
Base.ndims(::ArraySpec{N}) where N = N

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
    while divisor < max_divisor && value % (divisor * 2) == 0
        divisor *= 2
    end
    return divisor >= 2 ? divisor : 0  # Only return if at least divisible by 2
end

"""
    layout_may_alias_internally(sizes, strides) -> Bool

Whether two distinct in-bounds index tuples may map to the same linear
offset. Returns `false` only when the strided layout is provably injective:
each dimension's absolute stride must exceed the total span of all
smaller-strided dimensions (the standard non-overlapping strided layout
criterion, which C-/Fortran-contiguous and sliced layouts all satisfy). Zero
or repeated strides on dimensions with extent > 1, and layouts that fail the
(sufficient, not necessary) criterion, report `true` — conservative for
consumers that require non-overlap.
"""
function layout_may_alias_internally(sizes::NTuple{N, <:Integer},
                                     strides::NTuple{N, <:Integer}) where N
    # Express the sorted-stride criterion pairwise to avoid allocating and
    # sorting a temporary vector on every kernel launch. The division form
    # also avoids overflowing while computing a span that already reaches
    # the current stride.
    for i in 1:N
        size_i = Int(sizes[i])
        size_i > 1 || continue
        stride_i = abs(Int(strides[i]))
        iszero(stride_i) && return true

        span = 0
        for j in 1:N
            j == i && continue
            size_j = Int(sizes[j])
            size_j > 1 || continue
            stride_j = abs(Int(strides[j]))
            iszero(stride_j) && return true
            stride_j == stride_i && return true
            stride_j < stride_i || continue

            remaining = stride_i - 1 - span
            size_j - 1 > remaining ÷ stride_j && return true
            span += (size_j - 1) * stride_j
        end
    end
    return false
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

    ArraySpec{N}(alignment, contiguous, stride_div_by, shape_div_by,
                 layout_may_alias_internally(sizes, strides))
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
struct TileArray{T, N, S} <: AbstractTileArray{T, N}
    ptr::Ptr{T}
    sizes::NTuple{N, Int32}
    strides::NTuple{N, Int32}
end
Base.size(arr::TileArray) = arr.sizes
function Base.size(arr::TileArray{<:Any, N}, d::Integer) where N
    d < 1 && error("arraysize: dimension out of range") # from Array method
    return d > N ? Int32(1) : arr.sizes[d]
end
Base.length(arr::TileArray) = prod(size(arr))
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

Create a TileArray from a device array (CuArray or similar).
Automatically extracts pointer, sizes, strides, and computes ArraySpec.

This method works with any array type that supports:
- `pointer(arr)` - returns device pointer
- `size(arr)` - returns array dimensions
- `strides(arr)` - returns array strides
"""
function TileArray(arr::AbstractArray{T, N}) where {T, N}
    sizes = NTuple{N, Int32}(Int32.(size(arr)))
    strides_val = NTuple{N, Int32}(Int32.(strides(arr)))
    TileArray(device_pointer(arr), sizes, strides_val)
end

function TileArray(arr::PermutedDimsArray{T, N}) where {T, N}
    sizes = NTuple{N, Int32}(Int32.(size(arr)))
    strides_val = NTuple{N, Int32}(Int32.(strides(arr)))
    TileArray(device_pointer(parent(arr)), sizes, strides_val)
end

# Device arrays hand out device pointer types (e.g. `CuPtr`), which TileArray
# stores reinterpreted as `Ptr`. A `pointer` that already is a `Ptr` means host
# memory, which would compile fine but fault when dereferenced on device.
function device_pointer(arr::AbstractArray{T}) where T
    ptr = pointer(arr)
    ptr isa Ptr &&
        throw(ArgumentError("cannot create a TileArray from host memory ($(typeof(arr)))"))
    reinterpret(Ptr{T}, ptr)
end


"""
    Tile{T, Shape}

Represents a tile of data with element type `T` and static shape `Shape`.
Shape is a tuple type encoding the tile dimensions (e.g. `Tuple{16, 32}`).

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

Create a 0-dimensional (scalar) tile from a scalar value (`Number` or
`Ptr`). This is used internally to convert scalars to tiles for
broadcasting and to wrap pointer fields (`arr.ptr`) before passing
them to ptr-consuming intrinsics that expect `Tile{Ptr{T}, S}`.

In kernel code, this is compiled to a ConstantOp.
"""
@inline function Tile(val::T) where {T <: Union{Number, Ptr}}
    # Wrap scalar as 0D tile via from_scalar — this is eliminated by
    # scalar_elim_pass! along with all other from_scalar calls, so no
    # special-casing of Tile constructors is needed in the pass.
    cuTile.Intrinsics.from_scalar(val, Tuple{})
end

# No-op: pass-through for values already wrapped as Tile
@inline Tile(tile::Tile) = tile

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
struct Constant{T, V}
    function Constant{T, V}() where {T, V}
        # Ghost types have no runtime check on `V`, so an out-of-range integer
        # literal would silently truncate during codegen (e.g.
        # `Constant{Int8, 1024}` becoming `Int8(0)`). Reject mismatches up
        # front with the same `InexactError` Julia raises for `Int8(1024)`.
        # Floats and types-as-values are unaffected.
        if T <: Integer && V isa Integer && !(typemin(T) <= V <= typemax(T))
            throw(Base.InexactError(:Constant, T, V))
        end
        new{T, V}()
    end
end

# Convenience constructors that infer type from value
Constant(val::T) where {T} = Constant{T, val}()
Constant(val::Type{T}) where {T} = Constant{Type{T}, T}()

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

# Reconstruct Tile type with different element type and/or shape. The
# `<:Tile{T}` overload preserves the unionall: `similar_type(Tile{UInt32, S}
# where S, Int32) = Tile{Int32}` rather than falling through to the scalar
# fallback (which would otherwise lose the Tile-ness during inference of
# `bitcast`/`trunci` calls whose source has unbound Shape).
similar_type(::Type{Tile{T, Shape}}, ::Type{U}) where {T, Shape, U} = Tile{U, Shape}
similar_type(::Type{Tile{T, Shape}}, ::Type{U}, new_shape::Tuple) where {T, Shape, U} =
    Tile{U, Tuple{new_shape...}}
similar_type(::Type{<:Tile{T}}, ::Type{U}) where {T, U} = Tile{U}
similar_type(::Type, ::Type{T}) where {T} = T  # fallback for non-Tile types

"""
    bitwidth(::Type{T}) -> Int

Number of bits a single element of `T` occupies in a Tile IR tile. Used by the
whole-tile [`reinterpret`](@ref Base.reinterpret(::Type, ::Tile)) to scale the
tile shape across a change of element width (e.g. `UInt8` ↔ `Float4_E2M1FN`,
8 bits ↔ 4 bits).

The default is `8 * sizeof(T)`, which is correct for the standard integer and
floating-point types and for the byte-wide `Float8_*` formats. Sub-byte formats
whose `sizeof` rounds up to a whole byte (e.g. `Float4_E2M1FN`, 4 bits but
`sizeof == 1`) override this; the `Microfloats` extension forwards to
`Microfloats.bitwidth`, which derives the true width from the format's bit
fields. Matches the `bitwidth` convention used by `Microfloats`/`Narrow`.
"""
bitwidth(::Type{T}) where {T} = 8 * sizeof(T)


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

# Pack into TF32's 19 bits: Float32 layout, low 13 mantissa bits rounded
# (RNE) and dropped. Result occupies bits 0–18 of the returned UInt32.
function float_to_bits(value::Float64, ::Type{TFloat32})
    f32_bits = reinterpret(UInt32, Float32(value))
    sign = (f32_bits >> 31) & 0x1
    exp = (f32_bits >> 23) & 0xff
    mantissa23 = f32_bits & 0x007fffff

    if exp == 0xff
        mantissa10 = mantissa23 >> 13
        if mantissa23 != 0 && mantissa10 == 0
            # Truncated payload would turn NaN into Inf; restore a NaN bit.
            mantissa10 = UInt32(1)
        end
        return (sign << 18) | (exp << 10) | mantissa10
    end

    truncated = mantissa23 >> 13
    dropped = mantissa23 & 0x1fff
    half = UInt32(1) << 12
    if dropped > half || (dropped == half && (truncated & 0x1) != 0)
        truncated += 0x1
    end

    if truncated >= 0x400
        truncated = 0x0
        exp += 0x1
        if exp >= 0xff
            return (sign << 18) | (UInt32(0xff) << 10)
        end
    end

    return (sign << 18) | (exp << 10) | truncated
end


#=============================================================================
 Type Unions

 These unions define the supported element types for Tile IR operations,
 matching the types in the Tile IR spec (sections 8.7 and 8.8).
=============================================================================#

"""Scalar integer types supported by Tile IR (i8, i16, i32, i64)."""
const ScalarInt = Union{Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64}

"""Scalar floating-point types supported by Tile IR (f16, bf16, tf32, f32, f64)."""
const ScalarFloat = Union{Float16, BFloat16, Float32, Float64, TFloat32}

"""
Restricted floats — types whose op coverage is intentionally limited
(no general arithmetic, reductions, scans, …). Currently `TFloat32`;
future FP8/FP4 dtypes will join this union. Mirrors cuTile Python's
`NumericDTypeCategories.RestrictedFloat`.
"""
const RestrictedFloat = Union{TFloat32}

"""
    is_restricted_float(::Type) -> Bool

True if `T` is a restricted float. Used by `reduce` / `scan` (and other
arithmetic-requiring ops) to reject unsupported element types early
with a clear error rather than letting tileiras fail downstream.
"""
@inline is_restricted_float(::Type{T}) where {T} = T <: RestrictedFloat

"""Integer tile types."""
const IntTile{S} = Tile{T, S} where {T <: ScalarInt}

"""Floating-point tile types."""
const FloatTile{S} = Tile{T, S} where {T <: ScalarFloat}

"""Scalar or tile of element type T."""
const TileOrScalar{T} = Union{T, Tile{T}}

"""Integer values (scalar or tile)."""
const TileOrInt = TileOrScalar{<:ScalarInt}

"""Floating-point values (scalar or tile)."""
const TileOrFloat = TileOrScalar{<:ScalarFloat}


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


#=============================================================================
 ByTarget: Per-architecture optimization hints
=============================================================================#

"""
    ByTarget{T}

Per-architecture value that resolves to a concrete `T` based on the target
GPU's compute capability. Use with `@compiler_options` to specify
architecture-specific optimization hints.

# Example
```julia
function my_kernel(...)
    ct.@compiler_options num_ctas=ByTarget(v"10.0" => 2, v"12.0" => 4)
    ...
end
```
"""
struct ByTarget{T}
    default::Union{Some{T}, Nothing}
    targets::Dict{VersionNumber, T}
end

function ByTarget(pairs::Pair{VersionNumber, T}...; default=nothing) where T
    d = isnothing(default) ? nothing : Some{T}(default)
    ByTarget{T}(d, Dict{VersionNumber, T}(pairs...))
end

function ByTarget(pairs::Pair{VersionNumber}...; default=nothing)
    T = promote_type(typeof(default === nothing ? first(pairs).second : default),
                     map(p -> typeof(p.second), pairs)...)
    d = isnothing(default) ? nothing : Some{T}(T(default))
    ByTarget{T}(d, Dict{VersionNumber, T}(pairs...))
end

"""
    resolve(bt::ByTarget{T}, cap::VersionNumber) -> Union{T, Nothing}

Resolve a `ByTarget` value for a specific compute capability.
Returns the architecture-specific value, the default, or `nothing`.
"""
function resolve(bt::ByTarget{T}, cap::VersionNumber) where T
    val = get(bt.targets, cap, nothing)
    val !== nothing && return val
    bt.default !== nothing && return something(bt.default)
    return nothing
end

# Pass-through: plain values resolve to themselves
resolve(val, ::VersionNumber) = val
resolve(::Nothing, ::VersionNumber) = nothing

"""
    validate_hint(key::Symbol, val)

Validate a kernel optimization hint value. Throws `ArgumentError` for invalid values.

- `num_ctas`: power of 2 in [1, 16]
- `occupancy`: integer in [1, 32]
- `opt_level`: integer in [0, 3]
- `num_worker_warps`: either 4 or 8
"""
function validate_hint(key::Symbol, val)
    val === nothing && return
    if key === :num_ctas
        val isa Integer || throw(ArgumentError("num_ctas must be an integer, got $(typeof(val))"))
        1 <= val <= 16 || throw(ArgumentError("num_ctas must be between 1 and 16, got $val"))
        ispow2(val) || throw(ArgumentError("num_ctas must be a power of 2, got $val"))
    elseif key === :occupancy
        val isa Integer || throw(ArgumentError("occupancy must be an integer, got $(typeof(val))"))
        1 <= val <= 32 || throw(ArgumentError("occupancy must be between 1 and 32, got $val"))
    elseif key === :opt_level
        val isa Integer || throw(ArgumentError("opt_level must be an integer, got $(typeof(val))"))
        0 <= val <= 3 || throw(ArgumentError("opt_level must be between 0 and 3, got $val"))
    elseif key === :num_worker_warps
        val isa Integer || throw(ArgumentError("num_worker_warps must be an integer, got $(typeof(val))"))
        val in (4, 8) || throw(ArgumentError("num_worker_warps must be either 4 or 8, got $val"))
    else
        throw(ArgumentError("unknown hint key: $key"))
    end
end
