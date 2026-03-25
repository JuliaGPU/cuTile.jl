# Codegen types and utilities
#
# Core types (CGVal, CGCtx) and helper functions for Tile IR code generation.

#=============================================================================
 Type-safe shape wrappers: Julia (column-major) ↔ Tile IR (row-major)
=============================================================================#

# Tile IR is natively row-major: shapes are stored with the slowest-varying dimension first.
# Julia is column-major: shapes are stored with the fastest-varying dimension first.
# Converting between them is a simple reversal. The Shape{O} wrapper ensures we don't
# accidentally mix up conventions — IR operations accept only RowMajorShape, while
# user-facing shapes from Julia are ColMajorShape.

abstract type StorageOrder end
struct RowMajor <: StorageOrder end
struct ColMajor <: StorageOrder end

struct Shape{O<:StorageOrder}
    dims::Vector{Int}
end

const RowMajorShape = Shape{RowMajor}
const ColMajorShape = Shape{ColMajor}

# Conversion constructors
RowMajorShape(s::RowMajorShape) = s
RowMajorShape(s::ColMajorShape) = RowMajorShape(reverse(s.dims))
RowMajorShape(t::Tuple) = RowMajorShape(ColMajorShape(collect(Int, t)))

ColMajorShape(s::ColMajorShape) = s
ColMajorShape(s::RowMajorShape) = ColMajorShape(reverse(s.dims))
ColMajorShape(t::Tuple) = ColMajorShape(collect(Int, t))

# Forward common operations to .dims
Base.length(s::Shape) = length(s.dims)
Base.isempty(s::Shape) = isempty(s.dims)
Base.getindex(s::Shape, i) = s.dims[i]
Base.setindex!(s::Shape, v, i) = (s.dims[i] = v; s)
Base.copy(s::Shape{O}) where O = Shape{O}(copy(s.dims))
Base.:(==)(a::Shape{O}, b::Shape{O}) where O = a.dims == b.dims
Base.iterate(s::Shape, state...) = iterate(s.dims, state...)
Base.eachindex(s::Shape) = eachindex(s.dims)
Base.collect(s::Shape) = s.dims
TupleType(s::Shape) = Tuple{s.dims...}

# Scalar (0-D) shape — storage order is irrelevant for zero dimensions
struct ScalarShape end
Base.length(::ScalarShape) = 0
Base.isempty(::ScalarShape) = true
Base.collect(::ScalarShape) = Int[]
TupleType(::ScalarShape) = Tuple{}

Base.:(==)(::ScalarShape, ::ScalarShape) = true
Base.:(==)(::ScalarShape, ::Shape) = false
Base.:(==)(::Shape, ::ScalarShape) = false

# Cross-type conversions (must be after ScalarShape definition)
ColMajorShape(::ScalarShape) = ColMajorShape(Int[])

const TileShape = Union{RowMajorShape, ScalarShape}

#=============================================================================
 IRError: Exception type for IR compilation errors
=============================================================================#

"""
    IRError <: Exception

Exception thrown during Tile IR compilation for invalid IR, type mismatches,
or unsupported operations.
"""
struct IRError <: Exception
    msg::String
end
Base.showerror(io::IO, e::IRError) = print(io, "IRError: ", e.msg)

#=============================================================================
 CGVal: Unified value representation (analogous to Julia's jl_cgval_t)
=============================================================================#

"""
    CGVal

Unified value representation during Tile IR codegen (analogous to `jl_cgval_t` in the
Julia compiler). Every SSA value in the IR being compiled maps to a CGVal.

## Variants

A CGVal takes one of four forms, distinguished by field states:

| Variant              | `v`              | `type_id`    | `arg_ref`      | Notes                                          |
|:---------------------|:-----------------|:-------------|:---------------|:-----------------------------------------------|
| Concrete SSA value   | `Value`          | `TypeId`     | `nothing`      | Normal runtime value                           |
| Multi-value result   | `Vector{Value}`  | `nothing`    | `nothing`      | From loop/if ops; extracted via `getfield`      |
| Lazy argument ref    | `nothing`        | `nothing`    | `(idx, chain)` | Deferred field access into destructured args   |
| Ghost value          | `nothing`        | `TypeId(-1)` | `nothing`      | Zero-size type, compile-time only              |

## Dual-level type representation

CGVal deliberately separates **Julia-level type information** (`jltype`) from **IR-level
representation** (`shape`, `type_id`, `v`). `jltype` drives dispatch during interpretation
(determining which overlay methods are selected), while `shape`/`type_id`/`v` describe the
actual Tile IR value being compiled. Normally these agree — `shape == extract_tile_shape(jltype)`
— but they can be independently controlled when the interpretation requires a different
Julia type than what the IR carries.

The `to_scalar`/`from_scalar` pair is the primary use of this flexibility today:

- **`to_scalar`**: changes `jltype` from `Tile{T,S}` to scalar `T` while keeping the
  IR-side `shape`, `type_id`, and `v` unchanged. This lets the value flow through
  Julia's scalar overlay dispatch (e.g., `abs(::Float32)`) while the IR still operates on
  the shaped tile.

- **`from_scalar`**: restores `jltype` back to `Tile{T,S}`, re-aligning the two levels.

This same mechanism could be used in other contexts where the Julia dispatch type needs to
diverge from the underlying IR type (e.g., broadcasting semantics, type promotion views).

## Auxiliary fields

- `constant`: `Some(x)` for compile-time constants (ghost `Constant{T,V}` types), `nothing`
  otherwise.
- `tuple`: component refs (`SSAValue`s etc.) for tuple values used by `cat()` and similar.
"""
struct CGVal
    v::Union{Value, Vector{Value}, Nothing}  # Single value, multi-value, or nothing
    type_id::Union{TypeId, Nothing}  # Tile IR type (nothing for lazy refs or multi-value)
    jltype::Any               # Original Julia type
    shape::TileShape        # Tile shape (ScalarShape for scalars)
    # Lazy argument reference: (arg_idx, [field_indices...])
    # e.g., (1, [2, 1]) means "argument 1, field 2, sub-field 1"
    arg_ref::Union{Tuple{Int, Vector{Int}}, Nothing}
    constant::Union{Some, Nothing}  # Nothing = no constant, Some(x) = constant value x
    tuple::Union{Vector{Any}, Nothing}  # For tuples: component refs (SSAValue, etc.)
end

# Convenience constructors for concrete values
CGVal(v::Value, type_id::TypeId, @nospecialize(jltype)) =
    CGVal(v, type_id, jltype, ScalarShape(), nothing, nothing, nothing)

CGVal(v::Value, type_id::TypeId, @nospecialize(jltype), shape::TileShape) =
    CGVal(v, type_id, jltype, shape, nothing, nothing, nothing)

# Constructor for multi-value results (from loops, ifs)
CGVal(v::Vector{Value}, @nospecialize(jltype)) =
    CGVal(v, nothing, jltype, ScalarShape(), nothing, nothing, nothing)

# Constructor for lazy argument references
function arg_ref_value(arg_idx::Int, chain::Vector{Int}, @nospecialize(jltype))
    CGVal(nothing, nothing, jltype, ScalarShape(), (arg_idx, chain), nothing, nothing)
end

"""
    ghost_value(jltype[, constant]) -> CGVal

Create a ghost value (zero-size singleton with no runtime representation).
Optionally stores a compile-time constant value.
"""
ghost_value(@nospecialize(jltype)) = CGVal(nothing, TypeId(-1), jltype, ScalarShape(), nothing, nothing, nothing)
ghost_value(@nospecialize(jltype), constant) = CGVal(nothing, TypeId(-1), jltype, ScalarShape(), nothing, Some(constant), nothing)

"""
    tuple_value(jltype, component_refs, component_constants) -> CGVal

Create a tuple value with tracked component refs. Derives constant if all components have constants.
Used by intrinsics like cat() that need to access individual tuple elements.
"""
function tuple_value(@nospecialize(jltype), component_refs::Vector{Any}, component_constants::Vector{Any})
    # If all components have constants, derive the tuple constant
    constant = if all(!isnothing, component_constants)
        Some(Tuple(component_constants))
    else
        nothing
    end
    CGVal(nothing, TypeId(-1), jltype, ScalarShape(), nothing, constant, component_refs)
end

"""
    is_ghost(tv::CGVal) -> Bool

Check if a CGVal is a ghost (no runtime representation).
"""
is_ghost(tv::CGVal) = tv.v === nothing && tv.arg_ref === nothing

"""
    is_arg_ref(tv::CGVal) -> Bool

Check if a CGVal is a lazy argument reference.
"""
is_arg_ref(tv::CGVal) = tv.arg_ref !== nothing

#=============================================================================
 CGCtx: Compilation context
=============================================================================#

"""
    CGCtx

Holds all state during Tile IR code generation for a kernel function.
Maps Julia SSA values to CGVals and manages bytecode emission.
"""
mutable struct CGCtx
    # SSA value mapping: original Julia SSA index -> CGVal
    # Uses global/original indices everywhere (no local renumbering)
    # Loop/if ops store a CGVal with tuple_values field (extracted by getfield statements)
    values::Dict{Int, CGVal}
    args::Dict{Int, CGVal}        # Argument index -> CGVal
    slots::Dict{Int, CGVal}       # Slot number -> CGVal
    block_args::Dict{Int, CGVal}  # BlockArg id -> CGVal (for control flow)

    # Destructured argument handling: path-keyed flat values
    # Key: (arg_idx, path) where path is e.g. [1] or [1, 2] (field indices)
    arg_flat_values::Dict{Tuple{Int, Vector{Int}}, Vector{Value}}
    arg_types::Dict{Int, Type}

    # Cached TensorViews for TileArray arguments
    # Key: arg_idx::Int for top-level, or (arg_idx, path) for nested
    tensor_views::Dict{Any, Tuple{Value, TypeId}}

    # Bytecode infrastructure
    cb::CodeBuilder
    tt::TypeTable
    sci::StructuredIRCode

    # Memory ordering token
    token::Union{Value, Nothing}
    token_type::Union{TypeId, Nothing}

    # Type cache: Julia type -> TypeId
    type_cache::Dict{Type, TypeId}

    # Target architecture (e.g., v"10.0" for sm_100)
    sm_arch::Union{VersionNumber, Nothing}

    # Compilation cache (needed for combiner compilation)
    cache::CacheView
end

function CGCtx(; cb::CodeBuilder, tt::TypeTable, sci::StructuredIRCode,
                 token::Union{Value, Nothing} = nothing,
                 token_type::Union{TypeId, Nothing} = nothing,
                 type_cache::Dict{Type, TypeId} = Dict{Type, TypeId}(),
                 sm_arch::Union{VersionNumber, Nothing} = nothing,
                 cache::CacheView)
    CGCtx(
        Dict{Int, CGVal}(),
        Dict{Int, CGVal}(),
        Dict{Int, CGVal}(),
        Dict{Int, CGVal}(),
        Dict{Tuple{Int, Vector{Int}}, Vector{Value}}(),
        Dict{Int, Type}(),
        Dict{Any, Tuple{Value, TypeId}}(),
        cb, tt, sci, token, token_type, type_cache, sm_arch, cache,
    )
end


#=============================================================================
 Value lookup via indexing syntax
=============================================================================#

function Base.getindex(ctx::CGCtx, ssa::SSAValue)
    # Simple lookup by original Julia SSA index
    get(ctx.values, ssa.id, nothing)
end

function Base.getindex(ctx::CGCtx, arg::Argument)
    get(ctx.args, arg.n, nothing)
end

function Base.getindex(ctx::CGCtx, slot::SlotNumber)
    get(ctx.slots, slot.id, nothing)
end

function Base.setindex!(ctx::CGCtx, tv::CGVal, ssa::SSAValue)
    ctx.values[ssa.id] = tv
end

function Base.setindex!(ctx::CGCtx, tv::CGVal, arg::Argument)
    ctx.args[arg.n] = tv
end

function Base.setindex!(ctx::CGCtx, tv::CGVal, slot::SlotNumber)
    ctx.slots[slot.id] = tv
end

function Base.getindex(ctx::CGCtx, block_arg::BlockArg)
    get(ctx.block_args, block_arg.id, nothing)
end

function Base.setindex!(ctx::CGCtx, tv::CGVal, block_arg::BlockArg)
    ctx.block_args[block_arg.id] = tv
end

#=============================================================================
 Destructured argument helpers
=============================================================================#

"""
    get_arg_flat_values(ctx, arg_idx, path) -> Union{Vector{Value}, Nothing}

Get the flat Tile IR values for a destructured argument at the given path.
"""
function get_arg_flat_values(ctx::CGCtx, arg_idx::Int, path::Vector{Int})
    get(ctx.arg_flat_values, (arg_idx, path), nothing)
end

"""
    collect_child_values(ctx, arg_idx, path, n) -> Union{Vector{Value}, Nothing}

Collect values at `[path..., 1]` through `[path..., n]`, each expected to have
exactly one flat value. Returns `nothing` if any child is missing.
"""
function collect_child_values(ctx::CGCtx, arg_idx::Int, path::Vector{Int}, n::Int)
    result = Value[]
    for i in 1:n
        child = get_arg_flat_values(ctx, arg_idx, [path..., i])
        child === nothing && return nothing
        append!(result, child)
    end
    isempty(result) ? nothing : result
end

"""
    try_materialize_scalar(ctx, arg_idx, path, rt) -> Union{CGVal, Nothing}

If `path` maps to exactly one flat value, return a concrete CGVal for it.
"""
function try_materialize_scalar(ctx::CGCtx, arg_idx::Int, path::Vector{Int}, @nospecialize(rt))
    values = get_arg_flat_values(ctx, arg_idx, path)
    if values !== nothing && length(values) == 1
        type_id = tile_type_for_julia!(ctx, rt)
        return CGVal(values[1], type_id, rt)
    end
    nothing
end

"""
    resolve_arg_ref(ctx, arg_idx, chain, idx, rt) -> CGVal

Extend an arg_ref chain by `idx`, materializing to a concrete value if the path
maps to a leaf scalar, otherwise returning a new lazy arg_ref.
"""
function resolve_arg_ref(ctx::CGCtx, arg_idx::Int, chain::Vector{Int}, idx::Int, @nospecialize(rt))
    new_chain = [chain..., idx]
    cv = try_materialize_scalar(ctx, arg_idx, new_chain, rt)
    cv !== nothing && return cv
    return arg_ref_value(arg_idx, new_chain, rt)
end


"""
    is_destructured_arg(ctx, arg_idx) -> Bool

Check if an argument was destructured into multiple flat parameters.
"""
is_destructured_arg(ctx::CGCtx, arg_idx::Int) = haskey(ctx.arg_types, arg_idx)

"""
    get_arg_type(ctx, arg_idx) -> Union{Type, Nothing}

Get the original Julia type for a destructured argument.
"""
get_arg_type(ctx::CGCtx, arg_idx::Int) = get(ctx.arg_types, arg_idx, nothing)

#=============================================================================
 Type conversion utilities
=============================================================================#

"""
    require_concrete_type(T, context::String)

Ensure a type is fully concrete (not a UnionAll).
"""
function require_concrete_type(@nospecialize(T), context::String)
    T_unwrapped = CC.widenconst(T)
    if T_unwrapped isa UnionAll
        throw(IRError("Type must be fully concrete in $context, got partial type: $T"))
    end
    return T_unwrapped
end

"""
    tile_type_for_julia!(ctx, T; throw_error=true) -> TypeId or nothing

Get or create a Tile IR type for a Julia type. With `throw_error=false`, returns
`nothing` instead of throwing if the type has no Tile IR representation.
"""
function tile_type_for_julia!(ctx::CGCtx, @nospecialize(T); throw_error::Bool=true)
    actual_type = CC.widenconst(T)
    cached = get(ctx.type_cache, actual_type, nothing)
    cached !== nothing && return cached
    type_id = _tile_type_for_julia!(ctx.tt, actual_type)
    if type_id !== nothing
        ctx.type_cache[actual_type] = type_id
        return type_id
    end
    throw_error && throw(IRError("Unsupported Julia type for Tile IR: $actual_type"))
    return nothing
end

function _tile_type_for_julia!(tt::TypeTable, @nospecialize(T::Type))
    # Scalar types -> 0-D tile
    if T === Bool
        return tile_type!(tt, I1(tt), Int[])
    elseif T === Int8 || T === UInt8
        return tile_type!(tt, I8(tt), Int[])
    elseif T === Int16 || T === UInt16
        return tile_type!(tt, I16(tt), Int[])
    elseif T === Int32 || T === UInt32
        return tile_type!(tt, I32(tt), Int[])
    elseif T === Int64 || T === UInt64
        return tile_type!(tt, I64(tt), Int[])
    elseif T === Float16
        return tile_type!(tt, F16(tt), Int[])
    elseif T === BFloat16
        return tile_type!(tt, BF16(tt), Int[])
    elseif T === Float32
        return tile_type!(tt, F32(tt), Int[])
    elseif T === Float64
        return tile_type!(tt, F64(tt), Int[])
    end

    # Pointers -> 0-D tile of pointer type
    if T <: Ptr
        elem_dtype = julia_to_tile_dtype!(tt, eltype(T))
        ptr_type = pointer_type!(tt, elem_dtype)
        return tile_type!(tt, ptr_type, Int[])
    end

    # Tile{T, Shape} -> tile type with shape
    if T <: Tile
        if T isa UnionAll || !isa(T, DataType) || length(T.parameters) < 2
            throw(IRError("Tile type must be fully specified with element type and shape, got: $T. " *
                          "This indicates type instability in the kernel - ensure all tile operations have inferrable shapes."))
        end
        shape_param = size(T)
        if !(shape_param isa Tuple)
            throw(IRError("Tile shape must be a tuple, got: $shape_param"))
        end
        elem_dtype = julia_to_tile_dtype!(tt, eltype(T))
        shape = RowMajorShape(shape_param)
        return tile_type!(tt, elem_dtype, collect(shape))
    end

    return nothing
end

"""
    tile_type_and_shape_for_julia!(ctx, T) -> (TypeId, RowMajorShape)

Get the Tile IR type and shape for a Julia type.
"""
function tile_type_and_shape_for_julia!(ctx::CGCtx, @nospecialize(T))
    actual_type = CC.widenconst(T)
    type_id = tile_type_for_julia!(ctx, actual_type)

    # Extract shape from Tile types (in Tile IR row-major order)
    shape = if actual_type <: Tile
        RowMajorShape(size(actual_type))
    else
        ScalarShape()
    end

    return (type_id, shape)
end

#=============================================================================
 Struct destructuring helpers
=============================================================================#

"""
    is_ghost_type(T) -> Bool

Check if a type is a ghost type (zero-size singleton).
"""
function is_ghost_type(@nospecialize(T))
    try
        isbitstype(T) && sizeof(T) == 0
    catch
        false
    end
end


"""
    flat_field_count(T) -> Int

Count flat parameters a type expands to (recursive).
"""
function flat_field_count(@nospecialize(T))
    if is_ghost_type(T)
        return 0
    elseif isprimitivetype(T)
        return 1
    else
        count = 0
        for fi in 1:fieldcount(T)
            count += flat_field_count(fieldtype(T, fi))
        end
        return count
    end
end

"""
    flatten(x) -> Tuple

Flatten a value into a tuple of its kernel parameter leaf fields.
Ghost types return `()`, scalars return `(x,)`, structs/tuples recurse.
"""
flatten(x::TileArray) = (x.ptr, x.sizes..., x.strides...)
function flatten(x)
    T = typeof(x)
    is_ghost_type(T) && return ()
    isprimitivetype(T) && return (x,)
    result = Any[]
    for fi in 1:fieldcount(T)
        fval = getfield(x, fi)
        append!(result, flatten(fval))
    end
    return Tuple(result)
end

#-----------------------------------------------------------------------------
# Argument helpers
#-----------------------------------------------------------------------------

"""
    extract_argument_index(arg) -> Union{Int, Nothing}

Extract the raw argument index from a SlotNumber or Argument.
Returns the index that corresponds directly to `ir.argtypes[idx]`.
Note: index 1 is the function itself; user args start at index 2.
"""
function extract_argument_index(@nospecialize(arg))
    if arg isa SlotNumber
        return arg.id
    elseif arg isa Argument
        return arg.n
    end
    nothing
end

#-----------------------------------------------------------------------------
# Tile helpers
#-----------------------------------------------------------------------------

"""
    extract_tile_shape(T) -> RowMajorShape

Extract shape from a Tile{T, Shape} type in Tile IR (row-major) order.
Returns empty shape if not a Tile type.
"""
function extract_tile_shape(@nospecialize(T))
    T = CC.widenconst(T)
    if T <: Tile
        return RowMajorShape(size(T))
    end
    ScalarShape()
end
