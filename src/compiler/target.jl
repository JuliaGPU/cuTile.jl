# TileTarget and CodegenContext for cuTile compilation
#
# Holds the typed IR and compilation state for a kernel.

#=============================================================================
 TileTarget: Compilation target wrapping Julia typed IR
=============================================================================#

"""
    TileTarget

Holds information about a function being compiled to Tile IR.
"""
struct TileTarget
    mi::MethodInstance
    ci::CodeInfo
    rettype::Type
    argtypes::Vector{Any}
end

function TileTarget(@nospecialize(f), @nospecialize(argtypes::Type{<:Tuple}))
    ci, rettype = get_typed_ir(f, argtypes)
    mi = get_method_instance(f, argtypes)
    arg_types = argtypes === Tuple{} ? Any[] : collect(argtypes.parameters)
    TileTarget(mi, ci, rettype, arg_types)
end

# Accessors
code(target::TileTarget) = target.ci.code
ssatypes(target::TileTarget) = target.ci.ssavaluetypes
slottypes(target::TileTarget) = target.ci.slottypes
nargs(target::TileTarget) = length(target.argtypes)

#=============================================================================
 CGVal: Unified value representation (analogous to Julia's jl_cgval_t)
=============================================================================#

"""
    CGVal

Represents a value during Tile IR code generation, bundling the IR value
with its type information and metadata.

Similar to Julia compiler's `jl_cgval_t`, this provides a unified representation
for all values flowing through codegen. A CGVal can be either:
1. A concrete SSA value (v is set, arg_ref is nothing)
2. A lazy argument reference chain (v is nothing, arg_ref tracks the access path)
"""
struct CGVal
    v::Union{Value, Nothing}  # Tile IR value (nothing for ghost values or lazy refs)
    type_id::Union{TypeId, Nothing}  # Tile IR type (nothing for lazy refs)
    jltype::Any               # Original Julia type
    shape::Vector{Int}        # Tile shape (empty for scalars)
    # Lazy argument reference: (arg_idx, [:field, index, ...])
    # e.g., (1, [:sizes, 2]) means "argument 1, field :sizes, index 2"
    arg_ref::Union{Tuple{Int, Vector{Union{Symbol, Int}}}, Nothing}
    constant::Any             # Compile-time constant value (nothing if not known)
end

# Convenience constructors for concrete values (constant = nothing by default)
CGVal(v::Value, type_id::TypeId, @nospecialize(jltype)) =
    CGVal(v, type_id, jltype, Int[], nothing, nothing)

CGVal(v::Value, type_id::TypeId, @nospecialize(jltype), shape::Vector{Int}) =
    CGVal(v, type_id, jltype, shape, nothing, nothing)

# Constructor for lazy argument references (never have constants)
function arg_ref_value(arg_idx::Int, chain::Vector{Union{Symbol, Int}}, @nospecialize(jltype))
    CGVal(nothing, nothing, jltype, Int[], (arg_idx, chain), nothing)
end

"""
    ghost_value(jltype[, constant]) -> CGVal

Create a ghost value (zero-size singleton with no runtime representation).
Optionally stores a compile-time constant value.
"""
ghost_value(@nospecialize(jltype)) = CGVal(nothing, TypeId(-1), jltype, Int[], nothing, nothing)
ghost_value(@nospecialize(jltype), constant) = CGVal(nothing, TypeId(-1), jltype, Int[], nothing, constant)

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
 CodegenContext: Compilation context
=============================================================================#

"""
    CodegenContext

Holds all state during Tile IR code generation for a kernel function.
Maps Julia SSA values to CGVals and manages bytecode emission.
"""
mutable struct CodegenContext
    # SSA value mapping: original Julia SSA index -> CGVal
    # Uses global/original indices everywhere (no local renumbering)
    values::Dict{Int, CGVal}
    args::Dict{Int, CGVal}        # Argument index -> CGVal
    slots::Dict{Int, CGVal}       # Slot number -> CGVal
    block_args::Dict{Int, CGVal}  # BlockArg id -> CGVal (for control flow)

    # Destructured argument handling (for TileArray fields)
    arg_flat_values::Dict{Tuple{Int, Union{Nothing, Symbol}}, Vector{Value}}
    arg_types::Dict{Int, Type}

    # Bytecode infrastructure
    cb::CodeBuilder
    tt::TypeTable
    target::TileTarget

    # Memory ordering token
    token::Union{Value, Nothing}
    token_type::Union{TypeId, Nothing}

    # Type cache: Julia type -> TypeId
    type_cache::Dict{Type, TypeId}
end

function CodegenContext(writer::BytecodeWriter, target::TileTarget)
    CodegenContext(
        Dict{Int, CGVal}(),
        Dict{Int, CGVal}(),
        Dict{Int, CGVal}(),
        Dict{Int, CGVal}(),
        Dict{Tuple{Int, Union{Nothing, Symbol}}, Vector{Value}}(),
        Dict{Int, Type}(),
        CodeBuilder(writer.string_table, writer.constant_table, writer.type_table),
        writer.type_table,
        target,
        nothing,
        nothing,
        Dict{Type, TypeId}(),
    )
end

#=============================================================================
 Value lookup via indexing syntax
=============================================================================#

function Base.getindex(ctx::CodegenContext, ssa::SSAValue)
    # Simple lookup by original Julia SSA index
    get(ctx.values, ssa.id, nothing)
end

function Base.getindex(ctx::CodegenContext, arg::Argument)
    get(ctx.args, arg.n, nothing)
end

function Base.getindex(ctx::CodegenContext, slot::SlotNumber)
    get(ctx.slots, slot.id, nothing)
end

function Base.setindex!(ctx::CodegenContext, tv::CGVal, ssa::SSAValue)
    ctx.values[ssa.id] = tv
end

function Base.setindex!(ctx::CodegenContext, tv::CGVal, arg::Argument)
    ctx.args[arg.n] = tv
end

function Base.setindex!(ctx::CodegenContext, tv::CGVal, slot::SlotNumber)
    ctx.slots[slot.id] = tv
end

function Base.getindex(ctx::CodegenContext, block_arg::BlockArg)
    get(ctx.block_args, block_arg.id, nothing)
end

function Base.setindex!(ctx::CodegenContext, tv::CGVal, block_arg::BlockArg)
    ctx.block_args[block_arg.id] = tv
end

#=============================================================================
 Destructured argument helpers
=============================================================================#

"""
    get_arg_flat_values(ctx, arg_idx, field=nothing) -> Union{Vector{Value}, Nothing}

Get the flat Tile IR values for an argument or its field.
"""
function get_arg_flat_values(ctx::CodegenContext, arg_idx::Int, field::Union{Nothing, Symbol}=nothing)
    get(ctx.arg_flat_values, (arg_idx, field), nothing)
end

"""
    is_destructured_arg(ctx, arg_idx) -> Bool

Check if an argument was destructured into multiple flat parameters.
"""
is_destructured_arg(ctx::CodegenContext, arg_idx::Int) = haskey(ctx.arg_types, arg_idx)

"""
    get_arg_type(ctx, arg_idx) -> Union{Type, Nothing}

Get the original Julia type for a destructured argument.
"""
get_arg_type(ctx::CodegenContext, arg_idx::Int) = get(ctx.arg_types, arg_idx, nothing)

#=============================================================================
 Type conversion utilities
=============================================================================#

"""
    unwrap_type(T) -> Type

Unwrap type wrappers like Core.Const to get the actual type.
"""
function unwrap_type(@nospecialize(T))
    if T isa Core.Const
        return typeof(T.val)
    elseif T isa Core.PartialStruct
        return T.typ
    elseif T isa Type
        return T
    else
        return T
    end
end

"""
    require_concrete_type(T, context::String)

Ensure a type is fully concrete (not a UnionAll).
"""
function require_concrete_type(@nospecialize(T), context::String)
    T_unwrapped = unwrap_type(T)
    if T_unwrapped isa UnionAll
        error("Type must be fully concrete in $context, got partial type: $T")
    end
    return T_unwrapped
end

"""
    tile_type_for_julia!(ctx, T) -> TypeId

Get or create a Tile IR type for a Julia type.
"""
function tile_type_for_julia!(ctx::CodegenContext, @nospecialize(T))
    actual_type = unwrap_type(T)
    get!(ctx.type_cache, actual_type) do
        _tile_type_for_julia!(ctx.tt, actual_type)
    end
end

function _tile_type_for_julia!(tt::TypeTable, @nospecialize(T::Type))
    # Scalar types -> 0-D tile
    if T === Bool
        return tile_type!(tt, I1(tt), Int[])
    elseif T === Int32 || T === UInt32
        return tile_type!(tt, I32(tt), Int[])
    elseif T === Int64 || T === UInt64
        return tile_type!(tt, I64(tt), Int[])
    elseif T === Float16
        return tile_type!(tt, F16(tt), Int[])
    elseif T === Float32
        return tile_type!(tt, F32(tt), Int[])
    elseif T === Float64
        return tile_type!(tt, F64(tt), Int[])
    elseif T === Nothing
        return Token(tt)
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
            error("Tile type must be fully specified with element type and shape, got: $T. " *
                  "This indicates type instability in the kernel - ensure all tile operations have inferrable shapes.")
        end
        elem_type = T.parameters[1]
        shape_param = T.parameters[2]
        if !(shape_param isa Tuple)
            error("Tile shape must be a tuple, got: $shape_param")
        end
        elem_dtype = julia_to_tile_dtype!(tt, elem_type)
        shape = collect(Int, shape_param)
        return tile_type!(tt, elem_dtype, shape)
    end

    error("Unsupported Julia type for Tile IR: $T")
end

"""
    tile_type_and_shape_for_julia!(ctx, T) -> (TypeId, Vector{Int})

Get the Tile IR type and shape for a Julia type.
"""
function tile_type_and_shape_for_julia!(ctx::CodegenContext, @nospecialize(T))
    actual_type = unwrap_type(T)
    type_id = tile_type_for_julia!(ctx, actual_type)

    # Extract shape from Tile types
    shape = Int[]
    if actual_type <: Tile && length(actual_type.parameters) >= 2
        shape_param = actual_type.parameters[2]
        if shape_param isa Tuple
            shape = collect(Int, shape_param)
        end
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
    should_destructure(T) -> Bool

Check if a type should be destructured into flat parameters.
"""
function should_destructure(@nospecialize(T))
    T = unwrap_type(T)
    isstructtype(T) || return false
    is_ghost_type(T) && return false
    isprimitivetype(T) && return false
    T <: TileArray && return true
    return false
end

"""
    flat_field_count(T) -> Int

Count flat parameters a type expands to.
"""
flat_field_count(::Type{<:NTuple{N, T}}) where {N, T} = N
flat_field_count(::Type) = 1
