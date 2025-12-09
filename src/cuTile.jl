module cuTile

# Bytecode infrastructure
include("bytecode/encodings.jl")

# Compiler infrastructure
include("compiler/codegen.jl")

# Re-export bytecode components for direct use
export TypeTable, TypeId, CodeBuilder, BytecodeWriter, Value
export I1, I8, I16, I32, I64, F16, BF16, F32, F64, Token
export tile_type!, pointer_type!, tensor_view_type!, partition_view_type!, function_type!
export julia_to_tile_dtype!
export write_bytecode!, add_function!, finalize_function!
export encode_ConstantOp!, encode_GetTileBlockIdOp!, encode_ReturnOp!
export encode_AddFOp!, encode_AddIOp!, encode_SubFOp!, encode_MulFOp!, encode_MulIOp!
export encode_MakeTensorViewOp!, encode_MakePartitionViewOp!
export encode_LoadViewTkoOp!, encode_StoreViewTkoOp!
export RoundingMode, IntegerOverflow, MemoryOrderingSemantics, MemoryScope
export PaddingValue, PaddingMissing, PaddingZero

# Compiler exports
export compile_kernel, compile_to_cubin, compile_kernel_to_cubin, TileTarget, get_typed_ir

# API exports
export Tile, tile_shape

# Register intrinsics with the compiler after module initialization
function __init__()
    register_intrinsic!(:bid, bid)
    register_intrinsic!(:num_blocks, num_blocks)
    register_intrinsic!(:load, load)
    register_intrinsic!(:store, store)
    register_intrinsic!(:tile_add, tile_add)
    register_intrinsic!(:tile_sub, tile_sub)
    register_intrinsic!(:tile_mul, tile_mul)
    register_intrinsic!(:transpose, transpose)
end

#=============================================================================
 API Types
=============================================================================#

"""
    Constant{T}

Marker type for compile-time constant values.
"""
struct Constant{T}
    value::T
end

Base.convert(::Type{Constant{T}}, x::T) where T = Constant{T}(x)

"""
    Tile{T, Shape}

Represents a tile of data with element type `T` and static shape `Shape`.
Shape is a tuple of integers representing the tile dimensions.

This is a compile-time abstraction - at runtime in kernel code, tiles are
represented as Tile IR values. The struct exists to enable proper type
inference and operator dispatch.
"""
struct Tile{T, Shape}
    # No runtime fields - this is a phantom type for the compiler
    # The actual data lives in GPU registers during kernel execution

    # Inner constructor that's never actually called at runtime
    function Tile{T, Shape}() where {T, Shape}
        new{T, Shape}()
    end
end

# Type accessors
Base.eltype(::Type{Tile{T, Shape}}) where {T, Shape} = T
Base.eltype(::Tile{T, Shape}) where {T, Shape} = T
tile_shape(::Type{Tile{T, Shape}}) where {T, Shape} = Shape
tile_shape(::Tile{T, Shape}) where {T, Shape} = Shape

#=============================================================================
 Tile Arithmetic
=============================================================================#

# These are stub implementations that the compiler intercepts.
# They return a new Tile with the same shape, enabling proper type inference.

@noinline function tile_add(a::Tile{T, S}, b::Tile{T, S})::Tile{T, S} where {T, S}
    Tile{T, S}()
end

@noinline function tile_sub(a::Tile{T, S}, b::Tile{T, S})::Tile{T, S} where {T, S}
    Tile{T, S}()
end

@noinline function tile_mul(a::Tile{T, S}, b::Tile{T, S})::Tile{T, S} where {T, S}
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

# Marker for intrinsics that should be compiled to Tile IR.
# Uses inferencebarrier to prevent constant folding of the return value.
# At runtime this errors - but our compiler intercepts before runtime.
@noinline function new_intrinsic(::Type{T})::T where T
    # inferencebarrier on the zero value prevents the optimizer from knowing
    # what the return value is, so expressions using it can't be folded
    Base.inferencebarrier(zero(T))
end

"""
    bid(axis) -> Int32

Get the block ID along the given axis (0=x, 1=y, 2=z).
In kernel code, this is compiled to GetTileBlockIdOp.
"""
@noinline bid(axis::Integer)::Int32 = new_intrinsic(Int32)

"""
    num_blocks(axis) -> Int32

Get the grid size along the given axis (0=x, 1=y, 2=z).
In kernel code, this is compiled to GetNumTileBlocksOp.
"""
@noinline num_blocks(axis::Integer)::Int32 = new_intrinsic(Int32)

"""
    load(array, index, shape::NTuple{N, Int}) -> Tile{T, shape}

Load a tile from an array at the given index with the specified shape.
In kernel code, this is compiled to LoadViewTkoOp.

Returns a `Tile{T, Shape}` where T is the array element type and Shape
is the compile-time constant shape tuple.
"""
@noinline function load(array::AbstractArray{T}, index, shape::NTuple{N, Int}) where {T, N}
    # The shape must be a compile-time constant for the return type
    Tile{T, shape}()
end

"""
    store(array, index, tile::Tile) -> Nothing

Store a tile to an array at the given index.
In kernel code, this is compiled to StoreViewTkoOp.
"""
@noinline function store(array::AbstractArray{T}, index, tile::Tile{T})::Nothing where T
    # donotdelete prevents the optimizer from eliminating this call
    # even though it has no observable effects in Julia
    Base.donotdelete(array, index, tile)
    nothing
end

#=============================================================================
 Host Utilities
=============================================================================#

"""
    cdiv(a, b)

Ceiling division: ceil(a/b).
"""
cdiv(a, b) = cld(a, b)

"""
    compile_to_cubin(tile_bytecode::Vector{UInt8}; sm_arch="sm_100", opt_level=3) -> Vector{UInt8}

Compile Tile IR bytecode to a CUBIN using tileiras.
Returns the CUBIN bytes. Throws an error if compilation fails.
"""
function compile_to_cubin(tile_bytecode::Vector{UInt8}; sm_arch::String="sm_100", opt_level::Int=3)
    # Create temp files for input/output
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
    compile_kernel_to_cubin(f, argtypes; name=nothing, sm_arch="sm_100", opt_level=3) -> Vector{UInt8}

Compile a Julia kernel function directly to CUBIN.
"""
function compile_kernel_to_cubin(@nospecialize(f), @nospecialize(argtypes);
                                  name::Union{String, Nothing}=nothing,
                                  sm_arch::String="sm_100",
                                  opt_level::Int=3)
    tile_bytecode = compile_kernel(f, argtypes; name)
    return compile_to_cubin(tile_bytecode; sm_arch, opt_level)
end

end # module cuTile
