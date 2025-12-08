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

# Register intrinsics with the compiler after module initialization
function __init__()
    register_intrinsic!(:bid, bid)
    register_intrinsic!(:num_blocks, num_blocks)
    register_intrinsic!(:load, load)
    register_intrinsic!(:store, store)
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
    load(array, index, shape)

Load a tile from an array at the given index.
In kernel code, this is compiled to LoadViewTkoOp.
The return type is inferred from the array's element type.
"""
@noinline function load(array::AbstractArray{T}, index, shape)::T where T
    new_intrinsic(T)
end

"""
    store(array, index, tile) -> Nothing

Store a tile to an array at the given index.
In kernel code, this is compiled to StoreViewTkoOp.
"""
@noinline function store(array::AbstractArray{T}, index, tile::T)::Nothing where T
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

#=============================================================================
 Test Helpers
=============================================================================#

"""
    emit_empty_kernel() -> Vector{UInt8}

Generate bytecode for a minimal empty kernel that should be accepted by tileiras.
"""
function emit_empty_kernel()
    write_bytecode!(1) do writer, func_buf
        # Empty function with no parameters and no return value
        cb = add_function!(writer, func_buf, "empty_kernel", TypeId[], TypeId[];
                           is_entry=true)

        # Just return
        encode_ReturnOp!(cb)

        finalize_function!(func_buf, cb, writer.debug_info)
    end
end

"""
    emit_bid_kernel() -> Vector{UInt8}

Generate bytecode for a kernel that gets the block ID.
"""
function emit_bid_kernel()
    write_bytecode!(1) do writer, func_buf
        tt = writer.type_table
        i32_type = I32(tt)
        scalar_i32 = tile_type!(tt, i32_type, Int[])  # 0-D tile = scalar

        cb = add_function!(writer, func_buf, "bid_kernel", TypeId[], TypeId[];
                           is_entry=true)

        # Get block IDs
        bid_x, bid_y, bid_z = encode_GetTileBlockIdOp!(cb, scalar_i32, scalar_i32, scalar_i32)

        # Return (no return values)
        encode_ReturnOp!(cb)

        finalize_function!(func_buf, cb, writer.debug_info)
    end
end

"""
    emit_vadd_kernel(; tile_size=16) -> Vector{UInt8}

Generate bytecode for a vector addition kernel.
This is a manually-constructed kernel matching the structure of vadd.py.
"""
function emit_vadd_kernel(; tile_size::Int=16)
    write_bytecode!(1) do writer, func_buf
        tt = writer.type_table

        # Types we need
        f32_dtype = F32(tt)
        i32_dtype = I32(tt)
        i64_dtype = I64(tt)
        token_type = Token(tt)
        f32_ptr = pointer_type!(tt, f32_dtype)

        # Scalar types (0-D tiles)
        scalar_i32 = tile_type!(tt, i32_dtype, Int[])
        scalar_i64 = tile_type!(tt, i64_dtype, Int[])
        scalar_f32_ptr = tile_type!(tt, f32_ptr, Int[])

        # Tile type for the vector tiles
        tile_f32 = tile_type!(tt, f32_dtype, [tile_size])

        # TensorView: 1D array with dynamic size
        # Shape and strides are dynamic (DYNAMIC_SHAPE marker)
        tv_f32 = tensor_view_type!(tt, f32_dtype, [DYNAMIC_SHAPE], [DYNAMIC_SHAPE])

        # PartitionView for the tile shape
        pv_f32 = partition_view_type!(tt, [tile_size], tv_f32, [0], PaddingZero)

        # Function parameters:
        # - a_ptr: pointer to float32
        # - a_size: int64 (array size)
        # - a_stride: int64 (array stride)
        # - b_ptr, b_size, b_stride: same for b
        # - c_ptr, c_size, c_stride: same for c (output)
        param_types = [
            scalar_f32_ptr, scalar_i64, scalar_i64,  # a
            scalar_f32_ptr, scalar_i64, scalar_i64,  # b
            scalar_f32_ptr, scalar_i64, scalar_i64,  # c
        ]

        cb = add_function!(writer, func_buf, "vadd_kernel", param_types, TypeId[];
                           is_entry=true)

        # Create function parameter values
        params = make_block_args!(cb, length(param_types))
        a_ptr, a_size, a_stride = params[1], params[2], params[3]
        b_ptr, b_size, b_stride = params[4], params[5], params[6]
        c_ptr, c_size, c_stride = params[7], params[8], params[9]

        # Get block ID (pid)
        bid_x, _, _ = encode_GetTileBlockIdOp!(cb, scalar_i32, scalar_i32, scalar_i32)

        # Create tensor views from pointers
        # MakeTensorViewOp(result_type, base, dynamic_shape, dynamic_strides)
        a_view = encode_MakeTensorViewOp!(cb, tv_f32, a_ptr, [a_size], [a_stride])
        b_view = encode_MakeTensorViewOp!(cb, tv_f32, b_ptr, [b_size], [b_stride])
        c_view = encode_MakeTensorViewOp!(cb, tv_f32, c_ptr, [c_size], [c_stride])

        # Create partition views
        a_part = encode_MakePartitionViewOp!(cb, pv_f32, a_view)
        b_part = encode_MakePartitionViewOp!(cb, pv_f32, b_view)
        c_part = encode_MakePartitionViewOp!(cb, pv_f32, c_view)

        # Truncate bid from i32 to match index type if needed
        # (LoadViewTko expects i32 indices based on Python impl)
        index_type = scalar_i32
        index = [bid_x]

        # Load tiles
        a_tile, _ = encode_LoadViewTkoOp!(cb, tile_f32, token_type, a_part, index)
        b_tile, _ = encode_LoadViewTkoOp!(cb, tile_f32, token_type, b_part, index)

        # Add tiles
        result_tile = encode_AddFOp!(cb, tile_f32, a_tile, b_tile)

        # Store result
        encode_StoreViewTkoOp!(cb, token_type, result_tile, c_part, index)

        # Return
        encode_ReturnOp!(cb)

        finalize_function!(func_buf, cb, writer.debug_info)
    end
end

"""
    tileiras_path() -> String

Get the path to the tileiras binary.
"""
function tileiras_path()
    path = joinpath(@__DIR__, "..", "bin", "tileiras")
    isfile(path) || error("tileiras not found at $path")
    return path
end

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
        run(`$(tileiras_path()) $input_path -o $output_path --gpu-name $sm_arch -O$opt_level`)
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

"""
    hexdump(buf::Vector{UInt8}; width=16)

Print a hex dump of the buffer for debugging.
"""
function hexdump(buf::Vector{UInt8}; width::Int=16)
    for i in 1:width:length(buf)
        # Offset
        print(lpad(string(i-1, base=16), 8, '0'), "  ")

        # Hex bytes
        for j in i:min(i+width-1, length(buf))
            print(lpad(string(buf[j], base=16), 2, '0'), " ")
            if j == i + width÷2 - 1
                print(" ")
            end
        end

        # Padding for incomplete lines
        remaining = i + width - 1 - length(buf)
        if remaining > 0
            for _ in 1:remaining
                print("   ")
            end
            if i + width÷2 - 1 > length(buf)
                print(" ")
            end
        end

        # ASCII
        print(" |")
        for j in i:min(i+width-1, length(buf))
            c = Char(buf[j])
            print(isprint(c) && !isspace(c) ? c : '.')
        end
        println("|")
    end
end

end # module cuTile
