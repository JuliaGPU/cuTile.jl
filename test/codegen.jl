using CUDA
sm_arch = ct.default_sm_arch()

# Helper to disassemble cubin to SASS using cuobjdump
function disasm_sass(cubin::Vector{UInt8})
    mktempdir() do dir
        path = joinpath(dir, "kernel.cubin")
        write(path, cubin)
        read(`cuobjdump -sass $path`, String)
    end
end

@testset "load/store 1D" begin
    function copy_kernel(a::Ptr{Float32}, b::Ptr{Float32})
        pid = ct.bid(0)
        tile = ct.load(a, pid, (16,))
        ct.store(b, pid, tile)
        return
    end
    cubin = ct.compile(copy_kernel, Tuple{Ptr{Float32}, Ptr{Float32}}; sm_arch)
    sass = disasm_sass(cubin)
    @test contains(sass, "copy_kernel")
    @test contains(sass, "LDG")  # Load global (tile load)
    @test contains(sass, "STG")  # Store global (tile store)
end

@testset "load/store 2D" begin
    function copy_2d_kernel(a::Ptr{Float32}, b::Ptr{Float32})
        bidx = ct.bid(0)
        bidy = ct.bid(1)
        tile = ct.load(a, (bidx, bidy), (32, 32))
        ct.store(b, (bidx, bidy), tile)
        return
    end
    cubin = ct.compile(copy_2d_kernel, Tuple{Ptr{Float32}, Ptr{Float32}}; sm_arch)
    sass = disasm_sass(cubin)
    @test contains(sass, "copy_2d_kernel")
    @test contains(sass, "LDG")  # Load global (tile load)
    @test contains(sass, "STG")  # Store global (tile store)
end

@testset "add" begin
    function add_kernel(a::Ptr{Float32}, b::Ptr{Float32}, c::Ptr{Float32})
        pid = ct.bid(0)
        tile_a = ct.load(a, pid, (16,))
        tile_b = ct.load(b, pid, (16,))
        result = tile_a + tile_b
        ct.store(c, pid, result)
        return
    end
    cubin = ct.compile(add_kernel, Tuple{Ptr{Float32}, Ptr{Float32}, Ptr{Float32}}; sm_arch)
    sass = disasm_sass(cubin)
    @test contains(sass, "add_kernel")
    @test contains(sass, "LDG")   # Tile loads
    @test contains(sass, "FADD")  # Float add
    @test contains(sass, "STG")   # Tile store
end

@testset "sub" begin
    function sub_kernel(a::Ptr{Float32}, b::Ptr{Float32}, c::Ptr{Float32})
        pid = ct.bid(0)
        tile_a = ct.load(a, pid, (16,))
        tile_b = ct.load(b, pid, (16,))
        result = tile_a - tile_b
        ct.store(c, pid, result)
        return
    end
    cubin = ct.compile(sub_kernel, Tuple{Ptr{Float32}, Ptr{Float32}, Ptr{Float32}}; sm_arch)
    sass = disasm_sass(cubin)
    @test contains(sass, "sub_kernel")
    @test contains(sass, "LDG")   # Tile loads
    @test contains(sass, "FADD")  # Sub is FADD with negated operand
    @test contains(sass, "STG")   # Tile store
end

@testset "mul" begin
    function mul_kernel(a::Ptr{Float32}, b::Ptr{Float32}, c::Ptr{Float32})
        pid = ct.bid(0)
        tile_a = ct.load(a, pid, (16,))
        tile_b = ct.load(b, pid, (16,))
        result = tile_a * tile_b
        ct.store(c, pid, result)
        return
    end
    cubin = ct.compile(mul_kernel, Tuple{Ptr{Float32}, Ptr{Float32}, Ptr{Float32}}; sm_arch)
    sass = disasm_sass(cubin)
    @test contains(sass, "mul_kernel")
    @test contains(sass, "LDG")   # Tile loads
    @test contains(sass, "FMUL")  # Float multiply
    @test contains(sass, "STG")   # Tile store
end

@testset "load from TileArray 1D" begin
    function tilearray_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1})
        pid = ct.bid(0)
        tile = ct.load(a, pid, (16,))
        ct.store(b, pid, tile)
        return
    end
    spec = ct.ArraySpec{1}(16, true)
    argtypes = Tuple{ct.TileArray{Float32,1,spec}, ct.TileArray{Float32,1,spec}}
    cubin = ct.compile(tilearray_kernel, argtypes; sm_arch)
    sass = disasm_sass(cubin)
    @test contains(sass, "tilearray_kernel")
    @test contains(sass, "LDG")  # Tile load
    @test contains(sass, "STG")  # Tile store
end

@testset "load from TileArray 2D" begin
    function tilearray_2d_kernel(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,2})
        bidx = ct.bid(0)
        bidy = ct.bid(1)
        tile = ct.load(a, (bidx, bidy), (32, 32))
        ct.store(b, (bidx, bidy), tile)
        return
    end
    spec = ct.ArraySpec{2}(16, true)
    argtypes = Tuple{ct.TileArray{Float32,2,spec}, ct.TileArray{Float32,2,spec}}
    cubin = ct.compile(tilearray_2d_kernel, argtypes; sm_arch)
    sass = disasm_sass(cubin)
    @test contains(sass, "tilearray_2d_kernel")
    @test contains(sass, "LDG")  # Tile load
    @test contains(sass, "STG")  # Tile store
end

@testset "div" begin
    function div_kernel(a::Ptr{Float32}, b::Ptr{Float32}, c::Ptr{Float32})
        pid = ct.bid(0)
        tile_a = ct.load(a, pid, (16,))
        tile_b = ct.load(b, pid, (16,))
        result = tile_a / tile_b
        ct.store(c, pid, result)
        return
    end
    cubin = ct.compile(div_kernel, Tuple{Ptr{Float32}, Ptr{Float32}, Ptr{Float32}}; sm_arch)
    sass = disasm_sass(cubin)
    @test contains(sass, "div_kernel")
    @test contains(sass, "LDG")   # Tile loads
    # Division may be implemented as reciprocal + multiply
    @test contains(sass, "STG")   # Tile store
end

@testset "sqrt" begin
    function sqrt_kernel(a::Ptr{Float32}, b::Ptr{Float32})
        pid = ct.bid(0)
        tile = ct.load(a, pid, (16,))
        result = sqrt(tile)
        ct.store(b, pid, result)
        return
    end
    cubin = ct.compile(sqrt_kernel, Tuple{Ptr{Float32}, Ptr{Float32}}; sm_arch)
    sass = disasm_sass(cubin)
    @test contains(sass, "sqrt_kernel")
    @test contains(sass, "LDG")   # Tile load
    @test contains(sass, "STG")   # Tile store
end

@testset "reduce_sum" begin
    function reduce_sum_kernel(a::Ptr{Float32}, b::Ptr{Float32})
        pid = ct.bid(0)
        tile = ct.load(a, pid, (4, 16))  # 2D tile
        sums = ct.reduce_sum(tile, 1)    # Sum along axis 1 -> (4,)
        ct.store(b, pid, sums)
        return
    end
    cubin = ct.compile(reduce_sum_kernel, Tuple{Ptr{Float32}, Ptr{Float32}}; sm_arch)
    sass = disasm_sass(cubin)
    @test contains(sass, "reduce_sum_kernel")
    @test contains(sass, "LDG")   # Tile load
    @test contains(sass, "FADD")  # Float add (reduction)
    @test contains(sass, "STG")   # Tile store
end

@testset "reduce_max" begin
    function reduce_max_kernel(a::Ptr{Float32}, b::Ptr{Float32})
        pid = ct.bid(0)
        tile = ct.load(a, pid, (4, 16))  # 2D tile
        maxes = ct.reduce_max(tile, 1)   # Max along axis 1 -> (4,)
        ct.store(b, pid, maxes)
        return
    end
    cubin = ct.compile(reduce_max_kernel, Tuple{Ptr{Float32}, Ptr{Float32}}; sm_arch)
    sass = disasm_sass(cubin)
    @test contains(sass, "reduce_max_kernel")
    @test contains(sass, "LDG")   # Tile load
    @test contains(sass, "STG")   # Tile store
end

@testset "control flow" begin
    # Test structured IR restructuring + control flow encoding
    # Note: Full control flow emission is work-in-progress, these tests
    # verify the restructuring infrastructure is working

    @testset "restructuring produces IfOp" begin
        # Simple ternary expression generates structured IR with IfOp
        function ternary_func(x::Int32)
            return x > 0 ? x + 1 : x - 1
        end
        # Verify it can be restructured (this tests the IR layer)
        structured = ct.code_structured(ternary_func, Tuple{Int32})
        # Check that we have an IfOp in the structured IR
        has_ifop = any(item -> item isa ct.IfOp, structured.entry.body)
        @test has_ifop
    end

    @testset "restructuring handles terminating if-else" begin
        # Multiple return statements create terminating if-then-else
        function multi_return_func(x::Int32)
            if x > 0
                return x + 1
            else
                return x - 1
            end
        end
        structured = ct.code_structured(multi_return_func, Tuple{Int32})
        # Terminating if-else becomes an IfOp
        has_ifop = any(item -> item isa ct.IfOp, structured.entry.body)
        @test has_ifop
    end

    @testset "IfOp encoding produces bytecode" begin
        # Test that the bytecode encoding infrastructure works
        # by directly calling the encoder
        using cuTile: BytecodeWriter, CodeBuilder, TypeId, Value
        using cuTile: encode_IfOp!, encode_YieldOp!, tile_type!, I32

        writer = ct.BytecodeWriter()
        cb = ct.CodeBuilder(writer.string_table, writer.constant_table, writer.type_table)
        tt = writer.type_table

        # Create a dummy condition value
        cond_type = tile_type!(tt, I32(tt), Int[])
        cond_bytes = reinterpret(UInt8, [Int32(1)])
        cond = ct.encode_ConstantOp!(cb, cond_type, collect(cond_bytes))

        # Encode an IfOp with empty result types
        then_body = _ -> encode_YieldOp!(cb)
        else_body = _ -> encode_YieldOp!(cb)
        results = encode_IfOp!(then_body, else_body, cb, TypeId[], cond)

        @test isempty(results)  # No results from IfOp
        @test cb.num_ops >= 1   # At least the IfOp was encoded
    end

    @testset "LoopOp encoding produces bytecode" begin
        using cuTile: encode_LoopOp!, encode_BreakOp!, encode_ContinueOp!

        writer = ct.BytecodeWriter()
        cb = ct.CodeBuilder(writer.string_table, writer.constant_table, writer.type_table)
        tt = writer.type_table

        # Encode a LoopOp with no carried values
        body = _ -> encode_BreakOp!(cb)
        results = ct.encode_LoopOp!(body, cb, TypeId[], Value[])

        @test isempty(results)
        @test cb.num_ops >= 1
    end
end
