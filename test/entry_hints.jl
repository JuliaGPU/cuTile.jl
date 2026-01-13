@testset "Entry Hints" begin

    @testset "MLIR Encoding" begin
        # Setup: Define spec for concrete types
        spec1d = cuTile.ArraySpec{1}(16, true)

        function simple_kernel(a::cuTile.TileArray{Float32, 1, spec1d})
            pid = cuTile.bid(1)
            t = cuTile.load(a, pid, (16,))
            cuTile.store(a, pid, t)
            return nothing
        end

        argtypes = Tuple{cuTile.TileArray{Float32, 1, spec1d}}

        @testset "num_ctas only" begin
            bytecode = cuTile.emit_tileir(simple_kernel, argtypes; num_ctas=4)
            mlir = cuTile.disassemble_tileir(bytecode)
            @test occursin("optimization_hints=<sm_100 = {num_cta_in_cga = 4}>", mlir)
        end

        @testset "occupancy only" begin
            bytecode = cuTile.emit_tileir(simple_kernel, argtypes; occupancy=8)
            mlir = cuTile.disassemble_tileir(bytecode)
            @test occursin("optimization_hints=<sm_100 = {occupancy = 8}>", mlir)
        end

        @testset "both hints" begin
            bytecode = cuTile.emit_tileir(simple_kernel, argtypes;
                                          num_ctas=2, occupancy=4)
            mlir = cuTile.disassemble_tileir(bytecode)
            # Both should appear (order may vary)
            @test occursin("num_cta_in_cga = 2", mlir)
            @test occursin("occupancy = 4", mlir)
            @test occursin("optimization_hints=<sm_100 = {", mlir)
        end

        @testset "no hints" begin
            bytecode = cuTile.emit_tileir(simple_kernel, argtypes)
            mlir = cuTile.disassemble_tileir(bytecode)
            # Should NOT have optimization_hints attribute on entry function
            @test !occursin("optimization_hints", mlir)
        end

        @testset "architecture parameter" begin
            bytecode = cuTile.emit_tileir(simple_kernel, argtypes;
                                          sm_arch="sm_120", num_ctas=4)
            mlir = cuTile.disassemble_tileir(bytecode)
            @test occursin("optimization_hints=<sm_120 = {num_cta_in_cga = 4}>", mlir)
        end
    end

    @testset "Validation" begin
        spec1d = cuTile.ArraySpec{1}(16, true)

        function dummy_kernel(a::cuTile.TileArray{Float32, 1, spec1d})
            return nothing
        end

        argtypes = Tuple{cuTile.TileArray{Float32, 1, spec1d}}

        @testset "num_ctas validation" begin
            # Too small
            @test_throws "num_ctas must be between 1 and 16" begin
                cuTile.emit_tileir(dummy_kernel, argtypes; num_ctas=0)
            end

            # Too large
            @test_throws "num_ctas must be between 1 and 16" begin
                cuTile.emit_tileir(dummy_kernel, argtypes; num_ctas=17)
            end

            # Not power of 2
            @test_throws "num_ctas must be a power of 2" begin
                cuTile.emit_tileir(dummy_kernel, argtypes; num_ctas=3)
            end

            @test_throws "num_ctas must be a power of 2" begin
                cuTile.emit_tileir(dummy_kernel, argtypes; num_ctas=5)
            end

            # Valid values should succeed
            for valid_num_ctas in [1, 2, 4, 8, 16]
                bytecode = cuTile.emit_tileir(dummy_kernel, argtypes;
                                              num_ctas=valid_num_ctas)
                @test !isempty(bytecode)
            end
        end

        @testset "occupancy validation" begin
            # Too small
            @test_throws "occupancy must be between 1 and 32" begin
                cuTile.emit_tileir(dummy_kernel, argtypes; occupancy=0)
            end

            # Too large
            @test_throws "occupancy must be between 1 and 32" begin
                cuTile.emit_tileir(dummy_kernel, argtypes; occupancy=33)
            end

            # Valid boundaries
            bytecode1 = cuTile.emit_tileir(dummy_kernel, argtypes; occupancy=1)
            @test !isempty(bytecode1)

            bytecode32 = cuTile.emit_tileir(dummy_kernel, argtypes; occupancy=32)
            @test !isempty(bytecode32)
        end
    end

    # Integration tests only run if CUDA is available
    if isdefined(Main, :CUDA) && CUDA.functional()
        @testset "Integration" begin
            function vadd_kernel(a::cuTile.TileArray{Float32,1},
                                b::cuTile.TileArray{Float32,1},
                                c::cuTile.TileArray{Float32,1})
                pid = cuTile.bid(1)
                ta = cuTile.load(a, pid, (16,))
                tb = cuTile.load(b, pid, (16,))
                cuTile.store(c, pid, ta + tb)
                return nothing
            end

            n = 1024
            a = CUDA.ones(Float32, n)
            b = CUDA.ones(Float32, n) .* 2
            c = CUDA.zeros(Float32, n)

            @testset "launch with num_ctas" begin
                cuTile.launch(vadd_kernel, 64, a, b, c; num_ctas=2)
                CUDA.synchronize()
                @test Array(c) ≈ ones(Float32, n) .* 3
            end

            @testset "launch with occupancy" begin
                fill!(c, 0.0f0)
                cuTile.launch(vadd_kernel, 64, a, b, c; occupancy=4)
                CUDA.synchronize()
                @test Array(c) ≈ ones(Float32, n) .* 3
            end

            @testset "launch with both hints" begin
                fill!(c, 0.0f0)
                cuTile.launch(vadd_kernel, 64, a, b, c; num_ctas=4, occupancy=8)
                CUDA.synchronize()
                @test Array(c) ≈ ones(Float32, n) .* 3
            end
        end
    end

end
