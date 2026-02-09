using CUDA

@testset "invalidations" begin

@testset "redefine kernel" begin
    mod = @eval module $(gensym())
        import cuTile as ct
        function vadd_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1}, c::ct.TileArray{Float32,1})
            pid = ct.bid(1)
            ta = ct.load(a, (pid,), (16,))
            tb = ct.load(b, (pid,), (16,))
            ct.store(c, (pid,), ta + tb)
            return
        end
    end

    a = CUDA.ones(Float32, 1024)
    b = CUDA.ones(Float32, 1024)
    c = CUDA.zeros(Float32, 1024)

    ct.launch(mod.vadd_kernel, 64, a, b, c)
    @test Array(c) ≈ Array(a) + Array(b)

    @eval mod begin
        function vadd_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1}, c::ct.TileArray{Float32,1})
            pid = ct.bid(1)
            ta = ct.load(a, (pid,), (16,))
            tb = ct.load(b, (pid,), (16,))
            ct.store(c, (pid,), ta + tb * 2)
            return
        end
    end

    ct.launch(mod.vadd_kernel, 64, a, b, c)
    @test Array(c) ≈ Array(a) + Array(b) * 2
end

@testset "redefine called function" begin
    mod = @eval module $(gensym())
        import cuTile as ct
        combine(a, b) = a + b
        function vadd_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1}, c::ct.TileArray{Float32,1})
            pid = ct.bid(1)
            ta = ct.load(a, (pid,), (16,))
            tb = ct.load(b, (pid,), (16,))
            ct.store(c, (pid,), combine(ta, tb))
            return
        end
    end

    a = CUDA.ones(Float32, 1024)
    b = CUDA.ones(Float32, 1024)
    c = CUDA.zeros(Float32, 1024)

    ct.launch(mod.vadd_kernel, 64, a, b, c)
    @test Array(c) ≈ Array(a) + Array(b)

    @eval mod combine(a, b) = a + b * 2

    ct.launch(mod.vadd_kernel, 64, a, b, c)
    @test Array(c) ≈ Array(a) + Array(b) * 2
end

@testset "redefine reduce subprogram" begin
    mod = @eval module $(gensym())
        import cuTile as ct
        combine(a, b) = a + b
        function reduce_kernel(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,1})
            pid = ct.bid(1)
            tile = ct.load(a, (pid, 1), (1, 128))
            sums = reduce(combine, tile; dims=2, init=0.0f0)
            ct.store(b, pid, sums)
            return
        end
    end

    m, n = 64, 128
    a = CUDA.ones(Float32, m, n)
    b = CUDA.zeros(Float32, m)

    ct.launch(mod.reduce_kernel, m, a, b)
    @test all(Array(b) .≈ Float32(n))

    # Redefine to max (associative+commutative, tree-order independent)
    @eval mod combine(a, b) = max(a, b)

    ct.launch(mod.reduce_kernel, m, a, b)
    @test all(Array(b) .≈ 1.0f0)
end

@testset "redefine scan subprogram" begin
    mod = @eval module $(gensym())
        import cuTile as ct
        combine(a, b) = a + b
        function scan_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1})
            pid = ct.bid(1)
            tile = ct.load(a, pid, (128,))
            scanned = accumulate(combine, tile; dims=1, init=0.0f0)
            ct.store(b, pid, scanned)
            return
        end
    end

    n = 128
    a = CUDA.ones(Float32, n)
    b = CUDA.zeros(Float32, n)

    ct.launch(mod.scan_kernel, 1, a, b)
    expected = Float32.(cumsum(ones(Float32, n)))
    @test Array(b) ≈ expected

    # Redefine to max (associative+commutative, tree-order independent)
    @eval mod combine(a, b) = max(a, b)

    ct.launch(mod.scan_kernel, 1, a, b)
    # Running max over [1,1,...,1] with init=0 gives [1,1,...,1]
    @test all(Array(b) .≈ 1.0f0)
end

end # invalidations

@testset "reflection macros" begin
    function reflect_vadd(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                          c::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        tile_a = ct.load(a, pid, (16,))
        tile_b = ct.load(b, pid, (16,))
        ct.store(c, pid, tile_a + tile_b)
        return
    end

    n = 1024
    a = CUDA.rand(Float32, n)
    b = CUDA.rand(Float32, n)
    c = CUDA.zeros(Float32, n)

    # @device_code_tiled: check Tile IR output and verify execution
    @test @filecheck begin
        @check "entry @reflect_vadd"
        @check "tile<ptr<f32>>"
        @check "get_tile_block_id"
        @check "load_view"
        @check "addf"
        @check "store_view"
        buf = IOBuffer()
        ct.@device_code_tiled io=buf ct.launch(reflect_vadd, cld(n, 16), a, b, c)
        String(take!(buf))
    end
    @test Array(c) ≈ Array(a) + Array(b)

    # @device_code_structured: check StructuredIRCode output
    @test @filecheck begin
        @check "StructuredIRCode"
        @check "get_tile_block_id"
        @check "load_partition_view"
        @check "addf"
        @check "store_partition_view"
        buf = IOBuffer()
        ct.@device_code_structured io=buf ct.launch(reflect_vadd, cld(n, 16), a, b, c)
        String(take!(buf))
    end

    # @device_code_typed: check typed Julia IR output
    @test @filecheck begin
        @check "// reflect_vadd"
        @check "get_tile_block_id"
        @check "load_partition_view"
        @check "addf"
        @check "store_partition_view"
        buf = IOBuffer()
        ct.@device_code_typed io=buf ct.launch(reflect_vadd, cld(n, 16), a, b, c)
        String(take!(buf))
    end
end

@testset "assert" begin
    @testset "passing assertion with message" begin
        function assert_msg_kernel(a::ct.TileArray{Float32,1}, tile_size::ct.Constant{Int})
            bid = ct.bid(1)
            ct.@assert bid > Int32(0) "bid must be positive"
            t = ct.load(a, bid, (tile_size[],))
            ct.store(a, bid, t)
            return
        end

        a = CUDA.ones(Float32, 1024)
        ct.launch(assert_msg_kernel, cld(1024, 128), a, ct.Constant(128))
        CUDA.synchronize()
        @test all(Array(a) .== 1.0f0)
    end

    @testset "passing assertion without message" begin
        function assert_nomsg_kernel(a::ct.TileArray{Float32,1}, tile_size::ct.Constant{Int})
            bid = ct.bid(1)
            ct.@assert bid > Int32(0)
            t = ct.load(a, bid, (tile_size[],))
            ct.store(a, bid, t)
            return
        end

        a = CUDA.ones(Float32, 1024)
        ct.launch(assert_nomsg_kernel, cld(1024, 128), a, ct.Constant(128))
        CUDA.synchronize()
        @test all(Array(a) .== 1.0f0)
    end

    @testset "failing assertion" begin
        # Failed assertions crash the CUDA context, so we must test in a subprocess
        # (following the same pattern as cuTile Python's test_assert.py)
        script = """
        using CUDA
        import cuTile as ct

        function assert_fail_kernel(a::ct.TileArray{Float32,1}, tile_size::ct.Constant{Int})
            bid = ct.bid(1)
            ct.@assert bid > Int32(999999) "custom assert message"
            t = ct.load(a, bid, (tile_size[],))
            ct.store(a, bid, t)
            return
        end

        a = CUDA.ones(Float32, 1024)
        ct.launch(assert_fail_kernel, cld(1024, 128), a, ct.Constant(128))
        CUDA.synchronize()
        """
        cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) -e $script`
        output = Pipe()
        proc = run(pipeline(ignorestatus(cmd); stdout=output, stderr=output); wait=false)
        close(output.in)
        reader = @async read(output, String)
        wait(proc)
        result = fetch(reader)
        @test proc.exitcode != 0
        @test contains(result, "custom assert message")
    end
end
