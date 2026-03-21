# basic execution tests

using CUDA

@testset "compilation cache" begin
    function cached_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        tile = ct.load(a, pid, (16,))
        ct.store(b, pid, tile)
        return
    end

    n = 256
    tile_size = 16
    a = CUDA.rand(Float32, n)
    b = CUDA.zeros(Float32, n)

    # First launch triggers compilation
    ct.launch(cached_kernel, cld(n, tile_size), a, b)
    @test Array(b) ≈ Array(a)

    # Second launch should use cached CuFunction
    a2 = CUDA.rand(Float32, n)
    b2 = CUDA.zeros(Float32, n)
    ct.launch(cached_kernel, cld(n, tile_size), a2, b2)
    @test Array(b2) ≈ Array(a2)
end

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

    # @device_code_tiled with Constant arguments
    function reflect_const_vadd(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                                c::ct.TileArray{Float32,1}, tile_size::Int)
        pid = ct.bid(1)
        tile_a = ct.load(a, pid, (tile_size,))
        tile_b = ct.load(b, pid, (tile_size,))
        ct.store(c, pid, tile_a + tile_b)
        return
    end

    c2 = CUDA.zeros(Float32, n)
    @test @filecheck begin
        @check "entry @reflect_const_vadd"
        @check "load_view"
        @check "addf"
        @check "store_view"
        ct.@device_code_tiled ct.launch(reflect_const_vadd, cld(n, 16), a, b, c2,
                                        ct.Constant(16))
    end
    @test Array(c2) ≈ Array(a) + Array(b)

    # @device_code_tiled with reduce subprogram
    # Regression: compile hook tried to compile the reduce combiner (e.g. +) as a
    # standalone entry, which cuda-tile-translate rejects.
    function reflect_reduce(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        tile = ct.load(a, (pid, 1), (1, 128))
        ct.store(b, pid, sum(tile; dims=2))
        return
    end

    m, k = 64, 128
    a2d = CUDA.rand(Float32, m, k)
    b1d = CUDA.zeros(Float32, m)

    @test @filecheck begin
        @check "entry @reflect_reduce"
        @check "reduce"
        @check "addf"
        buf = IOBuffer()
        ct.@device_code_tiled io=buf ct.launch(reflect_reduce, m, a2d, b1d)
        String(take!(buf))
    end
end

@testset "assert" begin
    @testset "passing assertion with message" begin
        function assert_msg_kernel(a::ct.TileArray{Float32,1}, tile_size::Int)
            bid = ct.bid(1)
            ct.@assert bid > Int32(0) "bid must be positive"
            t = ct.load(a, bid, (tile_size,))
            ct.store(a, bid, t)
            return
        end

        a = CUDA.ones(Float32, 1024)
        ct.launch(assert_msg_kernel, cld(1024, 128), a, ct.Constant(128))
        CUDA.synchronize()
        @test all(Array(a) .== 1.0f0)
    end

    @testset "passing assertion without message" begin
        function assert_nomsg_kernel(a::ct.TileArray{Float32,1}, tile_size::Int)
            bid = ct.bid(1)
            ct.@assert bid > Int32(0)
            t = ct.load(a, bid, (tile_size,))
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

        function assert_fail_kernel(a::ct.TileArray{Float32,1}, tile_size::Int)
            bid = ct.bid(1)
            ct.@assert bid > Int32(999999) "custom assert message"
            t = ct.load(a, bid, (tile_size,))
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

@testset "Constant parameters" begin

@testset "1D with Constant tile size" begin
    function vadd_const_tile(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                             c::ct.TileArray{Float32,1}, tile::Int)
        pid = ct.bid(1)
        tile_a = ct.load(a, pid, (tile,))
        tile_b = ct.load(b, pid, (tile,))
        ct.store(c, pid, tile_a + tile_b)
        return
    end

    n = 1024
    tile_size = 32
    a = CUDA.rand(Float32, n)
    b = CUDA.rand(Float32, n)
    c = CUDA.zeros(Float32, n)

    ct.launch(vadd_const_tile, cld(n, tile_size), a, b, c, ct.Constant(tile_size))

    @test Array(c) ≈ Array(a) + Array(b)
end

@testset "2D with Constant tile sizes" begin
    function madd_const_tiles(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,2},
                              c::ct.TileArray{Float32,2},
                              tx::Int, ty::Int)
        bidx = ct.bid(1)
        bidy = ct.bid(2)
        tile_a = ct.load(a, (bidx, bidy), (tx, ty))
        tile_b = ct.load(b, (bidx, bidy), (tx, ty))
        ct.store(c, (bidx, bidy), tile_a + tile_b)
        return
    end

    m, n = 256, 256
    tile_x, tile_y = 64, 64
    a = CUDA.rand(Float32, m, n)
    b = CUDA.rand(Float32, m, n)
    c = CUDA.zeros(Float32, m, n)

    ct.launch(madd_const_tiles, (cld(m, tile_x), cld(n, tile_y)), a, b, c,
              ct.Constant(tile_x), ct.Constant(tile_y))

    @test Array(c) ≈ Array(a) + Array(b)
end

end

@testset "TileArray auto-conversion" begin
    # Test that CuArrays are automatically converted to TileArray
    function copy_kernel(src::ct.TileArray{Float32,1}, dst::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        tile = ct.load(src, pid, (16,))
        ct.store(dst, pid, tile)
        return
    end


    n = 512
    tile_size = 16
    src = CUDA.rand(Float32, n)
    dst = CUDA.zeros(Float32, n)

    # Pass CuArrays directly - should auto-convert
    ct.launch(copy_kernel, cld(n, tile_size), src, dst)

    @test Array(dst) ≈ Array(src)
end

const _EXEC_TEST_GLOBAL_CONST = Float32(1 / log(2))

@testset "global constant arithmetic" begin
    # Regression test for issue #77: scalar × global constant failed during codegen.
    function global_const_arith_kernel(a::ct.TileArray{Float32,1},
                                       b::ct.TileArray{Float32,1},
                                       scale::Float32)
        pid = ct.bid(1)
        tile = ct.load(a, pid, (16,))
        total_scale = scale * _EXEC_TEST_GLOBAL_CONST
        ct.store(b, pid, tile .* total_scale)
        return
    end

    n = 1024
    tile_size = 16
    scale = 2.5f0
    a = CUDA.rand(Float32, n)
    b = CUDA.zeros(Float32, n)

    ct.launch(global_const_arith_kernel, cld(n, tile_size), a, b, scale)

    @test Array(b) ≈ Array(a) .* (scale * _EXEC_TEST_GLOBAL_CONST)
end

@testset "kernel name with !" begin
    function kernel!()
        return
    end
    ct.launch(kernel!, 1)
end

@testset "non-Constant ghost type argument (nothing)" begin
    function ghost_nothing_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1}, ::Nothing)
        pid = ct.bid(1)
        tile = ct.load(a, pid, (16,))
        ct.store(b, pid, tile)
        return
    end

    n = 256
    a = CUDA.rand(Float32, n)
    b = CUDA.zeros(Float32, n)

    ct.launch(ghost_nothing_kernel, cld(n, 16), a, b, nothing)

    @test Array(b) ≈ Array(a)
end

@testset "non-Constant ghost type argument (Val)" begin
    function ghost_val_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1}, ::Val{n}) where n
        pid = ct.bid(1)
        tile = ct.load(a, pid, (n,))
        ct.store(b, pid, tile)
        return
    end

    n = 256
    a = CUDA.rand(Float32, n)
    b = CUDA.zeros(Float32, n)

    ct.launch(ghost_val_kernel, cld(n, 16), a, b, Val(16))

    @test Array(b) ≈ Array(a)
end

@testset "struct destructuring" begin
    @testset "TileArray + scalar field" begin
        struct ArrayWithScale{T, N, S}
            arr::ct.TileArray{T, N, S}
            scale::Float32
        end

        function scale_kernel(dest::ct.TileArray{Float32,1}, w::ArrayWithScale{Float32,1}, ts)
            bid = ct.bid(1)
            tile = ct.load(w.arr, bid, (ts[1],))
            ct.store(dest, bid, tile .* w.scale)
            return
        end

        A = CUDA.fill(Float32(3), 64)
        B = CUDA.zeros(Float32, 64)
        ct.launch(scale_kernel, 1, ct.TileArray(B),
                  ArrayWithScale(ct.TileArray(A), Float32(2)), ct.Constant((64,)))
        @test all(Array(B) .≈ 6.0f0)
    end

    @testset "two TileArrays in one struct" begin
        struct TwoArrays{T, N, S1, S2}
            a::ct.TileArray{T, N, S1}
            b::ct.TileArray{T, N, S2}
        end

        function add_two_kernel(dest::ct.TileArray{Float32,1}, pair::TwoArrays{Float32,1}, ts)
            bid = ct.bid(1)
            ta = ct.load(pair.a, bid, (ts[1],))
            tb = ct.load(pair.b, bid, (ts[1],))
            ct.store(dest, bid, ta .+ tb)
            return
        end

        X = CUDA.fill(Float32(2), 64)
        Y = CUDA.fill(Float32(3), 64)
        Z = CUDA.zeros(Float32, 64)
        ct.launch(add_two_kernel, 1, ct.TileArray(Z),
                  TwoArrays(ct.TileArray(X), ct.TileArray(Y)), ct.Constant((64,)))
        @test all(Array(Z) .≈ 5.0f0)
    end

    @testset "nested struct (struct inside struct)" begin
        struct InnerLayer{T, N, S}
            arr::ct.TileArray{T, N, S}
            bias::Float32
        end
        struct OuterLayer{T, N, S}
            inner::InnerLayer{T, N, S}
            multiplier::Float32
        end

        function nested_kernel(dest::ct.TileArray{Float32,1}, o::OuterLayer{Float32,1}, ts)
            bid = ct.bid(1)
            tile = ct.load(o.inner.arr, bid, (ts[1],))
            ct.store(dest, bid, (tile .+ o.inner.bias) .* o.multiplier)
            return
        end

        A = CUDA.fill(Float32(1), 64)
        B = CUDA.zeros(Float32, 64)
        ct.launch(nested_kernel, 1, ct.TileArray(B),
                  OuterLayer(InnerLayer(ct.TileArray(A), Float32(2)), Float32(3)), ct.Constant((64,)))
        @test all(Array(B) .≈ 9.0f0)
    end

    @testset "scalar-only struct" begin
        struct ScalarPair; a::Float32; b::Float32; end

        function scalar_struct_kernel(dest::ct.TileArray{Float32,1}, src::ct.TileArray{Float32,1},
                                      sp::ScalarPair, ts)
            bid = ct.bid(1)
            tile = ct.load(src, bid, (ts[1],))
            ct.store(dest, bid, tile .* sp.a .+ sp.b)
            return
        end

        A = CUDA.fill(Float32(4), 64)
        B = CUDA.zeros(Float32, 64)
        ct.launch(scalar_struct_kernel, 1, ct.TileArray(B), ct.TileArray(A),
                  ScalarPair(Float32(2), Float32(1)), ct.Constant((64,)))
        @test all(Array(B) .≈ 9.0f0)
    end

    @testset "heterogeneous tuple in struct" begin
        struct HetTupleWrapper{A, B}; a::A; b::B; end

        function het_tuple_kernel(dest::ct.TileArray{Float32,1,S},
                                  w::HetTupleWrapper{ct.TileArray{Float32,1,S2}, Int32},
                                  ts) where {S, S2}
            bid = ct.bid(1)
            tile = ct.load(w.a, bid, (ts[1],))
            ct.store(dest, bid, tile .+ w.b)
            return
        end

        A = CUDA.ones(Float32, 64)
        B = CUDA.ones(Float32, 64)
        ct.launch(het_tuple_kernel, 1, ct.TileArray(A),
                  HetTupleWrapper(ct.TileArray(B), Int32(5)), ct.Constant((64,)))
        @test all(Array(A) .≈ 6.0f0)
    end
end
