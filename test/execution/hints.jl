using CUDA

@testset "Entry Hints" begin

@testset "launch with num_ctas" begin
    function vadd_kernel_num_ctas(a::ct.TileArray{Float32,1},
                        b::ct.TileArray{Float32,1},
                        c::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        ta = ct.load(a, pid, (16,))
        tb = ct.load(b, pid, (16,))
        ct.store(c, pid, ta + tb)
        return nothing
    end

    n = 1024
    a = CUDA.ones(Float32, n)
    b = CUDA.ones(Float32, n) .* 2
    c = CUDA.zeros(Float32, n)

    ct.launch(vadd_kernel_num_ctas, 64, a, b, c; num_ctas=2)

    @test Array(c) ≈ ones(Float32, n) .* 3
end

@testset "launch with occupancy" begin
    function vadd_kernel_occupancy(a::ct.TileArray{Float32,1},
                        b::ct.TileArray{Float32,1},
                        c::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        ta = ct.load(a, pid, (16,))
        tb = ct.load(b, pid, (16,))
        ct.store(c, pid, ta + tb)
        return nothing
    end

    n = 1024
    a = CUDA.ones(Float32, n)
    b = CUDA.ones(Float32, n) .* 2
    c = CUDA.zeros(Float32, n)

    ct.launch(vadd_kernel_occupancy, 64, a, b, c; occupancy=4)

    @test Array(c) ≈ ones(Float32, n) .* 3
end

@testset "launch with both hints" begin
    function vadd_kernel_both_hints(a::ct.TileArray{Float32,1},
                        b::ct.TileArray{Float32,1},
                        c::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        ta = ct.load(a, pid, (16,))
        tb = ct.load(b, pid, (16,))
        ct.store(c, pid, ta + tb)
        return nothing
    end

    n = 1024
    a = CUDA.ones(Float32, n)
    b = CUDA.ones(Float32, n) .* 2
    c = CUDA.zeros(Float32, n)

    ct.launch(vadd_kernel_both_hints, 64, a, b, c; num_ctas=4, occupancy=8)

    @test Array(c) ≈ ones(Float32, n) .* 3
end

end


@testset "Load / Store Optimization Hints" begin

@testset "load with latency hint" begin
    function vadd_with_load_latency(a::ct.TileArray{Float32,1},
                                    b::ct.TileArray{Float32,1},
                                    c::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        ta = ct.load(a, pid, (16,); latency=5)
        tb = ct.load(b, pid, (16,); latency=3)
        ct.store(c, pid, ta + tb)
        return nothing
    end

    n = 1024
    a = CUDA.ones(Float32, n)
    b = CUDA.ones(Float32, n) .* 2
    c = CUDA.zeros(Float32, n)

    ct.launch(vadd_with_load_latency, 64, a, b, c)

    @test Array(c) ≈ ones(Float32, n) .* 3
end

@testset "load with allow_tma=false" begin
    function vadd_no_tma(a::ct.TileArray{Float32,1},
                         b::ct.TileArray{Float32,1},
                         c::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        ta = ct.load(a, pid, (16,); allow_tma=false)
        tb = ct.load(b, pid, (16,); allow_tma=false)
        ct.store(c, pid, ta + tb)
        return nothing
    end

    n = 1024
    a = CUDA.ones(Float32, n)
    b = CUDA.ones(Float32, n) .* 2
    c = CUDA.zeros(Float32, n)

    ct.launch(vadd_no_tma, 64, a, b, c)

    @test Array(c) ≈ ones(Float32, n) .* 3
end

@testset "load with both hints" begin
    function vadd_both_load_hints(a::ct.TileArray{Float32,1},
                                  b::ct.TileArray{Float32,1},
                                  c::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        ta = ct.load(a, pid, (16,); latency=7, allow_tma=false)
        tb = ct.load(b, pid, (16,); latency=4, allow_tma=true)
        ct.store(c, pid, ta + tb)
        return nothing
    end

    n = 1024
    a = CUDA.ones(Float32, n)
    b = CUDA.ones(Float32, n) .* 2
    c = CUDA.zeros(Float32, n)

    ct.launch(vadd_both_load_hints, 64, a, b, c)

    @test Array(c) ≈ ones(Float32, n) .* 3
end

@testset "store with latency hint" begin
    function copy_with_store_latency(a::ct.TileArray{Float32,1},
                                     b::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        ta = ct.load(a, pid, (16,))
        ct.store(b, pid, ta; latency=2)
        return nothing
    end

    n = 1024
    a = CUDA.rand(Float32, n)
    b = CUDA.zeros(Float32, n)

    ct.launch(copy_with_store_latency, 64, a, b)

    @test Array(b) ≈ Array(a)
end

@testset "store with allow_tma=false" begin
    function copy_no_tma_store(a::ct.TileArray{Float32,1},
                               b::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        ta = ct.load(a, pid, (16,))
        ct.store(b, pid, ta; allow_tma=false)
        return nothing
    end

    n = 1024
    a = CUDA.rand(Float32, n)
    b = CUDA.zeros(Float32, n)

    ct.launch(copy_no_tma_store, 64, a, b)

    @test Array(b) ≈ Array(a)
end

@testset "different hints on load and store" begin
    function vadd_mixed_hints(a::ct.TileArray{Float32,1},
                              b::ct.TileArray{Float32,1},
                              c::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        # Load with high latency, no TMA
        ta = ct.load(a, pid, (16,); latency=8, allow_tma=false)
        tb = ct.load(b, pid, (16,); latency=6, allow_tma=false)
        # Store with low latency, allow TMA
        ct.store(c, pid, ta + tb; latency=2, allow_tma=true)
        return nothing
    end

    n = 1024
    a = CUDA.ones(Float32, n)
    b = CUDA.ones(Float32, n) .* 2
    c = CUDA.zeros(Float32, n)

    ct.launch(vadd_mixed_hints, 64, a, b, c)

    @test Array(c) ≈ ones(Float32, n) .* 3
end

# Pointer-based operations (gather/scatter) with latency hints
@testset "gather with latency hint" begin
    function gather_with_latency(a::ct.TileArray{Float32,1},
                                 b::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        base = (pid - 1) * 16
        indices = base .+ ct.arange((16,), Int32)
        tile = ct.gather(a, indices; latency=5)
        ct.store(b, pid, tile)
        return nothing
    end

    n = 1024
    a = CUDA.rand(Float32, n)
    b = CUDA.zeros(Float32, n)

    ct.launch(gather_with_latency, 64, a, b)

    @test Array(b) ≈ Array(a)
end

@testset "scatter with latency hint" begin
    function scatter_with_latency(a::ct.TileArray{Float32,1},
                                  b::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        tile = ct.load(a, pid, (16,))
        base = (pid - 1) * 16
        indices = base .+ ct.arange((16,), Int32)
        ct.scatter(b, indices, tile; latency=3)
        return nothing
    end

    n = 1024
    a = CUDA.rand(Float32, n)
    b = CUDA.zeros(Float32, n)

    ct.launch(scatter_with_latency, 64, a, b)

    @test Array(b) ≈ Array(a)
end

end
