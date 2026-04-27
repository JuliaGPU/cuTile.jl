# tests for different data types

using CUDA

@testset "Float64" begin
    function vadd_f64(a::ct.TileArray{Float64,1}, b::ct.TileArray{Float64,1},
                      c::ct.TileArray{Float64,1})
        pid = ct.bid(1)
        tile_a = ct.load(a, pid, (16,))
        tile_b = ct.load(b, pid, (16,))
        ct.store(c, pid, tile_a + tile_b)
        return
    end

    n = 1024
    tile_size = 16
    a = CUDA.rand(Float64, n)
    b = CUDA.rand(Float64, n)
    c = CUDA.zeros(Float64, n)

    ct.launch(vadd_f64, cld(n, tile_size), a, b, c)

    @test Array(c) ≈ Array(a) + Array(b)
end

@testset "Float16" begin
    function vadd_f16(a::ct.TileArray{Float16,1}, b::ct.TileArray{Float16,1},
                      c::ct.TileArray{Float16,1})
        pid = ct.bid(1)
        tile_a = ct.load(a, pid, (16,))
        tile_b = ct.load(b, pid, (16,))
        ct.store(c, pid, tile_a + tile_b)
        return
    end

    n = 1024
    tile_size = 16
    a = CUDA.rand(Float16, n)
    b = CUDA.rand(Float16, n)
    c = CUDA.zeros(Float16, n)

    ct.launch(vadd_f16, cld(n, tile_size), a, b, c)

    @test Array(c) ≈ Array(a) + Array(b)
end

@testset "BFloat16" begin
    function vadd_bf16(a::ct.TileArray{ct.BFloat16,1}, b::ct.TileArray{ct.BFloat16,1},
                      c::ct.TileArray{ct.BFloat16,1})
        pid = ct.bid(1)
        tile_a = ct.load(a, pid, (16,))
        tile_b = ct.load(b, pid, (16,))
        ct.store(c, pid, tile_a + tile_b)
        return
    end

    n = 1024
    tile_size = 16
    a = CUDA.rand(ct.BFloat16, n)
    b = CUDA.rand(ct.BFloat16, n)
    c = CUDA.zeros(ct.BFloat16, n)

    ct.launch(vadd_bf16, cld(n, tile_size), a, b, c)

    @test Array(c) ≈ Array(a) + Array(b)

    # Broadcast goes through BFloat16s' `-(::T,::T) = Base.sub_float(x, y)`,
    # which Julia inlines as a literal IntrinsicFunction in args[1] (rather
    # than a GlobalRef like the IEEEFloat path). Exercises canonicalize's
    # intrinsic lowering on that callee form.
    function vsub_bf16_bcast(a::ct.TileArray{ct.BFloat16,1}, b::ct.TileArray{ct.BFloat16,1},
                             c::ct.TileArray{ct.BFloat16,1})
        pid = ct.bid(1)
        tile_a = ct.load(a, pid, (16,))
        tile_b = ct.load(b, pid, (16,))
        ct.store(c, pid, tile_a .- tile_b)
        return
    end

    d = CUDA.zeros(ct.BFloat16, n)
    ct.launch(vsub_bf16_bcast, cld(n, tile_size), a, b, d)

    @test Array(d) ≈ Array(a) .- Array(b)
end
