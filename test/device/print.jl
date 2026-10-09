# device print tests
#
# synchronizing the context flushes kernel output. synchronizing a stream doesn't when it
# has already finished, as that doesn't involve the driver.

using CUDA

@testset "print constant string" begin
    function print_const_kernel(a::ct.TileArray{Float32,1})
        bid = ct.bid(1)
        tile = ct.load(a, bid, (16,))
        print("hello world\n")
        ct.store(a, bid, tile)
        return
    end

    a = CUDA.ones(Float32, 16)
    @test @filecheck begin
        @check "hello world"
        @cuda backend=cuTile print_const_kernel(a)
        CUDA.synchronize(CUDA.context())
    end
end

@testset "println with tile" begin
    function print_tile_kernel(a::ct.TileArray{Float32,1})
        bid = ct.bid(1)
        tile = ct.load(a, bid, (16,))
        println("tile=", tile)
        ct.store(a, bid, tile)
        return
    end

    a = CUDA.ones(Float32, 16)
    @test @filecheck begin
        @check "tile=["
        @check "1.000000"
        @cuda backend=cuTile print_tile_kernel(a)
        CUDA.synchronize(CUDA.context())
    end
end

@testset "print bid (scalar tile)" begin
    function print_bid_kernel(a::ct.TileArray{Float32,1})
        bid = ct.bid(1)
        tile = ct.load(a, bid, (16,))
        println("bid=", bid)
        ct.store(a, bid, tile)
        return
    end

    a = CUDA.ones(Float32, 16)
    @test @filecheck begin
        @check "bid=1"
        @cuda backend=cuTile print_bid_kernel(a)
        CUDA.synchronize(CUDA.context())
    end
end

@testset "string interpolation" begin
    function interp_kernel(a::ct.TileArray{Float32,1})
        bid = ct.bid(1)
        tile = ct.load(a, bid, (16,))
        println("bid=$bid")
        ct.store(a, bid, tile)
        return
    end

    a = CUDA.ones(Float32, 16)
    @test @filecheck begin
        @check "bid=1"
        @cuda backend=cuTile interp_kernel(a)
        CUDA.synchronize(CUDA.context())
    end
end

@testset "multiple prints" begin
    function multi_print_kernel(a::ct.TileArray{Float32,1})
        bid = ct.bid(1)
        tile = ct.load(a, bid, (16,))
        println("first")
        println("second")
        ct.store(a, bid, tile)
        return
    end

    a = CUDA.ones(Float32, 16)
    @test @filecheck begin
        @check "first"
        @check "second"
        @cuda backend=cuTile multi_print_kernel(a)
        CUDA.synchronize(CUDA.context())
    end
end
