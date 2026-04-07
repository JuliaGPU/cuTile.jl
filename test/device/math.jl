# math primitives

using CUDA


@testset "bitwise operations" begin

@testset "andi (bitwise AND)" begin
    function bitwise_and_kernel(a::ct.TileArray{Int32,1}, b::ct.TileArray{Int32,1},
                                c::ct.TileArray{Int32,1})
        pid = ct.bid(1)
        ta = ct.load(a, pid, (16,))
        tb = ct.load(b, pid, (16,))
        ct.store(c, pid, map(&, ta, tb))
        return
    end

    n = 1024
    tile_size = 16
    a = CuArray(rand(Int32(0):Int32(0x7fff_ffff), n))
    b = CuArray(rand(Int32(0):Int32(0x7fff_ffff), n))
    c = CUDA.zeros(Int32, n)

    ct.launch(bitwise_and_kernel, cld(n, tile_size), a, b, c)

    @test Array(c) == Array(a) .& Array(b)
end

@testset "ori (bitwise OR)" begin
    function bitwise_or_kernel(a::ct.TileArray{Int32,1}, b::ct.TileArray{Int32,1},
                               c::ct.TileArray{Int32,1})
        pid = ct.bid(1)
        ta = ct.load(a, pid, (16,))
        tb = ct.load(b, pid, (16,))
        ct.store(c, pid, map(|, ta, tb))
        return
    end

    n = 1024
    tile_size = 16
    a = CuArray(rand(Int32(0):Int32(0x7fff_ffff), n))
    b = CuArray(rand(Int32(0):Int32(0x7fff_ffff), n))
    c = CUDA.zeros(Int32, n)

    ct.launch(bitwise_or_kernel, cld(n, tile_size), a, b, c)

    @test Array(c) == Array(a) .| Array(b)
end

@testset "xori (bitwise XOR)" begin
    function bitwise_xor_kernel(a::ct.TileArray{Int32,1}, b::ct.TileArray{Int32,1},
                                c::ct.TileArray{Int32,1})
        pid = ct.bid(1)
        ta = ct.load(a, pid, (16,))
        tb = ct.load(b, pid, (16,))
        ct.store(c, pid, map(xor, ta, tb))
        return
    end

    n = 1024
    tile_size = 16
    a = CuArray(rand(Int32(0):Int32(0x7fff_ffff), n))
    b = CuArray(rand(Int32(0):Int32(0x7fff_ffff), n))
    c = CUDA.zeros(Int32, n)

    ct.launch(bitwise_xor_kernel, cld(n, tile_size), a, b, c)

    @test Array(c) == Array(a) .⊻ Array(b)
end

@testset "shli (shift left)" begin
    function shift_left_kernel(a::ct.TileArray{Int32,1}, b::ct.TileArray{Int32,1})
        pid = ct.bid(1)
        tile = ct.load(a, pid, (16,))
        ct.store(b, pid, map(x -> x << Int32(4), tile))
        return
    end

    n = 1024
    tile_size = 16
    a = CuArray(rand(Int32(0):Int32(0x0fff_ffff), n))
    b = CUDA.zeros(Int32, n)

    ct.launch(shift_left_kernel, cld(n, tile_size), a, b)

    @test Array(b) == Array(a) .<< Int32(4)
end

@testset "shri (shift right)" begin
    function shift_right_kernel(a::ct.TileArray{Int32,1}, b::ct.TileArray{Int32,1})
        pid = ct.bid(1)
        tile = ct.load(a, pid, (16,))
        ct.store(b, pid, map(x -> x >> Int32(8), tile))
        return
    end

    n = 1024
    tile_size = 16
    a = CuArray(rand(Int32(0):Int32(0x7fff_ffff), n))
    b = CUDA.zeros(Int32, n)

    ct.launch(shift_right_kernel, cld(n, tile_size), a, b)

    @test Array(b) == Array(a) .>> Int32(8)
end

@testset "combined bitwise ops" begin
    # (a & b) | (a ^ b) \u2014 exercises all three ops in a single kernel
    function combined_bitwise_kernel(a::ct.TileArray{Int32,1}, b::ct.TileArray{Int32,1},
                                     c::ct.TileArray{Int32,1})
        pid = ct.bid(1)
        ta = ct.load(a, pid, (16,))
        tb = ct.load(b, pid, (16,))
        ct.store(c, pid, map(|, map(&, ta, tb), map(xor, ta, tb)))
        return
    end

    n = 1024
    tile_size = 16
    a = CuArray(rand(Int32(0):Int32(0x7fff_ffff), n))
    b = CuArray(rand(Int32(0):Int32(0x7fff_ffff), n))
    c = CUDA.zeros(Int32, n)

    ct.launch(combined_bitwise_kernel, cld(n, tile_size), a, b, c)

    @test Array(c) == (Array(a) .& Array(b)) .| (Array(a) .⊻ Array(b))
end

@testset "bitwise NOT (~)" begin
    function bitwise_not_kernel(a::ct.TileArray{Int32,1}, b::ct.TileArray{Int32,1})
        pid = ct.bid(1)
        tile = ct.load(a, pid, (16,))
        ct.store(b, pid, map(~, tile))
        return
    end

    n = 1024
    tile_size = 16
    a = CuArray(rand(Int32(0):Int32(0x7fff_ffff), n))
    b = CUDA.zeros(Int32, n)

    ct.launch(bitwise_not_kernel, cld(n, tile_size), a, b)

    @test Array(b) == .~Array(a)
end

end


@testset "isnan" begin
    function isnan_kernel(a::ct.TileArray{Float32,1}, out::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        ta = ct.load(a, pid, (16,))
        ct.store(out, pid, ct.where(isnan.(ta), 1.0f0, 0.0f0))
        return
    end

    n = 1024
    a = CUDA.rand(Float32, n)
    CUDA.@allowscalar a[1:16:end] .= NaN32
    out = CUDA.zeros(Float32, n)

    ct.launch(isnan_kernel, cld(n, 16), a, out)

    @test Array(out) == Float32.(isnan.(Array(a)))
end
