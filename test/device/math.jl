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

    @cuda backend=cuTile blocks=cld(n, tile_size) bitwise_and_kernel(a, b, c)

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

    @cuda backend=cuTile blocks=cld(n, tile_size) bitwise_or_kernel(a, b, c)

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

    @cuda backend=cuTile blocks=cld(n, tile_size) bitwise_xor_kernel(a, b, c)

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

    @cuda backend=cuTile blocks=cld(n, tile_size) shift_left_kernel(a, b)

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

    @cuda backend=cuTile blocks=cld(n, tile_size) shift_right_kernel(a, b)

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

    @cuda backend=cuTile blocks=cld(n, tile_size) combined_bitwise_kernel(a, b, c)

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

    @cuda backend=cuTile blocks=cld(n, tile_size) bitwise_not_kernel(a, b)

    @test Array(b) == .~Array(a)
end

end


@testset "isnan" begin
    function isnan_kernel(a::ct.TileArray{Float32,1}, out::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        ta = ct.load(a, pid, (16,))
        ct.store(out, pid, ifelse.(isnan.(ta), 1.0f0, 0.0f0))
        return
    end

    n = 1024
    a = CUDA.rand(Float32, n)
    CUDA.@allowscalar a[1:16:end] .= NaN32
    out = CUDA.zeros(Float32, n)

    @cuda backend=cuTile blocks=cld(n, 16) isnan_kernel(a, out)

    @test Array(out) == Float32.(isnan.(Array(a)))
end

# `x ^ n` with a runtime integer exponent. Used to fail codegen outright: the
# overlay branched on the exponent, which is control flow once it is a tile.
@testset "float ^ integer exponent" begin
    function pow_int_kernel(a::ct.TileArray{Float32,1}, e::ct.TileArray{Int32,1},
                            out::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        ct.store(out, pid, ct.load(a, pid, (16,)) .^ ct.load(e, pid, (16,)))
        return
    end

    # Negative bases, signed zero, infinities and NaN — Base defines all of
    # these for an integer exponent, including `x^0 == 1` for any `x`.
    bases = Float32[2, -2, 0.5, -0.5, 0, -0.0, 1, -1, 3, -3, Inf, -Inf, NaN, 7, -7, 1f10]
    exps = Int32[0, 1, 2, 3, -1, -2, 5, -3, 0, 4, 0, 3, 0, 7, 7, 3]

    a = CuArray(bases)
    e = CuArray(exps)
    out = CUDA.zeros(Float32, 16)

    @cuda backend=cuTile pow_int_kernel(a, e, out)

    got = Array(out)
    want = bases .^ exps
    @test all(got[i] === want[i] || (isnan(got[i]) && isnan(want[i])) ||
              isapprox(got[i], want[i]; rtol=1f-5) for i in eachindex(got))
end

@testset "float ^ Int64 exponent" begin
    # Int64 is Julia's default integer, so it is the common case here.
    function pow_int64_kernel(a::ct.TileArray{Float32,1}, e::ct.TileArray{Int64,1},
                              out::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        ct.store(out, pid, ct.load(a, pid, (16,)) .^ ct.load(e, pid, (16,)))
        return
    end

    n = 256
    a = CUDA.rand(Float32, n) .+ 0.5f0
    e = CuArray(rand(-4:8, n))
    out = CUDA.zeros(Float32, n)

    @cuda backend=cuTile blocks=cld(n, 16) pow_int64_kernel(a, e, out)

    @test Array(out) ≈ Array(a) .^ Array(e) rtol=1f-4
end

# fpowi is cheaper than converting and calling pow, but drifts for large
# exponents, so `^` does not use it; check the intrinsic itself still works.
@testset "Intrinsics.powi" begin
    function powi_kernel(a::ct.TileArray{Float32,1}, e::ct.TileArray{Int32,1},
                         out::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        ct.store(out, pid, ct.Intrinsics.powi(ct.load(a, pid, (16,)),
                                              ct.load(e, pid, (16,))))
        return
    end

    n = 256
    a = CUDA.rand(Float32, n) .+ 0.5f0
    e = CuArray(Int32.(rand(-4:8, n)))
    out = CUDA.zeros(Float32, n)

    @cuda backend=cuTile blocks=cld(n, 16) powi_kernel(a, e, out)

    @test Array(out) ≈ Array(a) .^ Array(e) rtol=1f-3
end
