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

@testset "not_int on scalars" begin
    # `!` on a Bool and `~` on an integer both lower through `not_int`, which
    # needs a per-type all-ones constant — so unlike the binary ops above, its
    # rules are type-directed and have to cover the scalar form explicitly.
    function scalar_not_kernel(a::ct.TileArray{Int32,1}, b::ct.TileArray{Int32,1})
        pid = ct.bid(1)
        flag = pid > Int32(1)
        value = a[pid]
        ct.store(b, pid, (!flag ? Int32(1) : Int32(0)) + ~value)
        return
    end

    n = 64
    a = CuArray(rand(Int32(0):Int32(0x7fff_ffff), n))
    b = CUDA.zeros(Int32, n)

    @cuda backend=cuTile blocks=n scalar_not_kernel(a, b)

    expected = (.~Array(a)) .+ Int32[i == 1 for i in 1:n]
    @test Array(b) == expected
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

@testset "float ^ integer exponent" begin
    function pow_int_kernel(a::ct.TileArray{Float32,1}, e::ct.TileArray{Int32,1},
                            out::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        ct.store(out, pid, ct.load(a, pid, (16,)) .^ ct.load(e, pid, (16,)))
        return
    end

    bases = Float32[2, -2, 0.5, -0.5, 0, -0.0, 1, -1, 3, -3, Inf, -Inf, NaN, 7, -7, -1]
    exps = Int32[0, 1, 2, 3, -1, -2, 5, -3, 0, 4, 0, 3, 0, 7, 7, 16_777_217]

    a = CuArray(bases)
    e = CuArray(exps)
    out = CUDA.zeros(Float32, 16)

    @cuda backend=cuTile pow_int_kernel(a, e, out)

    got = Array(out)
    want = bases .^ exps
    @test all(got[i] === want[i] || (isnan(got[i]) && isnan(want[i])) ||
              isapprox(got[i], want[i]; rtol=1f-5) for i in firstindex(got):lastindex(got)-1)
    @test got[end] === -1f0
end

@testset "float ^ Int64 exponent" begin
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

if cuTile.bytecode_version() >= v"13.4"
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
end

@testset "Base math functions over tiles" begin
    for (name, f, gen) in [
        # These branch per element. Inputs stay within each domain, outside of
        # which Base throws.
        ("log1p",    log1p,             () -> 4 .* rand(Float32, 256) .- 0.999f0),
        ("expm1",    expm1,             () -> 4 .* randn(Float32, 256)),
        ("asin",     asin,              () -> 2 .* rand(Float32, 256) .- 1),
        ("acos",     acos,              () -> 2 .* rand(Float32, 256) .- 1),
        ("hypot",    x -> hypot(x, 2x), () -> randn(Float32, 256)),
        ("atan",     atan,              () -> 4 .* randn(Float32, 256)),
        ("copysign", x -> copysign(2f0, x), () -> randn(Float32, 256)),
        ("muladd",   x -> muladd(x, x, 1f0), () -> randn(Float32, 256)),
        ("Float64 muladd", x -> Float32(muladd(Float64(x), 2.0, 1.0)), () -> randn(Float32, 256)),
        ("isinf",    x -> Float32(isinf(x)),
                     () -> [randn(Float32, 252); Inf32; -Inf32; NaN32; floatmax(Float32)]),
    ]
        @testset "$name" begin
            function base_math_kernel(out::ct.TileArray{Float32,1}, x::ct.TileArray{Float32,1}, f)
                ct.store(out, 1, f.(ct.load(x, 1, (256,))))
                return
            end
            x = gen()
            out = CUDA.zeros(Float32, 256)
            @cuda backend=cuTile base_math_kernel(out, CuArray(x), f)
            @test Array(out) ≈ f.(x) rtol=1f-5
        end
    end

    # Edge cases: tiny arguments, the boundaries of the argument reduction, -1,
    # overflow, and special values. Results are within 1 ulp of the correctly
    # rounded value.
    function edge_kernel(out::ct.TileArray{Float32,2}, x::ct.TileArray{Float32,2})
        ct.store(out, (1, 1), log1p.(ct.load(x, (1, 1), (32, 1))))
        ct.store(out, (1, 2), expm1.(ct.load(x, (1, 2), (32, 1))))
        return
    end
    small = Float32[0, -0f0, 1f-30, -1f-30, 1f-10, -1f-10, 1f-7, -1f-7, 3f-8, -3f-8, 1f-4, -1f-4]
    log1p_x = vcat(small, Float32[0.4142, 0.4143, -0.2929, -0.293, -0.5, -0.9999, -1, 0.5,
                                  1, 10, 1f6, 1f30, floatmax(Float32), Inf, -2, NaN, 0.3, 3, 2, 0.7])
    expm1_x = vcat(small, Float32[0.34, 0.35, -0.35, 1.03, 1.04, -1.04, 18, 19, -18, -19,
                                  88.72, 88.7228, 88.723, 100, -100, Inf, -Inf, NaN, 5, -5])
    out = CUDA.zeros(Float32, 32, 2)
    @cuda backend=cuTile edge_kernel(out, CuArray([log1p_x expm1_x]))
    O = Array(out)
    rounded(f, x) = f === log1p && x < -1 ? NaN32 : Float32(f(Float64(x)))
    @test all(isapprox.(O[:, 1], rounded.(log1p, log1p_x); rtol=eps(Float32), nans=true))
    @test all(isapprox.(O[:, 2], rounded.(expm1, expm1_x); rtol=eps(Float32), nans=true))

    # Float16 rounds the Float32 result.
    function half_kernel(out::ct.TileArray{Float16,2}, x::ct.TileArray{Float16,2})
        ct.store(out, (1, 1), log1p.(ct.load(x, (1, 1), (256, 1))))
        ct.store(out, (1, 2), expm1.(ct.load(x, (1, 2), (256, 1))))
        return
    end
    xh = Float16.([rand(Float32, 256) .* 4 .- 0.99f0 randn(Float32, 256) .* 4])
    outh = CUDA.zeros(Float16, 256, 2)
    @cuda backend=cuTile half_kernel(outh, CuArray(xh))
    @test Array(outh) ≈ [log1p.(xh[:, 1]) expm1.(xh[:, 2])] rtol=eps(Float16)
end
