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

    @cuda backend=cuTile blocks=cld(n, tile_size) vadd_f64(a, b, c)

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

    @cuda backend=cuTile blocks=cld(n, tile_size) vadd_f16(a, b, c)

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

    @cuda backend=cuTile blocks=cld(n, tile_size) vadd_bf16(a, b, c)

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
    @cuda backend=cuTile blocks=cld(n, tile_size) vsub_bf16_bcast(a, b, d)

    @test Array(d) ≈ Array(a) .- Array(b)
end

function round_f64_f32(a::ct.TileArray{Float64,1}, b::ct.TileArray{Float32,1},
                       mode::Base.Rounding.RoundingMode)
    tile = ct.load(a, ct.bid(1), (2,))
    ct.store(b, ct.bid(1), Float32.(tile, mode))
    return
end

function round_f32_tf32(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1})
    tile = ct.load(a, ct.bid(1), (2,))
    ct.store(b, ct.bid(1), Float32.(ct.TFloat32.(tile, RoundNearestTiesAway)))
    return
end

function round_scalar_f64_f32(x::Float64, b::ct.TileArray{Float32,1})
    ct.store(b, Int32(1), Float32(x, RoundDown))
    return
end

@testset "conversion rounding" begin
    low = 1.0f0
    high = nextfloat(low)
    value = Float64(low) + Float64(high - low) * 0.6
    a = CuArray([-value, value])
    b = CUDA.zeros(Float32, 2)

    # Round-to-nearest is the default mode, supported on every bytecode version.
    @cuda backend=cuTile round_f64_f32(a, b, ct.Constant(RoundNearest))
    @test Array(b) == Float32[-high, high]

    # The directed modes and ties-away need v13.4.
    if ct.bytecode_version() >= v"13.4"
        for (mode, expected) in ((RoundDown, Float32[-high, low]),
                                 (RoundUp, Float32[-low, high]),
                                 (RoundToZero, Float32[-low, low]))
            @cuda backend=cuTile round_f64_f32(a, b, ct.Constant(mode))
            @test Array(b) == expected
        end

        tie = Float32(1 + 2.0^-11)
        @cuda backend=cuTile round_f32_tf32(CuArray([-tie, tie]), b)
        @test Array(b) == Float32[-(1 + 2.0^-10), 1 + 2.0^-10]

        scalar = CUDA.zeros(Float32, 1)
        @cuda backend=cuTile round_scalar_f64_f32(value, scalar)
        @test Array(scalar) == Float32[low]
    end
end
