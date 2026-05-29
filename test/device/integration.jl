# more comprehensive integration tests

using CUDA

@testset "basic matmul" begin
    function matmul_kernel(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,2},
                           c::ct.TileArray{Float32,2})
        bidx = ct.bid(1)
        bidy = ct.bid(2)
        # Load tiles: a is (M, K), b is (K, N)
        tile_a = ct.load(a, (bidx, 1), (32, 16))
        tile_b = ct.load(b, (1, bidy), (16, 32))
        # matmul: c = a @ b (using * operator)
        result = tile_a * tile_b
        ct.store(c, (bidx, bidy), result)
        return
    end

    M, K, N = 64, 16, 64
    a = CUDA.rand(Float32, M, K)
    b = CUDA.rand(Float32, K, N)
    c = CUDA.zeros(Float32, M, N)

    grid_x = cld(M, 32)
    grid_y = cld(N, 32)
    @cuda backend=cuTile blocks=(grid_x, grid_y, 1) matmul_kernel(a, b, c)

    # Verify against CPU reference
    a_cpu = Array(a)
    b_cpu = Array(b)
    c_cpu = Array(c)
    c_ref = a_cpu * b_cpu

    @test c_cpu ≈ c_ref
end

@testset "i8/u8 matmul (mmai)" begin
    # i8/u8 × i8/u8 → i32 lowers to `cuda_tile.mmai`. The accumulator must be
    # Int32 (the `*` operator would pick the input dtype as acc and fail), so we
    # use `muladd` with an explicit i32 acc. Per-operand signedness is derived
    # from the Julia element type, so we feed values that differ under signed vs
    # unsigned interpretation (negatives, and magnitudes > 127) to confirm each
    # operand is interpreted correctly. K = 16, |product| ≤ 255² · 16 ≈ 1.0e6,
    # well within Int32, so the result is exact.
    function mmai(a::ct.TileArray{T1,2}, b::ct.TileArray{T2,2}, c::ct.TileArray{Int32,2}) where {T1,T2}
        ta = ct.load(a, (1, 1), (16, 16))
        tb = ct.load(b, (1, 1), (16, 16))
        tc = muladd(ta, tb, zeros(Int32, (16, 16)))
        ct.store(c, (1, 1), tc)
        return
    end

    M = K = N = 16
    @testset "signed × signed" begin
        a = rand(Int8(-128):Int8(127), M, K); b = rand(Int8(-128):Int8(127), K, N)
        c = CUDA.zeros(Int32, M, N)
        @cuda backend=cuTile blocks=1 mmai(CuArray(a), CuArray(b), c)
        @test Array(c) == Int32.(a) * Int32.(b)
    end
    @testset "unsigned × unsigned" begin
        a = rand(UInt8(0):UInt8(255), M, K); b = rand(UInt8(0):UInt8(255), K, N)
        c = CUDA.zeros(Int32, M, N)
        @cuda backend=cuTile blocks=1 mmai(CuArray(a), CuArray(b), c)
        @test Array(c) == Int32.(a) * Int32.(b)
    end
    @testset "unsigned × signed" begin
        a = rand(UInt8(0):UInt8(255), M, K); b = rand(Int8(-128):Int8(127), K, N)
        c = CUDA.zeros(Int32, M, N)
        @cuda backend=cuTile blocks=1 mmai(CuArray(a), CuArray(b), c)
        @test Array(c) == Int32.(a) * Int32.(b)
    end
end
