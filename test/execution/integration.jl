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
    ct.launch(matmul_kernel, (grid_x, grid_y, 1), a, b, c)

    # Verify against CPU reference
    a_cpu = Array(a)
    b_cpu = Array(b)
    c_cpu = Array(c)
    c_ref = a_cpu * b_cpu

    @test c_cpu ≈ c_ref
end
