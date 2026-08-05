using CUDA

@testset "Tiled mapreduce" begin
    @testset "sum 1D" begin
        A = CUDA.rand(Float32, 1024)
        @test sum(ct.Tiled(A)) ≈ sum(A)
    end

    @testset "sum 1D dims=1" begin
        A = CUDA.rand(Float32, 1024)
        R = sum(ct.Tiled(A); dims=1)
        @test R isa CuArray
        @test Array(R) ≈ Array(sum(A; dims=1))
    end

    @testset "sum 2D dims=1" begin
        A = CUDA.rand(Float32, 128, 256)
        R = sum(ct.Tiled(A); dims=1)
        @test size(R) == (1, 256)
        @test Array(R) ≈ Array(sum(A; dims=1))
    end

    @testset "sum 2D dims=2" begin
        A = CUDA.rand(Float32, 128, 256)
        R = sum(ct.Tiled(A); dims=2)
        @test size(R) == (128, 1)
        @test Array(R) ≈ Array(sum(A; dims=2))
    end

    @testset "sum 2D dims=(1,2)" begin
        A = CUDA.rand(Float32, 128, 256)
        R = sum(ct.Tiled(A); dims=(1, 2))
        @test size(R) == (1, 1)
        @test Array(R) ≈ Array(sum(A; dims=(1, 2)))
    end

    @testset "sum 2D dims=(2,)" begin
        A = CUDA.rand(Float32, 128, 256)
        R = sum(ct.Tiled(A); dims=(2,))
        @test size(R) == (128, 1)
        @test Array(R) ≈ Array(sum(A; dims=2))
    end

    @testset "sum 2D dims=()" begin
        A = CUDA.rand(Float32, 128, 256)
        R = sum(ct.Tiled(A); dims=())
        @test size(R) == (128, 256)
        @test Array(R) ≈ Array(A)
    end

    @testset "dims beyond ndims (Base parity)" begin
        A = CUDA.rand(Float32, 64, 32)
        R = sum(ct.Tiled(A); dims=3)
        @test size(R) == (64, 32)
        @test Array(R) ≈ Array(A)
        R = sum(ct.Tiled(A); dims=(2, 3))
        @test size(R) == (64, 1)
        @test Array(R) ≈ Array(sum(A; dims=2))
        @test_throws ArgumentError sum(ct.Tiled(A); dims=0)
    end

    @testset "maximum 3D dims=(1,3)" begin
        A = CUDA.rand(Float32, 32, 16, 8)
        R = maximum(ct.Tiled(A); dims=(1, 3))
        @test size(R) == (1, 16, 1)
        @test Array(R) ≈ Array(maximum(A; dims=(1, 3)))
    end

    @testset "sum 2D full" begin
        A = CUDA.rand(Float32, 128, 256)
        @test sum(ct.Tiled(A)) ≈ sum(A)
    end

    @testset "maximum 1D" begin
        A = CUDA.rand(Float32, 1024)
        @test maximum(ct.Tiled(A)) ≈ maximum(A)
    end

    @testset "minimum 1D" begin
        A = CUDA.rand(Float32, 1024)
        @test minimum(ct.Tiled(A)) ≈ minimum(A)
    end

    @testset "maximum 2D dims=1" begin
        A = CUDA.rand(Float32, 64, 128)
        R = maximum(ct.Tiled(A); dims=1)
        @test size(R) == (1, 128)
        @test Array(R) ≈ Array(maximum(A; dims=1))
    end

    @testset "minimum 2D dims=2" begin
        A = CUDA.rand(Float32, 64, 128)
        R = minimum(ct.Tiled(A); dims=2)
        @test size(R) == (64, 1)
        @test Array(R) ≈ Array(minimum(A; dims=2))
    end

    @testset "prod 1D" begin
        # Use aligned size (power of 2) and values close to 1 to avoid overflow
        A = CUDA.ones(Float32, 32) .+ CUDA.rand(Float32, 32) .* 0.01f0
        @test prod(ct.Tiled(A)) ≈ prod(A) rtol=1e-3
    end

    @testset "max/min with OOB padding" begin
        # Non-aligned size forces OOB elements in the last tile.
        # Padding must use the identity (NegInf/PosInf), not zero.
        A = CUDA.fill(Float32(-5.0), 1000)
        @test maximum(ct.Tiled(A)) == -5.0f0
        B = CUDA.fill(Float32(5.0), 1000)
        @test minimum(ct.Tiled(B)) == 5.0f0
    end

    @testset "non-aligned sizes" begin
        A = CUDA.rand(Float32, 1000)
        @test sum(ct.Tiled(A)) ≈ sum(A)
    end

    @testset "non-aligned 2D" begin
        A = CUDA.rand(Float32, 100, 200)
        @test sum(ct.Tiled(A)) ≈ sum(A)
        @test Array(sum(ct.Tiled(A); dims=1)) ≈ Array(sum(A; dims=1))
    end

    @testset "mapreduce with abs" begin
        A = CUDA.rand(Float32, 1024) .- 0.5f0
        @test mapreduce(abs, +, ct.Tiled(A)) ≈ mapreduce(abs, +, A)
    end

    @testset "large array (multiple tiles)" begin
        A = CUDA.rand(Float32, 8192)
        @test sum(ct.Tiled(A)) ≈ sum(A)
    end

    @testset "sum! in-place" begin
        A = CUDA.rand(Float32, 64, 128)
        R = CUDA.zeros(Float32, 1, 128)
        sum!(R, ct.Tiled(A))
        @test Array(R) ≈ Array(sum(A; dims=1))
    end

    @testset "maximum! in-place" begin
        A = CUDA.rand(Float32, 64, 128)
        R = CUDA.zeros(Float32, 1, 128)
        maximum!(R, ct.Tiled(A))
        @test Array(R) ≈ Array(maximum(A; dims=1))
    end

    @testset "any" begin
        A = CUDA.zeros(Bool, 1024)
        @test any(ct.Tiled(A)) == false
        B = copy(A)
        CUDA.@allowscalar B[512] = true
        @test any(ct.Tiled(B)) == true
    end

    @testset "all" begin
        A = CUDA.ones(Bool, 1024)
        @test all(ct.Tiled(A)) == true
        B = copy(A)
        CUDA.@allowscalar B[512] = false
        @test all(ct.Tiled(B)) == false
    end

    @testset "count" begin
        A = CUDA.zeros(Bool, 1024)
        for i in [1, 100, 500, 999]
            CUDA.@allowscalar A[i] = true
        end
        @test count(ct.Tiled(A)) == 4
    end
end
