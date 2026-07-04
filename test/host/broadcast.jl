using CUDA

@testset "Tiled broadcast" begin
    @testset "1D element-wise" begin
        n = 1024
        A = CUDA.rand(Float32, n)
        B = CUDA.rand(Float32, n)
        C = CUDA.zeros(Float32, n)
        ct.Tiled(C) .= ct.Tiled(A) .+ ct.Tiled(B)
        @test Array(C) ≈ Array(A) .+ Array(B)
    end

    @testset "fused multi-op" begin
        n = 1024
        A = CUDA.rand(Float32, n) .+ 0.1f0
        C = CUDA.zeros(Float32, n)
        ct.Tiled(C) .= ct.Tiled(A) .+ ct.Tiled(A) .* sin.(ct.Tiled(A))
        @test Array(C) ≈ Array(A) .+ Array(A) .* sin.(Array(A)) rtol=1e-5
    end

    @testset "scalar broadcast" begin
        n = 1024
        A = CUDA.rand(Float32, n)
        C = CUDA.zeros(Float32, n)
        ct.Tiled(C) .= ct.Tiled(A) .+ 1.0f0
        @test Array(C) ≈ Array(A) .+ 1.0f0
    end

    @testset "2D element-wise" begin
        m, n = 128, 256
        A = CUDA.rand(Float32, m, n)
        B = CUDA.rand(Float32, m, n)
        C = CUDA.zeros(Float32, m, n)
        ct.Tiled(C) .= ct.Tiled(A) .+ ct.Tiled(B)
        @test Array(C) ≈ Array(A) .+ Array(B)
    end

    @testset "3D element-wise" begin
        A = CUDA.rand(Float32, 64, 64, 4)
        B = CUDA.rand(Float32, 64, 64, 4)
        C = CUDA.zeros(Float32, 64, 64, 4)
        ct.Tiled(C) .= ct.Tiled(A) .+ ct.Tiled(B)
        @test Array(C) ≈ Array(A) .+ Array(B)
    end

    @testset "ct.@. expands to Tiled" begin
        ex = @macroexpand ct.@. C = A + B
        # The macro should produce Tiled() wrapping, not plain dotted calls
        @test occursin("Tiled", string(ex))
    end

    @testset "ct.@. in-place" begin
        n = 1024
        A = CUDA.rand(Float32, n)
        B = CUDA.rand(Float32, n)
        C = CUDA.zeros(Float32, n)
        ct.@. C = A + B
        @test Array(C) ≈ Array(A) .+ Array(B)
    end

    @testset "ct.@. with function" begin
        n = 1024
        A = CUDA.rand(Float32, n) .+ 0.1f0
        C = CUDA.zeros(Float32, n)
        ct.@. C = A + sin(A)
        @test Array(C) ≈ Array(A) .+ sin.(Array(A)) rtol=1e-5
    end

    @testset "ct.@. with scalar" begin
        n = 1024
        A = CUDA.rand(Float32, n)
        C = CUDA.zeros(Float32, n)
        ct.@. C = A + 2.0f0
        @test Array(C) ≈ Array(A) .+ 2.0f0
    end

    @testset "allocating copy" begin
        n = 1024
        A = CUDA.rand(Float32, n)
        B = CUDA.rand(Float32, n)
        C = ct.Tiled(A) .+ ct.Tiled(B)
        @test C isa CuArray
        @test Array(C) ≈ Array(A) .+ Array(B)
    end

    @testset "allocating ct.@." begin
        n = 1024
        A = CUDA.rand(Float32, n)
        B = CUDA.rand(Float32, n)
        C = ct.@. A + B
        @test C isa CuArray
        @test Array(C) ≈ Array(A) .+ Array(B)
    end

    @testset "leading singleton dim" begin
        A = CUDA.rand(Float32, 1, 1024)
        B = similar(A)
        ct.Tiled(B) .= ct.Tiled(A) .+ 1.0f0
        @test Array(B) ≈ Array(A) .+ 1.0f0
    end

    @testset "double leading singleton" begin
        A = CUDA.rand(Float32, 1, 1, 512)
        B = similar(A)
        ct.Tiled(B) .= ct.Tiled(A) .* 2.0f0
        @test Array(B) ≈ Array(A) .* 2.0f0
    end

    @testset "small leading dim" begin
        A = CUDA.rand(Float32, 4, 1024)
        B = similar(A)
        ct.Tiled(B) .= ct.Tiled(A) .+ ct.Tiled(A)
        @test Array(B) ≈ 2 .* Array(A)
    end

    @testset "non-aligned size" begin
        A = CUDA.rand(Float32, 1000)
        B = CUDA.rand(Float32, 1000)
        C = CUDA.zeros(Float32, 1000)
        ct.Tiled(C) .= ct.Tiled(A) .+ ct.Tiled(B)
        @test Array(C) ≈ Array(A) .+ Array(B)
    end

    @testset "shape mismatch throws" begin
        A = CUDA.rand(Float32, 11)
        C = CUDA.zeros(Float32, 10)
        @test_throws DimensionMismatch ct.Tiled(C) .= ct.Tiled(A)
        @test_throws DimensionMismatch ct.Tiled(C) .= ct.Tiled(C) .+ ct.Tiled(A)

        A2 = CUDA.rand(Float32, 8, 16)
        C2 = CUDA.zeros(Float32, 8, 8)
        @test_throws DimensionMismatch ct.Tiled(C2) .= ct.Tiled(A2) .* 2.0f0
    end

    @testset "size-1 dim expansion (row)" begin
        m, n = 64, 128
        A = CUDA.rand(Float32, 1, n)
        B = CUDA.rand(Float32, m, n)
        C = CUDA.zeros(Float32, m, n)
        ct.Tiled(C) .= ct.Tiled(A) .+ ct.Tiled(B)
        @test Array(C) ≈ Array(A) .+ Array(B)
    end

    @testset "size-1 dim expansion (column)" begin
        m, n = 64, 128
        A = CUDA.rand(Float32, m, 1)
        B = CUDA.rand(Float32, m, n)
        C = CUDA.zeros(Float32, m, n)
        ct.Tiled(C) .= ct.Tiled(A) .* ct.Tiled(B)
        @test Array(C) ≈ Array(A) .* Array(B)
    end

    @testset "size-1 expansion only" begin
        m, n = 32, 512
        A = CUDA.rand(Float32, 1, n)
        C = CUDA.zeros(Float32, m, n)
        ct.Tiled(C) .= ct.Tiled(A) .+ 1.0f0
        @test Array(C) ≈ repeat(Array(A) .+ 1.0f0, m, 1)
    end

    @testset "rank expansion (vector + matrix)" begin
        m, n = 64, 128
        A = CUDA.rand(Float32, m)
        B = CUDA.rand(Float32, m, n)
        C = CUDA.zeros(Float32, m, n)
        ct.Tiled(C) .= ct.Tiled(A) .+ ct.Tiled(B)
        @test Array(C) ≈ Array(A) .+ Array(B)
    end

    @testset "allocating with expansion" begin
        m, n = 64, 128
        A = CUDA.rand(Float32, 1, n)
        B = CUDA.rand(Float32, m, 1)
        C = ct.Tiled(A) .+ ct.Tiled(B)
        @test size(C) == (m, n)
        @test Array(C) ≈ Array(A) .+ Array(B)
    end

    @testset "scalar RHS fill" begin
        C = CUDA.rand(Float32, 1000)
        ct.Tiled(C) .= 0
        @test all(iszero, Array(C))

        D = CUDA.zeros(Float32, 33, 65)
        ct.Tiled(D) .= 2.5f0
        @test all(==(2.5f0), Array(D))
    end

    @testset "empty array no-op" begin
        C = CUDA.zeros(Float32, 0)
        ct.Tiled(C) .= ct.Tiled(C) .+ 1.0f0  # must not launch a 0-block grid
        @test isempty(Array(C))

        D = CUDA.zeros(Float32, 4, 0)
        ct.Tiled(D) .= 1.0f0
        @test isempty(Array(D))
    end

    @testset "0-dim arrays" begin
        C = CUDA.zeros(Float32)
        ct.Tiled(C) .= 1.0f0
        @test Array(C)[] == 1.0f0

        A = CUDA.fill(2.0f0)
        ct.Tiled(C) .= ct.Tiled(A) .+ 1.0f0
        @test Array(C)[] == 3.0f0
    end

    @testset "host-side leaves rejected" begin
        C = CUDA.zeros(Float32, 16)
        A_cpu = rand(Float32, 16)
        @test_throws ArgumentError ct.Tiled(C) .= ct.Tiled(A_cpu) .* 2.0f0
        @test_throws ArgumentError ct.Tiled(C) .= ct.Tiled(C) .+ (1:16)
        @test_throws ArgumentError ct.Tiled(A_cpu) .= 0  # CPU destination
    end

    @testset "ct.@. returns destination array" begin
        n = 256
        A = CUDA.rand(Float32, n)
        B = CUDA.rand(Float32, n)
        C = CUDA.zeros(Float32, n)
        ret = ct.@. C = A + B
        @test ret === C
        @test Array(C) ≈ Array(A) .+ Array(B)
    end
end
