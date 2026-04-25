# End-to-end correctness tests for `permutedims` / `transpose` / `reshape`
# on TileArrays.

using CUDA

@testset "permutedims — 2D" begin
    # Permute a 2D TileArray, copy a 4×4 tile from the permuted view into
    # a 4×4 destination. Compared against CPU `permutedims`.
    function kern(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,2})
        ap = permutedims(a, (2, 1))
        t = ct.load(ap, (1, 1), (4, 4))
        ct.store(b, (1, 1), t)
        return
    end
    a = CUDA.rand(Float32, 8, 4)
    b = CUDA.zeros(Float32, 4, 4)
    ct.launch(kern, 1, a, b)
    @test Array(b) == permutedims(Array(a), (2, 1))[1:4, 1:4]
end

@testset "transpose — 2D" begin
    function kern(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,2})
        at = transpose(a)
        t = ct.load(at, (1, 1), (4, 4))
        ct.store(b, (1, 1), t)
        return
    end
    a = CUDA.rand(Float32, 8, 4)
    b = CUDA.zeros(Float32, 4, 4)
    ct.launch(kern, 1, a, b)
    @test Array(b) == permutedims(Array(a), (2, 1))[1:4, 1:4]
end

@testset "reshape — 1D → 2D" begin
    # Reshape a 1D vector to 2D and load a tile from it.
    function kern(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,2})
        ar = reshape(a, (4, 4))
        t = ct.load(ar, (1, 1), (4, 4))
        ct.store(b, (1, 1), t)
        return
    end
    a = CUDA.rand(Float32, 16)
    b = CUDA.zeros(Float32, 4, 4)
    ct.launch(kern, 1, a, b)
    @test Array(b) == reshape(Array(a), 4, 4)
end

@testset "reshape — 2D → 1D" begin
    # Round-trip: 2D → 1D → 2D.
    function kern(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,1})
        ar = reshape(a, (16,))
        t = ct.load(ar, 1, (16,))
        ct.store(b, 1, t)
        return
    end
    a = CUDA.rand(Float32, 4, 4)
    b = CUDA.zeros(Float32, 16)
    ct.launch(kern, 1, a, b)
    @test Array(b) == reshape(Array(a), 16)
end

@testset "permutedims — 3D" begin
    # 3D permutation with a non-cyclic perm.
    function kern(a::ct.TileArray{Float32,3}, b::ct.TileArray{Float32,3})
        ap = permutedims(a, (3, 1, 2))
        t = ct.load(ap, (1, 1, 1), (2, 2, 2))
        ct.store(b, (1, 1, 1), t)
        return
    end
    a = CUDA.rand(Float32, 4, 4, 4)
    b = CUDA.zeros(Float32, 4, 4, 4)
    ct.launch(kern, 1, a, b)
    expected = permutedims(Array(a), (3, 1, 2))
    @test Array(b)[1:2, 1:2, 1:2] == expected[1:2, 1:2, 1:2]
end
