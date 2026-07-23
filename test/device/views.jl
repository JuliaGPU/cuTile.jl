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
    @cuda backend=cuTile kern(a, b)
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
    @cuda backend=cuTile kern(a, b)
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
    @cuda backend=cuTile kern(a, b)
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
    @cuda backend=cuTile kern(a, b)
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
    @cuda backend=cuTile kern(a, b)
    expected = permutedims(Array(a), (3, 1, 2))
    @test Array(b)[1:2, 1:2, 1:2] == expected[1:2, 1:2, 1:2]
end

@testset "eachtile — adjacent, overlapping, and gapped windows" begin
    function copy_windows(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1}, n::Int32)
        src = eachtile(a, (4,); step=(2,))
        dst = eachtile(b, (4,); step=(2,))
        for i in 1:n
            dst[i] = src[i]
        end
        return
    end

    a = CUDA.rand(Float32, 16)
    b = CUDA.zeros(Float32, 16)
    @cuda backend=cuTile copy_windows(a, b, Int32(8))
    @test Array(b) == Array(a)

    function copy_gaps(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1})
        src = eachtile(a, (4,); step=(8,))
        dst = eachtile(b, (4,); step=(8,))
        # On device, `size` is overlaid to query the backend index space
        # (get_index_space_shape) rather than baking in `cld`.
        for i in 1:size(src, 1)
            dst[i] = src[i]
        end
        return
    end

    fill!(b, 0)
    @cuda backend=cuTile copy_gaps(a, b)
    expected = zeros(Float32, 16)
    expected[1:4] .= Array(a)[1:4]
    expected[9:12] .= Array(a)[9:12]
    @test Array(b) == expected
end

@testset "eachtile — asymmetric 2D windows" begin
    function copy_window(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,2})
        src = eachtile(a, (4, 8); step=(3, 2), padding_mode=ct.PaddingMode.Zero)
        dst = eachtile(b, (4, 8); step=(3, 2))
        ct.store(dst, (2, 3), ct.load(src, (2, 3)))
        return
    end

    a = CUDA.rand(Float32, 12, 12)
    b = CUDA.zeros(Float32, 12, 12)
    @cuda backend=cuTile copy_window(a, b)
    expected = zeros(Float32, 12, 12)
    expected[4:7, 5:12] .= Array(a)[4:7, 5:12]
    @test Array(b) == expected
end

@testset "eachtile — partial edges and requested-rank normalization" begin
    function copy_partial(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1})
        src = eachtile(a, (4,); step=(4,), padding_mode=ct.PaddingMode.Zero)
        ct.store(b, 1, ct.load(src, 2))
        return
    end

    a = CUDA.rand(Float32, 6)
    b = CUDA.zeros(Float32, 4)
    @cuda backend=cuTile copy_partial(a, b)
    @test Array(b) == vcat(Array(a)[5:6], zeros(Float32, 2))

    function copy_normalized(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,2})
        src = eachtile(a, (4,); step=(2,))
        dst = eachtile(b, (4,); step=(2,))
        dst[2, 1] = src[2, 1]
        return
    end

    a2 = CUDA.rand(Float32, 8, 1)
    b2 = CUDA.zeros(Float32, 8, 1)
    @cuda backend=cuTile copy_normalized(a2, b2)
    expected = zeros(Float32, 8, 1)
    expected[3:6, 1] .= Array(a2)[3:6, 1]
    @test Array(b2) == expected
end
