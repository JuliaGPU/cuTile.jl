# End-to-end correctness tests for @view / slice on TileArrays.

using CUDA

@testset "slice — 1D static copy" begin
    # Copy a[3:10] into b[1:8].
    function kern(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1})
        sub = @view a[3:10]
        t = ct.load(sub, 1, (8,))
        ct.store(b, 1, t)
        return
    end

    a = CUDA.rand(Float32, 16)
    b = CUDA.zeros(Float32, 8)
    ct.launch(kern, 1, a, b)
    @test Array(b) == Array(a)[3:10]
end

@testset "slice — 1D dynamic copy" begin
    # Runtime start/stop: copy a[i:j].
    function kern(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                  i::Int32, j::Int32)
        sub = @view a[i:j]
        t = ct.load(sub, 1, (4,))
        ct.store(b, 1, t)
        return
    end

    n = 32
    a = CUDA.rand(Float32, n)
    b = CUDA.zeros(Float32, 4)
    # Pull a[10:13] (4 elements).
    ct.launch(kern, 1, a, b, Int32(10), Int32(13))
    @test Array(b) == Array(a)[10:13]
end

@testset "slice — 2D row-slice" begin
    # Copy a[r1:r2, :] into b.
    function kern(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,2},
                  r1::Int32, r2::Int32)
        sub = @view a[r1:r2, :]
        t = ct.load(sub, (1, 1), (4, 4))
        ct.store(b, (1, 1), t)
        return
    end

    a = CUDA.rand(Float32, 8, 4)
    b = CUDA.zeros(Float32, 4, 4)
    ct.launch(kern, 1, a, b, Int32(3), Int32(6))
    @test Array(b) == Array(a)[3:6, :]
end

@testset "slice — 2D col-slice" begin
    function kern(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,2},
                  c1::Int32, c2::Int32)
        sub = @view a[:, c1:c2]
        t = ct.load(sub, (1, 1), (4, 4))
        ct.store(b, (1, 1), t)
        return
    end

    a = CUDA.rand(Float32, 4, 8)
    b = CUDA.zeros(Float32, 4, 4)
    ct.launch(kern, 1, a, b, Int32(2), Int32(5))
    @test Array(b) == Array(a)[:, 2:5]
end

@testset "slice — 2D chained (both axes)" begin
    function kern(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,2},
                  r1::Int32, r2::Int32, c1::Int32, c2::Int32)
        sub = @view a[r1:r2, c1:c2]
        t = ct.load(sub, (1, 1), (4, 4))
        ct.store(b, (1, 1), t)
        return
    end

    a = CUDA.rand(Float32, 8, 8)
    b = CUDA.zeros(Float32, 4, 4)
    ct.launch(kern, 1, a, b, Int32(3), Int32(6), Int32(2), Int32(5))
    @test Array(b) == Array(a)[3:6, 2:5]
end

@testset "slice — store through @view" begin
    # Store into a[i:j] — validates that slice results work for stores too.
    function kern(a::ct.TileArray{Float32,1}, src::ct.TileArray{Float32,1},
                  i::Int32, j::Int32)
        sub = @view a[i:j]
        t = ct.load(src, 1, (4,))
        ct.store(sub, 1, t)
        return
    end

    a = CUDA.zeros(Float32, 16)
    src = CUDA.rand(Float32, 4)
    ct.launch(kern, 1, a, src, Int32(5), Int32(8))
    got = Array(a)
    want = zeros(Float32, 16)
    want[5:8] .= Array(src)
    @test got == want
end

@testset "slice — nested view" begin
    # Slicing a slice: view(view(a, r1, :), :, r2) — exercises the arg_ref
    # chain across two successive slice intrinsics. Uses the `view` function
    # rather than `@view` for legibility.
    function kern(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,2},
                  r1s::Int32, r1e::Int32, c2s::Int32, c2e::Int32)
        outer = view(a, r1s:r1e, :)
        inner = view(outer, :, c2s:c2e)
        t = ct.load(inner, (1, 1), (4, 4))
        ct.store(b, (1, 1), t)
        return
    end

    a = CUDA.rand(Float32, 8, 8)
    b = CUDA.zeros(Float32, 4, 4)
    ct.launch(kern, 1, a, b, Int32(3), Int32(6), Int32(2), Int32(5))
    @test Array(b) == Array(a)[3:6, 2:5]
end

