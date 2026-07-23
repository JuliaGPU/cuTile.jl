using CUDA

@testset "GatherScatterView — sparse rows with dynamic dense start" begin
    function gather_rows(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,2}, col_start::Int32)
        rows = ct.arange(4; start=1, step=2)
        selected = @view a[rows, col_start:col_start + Int32(3)]
        tile = ct.load(selected, (4, 4); padding_mode=ct.PaddingMode.Zero)
        ct.store(b, (1, 1), tile)
        return
    end

    a = CUDA.CuArray(reshape(Float32.(1:64), 8, 8))
    b = CUDA.zeros(Float32, 4, 4)
    @cuda backend=cuTile gather_rows(a, b, Int32(2))
    @test Array(b) == Array(a)[[1, 3, 5, 7], 2:5]
end

@testset "GatherScatterView — Colon dense dimension" begin
    function gather_full_cols(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,2})
        rows = ct.arange(4; start=1, step=2)
        selected = @view a[rows, :]
        tile = ct.load(selected, (4, 8); padding_mode=ct.PaddingMode.Zero)
        ct.store(b, (1, 1), tile)
        return
    end

    a = CUDA.CuArray(reshape(Float32.(1:64), 8, 8))
    b = CUDA.zeros(Float32, 4, 8)
    @cuda backend=cuTile gather_full_cols(a, b)
    @test Array(b) == Array(a)[[1, 3, 5, 7], :]
end

@testset "GatherScatterView — sparse columns store" begin
    function scatter_columns(src::ct.TileArray{Float32,2}, dst::ct.TileArray{Float32,2})
        cols = ct.arange(4; start=1, step=2)
        selected = view(dst, Int32(3):Int32(6), cols)
        tile = ct.load(src, (1, 1), (4, 4))
        ct.store(selected, tile)
        return
    end

    src = CUDA.CuArray(reshape(Float32.(1:16), 4, 4))
    dst = CUDA.zeros(Float32, 8, 8)
    @cuda backend=cuTile scatter_columns(src, dst)
    expected = zeros(Float32, 8, 8)
    expected[3:6, [1, 3, 5, 7]] .= Array(src)
    @test Array(dst) == expected
end

@testset "GatherScatterView — Colon dense dimension store" begin
    function scatter_full_cols(src::ct.TileArray{Float32,2}, dst::ct.TileArray{Float32,2})
        rows = ct.arange(4; start=1, step=2)
        selected = @view dst[rows, :]
        tile = ct.load(src, (1, 1), (4, 8))
        ct.store(selected, tile)
        return
    end

    src = CUDA.CuArray(reshape(Float32.(1:32), 4, 8))
    dst = CUDA.zeros(Float32, 8, 8)
    @cuda backend=cuTile scatter_full_cols(src, dst)
    expected = zeros(Float32, 8, 8)
    expected[[1, 3, 5, 7], :] .= Array(src)
    @test Array(dst) == expected
end

@testset "GatherScatterView — partial out-of-bounds load" begin
    function gather_oob(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,2})
        rows = ct.arange(4; start=7)
        selected = @view a[rows, Int32(6):Int32(9)]
        tile = ct.load(selected, (4, 4); padding_mode=ct.PaddingMode.Zero)
        ct.store(b, (1, 1), tile)
        return
    end

    a = CUDA.CuArray(reshape(Float32.(1:64), 8, 8))
    b = CUDA.zeros(Float32, 4, 4)
    @cuda backend=cuTile gather_oob(a, b)
    expected = zeros(Float32, 4, 4)
    for j in 1:4, i in 1:4
        row, col = 6 + i, 5 + j
        row <= 8 && col <= 8 && (expected[i, j] = Array(a)[row, col])
    end
    @test Array(b) == expected
end

@testset "GatherScatterView — partial out-of-bounds store" begin
    function scatter_oob(src::ct.TileArray{Float32,2}, dst::ct.TileArray{Float32,2})
        rows = ct.arange(4; start=7)
        selected = view(dst, rows, Int32(6):Int32(9))
        tile = ct.load(src, (1, 1), (4, 4))
        ct.store(selected, tile)
        return
    end

    src = CUDA.CuArray(reshape(Float32.(1:16), 4, 4))
    dst = CUDA.zeros(Float32, 8, 8)
    @cuda backend=cuTile scatter_oob(src, dst)
    expected = zeros(Float32, 8, 8)
    expected[7:8, 6:8] .= Array(src)[1:2, 1:3]
    @test Array(dst) == expected
end

@testset "GatherScatterView — middle sparse 3D axis" begin
    function gather_middle(a::ct.TileArray{Float32,3}, b::ct.TileArray{Float32,3})
        depth = ct.arange(4; start=1, step=2)
        selected = view(a, Int32(1):Int32(2), depth, Int32(3):Int32(4))
        tile = ct.load(selected, (2, 4, 2))
        ct.store(b, (1, 1, 1), tile)
        return
    end

    a = CUDA.CuArray(reshape(Float32.(1:128), 4, 8, 4))
    b = CUDA.zeros(Float32, 2, 4, 2)
    @cuda backend=cuTile gather_middle(a, b)
    @test Array(b) == Array(a)[1:2, [1, 3, 5, 7], 3:4]
end
