spec1d = ct.ArraySpec{1}(16, true)
spec2d = ct.ArraySpec{2}(16, true)
Array1D32 = ct.TileArray{Float32, 1, Int32, spec1d}
Array1D64 = ct.TileArray{Float32, 1, Int64, spec1d}
Array2D64 = ct.TileArray{Float32, 2, Int64, spec2d}

@testset "Int64 indexing — partition views" begin
    @test @filecheck begin
        @check_label "entry"
        @check "make_tensor_view"
        @check "tile<i64> -> tensor_view"
        @check "exti {{.*}} signed : tile<i32> -> tile<i64>"
        @check "load_view_tko"
        @check "tile<i64> -> tile<16xf32>"
        code_tiled(Tuple{Array1D64}; bytecode_version=v"13.3") do a
            index = ct.bid(1)
            tile = ct.load(a, index, (16,))
            ct.store(a, index, tile)
            return
        end
    end

    @test_throws "Int64-indexed TileArray requires Tile IR bytecode v13.3+" code_tiled(
        Tuple{Array1D64}; bytecode_version=v"13.2") do a
        ct.store(a, 1, ct.load(a, 1, (16,)))
        return
    end
end

@testset "Int64 indexing — mixed array widths" begin
    @test @filecheck begin
        @check_label "entry"
        @check "make_tensor_view"
        @check "tile<i32> -> tensor_view"
        @check "make_tensor_view"
        @check "tile<i64> -> tensor_view"
        code_tiled(Tuple{Array1D32, Array1D64}; bytecode_version=v"13.3") do a32, a64
            ct.store(a32, 1, ct.load(a32, 1, (16,)))
            ct.store(a64, 1, ct.load(a64, 1, (16,)))
            return
        end
    end
end

@testset "Int64 indexing — strided and gather/scatter views" begin
    @test @filecheck begin
        @check_label "entry"
        @check "make_strided_view"
        @check "load_view_tko"
        @check "tile<i64> -> tile<4x4xf32>"
        @check "make_gather_scatter_view"
        @check "tile<i64>, tile<4xi64> -> tile<4x4xf32>"
        code_tiled(Tuple{Array2D64}; bytecode_version=v"13.3") do a
            tiles = eachtile(a, (4, 4); step=(2, 3))
            tile = ct.load(tiles, (ct.bid(1), ct.bid(2)))
            ct.store(tiles, (ct.bid(1), ct.bid(2)), tile)

            rows = ct.arange(4)
            sparse = @view a[rows, Int32(1):Int32(4)]
            gathered = ct.load(sparse, (4, 4))
            ct.store(sparse, gathered)
            return
        end
    end
end

@testset "Int64 indexing — pointer and atomic paths" begin
    @test @filecheck begin
        @check_label "entry"
        @check "offset {{.*}} tile<4x4xi64> -> tile<4x4xptr<f32>>"
        @check "cmpi less_than {{.*}} unsigned : tile<4x4xi64>"
        @check "atomic_rmw_tko"
        @check "atomic_red_view_tko"
        @check "tile<i64> -> token"
        code_tiled(Tuple{Array2D64}; bytecode_version=v"13.3") do a
            indices = ct.arange(4)
            rows = reshape(indices, (4, 1))
            cols = reshape(indices, (1, 4))
            tile = ct.gather(a, (rows, cols))
            ct.scatter(a, (rows, cols), tile)
            ct.atomic_add(a, (rows, cols), tile)
            ct.atomic_store_add(a, (ct.bid(1), 1), tile)
            return
        end
    end
end
