# array literal syntax ([1,2,3,4], [1 3; 2 4], hvncat) producing constant tiles

using CUDA

@testset "vector literals" begin
    function vect_kernel(out::ct.TileArray{Int64,1})
        ct.store(out, ct.bid(1), [1, 2, 3, 4])
        return
    end
    out = CUDA.zeros(Int64, 4)
    @cuda backend=cuTile blocks=1 vect_kernel(out)
    @test Array(out) == [1, 2, 3, 4]

    function typed_vect_kernel(out::ct.TileArray{Float32,1})
        ct.store(out, ct.bid(1), Float32[1.5, 2, 3, 4])
        return
    end
    out = CUDA.zeros(Float32, 4)
    @cuda backend=cuTile blocks=1 typed_vect_kernel(out)
    @test Array(out) == Float32[1.5, 2, 3, 4]

    function vcat_kernel(out::ct.TileArray{Float64,1})
        ct.store(out, ct.bid(1), [1.0; 2.5; 3.0; 4.0])
        return
    end
    out = CUDA.zeros(Float64, 4)
    @cuda backend=cuTile blocks=1 vcat_kernel(out)
    @test Array(out) == [1.0; 2.5; 3.0; 4.0]

    # heterogeneous elements promote like Base.vect
    function promote_kernel(out::ct.TileArray{Float64,1})
        ct.store(out, ct.bid(1), [1, 2.5, 3, 4])
        return
    end
    out = CUDA.zeros(Float64, 4)
    @cuda backend=cuTile blocks=1 promote_kernel(out)
    @test Array(out) == [1, 2.5, 3, 4]
end

@testset "matrix literals" begin
    function hvcat_kernel(out::ct.TileArray{Int64,2})
        ct.store(out, ct.bid(1), [1 3; 2 4])
        return
    end
    out = CUDA.zeros(Int64, 2, 2)
    @cuda backend=cuTile blocks=1 hvcat_kernel(out)
    @test Array(out) == [1 3; 2 4]

    # non-square to catch transposition bugs
    function hvcat_rect_kernel(out::ct.TileArray{Float32,2})
        ct.store(out, ct.bid(1), Float32[1 2; 3 4; 5 6; 7 8])
        return
    end
    out = CUDA.zeros(Float32, 4, 2)
    @cuda backend=cuTile blocks=1 hvcat_rect_kernel(out)
    @test Array(out) == Float32[1 2; 3 4; 5 6; 7 8]

    function hcat_kernel(out::ct.TileArray{Int32,2})
        ct.store(out, ct.bid(1), Int32[1 2 3 4])
        return
    end
    out = CUDA.zeros(Int32, 1, 4)
    @cuda backend=cuTile blocks=1 hcat_kernel(out)
    @test Array(out) == Int32[1 2 3 4]
end

@testset "hvncat literals" begin
    function hvncat_colfirst_kernel(out::ct.TileArray{Int64,2})
        ct.store(out, ct.bid(1), [1; 2;; 3; 4])
        return
    end
    out = CUDA.zeros(Int64, 2, 2)
    @cuda backend=cuTile blocks=1 hvncat_colfirst_kernel(out)
    @test Array(out) == [1; 2;; 3; 4]

    function hvncat_3d_kernel(out::ct.TileArray{Int64,3})
        ct.store(out, ct.bid(1), [1 2; 3 4;;; 5 6; 7 8])
        return
    end
    out = CUDA.zeros(Int64, 2, 2, 2)
    @cuda backend=cuTile blocks=1 hvncat_3d_kernel(out)
    @test Array(out) == [1 2; 3 4;;; 5 6; 7 8]
end

@testset "bool literals" begin
    # non-splat i1 constants are bit-packed; cover both sub-byte (4 elements)
    # and multi-byte (16 elements) packing
    function bool_2x2_kernel(out::ct.TileArray{Float32,2})
        m = [true false; true true]
        ct.store(out, ct.bid(1), Float32.(m))
        return
    end
    out = CUDA.zeros(Float32, 2, 2)
    @cuda backend=cuTile blocks=1 bool_2x2_kernel(out)
    @test Array(out) == Float32.([true false; true true])

    bools16 = (true, false, true, true, false, false, true, false,
               false, true, false, false, true, true, false, true)
    function bool_16_kernel(out::ct.TileArray{Float32,1})
        m = [true, false, true, true, false, false, true, false,
             false, true, false, false, true, true, false, true]
        ct.store(out, ct.bid(1), Float32.(m))
        return
    end
    out = CUDA.zeros(Float32, 16)
    @cuda backend=cuTile blocks=1 bool_16_kernel(out)
    @test Array(out) == Float32[bools16...]
end

@testset "literals in arithmetic" begin
    function arith_kernel(a::ct.TileArray{Float32,1}, out::ct.TileArray{Float32,1})
        tile = ct.load(a, ct.bid(1), (4,))
        ct.store(out, ct.bid(1), tile .+ Float32[10, 20, 30, 40])
        return
    end
    a = CuArray(Float32[1, 2, 3, 4])
    out = CUDA.zeros(Float32, 4)
    @cuda backend=cuTile blocks=1 arith_kernel(a, out)
    @test Array(out) == Float32[11, 22, 33, 44]
end
