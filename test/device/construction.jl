# array construction syntax: scalar literals ([1,2,3,4], [1 3; 2 4], hvncat)
# producing constant tiles, runtime scalar elements, and bracket-syntax tile
# concatenation ([t1; t2], [t1 t2; t3 t4], ...)

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

@testset "single-dim hvncat syntax" begin
    function hvncat_dim_kernel(out::ct.TileArray{Int64,2})
        ct.store(out, ct.bid(1), [1;; 2])
        return
    end
    out = CUDA.zeros(Int64, 1, 2)
    @cuda backend=cuTile blocks=1 hvncat_dim_kernel(out)
    @test Array(out) == [1;; 2]
end

@testset "runtime scalar elements" begin
    function runtime_vect_kernel(out::ct.TileArray{Float64,1}, x::Float64, y::Float64)
        ct.store(out, ct.bid(1), [x, y, 2.0, 4.0])
        return
    end
    out = CUDA.zeros(Float64, 4)
    @cuda backend=cuTile blocks=1 runtime_vect_kernel(out, 1.5, 2.5)
    @test Array(out) == [1.5, 2.5, 2.0, 4.0]

    # matrix with runtime elements exercises the row-major reorder on the
    # runtime (cat tree) path
    function runtime_hvcat_kernel(out::ct.TileArray{Float32,2}, x::Float32)
        ct.store(out, ct.bid(1), Float32[x 3; 2 4])
        return
    end
    out = CUDA.zeros(Float32, 2, 2)
    @cuda backend=cuTile blocks=1 runtime_hvcat_kernel(out, 1.5f0)
    @test Array(out) == Float32[1.5 3; 2 4]
end

@testset "tile concatenation" begin
    function tile_vcat_kernel(a::ct.TileArray{Float32,1}, out::ct.TileArray{Float32,1})
        t1 = ct.load(a, (1,), (4,))
        t2 = ct.load(a, (2,), (4,))
        ct.store(out, ct.bid(1), [t1; t2])
        return
    end
    a = CuArray(Float32.(1:8))
    out = CUDA.zeros(Float32, 8)
    @cuda backend=cuTile blocks=1 tile_vcat_kernel(a, out)
    @test Array(out) == Float32.(1:8)

    # vcat of four tiles keeps intermediates power-of-two via the balanced tree
    function tile_vcat4_kernel(a::ct.TileArray{Float32,1}, out::ct.TileArray{Float32,1})
        t1 = ct.load(a, (1,), (2,))
        t2 = ct.load(a, (2,), (2,))
        t3 = ct.load(a, (3,), (2,))
        t4 = ct.load(a, (4,), (2,))
        ct.store(out, ct.bid(1), [t1; t2; t3; t4])
        return
    end
    out = CUDA.zeros(Float32, 8)
    @cuda backend=cuTile blocks=1 tile_vcat4_kernel(a, out)
    @test Array(out) == Float32.(1:8)

    # hcat treats vectors as columns, like Julia
    function tile_hcat_kernel(a::ct.TileArray{Float32,1}, out::ct.TileArray{Float32,2})
        t1 = ct.load(a, (1,), (4,))
        t2 = ct.load(a, (2,), (4,))
        ct.store(out, ct.bid(1), [t1 t2])
        return
    end
    out = CUDA.zeros(Float32, 4, 2)
    @cuda backend=cuTile blocks=1 tile_hcat_kernel(a, out)
    @test Array(out) == [Float32.(1:4) Float32.(5:8)]

    # hvcat builds block matrices
    function tile_hvcat_kernel(a::ct.TileArray{Float32,2}, out::ct.TileArray{Float32,2})
        t11 = ct.load(a, (1, 1), (2, 2))
        t12 = ct.load(a, (1, 2), (2, 2))
        t21 = ct.load(a, (2, 1), (2, 2))
        t22 = ct.load(a, (2, 2), (2, 2))
        ct.store(out, ct.bid(1), [t11 t12; t21 t22])
        return
    end
    blocks = CuArray(reshape(Float32.(1:16), 4, 4))
    out = CUDA.zeros(Float32, 4, 4)
    @cuda backend=cuTile blocks=1 tile_hvcat_kernel(blocks, out)
    @test Array(out) == reshape(Float32.(1:16), 4, 4)

    # hvncat(dim) stacks tiles along a new trailing dimension
    function tile_stack_kernel(a::ct.TileArray{Float32,2}, out::ct.TileArray{Float32,3})
        t1 = ct.load(a, (1, 1), (2, 2))
        t2 = ct.load(a, (2, 1), (2, 2))
        ct.store(out, ct.bid(1), [t1;;; t2])
        return
    end
    m = CuArray(reshape(Float32.(1:8), 4, 2))
    out = CUDA.zeros(Float32, 2, 2, 2)
    @cuda backend=cuTile blocks=1 tile_stack_kernel(m, out)
    host = reshape(Float32.(1:8), 4, 2)
    @test Array(out) == cat(host[1:2, :], host[3:4, :]; dims=3)
end

@testset "mixed scalar/tile blocks" begin
    # scalars lift to unit tiles ([x; t] etc.)
    function mixed_vcat_kernel(a::ct.TileArray{Float32,1}, out::ct.TileArray{Float32,1}, x::Float32)
        t = ct.load(a, ct.bid(1), (1,))
        ct.store(out, ct.bid(1), [x; t])
        return
    end
    a = CuArray(Float32[42])
    out = CUDA.zeros(Float32, 2)
    @cuda backend=cuTile blocks=1 mixed_vcat_kernel(a, out, 7.5f0)
    @test Array(out) == Float32[7.5, 42]

    # block matrix from 1x1 scalar and tile blocks, with eltype promotion
    function mixed_hvcat_kernel(out::ct.TileArray{Float64,2}, x::Float32)
        ct.store(out, ct.bid(1), [x [1.0]; [2.0] [x]])
        return
    end
    out = CUDA.zeros(Float64, 2, 2)
    @cuda backend=cuTile blocks=1 mixed_hvcat_kernel(out, 7.5f0)
    @test Array(out) == [7.5 1.0; 2.0 7.5]
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
