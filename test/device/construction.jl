using CUDA

@testset "scalar literals" begin
    function scalar_literals_kernel(vec::ct.TileArray{Float64,1},
                                    mat::ct.TileArray{Float32,2},
                                    cube::ct.TileArray{Int32,3}, x::Float64)
        ct.store(vec, 1, [x, 2, 3, 4])
        ct.store(mat, 1, Float32[1 2; 3 4; 5 6; 7 8])
        ct.store(cube, 1, Int32[1 2; 3 4;;; 5 6; 7 8])
        return
    end

    vec = CUDA.zeros(Float64, 4)
    mat = CUDA.zeros(Float32, 4, 2)
    cube = CUDA.zeros(Int32, 2, 2, 2)
    @cuda backend=cuTile blocks=1 scalar_literals_kernel(vec, mat, cube, 1.5)

    @test Array(vec) == [1.5, 2, 3, 4]
    @test Array(mat) == Float32[1 2; 3 4; 5 6; 7 8]
    @test Array(cube) == Int32[1 2; 3 4;;; 5 6; 7 8]
end

@testset "constant literals" begin
    function constant_literals_kernel(flags::ct.TileArray{Bool,1},
                                      filled::ct.TileArray{Float32,1})
        ct.store(flags, 1, Bool[true, false, false, true])
        ct.store(filled, 1, vcat(zeros(Float32, (2,)), ones(Float32, (2,))))
        return
    end

    flags = CUDA.zeros(Bool, 4)
    filled = CUDA.zeros(Float32, 4)
    @cuda backend=cuTile blocks=1 constant_literals_kernel(flags, filled)

    @test Array(flags) == Bool[true, false, false, true]
    @test Array(filled) == Float32[0, 0, 1, 1]
end

@testset "tile concatenation" begin
    function tile_concatenation_kernel(input::ct.TileArray{Float32,2},
                                       block::ct.TileArray{Float32,2},
                                       stack::ct.TileArray{Float32,3},
                                       diagonal::ct.TileArray{Float32,2})
        t = ct.load(input, 1, (2, 2))
        ct.store(block, 1, [t t; t t])
        ct.store(stack, 1, [t t;;; t t])
        ct.store(diagonal, 1, cat(t, t; dims=(1, 2)))
        return
    end

    input = CUDA.rand(Float32, 2, 2)
    block = CUDA.zeros(Float32, 4, 4)
    stack = CUDA.zeros(Float32, 2, 4, 2)
    diagonal = CUDA.zeros(Float32, 4, 4)
    @cuda backend=cuTile blocks=1 tile_concatenation_kernel(input, block, stack, diagonal)

    host = Array(input)
    @test Array(block) == [host host; host host]
    @test Array(stack) == [host host;;; host host]
    @test Array(diagonal) == cat(host, host; dims=(1, 2))
end

@testset "mixed concatenation" begin
    function mixed_kernel(out::ct.TileArray{Float32,2},
                          vector::ct.TileArray{Float32,1},
                          wide::ct.TileArray{Float32,1}, x::Float64)
        t = ct.load(vector, 1, (2,))
        ct.store(out, 1, Float32[x [2]; [3 4]])
        ct.store(wide, 1, cat(t, Float32(x), Float32(x); dims=1))
        return
    end

    out = CUDA.zeros(Float32, 2, 2)
    vector = CUDA.rand(Float32, 2)
    wide = CUDA.zeros(Float32, 4)
    @cuda backend=cuTile blocks=1 mixed_kernel(out, vector, wide, 1.5)

    @test Array(out) == Float32[1.5 2; 3 4]
    @test Array(wide) == [Array(vector); 1.5f0; 1.5f0]
end

@testset "empty concatenation" begin
    function empty_concatenation_kernel(input::ct.TileArray{Float32,2},
                                        input_vector::ct.TileArray{Float32,1},
                                        block::ct.TileArray{Float32,2},
                                        promoted::ct.TileArray{Float64,1},
                                        shape::ct.TileArray{Int32,1})
        tile = ct.load(input, 1, (2, 2))
        vector = ct.load(input_vector, 1, (4,))
        ct.store(block, 1, [Float32[] Float32[]; tile])
        ct.store(promoted, 1, vcat(Float64[], vector, Float64[]))

        empty = hcat(Float32[], Float32[], Float32[])
        ct.store(shape, 1,
                 Int32[length(empty), size(empty, 1), size(empty, 2), ndims(empty)])
        return
    end

    input = CUDA.rand(Float32, 2, 2)
    input_vector = CUDA.rand(Float32, 4)
    block = CUDA.zeros(Float32, 2, 2)
    promoted = CUDA.zeros(Float64, 4)
    shape = CUDA.zeros(Int32, 4)
    @cuda backend=cuTile blocks=1 empty_concatenation_kernel(
        input, input_vector, block, promoted, shape)

    host = Array(input)
    @test Array(block) == [Float32[] Float32[]; host]
    @test Array(promoted) == vcat(Float64[], Array(input_vector), Float64[])
    @test Array(shape) == Int32[0, 0, 3, 2]
end
