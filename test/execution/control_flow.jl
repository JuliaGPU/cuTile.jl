# control flow tests

using CUDA

@testset "early returns" begin

@testset "early return — taken" begin
    function early_return_skip(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1}, flag::Int32)
        pid = ct.bid(1)
        tile = ct.load(a, pid, (16,))
        if flag == Int32(0)
            return nothing
        end
        ct.store(b, pid, tile .* 2.0f0)
        return nothing
    end

    a = CUDA.rand(Float32, 64)
    b = CUDA.zeros(Float32, 64)
    ct.launch(early_return_skip, 4, a, b, Int32(0))
    @test all(Array(b) .== 0.0f0)
end

@testset "early return — not taken" begin
    function early_return_store(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1}, flag::Int32)
        pid = ct.bid(1)
        tile = ct.load(a, pid, (16,))
        if flag == Int32(0)
            return nothing
        end
        ct.store(b, pid, tile .* 2.0f0)
        return nothing
    end

    a = CUDA.rand(Float32, 64)
    b = CUDA.zeros(Float32, 64)
    ct.launch(early_return_store, 4, a, b, Int32(1))
    @test Array(b) ≈ Array(a) .* 2.0f0
end

@testset "multiple early returns" begin
    function multi_early_return(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                                flag1::Int32, flag2::Int32)
        pid = ct.bid(1)
        tile = ct.load(a, pid, (16,))
        if flag1 == Int32(0)
            return nothing
        end
        if flag2 == Int32(0)
            return nothing
        end
        ct.store(b, pid, tile .* 2.0f0)
        return nothing
    end

    a = CUDA.rand(Float32, 64)

    b1 = CUDA.zeros(Float32, 64)
    ct.launch(multi_early_return, 4, a, b1, Int32(1), Int32(1))
    @test Array(b1) ≈ Array(a) .* 2.0f0

    b2 = CUDA.zeros(Float32, 64)
    ct.launch(multi_early_return, 4, a, b2, Int32(0), Int32(1))
    @test all(Array(b2) .== 0.0f0)

    b3 = CUDA.zeros(Float32, 64)
    ct.launch(multi_early_return, 4, a, b3, Int32(1), Int32(0))
    @test all(Array(b3) .== 0.0f0)
end

end

@testset "scalar indexing as loop bound" begin
    function scalar_index_loop_kernel(data::ct.TileArray{Float32,1},
                                      lengths::ct.TileArray{Int32,1},
                                      out::ct.TileArray{Float32,1})
        bid = ct.bid(1)
        len = lengths[bid]
        acc = ct.zeros((16,), Float32)
        j = Int32(1)
        while j <= len
            tile = ct.load(data, j, (16,))
            acc = acc .+ tile
            j += Int32(1)
        end
        ct.store(out, bid, acc)
        return
    end

    # 3 blocks, each sums a different number of tiles
    n_tiles = Int32[2, 3, 1]
    data = CUDA.rand(Float32, 48)  # 3 tiles of 16
    lengths = CuArray(n_tiles)
    out = CUDA.zeros(Float32, 48)

    ct.launch(scalar_index_loop_kernel, 3, data, lengths, out)

    data_cpu = Array(data)
    out_cpu = Array(out)
    for bid in 1:3
        expected = zeros(Float32, 16)
        for j in 1:n_tiles[bid]
            expected .+= data_cpu[(j-1)*16+1 : j*16]
        end
        @test out_cpu[(bid-1)*16+1 : bid*16] ≈ expected
    end
end

@testset "for loop accumulation" begin
    function for_loop_acc_kernel(data::ct.TileArray{Float32,1},
                                 out::ct.TileArray{Float32,1},
                                 n_iters::Int32)
        pid = ct.bid(1)
        acc = ct.zeros((16,), Float32)
        for i in Int32(1):n_iters
            acc = acc .+ ct.load(data, i, (16,))
        end
        ct.store(out, pid, acc)
        return
    end

    n_iters = Int32(4)
    data = CUDA.rand(Float32, 64)   # 4 tiles of 16
    out = CUDA.zeros(Float32, 16)

    ct.launch(for_loop_acc_kernel, 1, data, out, n_iters)

    data_cpu = Array(data)
    expected = sum(reshape(data_cpu, 16, 4); dims=2)[:, 1]
    @test Array(out) ≈ expected
end

@testset "for loop with constant bound" begin
    function for_loop_const_kernel(data::ct.TileArray{Float32,1},
                                   out::ct.TileArray{Float32,1},
                                   N::Int)
        pid = ct.bid(1)
        acc = ct.zeros((16,), Float32)
        for i in Int32(1):Int32(N[])
            acc = acc .+ ct.load(data, i, (16,))
        end
        ct.store(out, pid, acc)
        return
    end

    data = CUDA.rand(Float32, 48)   # 3 tiles of 16
    out = CUDA.zeros(Float32, 16)

    ct.launch(for_loop_const_kernel, 1, data, out, ct.Constant(3))

    data_cpu = Array(data)
    expected = sum(reshape(data_cpu, 16, 3); dims=2)[:, 1]
    @test Array(out) ≈ expected
end

@testset "for loop store each iteration" begin
    function for_loop_store_kernel(inp::ct.TileArray{Float32,1},
                                   out::ct.TileArray{Float32,1},
                                   n_iters::Int32)
        pid = ct.bid(1)
        acc = ct.zeros((16,), Float32)
        for i in Int32(1):n_iters
            acc = acc .+ ct.load(inp, i, (16,))
        end
        ct.store(out, pid, acc)
        return
    end

    n_iters = Int32(2)
    inp = CUDA.ones(Float32, 32)    # 2 tiles of 16, all ones
    out = CUDA.zeros(Float32, 16)

    ct.launch(for_loop_store_kernel, 1, inp, out, n_iters)

    @test all(Array(out) .≈ 2.0f0)
end
