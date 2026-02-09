using CUDA

@testset "sum along axis 2" begin
    function sum_kernel(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        tile = ct.load(a, (pid, 1), (1, 128))
        sums = sum(tile; dims=2)
        ct.store(b, pid, sums)
        return
    end

    m, n = 64, 128
    a = CUDA.rand(Float32, m, n)
    b = CUDA.zeros(Float32, m)

    ct.launch(sum_kernel, m, a, b)

    a_cpu = Array(a)
    b_cpu = Array(b)
    for i in 1:m
        @test b_cpu[i] ≈ sum(a_cpu[i, :]) rtol=1e-3
    end
end

@testset "sum along axis 1" begin
    function sum_axis1_kernel(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        tile = ct.load(a, (1, pid), (64, 1))
        sums = sum(tile; dims=1)
        ct.store(b, pid, sums)
        return
    end

    m, n = 64, 128
    a = CUDA.rand(Float32, m, n)
    b = CUDA.zeros(Float32, n)

    ct.launch(sum_axis1_kernel, n, a, b)

    a_cpu = Array(a)
    b_cpu = Array(b)
    for j in 1:n
        @test b_cpu[j] ≈ sum(a_cpu[:, j]) rtol=1e-3
    end
end

@testset "maximum along axis 2" begin
    function maximum_kernel(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        tile = ct.load(a, (pid, 1), (1, 128))
        maxes = maximum(tile; dims=2)
        ct.store(b, pid, maxes)
        return
    end

    m, n = 64, 128
    a = CUDA.rand(Float32, m, n)
    b = CUDA.zeros(Float32, m)

    ct.launch(maximum_kernel, m, a, b)

    a_cpu = Array(a)
    b_cpu = Array(b)
    for i in 1:m
        @test b_cpu[i] ≈ maximum(a_cpu[i, :])
    end
end

@testset "minimum along axis 2" begin
    function minimum_kernel(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        tile = ct.load(a, (pid, 1), (1, 128))
        mins = minimum(tile; dims=2)
        ct.store(b, pid, mins)
        return
    end

    m, n = 64, 128
    a = CUDA.rand(Float32, m, n)
    b = CUDA.zeros(Float32, m)

    ct.launch(minimum_kernel, m, a, b)

    a_cpu = Array(a)
    b_cpu = Array(b)
    for i in 1:m
        @test b_cpu[i] ≈ minimum(a_cpu[i, :])
    end
end

@testset "prod along axis 2" begin
    function prod_kernel(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        tile = ct.load(a, (pid, 1), (1, 128))
        prods = prod(tile; dims=2)
        ct.store(b, pid, prods)
        return
    end

    m, n = 64, 128
    # Use small values to avoid overflow/underflow
    a = CuArray(rand(Float32, m, n) .* 0.1f0 .+ 0.95f0)
    b = CUDA.zeros(Float32, m)

    ct.launch(prod_kernel, m, a, b)

    a_cpu = Array(a)
    b_cpu = Array(b)
    for i in 1:m
        @test b_cpu[i] ≈ prod(a_cpu[i, :]) rtol=1e-2
    end
end

@testset "reduce with custom combiner" begin
    function custom_reduce_kernel(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        tile = ct.load(a, (pid, 1), (1, 128))
        sums = reduce((x, y) -> x + y, tile; dims=2, init=0.0f0)
        ct.store(b, pid, sums)
        return
    end

    m, n = 64, 128
    a = CUDA.rand(Float32, m, n)
    b = CUDA.zeros(Float32, m)

    ct.launch(custom_reduce_kernel, m, a, b)

    a_cpu = Array(a)
    b_cpu = Array(b)
    for i in 1:m
        @test b_cpu[i] ≈ sum(a_cpu[i, :]) rtol=1e-3
    end
end

@testset "map(abs, tile)" begin
    function map_abs_kernel(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,2})
        pid = ct.bid(1)
        tile = ct.load(a, (pid, 1), (1, 128))
        result = map(abs, tile)
        ct.store(b, (pid, 1), result)
        return
    end

    m, n = 64, 128
    a = CUDA.rand(Float32, m, n) .- 0.5f0
    b = CUDA.zeros(Float32, m, n)

    ct.launch(map_abs_kernel, m, a, b)

    @test Array(b) ≈ abs.(Array(a)) rtol=1e-5
end

@testset "mapreduce(abs, +, tile)" begin
    function mapreduce_abs_kernel(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        tile = ct.load(a, (pid, 1), (1, 128))
        sums = mapreduce(abs, +, tile; dims=2, init=0.0f0)
        ct.store(b, pid, sums)
        return
    end

    m, n = 64, 128
    a = CUDA.rand(Float32, m, n) .- 0.5f0
    b = CUDA.zeros(Float32, m)

    ct.launch(mapreduce_abs_kernel, m, a, b)

    a_cpu = Array(a)
    b_cpu = Array(b)
    for i in 1:m
        @test b_cpu[i] ≈ sum(abs, a_cpu[i, :]) rtol=1e-3
    end
end

@testset "mapreduce(x -> x * x, +, tile)" begin
    function mapreduce_sq_kernel(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        tile = ct.load(a, (pid, 1), (1, 128))
        sums = mapreduce(x -> x * x, +, tile; dims=2, init=0.0f0)
        ct.store(b, pid, sums)
        return
    end

    m, n = 64, 128
    a = CUDA.rand(Float32, m, n)
    b = CUDA.zeros(Float32, m)

    ct.launch(mapreduce_sq_kernel, m, a, b)

    a_cpu = Array(a)
    b_cpu = Array(b)
    for i in 1:m
        @test b_cpu[i] ≈ sum(x -> x^2, a_cpu[i, :]) rtol=1e-3
    end
end

@testset "dropdims" begin
    # Mean-subtract pattern: reduce row to get mean, dropdims the singleton,
    # then broadcast-subtract from the original tile and store the column norms.
    function dropdims_kernel(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        tile = ct.load(a, (pid, 1), (1, 128))            # (1, 128)
        row_sum = sum(tile; dims=2)                       # (1, 1)
        row_sum_1d = dropdims(row_sum; dims=2)            # (1,)
        ct.store(b, pid, row_sum_1d)
        return
    end

    m, n = 64, 128
    a = CUDA.rand(Float32, m, n)
    b = CUDA.zeros(Float32, m)

    ct.launch(dropdims_kernel, m, a, b)

    a_cpu = Array(a)
    b_cpu = Array(b)
    for i in 1:m
        @test b_cpu[i] ≈ sum(a_cpu[i, :]) rtol=1e-3
    end
end

@testset "1D cumsum (forward)" begin
    function cumsum_1d_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                              tile_size::ct.Constant{Int})
        bid = ct.bid(1)
        tile = ct.load(a, bid, (tile_size[],))
        result = cumsum(tile; dims=1)
        ct.store(b, bid, result)
        return nothing
    end

    sz = 32
    N = 1024
    a = CUDA.rand(Float32, N)
    b = CUDA.zeros(Float32, N)

    ct.launch(cumsum_1d_kernel, cld(N, sz), a, b, ct.Constant(sz))

    # Per-tile cumulative sum
    a_cpu = Array(a)
    b_cpu = Array(b)
    a_reshaped = reshape(a_cpu, sz, :)
    expected = mapslices(x -> accumulate(+, x), a_reshaped, dims=1)
    @test b_cpu ≈ vec(expected) rtol=1e-3
end

@testset "2D cumsum along axis 1" begin
    function cumsum_2d_axis1_kernel(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,2})
        pid = ct.bid(1)
        tile = ct.load(a, (pid, 1), (4, 8))
        result = cumsum(tile; dims=1)
        ct.store(b, (pid, 1), result)
        return nothing
    end

    m, n = 32, 8
    a = CUDA.rand(Float32, m, n)
    b = CUDA.zeros(Float32, m, n)

    ct.launch(cumsum_2d_axis1_kernel, cld(m, 4), a, b)

    a_cpu = Array(a)
    b_cpu = Array(b)
    # cumsum along dim 1 within each 4-row tile
    for bid in 0:(cld(m, 4)-1)
        rows = (bid*4+1):(bid*4+4)
        for j in 1:n
            @test b_cpu[rows, j] ≈ accumulate(+, a_cpu[rows, j]) rtol=1e-3
        end
    end
end

@testset "1D reverse cumsum" begin
    function reverse_cumsum_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                                    tile_size::ct.Constant{Int})
        bid = ct.bid(1)
        tile = ct.load(a, bid, (tile_size[],))
        result = cumsum(tile; dims=1, rev=true)
        ct.store(b, bid, result)
        return nothing
    end

    sz = 32
    N = 1024
    a = CUDA.rand(Float32, N)
    b = CUDA.zeros(Float32, N)

    ct.launch(reverse_cumsum_kernel, cld(N, sz), a, b, ct.Constant(sz))

    a_cpu = Array(a)
    b_cpu = Array(b)
    a_reshaped = reshape(a_cpu, sz, :)
    expected = mapslices(x -> reverse(accumulate(+, reverse(x))), a_reshaped, dims=1)
    @test b_cpu ≈ vec(expected) rtol=1e-3
end

@testset "1D cumprod" begin
    function cumprod_1d_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                               tile_size::ct.Constant{Int})
        bid = ct.bid(1)
        tile = ct.load(a, bid, (tile_size[],))
        result = cumprod(tile; dims=1)
        ct.store(b, bid, result)
        return nothing
    end

    sz = 32
    N = 1024
    # Use values close to 1.0 to avoid overflow/underflow
    a = CuArray(rand(Float32, N) .* 0.1f0 .+ 0.95f0)
    b = CUDA.zeros(Float32, N)

    ct.launch(cumprod_1d_kernel, cld(N, sz), a, b, ct.Constant(sz))

    a_cpu = Array(a)
    b_cpu = Array(b)
    a_reshaped = reshape(a_cpu, sz, :)
    expected = mapslices(x -> accumulate(*, x), a_reshaped, dims=1)
    @test b_cpu ≈ vec(expected) rtol=1e-2
end

@testset "1D reduce operations" begin
    TILE_SIZE = 32
    N = 1024

    function reduce_sum_1d(a::ct.TileArray{T,1}, b::ct.TileArray{T,1},
                           tileSz::ct.Constant{Int}) where {T}
        ct.store(b, ct.bid(1), sum(ct.load(a, ct.bid(1), (tileSz[],)); dims=1))
        return nothing
    end

    function reduce_max_1d(a::ct.TileArray{T,1}, b::ct.TileArray{T,1},
                           tileSz::ct.Constant{Int}) where {T}
        ct.store(b, ct.bid(1), maximum(ct.load(a, ct.bid(1), (tileSz[],)); dims=1))
        return nothing
    end

    function cpu_reduce(a_reshaped::AbstractArray{T}, op) where {T}
        result = mapslices(op, a_reshaped, dims=1)[:]
        # For unsigned sum, apply mask to handle overflow
        if T <: Unsigned && op === sum
            result .= result .& typemax(T)
        end
        return result
    end

    TEST_TYPES = [Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64, Float16, Float32, Float64]

    TEST_OPS = [
        (reduce_sum_1d, sum),
        (reduce_max_1d, maximum),
    ]

    @testset "Type: $elType, Operation: $gpu_kernel" for elType in TEST_TYPES, (gpu_kernel, cpu_op) in TEST_OPS
        # Generate input data with type-appropriate ranges to avoid overflow
        if elType == UInt8
            a_gpu = CuArray{UInt8}(rand(UInt8(0):UInt8(7), N))
        elseif elType == Int8
            a_gpu = CuArray{Int8}(rand(-3:3, N))
        elseif elType == Int16
            a_gpu = CuArray{Int16}(rand(-800:800, N))
        elseif elType == UInt16
            a_gpu = CuArray{UInt16}(rand(1:2000, N))
        elseif elType <: Integer && elType <: Signed
            a_gpu = CuArray{elType}(rand(-1000:1000, N))
        else
            a_gpu = CUDA.rand(elType, N)
        end
        b_gpu = CUDA.zeros(elType, cld(N, TILE_SIZE))

        ct.launch(gpu_kernel, cld(N, TILE_SIZE), a_gpu, b_gpu, ct.Constant(TILE_SIZE))

        a_cpu = Array(a_gpu)
        b_cpu = Array(b_gpu)
        a_reshaped = reshape(a_cpu, TILE_SIZE, :)
        cpu_result = cpu_reduce(a_reshaped, cpu_op)

        if elType <: AbstractFloat
            @test b_cpu ≈ cpu_result rtol=1e-3
        else
            @test b_cpu == cpu_result
        end
    end
end

@testset "1D scan (cumsum)" begin
    TILE_SIZE = 32
    N = 1024

    function scan_kernel(a::ct.TileArray{T,1}, b::ct.TileArray{T,1}, tileSz::ct.Constant{Int}) where {T}
        ct.store(b, ct.bid(1), cumsum(ct.load(a, ct.bid(1), (tileSz[],)); dims=1))
        return nothing
    end

    TEST_TYPES = [Float16, Float32, Float64, Int32, Int64, UInt32, UInt64]

    @testset "Type: $elType" for elType in TEST_TYPES
        # Type-appropriate input generation (small values to avoid overflow in cumsum)
        if elType <: Integer && elType <: Signed
            a_gpu = CuArray{elType}(rand(elType(-3):elType(3), N))
        elseif elType <: Integer
            a_gpu = CuArray{elType}(rand(elType(0):elType(7), N))
        else
            a_gpu = CUDA.rand(elType, N)
        end
        b_gpu = CUDA.zeros(elType, N)

        ct.launch(scan_kernel, cld(N, TILE_SIZE), a_gpu, b_gpu, ct.Constant(TILE_SIZE))

        a_cpu = Array(a_gpu)
        b_cpu = Array(b_gpu)

        # CPU reference: per-tile cumulative sum
        a_reshaped = reshape(a_cpu, TILE_SIZE, :)
        expected = mapslices(x -> accumulate(+, x), a_reshaped, dims=1)

        if elType <: AbstractFloat
            @test b_cpu ≈ vec(expected) rtol=1e-3
        else
            @test b_cpu == vec(expected)
        end
    end
end

@testset "any / all" begin
    TILE_SIZE = 16

    function any_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Int32,1},
                        tileSz::ct.Constant{Int})
        tile = ct.load(a, ct.bid(1), (tileSz[],))
        mask = tile .> 0.0f0
        result = any(mask; dims=1)
        ct.store(b, ct.bid(1), convert(ct.Tile{Int32}, result))
        return nothing
    end

    function all_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Int32,1},
                        tileSz::ct.Constant{Int})
        tile = ct.load(a, ct.bid(1), (tileSz[],))
        mask = tile .> 0.0f0
        result = all(mask; dims=1)
        ct.store(b, ct.bid(1), convert(ct.Tile{Int32}, result))
        return nothing
    end

    N = 64
    n_blocks = cld(N, TILE_SIZE)

    # All positive → any=true, all=true
    a_pos = CUDA.ones(Float32, N)
    b_any = CUDA.zeros(Int32, n_blocks)
    b_all = CUDA.zeros(Int32, n_blocks)
    ct.launch(any_kernel, n_blocks, a_pos, b_any, ct.Constant(TILE_SIZE))
    ct.launch(all_kernel, n_blocks, a_pos, b_all, ct.Constant(TILE_SIZE))
    @test all(Array(b_any) .== 1)
    @test all(Array(b_all) .== 1)

    # All negative → any=false, all=false
    a_neg = CUDA.fill(Float32(-1), N)
    b_any = CUDA.zeros(Int32, n_blocks)
    b_all = CUDA.zeros(Int32, n_blocks)
    ct.launch(any_kernel, n_blocks, a_neg, b_any, ct.Constant(TILE_SIZE))
    ct.launch(all_kernel, n_blocks, a_neg, b_all, ct.Constant(TILE_SIZE))
    @test all(Array(b_any) .== 0)
    @test all(Array(b_all) .== 0)

    # Mixed → any=true, all=false (first element positive, rest negative)
    a_mix = CUDA.fill(Float32(-1), N)
    # Set first element of each tile to positive
    a_mix_cpu = Array(a_mix)
    for i in 1:TILE_SIZE:N
        a_mix_cpu[i] = 1.0f0
    end
    a_mix = CuArray(a_mix_cpu)
    b_any = CUDA.zeros(Int32, n_blocks)
    b_all = CUDA.zeros(Int32, n_blocks)
    ct.launch(any_kernel, n_blocks, a_mix, b_any, ct.Constant(TILE_SIZE))
    ct.launch(all_kernel, n_blocks, a_mix, b_all, ct.Constant(TILE_SIZE))
    @test all(Array(b_any) .== 1)
    @test all(Array(b_all) .== 0)
end

@testset "count" begin
    TILE_SIZE = 16

    function count_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Int32,1},
                          tileSz::ct.Constant{Int})
        tile = ct.load(a, ct.bid(1), (tileSz[],))
        result = count(tile .> 0.0f0; dims=1)
        ct.store(b, ct.bid(1), result)
        return nothing
    end

    N = 64
    n_blocks = cld(N, TILE_SIZE)

    # Known pattern: 3 positive per tile
    a_cpu = fill(Float32(-1), N)
    for i in 1:TILE_SIZE:N
        a_cpu[i] = 1.0f0
        a_cpu[i+1] = 2.0f0
        a_cpu[i+2] = 3.0f0
    end
    a = CuArray(a_cpu)
    b = CUDA.zeros(Int32, n_blocks)

    ct.launch(count_kernel, n_blocks, a, b, ct.Constant(TILE_SIZE))

    @test all(Array(b) .== 3)
end

@testset "argmax / argmin" begin
    TILE_SIZE = 16

    function argmax_kernel(a::ct.TileArray{Float32,2}, b::ct.TileArray{Int32,2})
        tile = ct.load(a, ct.bid(1), (4, 16))
        result = argmax(tile; dims=2)
        ct.store(b, ct.bid(1), result)
        return nothing
    end

    function argmin_kernel(a::ct.TileArray{Float32,2}, b::ct.TileArray{Int32,2})
        tile = ct.load(a, ct.bid(1), (4, 16))
        result = argmin(tile; dims=2)
        ct.store(b, ct.bid(1), result)
        return nothing
    end

    m, n = 4, 16
    # Create data with known argmax/argmin positions
    a_cpu = zeros(Float32, m, n)
    for row in 1:m
        for col in 1:n
            a_cpu[row, col] = Float32(col)  # max at col 16, min at col 1
        end
    end
    a = CuArray(a_cpu)
    b_max = CUDA.zeros(Int32, m, 1)
    b_min = CUDA.zeros(Int32, m, 1)

    ct.launch(argmax_kernel, 1, a, b_max)
    ct.launch(argmin_kernel, 1, a, b_min)

    b_max_cpu = Array(b_max)
    b_min_cpu = Array(b_min)

    # argmax should return 16 (1-indexed) for all rows
    @test all(b_max_cpu .== 16)
    # argmin should return 1 (1-indexed) for all rows
    @test all(b_min_cpu .== 1)

    # Test with random data
    a_rand = CUDA.rand(Float32, m, n)
    b_max_rand = CUDA.zeros(Int32, m, 1)
    b_min_rand = CUDA.zeros(Int32, m, 1)

    ct.launch(argmax_kernel, 1, a_rand, b_max_rand)
    ct.launch(argmin_kernel, 1, a_rand, b_min_rand)

    a_rand_cpu = Array(a_rand)
    # Compare with CPU argmax/argmin (Julia returns CartesianIndex, extract column)
    for row in 1:m
        expected_max = argmax(a_rand_cpu[row, :])
        expected_min = argmin(a_rand_cpu[row, :])
        @test Array(b_max_rand)[row, 1] == expected_max
        @test Array(b_min_rand)[row, 1] == expected_min
    end
end
