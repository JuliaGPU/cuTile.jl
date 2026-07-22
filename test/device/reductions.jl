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

    @cuda backend=cuTile blocks=m sum_kernel(a, b)

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

    @cuda backend=cuTile blocks=n sum_axis1_kernel(a, b)

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

    @cuda backend=cuTile blocks=m maximum_kernel(a, b)

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

    @cuda backend=cuTile blocks=m minimum_kernel(a, b)

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

    @cuda backend=cuTile blocks=m prod_kernel(a, b)

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

    @cuda backend=cuTile blocks=m custom_reduce_kernel(a, b)

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

    @cuda backend=cuTile blocks=m map_abs_kernel(a, b)

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

    @cuda backend=cuTile blocks=m mapreduce_abs_kernel(a, b)

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

    @cuda backend=cuTile blocks=m mapreduce_sq_kernel(a, b)

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

    @cuda backend=cuTile blocks=m dropdims_kernel(a, b)

    a_cpu = Array(a)
    b_cpu = Array(b)
    for i in 1:m
        @test b_cpu[i] ≈ sum(a_cpu[i, :]) rtol=1e-3
    end
end

@testset "1D cumsum (forward)" begin
    function cumsum_1d_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                              tile_size::Int)
        bid = ct.bid(1)
        tile = ct.load(a, bid, (tile_size,))
        result = cumsum(tile; dims=1)
        ct.store(b, bid, result)
        return nothing
    end

    sz = 32
    N = 1024
    a = CUDA.rand(Float32, N)
    b = CUDA.zeros(Float32, N)

    @cuda backend=cuTile blocks=cld(N, sz) cumsum_1d_kernel(a, b, ct.Constant(sz))

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

    @cuda backend=cuTile blocks=cld(m, 4) cumsum_2d_axis1_kernel(a, b)

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
                                    tile_size::Int)
        bid = ct.bid(1)
        tile = ct.load(a, bid, (tile_size,))
        result = cumsum(tile; dims=1, rev=true)
        ct.store(b, bid, result)
        return nothing
    end

    sz = 32
    N = 1024
    a = CUDA.rand(Float32, N)
    b = CUDA.zeros(Float32, N)

    @cuda backend=cuTile blocks=cld(N, sz) reverse_cumsum_kernel(a, b, ct.Constant(sz))

    a_cpu = Array(a)
    b_cpu = Array(b)
    a_reshaped = reshape(a_cpu, sz, :)
    expected = mapslices(x -> reverse(accumulate(+, reverse(x))), a_reshaped, dims=1)
    @test b_cpu ≈ vec(expected) rtol=1e-3
end

@testset "1D cumprod" begin
    function cumprod_1d_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                               tile_size::Int)
        bid = ct.bid(1)
        tile = ct.load(a, bid, (tile_size,))
        result = cumprod(tile; dims=1)
        ct.store(b, bid, result)
        return nothing
    end

    sz = 32
    N = 1024
    # Use values close to 1.0 to avoid overflow/underflow
    a = CuArray(rand(Float32, N) .* 0.1f0 .+ 0.95f0)
    b = CUDA.zeros(Float32, N)

    @cuda backend=cuTile blocks=cld(N, sz) cumprod_1d_kernel(a, b, ct.Constant(sz))

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
                           tileSz::Int) where {T}
        ct.store(b, ct.bid(1), sum(ct.load(a, ct.bid(1), (tileSz,)); dims=1))
        return nothing
    end

    function reduce_max_1d(a::ct.TileArray{T,1}, b::ct.TileArray{T,1},
                           tileSz::Int) where {T}
        ct.store(b, ct.bid(1), maximum(ct.load(a, ct.bid(1), (tileSz,)); dims=1))
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

        @cuda backend=cuTile blocks=cld(N, TILE_SIZE) gpu_kernel(a_gpu, b_gpu, ct.Constant(TILE_SIZE))

        a_cpu = Array(a_gpu)
        b_cpu = Array(b_gpu)
        a_reshaped = reshape(a_cpu, TILE_SIZE, :)
        cpu_result = cpu_reduce(a_reshaped, cpu_op)

        if elType <: AbstractFloat
            @test b_cpu ≈ cpu_result
        else
            @test b_cpu == cpu_result
        end
    end
end

@testset "1D scan (cumsum)" begin
    TILE_SIZE = 32
    N = 1024

    function scan_kernel(a::ct.TileArray{T,1}, b::ct.TileArray{T,1}, tileSz::Int) where {T}
        ct.store(b, ct.bid(1), cumsum(ct.load(a, ct.bid(1), (tileSz,)); dims=1))
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

        @cuda backend=cuTile blocks=cld(N, TILE_SIZE) scan_kernel(a_gpu, b_gpu, ct.Constant(TILE_SIZE))

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
                        tileSz::Int)
        tile = ct.load(a, ct.bid(1), (tileSz,))
        mask = tile .> 0.0f0
        result = any(mask; dims=1)
        ct.store(b, ct.bid(1), convert(ct.Tile{Int32}, result))
        return nothing
    end

    function all_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Int32,1},
                        tileSz::Int)
        tile = ct.load(a, ct.bid(1), (tileSz,))
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
    @cuda backend=cuTile blocks=n_blocks any_kernel(a_pos, b_any, ct.Constant(TILE_SIZE))
    @cuda backend=cuTile blocks=n_blocks all_kernel(a_pos, b_all, ct.Constant(TILE_SIZE))
    @test all(Array(b_any) .== 1)
    @test all(Array(b_all) .== 1)

    # All negative → any=false, all=false
    a_neg = CUDA.fill(Float32(-1), N)
    b_any = CUDA.zeros(Int32, n_blocks)
    b_all = CUDA.zeros(Int32, n_blocks)
    @cuda backend=cuTile blocks=n_blocks any_kernel(a_neg, b_any, ct.Constant(TILE_SIZE))
    @cuda backend=cuTile blocks=n_blocks all_kernel(a_neg, b_all, ct.Constant(TILE_SIZE))
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
    @cuda backend=cuTile blocks=n_blocks any_kernel(a_mix, b_any, ct.Constant(TILE_SIZE))
    @cuda backend=cuTile blocks=n_blocks all_kernel(a_mix, b_all, ct.Constant(TILE_SIZE))
    @test all(Array(b_any) .== 1)
    @test all(Array(b_all) .== 0)
end

@testset "count" begin
    TILE_SIZE = 16

    function count_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Int32,1},
                          tileSz::Int)
        tile = ct.load(a, ct.bid(1), (tileSz,))
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

    @cuda backend=cuTile blocks=n_blocks count_kernel(a, b, ct.Constant(TILE_SIZE))

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

    @cuda backend=cuTile argmax_kernel(a, b_max)
    @cuda backend=cuTile argmin_kernel(a, b_min)

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

    @cuda backend=cuTile argmax_kernel(a_rand, b_max_rand)
    @cuda backend=cuTile argmin_kernel(a_rand, b_min_rand)

    a_rand_cpu = Array(a_rand)
    # Compare with CPU argmax/argmin (Julia returns CartesianIndex, extract column)
    for row in 1:m
        expected_max = argmax(a_rand_cpu[row, :])
        expected_min = argmin(a_rand_cpu[row, :])
        @test Array(b_max_rand)[row, 1] == expected_max
        @test Array(b_min_rand)[row, 1] == expected_min
    end
end

@testset "NaN reduction semantics" begin
    function nan_reduction_kernel(a::ct.TileArray{Float32,2},
                                  idx_default::ct.TileArray{Int32,2},
                                  idx_propagate::ct.TileArray{Int32,2},
                                  val_default::ct.TileArray{Float32,2},
                                  val_propagate::ct.TileArray{Float32,2})
        tile = ct.load(a, (1, 1), (2, 16))
        ct.store(idx_default, (1, 1), argmax(tile; dims=2))
        ct.store(idx_default, (1, 2), argmin(tile; dims=2))
        ct.store(idx_propagate, (1, 1), argmax(tile; dims=2, propagate_nan=true))
        ct.store(idx_propagate, (1, 2), argmin(tile; dims=2, propagate_nan=true))
        ct.store(val_default, (1, 1), maximum(tile; dims=2))
        ct.store(val_default, (1, 2), minimum(tile; dims=2))
        ct.store(val_propagate, (1, 1), maximum(tile; dims=2, propagate_nan=true))
        ct.store(val_propagate, (1, 2), minimum(tile; dims=2, propagate_nan=true))
        return nothing
    end

    a_cpu = repeat(reshape(Float32.(1:16), 1, 16), 2, 1)
    a_cpu[1, 1] = NaN
    a_cpu[2, 5] = NaN
    a_cpu[2, 9] = NaN
    a = CuArray(a_cpu)
    idx_default = CUDA.zeros(Int32, 2, 2)
    idx_propagate = CUDA.zeros(Int32, 2, 2)
    val_default = CUDA.zeros(Float32, 2, 2)
    val_propagate = CUDA.zeros(Float32, 2, 2)

    @cuda backend=cuTile nan_reduction_kernel(
        a, idx_default, idx_propagate, val_default, val_propagate)

    @test Array(idx_default) == Int32[16 2; 16 1]
    @test Array(idx_propagate) == Int32[1 1; 5 5]
    @test Array(val_default) == Float32[16 2; 16 1]
    @test all(isnan, Array(val_propagate))
end

@testset "sum without dims (1D)" begin
    function sum_no_dims_1d(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1}, tileSz::Int)
        tile = ct.load(a, ct.bid(1), (tileSz,))
        b[ct.bid(1)] = sum(tile)
        return nothing
    end

    sz = 32; N = 1024
    a = CUDA.rand(Float32, N)
    b = CUDA.zeros(Float32, cld(N, sz))
    @cuda backend=cuTile blocks=cld(N, sz) sum_no_dims_1d(a, b, ct.Constant(sz))

    a_cpu = reshape(Array(a), sz, :)
    @test Array(b) ≈ vec(sum(a_cpu; dims=1)) rtol=1e-3
end

@testset "sum without dims (2D)" begin
    function sum_no_dims_2d(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        tile = ct.load(a, (pid, 1), (1, 128))
        b[pid] = sum(tile)
        return nothing
    end

    m, n = 64, 128
    a = CUDA.rand(Float32, m, n)
    b = CUDA.zeros(Float32, m)
    @cuda backend=cuTile blocks=m sum_no_dims_2d(a, b)

    a_cpu = Array(a)
    for i in 1:m
        @test Array(b)[i] ≈ sum(a_cpu[i, :]) rtol=1e-3
    end
end

@testset "any without dims (1D)" begin
    function any_no_dims_1d(a::ct.TileArray{Float32,1}, b::ct.TileArray{Int32,1}, tileSz::Int)
        tile = ct.load(a, ct.bid(1), (tileSz,))
        # Store as Int32 since we can't store Bool scalars directly
        b[ct.bid(1)] = Int32(any(tile .> 0.0f0))
        return nothing
    end

    sz = 32; N = 1024
    a = CUDA.rand(Float32, N) .- 0.5f0  # some positive, some negative
    b = CUDA.zeros(Int32, cld(N, sz))
    @cuda backend=cuTile blocks=cld(N, sz) any_no_dims_1d(a, b, ct.Constant(sz))

    a_cpu = reshape(Array(a), sz, :)
    for i in 1:cld(N, sz)
        @test Array(b)[i] == Int32(any(a_cpu[:, i] .> 0.0f0))
    end
end

@testset "all without dims (1D)" begin
    function all_no_dims_1d(a::ct.TileArray{Float32,1}, b::ct.TileArray{Int32,1}, tileSz::Int)
        tile = ct.load(a, ct.bid(1), (tileSz,))
        b[ct.bid(1)] = Int32(all(tile .> 0.0f0))
        return nothing
    end

    sz = 32; N = 1024
    a = CUDA.rand(Float32, N)  # all positive
    b = CUDA.zeros(Int32, cld(N, sz))
    @cuda backend=cuTile blocks=cld(N, sz) all_no_dims_1d(a, b, ct.Constant(sz))

    a_cpu = reshape(Array(a), sz, :)
    for i in 1:cld(N, sz)
        @test Array(b)[i] == Int32(all(a_cpu[:, i] .> 0.0f0))
    end
end

@testset "scalar arithmetic on reduction result" begin
    # Regression test: sum(tile) produces a 0D tile via reshape(tile, ()),
    # and dividing that by a scalar kernel arg must not fail due to shape
    # kind mismatch between the two 0D values (both must be RowMajorShape).
    function scalar_reduce_div(X::ct.TileArray{Float32,2}, Y::ct.TileArray{Float32,2},
                               M::Int, TILE_M::Int)
        bid_n = ct.bid(1)
        x = ct.load(X, (1, bid_n), (TILE_M,))
        rstd = 1 / sqrt(sum(x .* x) / M .+ 1.0f-6)
        ct.store(Y, (1, bid_n), x .* rstd)
        return
    end

    M, N = 128, 4
    X = CUDA.randn(Float32, M, N)
    Y = similar(X)
    @cuda backend=cuTile blocks=N scalar_reduce_div(X, Y, ct.Constant(M), ct.Constant(M))

    X_cpu = Array(X)
    Y_cpu = Array(Y)
    for j in 1:N
        col = X_cpu[:, j]
        rstd = 1 / sqrt(sum(col .^ 2) / M + 1f-6)
        @test Y_cpu[:, j] ≈ col .* rstd rtol=1e-3
    end
end

@testset "sum without dims (3D)" begin
    function sum_no_dims_3d(a::ct.TileArray{Float32,3}, b::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        tile = ct.load(a, (pid, 1, 1), (1, 8, 16))
        b[pid] = sum(tile)
        return nothing
    end

    d1, d2, d3 = 4, 8, 16
    a = CUDA.rand(Float32, d1, d2, d3)
    b = CUDA.zeros(Float32, d1)
    @cuda backend=cuTile blocks=d1 sum_no_dims_3d(a, b)

    a_cpu = Array(a)
    for i in 1:d1
        @test Array(b)[i] ≈ sum(a_cpu[i, :, :]) rtol=1e-3
    end
end

@testset "maximum without dims (3D)" begin
    function maximum_no_dims_3d(a::ct.TileArray{Float32,3}, b::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        tile = ct.load(a, (pid, 1, 1), (1, 8, 16))
        b[pid] = maximum(tile)
        return nothing
    end

    d1, d2, d3 = 4, 8, 16
    a = CUDA.rand(Float32, d1, d2, d3)
    b = CUDA.zeros(Float32, d1)
    @cuda backend=cuTile blocks=d1 maximum_no_dims_3d(a, b)

    a_cpu = Array(a)
    for i in 1:d1
        @test Array(b)[i] ≈ maximum(a_cpu[i, :, :]) rtol=1e-3
    end
end

@testset "count without dims (1D)" begin
    function count_no_dims_1d(a::ct.TileArray{Float32,1}, b::ct.TileArray{Int32,1}, tileSz::Int)
        tile = ct.load(a, ct.bid(1), (tileSz,))
        b[ct.bid(1)] = count(tile .> 0.0f0)
        return nothing
    end

    sz = 32; N = 1024
    a = CUDA.rand(Float32, N) .- 0.5f0
    b = CUDA.zeros(Int32, cld(N, sz))
    @cuda backend=cuTile blocks=cld(N, sz) count_no_dims_1d(a, b, ct.Constant(sz))

    a_cpu = reshape(Array(a), sz, :)
    for i in 1:cld(N, sz)
        @test Array(b)[i] == count(a_cpu[:, i] .> 0.0f0)
    end
end
