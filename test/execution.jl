@testset "execution" begin

using CUDA

@testset "launch" begin

@testset "1D vector add" begin
    function vadd_1d(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                     c::ct.TileArray{Float32,1})
        pid = ct.bid(0)
        tile_a = ct.load(a, pid, (16,))
        tile_b = ct.load(b, pid, (16,))
        ct.store(c, pid, tile_a + tile_b)
        return
    end

    n = 1024
    tile_size = 16
    a = CUDA.rand(Float32, n)
    b = CUDA.rand(Float32, n)
    c = CUDA.zeros(Float32, n)

    ct.launch(vadd_1d, cld(n, tile_size), a, b, c)

    @test Array(c) ≈ Array(a) + Array(b)
end

@testset "1D vector sub" begin
    function vsub_1d(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                     c::ct.TileArray{Float32,1})
        pid = ct.bid(0)
        tile_a = ct.load(a, pid, (16,))
        tile_b = ct.load(b, pid, (16,))
        ct.store(c, pid, tile_a - tile_b)
        return
    end

    n = 1024
    tile_size = 16
    a = CUDA.rand(Float32, n)
    b = CUDA.rand(Float32, n)
    c = CUDA.zeros(Float32, n)

    ct.launch(vsub_1d, cld(n, tile_size), a, b, c)

    @test Array(c) ≈ Array(a) - Array(b)
end

@testset "1D vector mul" begin
    function vmul_1d(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                     c::ct.TileArray{Float32,1})
        pid = ct.bid(0)
        tile_a = ct.load(a, pid, (16,))
        tile_b = ct.load(b, pid, (16,))
        ct.store(c, pid, tile_a * tile_b)
        return
    end

    n = 1024
    tile_size = 16
    a = CUDA.rand(Float32, n)
    b = CUDA.rand(Float32, n)
    c = CUDA.zeros(Float32, n)

    ct.launch(vmul_1d, cld(n, tile_size), a, b, c)

    @test Array(c) ≈ Array(a) .* Array(b)
end

@testset "2D matrix add" begin
    function madd_2d(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,2},
                     c::ct.TileArray{Float32,2})
        bidx = ct.bid(0)
        bidy = ct.bid(1)
        tile_a = ct.load(a, (bidx, bidy), (32, 32))
        tile_b = ct.load(b, (bidx, bidy), (32, 32))
        ct.store(c, (bidx, bidy), tile_a + tile_b)
        return
    end

    m, n = 256, 256
    tile_x, tile_y = 32, 32
    a = CUDA.rand(Float32, m, n)
    b = CUDA.rand(Float32, m, n)
    c = CUDA.zeros(Float32, m, n)

    ct.launch(madd_2d, (cld(m, tile_x), cld(n, tile_y)), a, b, c)

    @test Array(c) ≈ Array(a) + Array(b)
end

@testset "transpose" begin
    function transpose_kernel(x::ct.TileArray{Float32,2}, y::ct.TileArray{Float32,2})
        bidx = ct.bid(0)
        bidy = ct.bid(1)
        tile = ct.load(x, (bidx, bidy), (32, 32))
        transposed = ct.transpose(tile)
        ct.store(y, (bidy, bidx), transposed)
        return
    end

    m, n = 256, 128
    tile_size = 32
    x = CUDA.rand(Float32, m, n)
    y = CUDA.zeros(Float32, n, m)

    ct.launch(transpose_kernel, (cld(m, tile_size), cld(n, tile_size)), x, y)

    @test Array(y) ≈ transpose(Array(x))
end

end

@testset "Constant parameters" begin

@testset "1D with Constant tile size" begin
    function vadd_const_tile(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                             c::ct.TileArray{Float32,1}, tile::ct.Constant{Int})
        pid = ct.bid(0)
        tile_a = ct.load(a, pid, (tile[],))
        tile_b = ct.load(b, pid, (tile[],))
        ct.store(c, pid, tile_a + tile_b)
        return
    end

    n = 1024
    tile_size = 32
    a = CUDA.rand(Float32, n)
    b = CUDA.rand(Float32, n)
    c = CUDA.zeros(Float32, n)

    ct.launch(vadd_const_tile, cld(n, tile_size), a, b, c, ct.Constant(tile_size))

    @test Array(c) ≈ Array(a) + Array(b)
end

@testset "2D with Constant tile sizes" begin
    function madd_const_tiles(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,2},
                              c::ct.TileArray{Float32,2},
                              tx::ct.Constant{Int}, ty::ct.Constant{Int})
        bidx = ct.bid(0)
        bidy = ct.bid(1)
        tile_a = ct.load(a, (bidx, bidy), (tx[], ty[]))
        tile_b = ct.load(b, (bidx, bidy), (tx[], ty[]))
        ct.store(c, (bidx, bidy), tile_a + tile_b)
        return
    end

    m, n = 256, 256
    tile_x, tile_y = 64, 64
    a = CUDA.rand(Float32, m, n)
    b = CUDA.rand(Float32, m, n)
    c = CUDA.zeros(Float32, m, n)

    ct.launch(madd_const_tiles, (cld(m, tile_x), cld(n, tile_y)), a, b, c,
              ct.Constant(tile_x), ct.Constant(tile_y))

    @test Array(c) ≈ Array(a) + Array(b)
end

end

@testset "data types" begin

@testset "Float64" begin
    function vadd_f64(a::ct.TileArray{Float64,1}, b::ct.TileArray{Float64,1},
                      c::ct.TileArray{Float64,1})
        pid = ct.bid(0)
        tile_a = ct.load(a, pid, (16,))
        tile_b = ct.load(b, pid, (16,))
        ct.store(c, pid, tile_a + tile_b)
        return
    end

    n = 1024
    tile_size = 16
    a = CUDA.rand(Float64, n)
    b = CUDA.rand(Float64, n)
    c = CUDA.zeros(Float64, n)

    ct.launch(vadd_f64, cld(n, tile_size), a, b, c)

    @test Array(c) ≈ Array(a) + Array(b)
end

@testset "Float16" begin
    function vadd_f16(a::ct.TileArray{Float16,1}, b::ct.TileArray{Float16,1},
                      c::ct.TileArray{Float16,1})
        pid = ct.bid(0)
        tile_a = ct.load(a, pid, (16,))
        tile_b = ct.load(b, pid, (16,))
        ct.store(c, pid, tile_a + tile_b)
        return
    end

    n = 1024
    tile_size = 16
    a = CUDA.rand(Float16, n)
    b = CUDA.rand(Float16, n)
    c = CUDA.zeros(Float16, n)

    ct.launch(vadd_f16, cld(n, tile_size), a, b, c)

    @test Array(c) ≈ Array(a) + Array(b)
end

end

@testset "compilation cache" begin
    function cached_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1})
        pid = ct.bid(0)
        tile = ct.load(a, pid, (16,))
        ct.store(b, pid, tile)
        return
    end

    n = 256
    tile_size = 16
    a = CUDA.rand(Float32, n)
    b = CUDA.zeros(Float32, n)

    # First launch triggers compilation
    ct.launch(cached_kernel, cld(n, tile_size), a, b)
    @test Array(b) ≈ Array(a)

    # Second launch should use cached CuFunction
    a2 = CUDA.rand(Float32, n)
    b2 = CUDA.zeros(Float32, n)
    ct.launch(cached_kernel, cld(n, tile_size), a2, b2)
    @test Array(b2) ≈ Array(a2)
end

@testset "TileArray auto-conversion" begin
    # Test that CuArrays are automatically converted to TileArray
    function copy_kernel(src::ct.TileArray{Float32,1}, dst::ct.TileArray{Float32,1})
        pid = ct.bid(0)
        tile = ct.load(src, pid, (16,))
        ct.store(dst, pid, tile)
        return
    end

    n = 512
    tile_size = 16
    src = CUDA.rand(Float32, n)
    dst = CUDA.zeros(Float32, n)

    # Pass CuArrays directly - should auto-convert
    ct.launch(copy_kernel, cld(n, tile_size), src, dst)

    @test Array(dst) ≈ Array(src)
end

end

@testset "math operations" begin

@testset "1D vector div" begin
    function vdiv_1d(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                     c::ct.TileArray{Float32,1})
        pid = ct.bid(0)
        tile_a = ct.load(a, pid, (16,))
        tile_b = ct.load(b, pid, (16,))
        ct.store(c, pid, tile_a / tile_b)
        return
    end

    n = 1024
    tile_size = 16
    a = CUDA.rand(Float32, n)
    b = CUDA.rand(Float32, n) .+ 0.1f0  # Ensure non-zero
    c = CUDA.zeros(Float32, n)

    ct.launch(vdiv_1d, cld(n, tile_size), a, b, c)

    @test Array(c) ≈ Array(a) ./ Array(b) rtol=1e-5
end

@testset "1D sqrt" begin
    function vsqrt_1d(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1})
        pid = ct.bid(0)
        tile = ct.load(a, pid, (16,))
        ct.store(b, pid, sqrt(tile))
        return
    end

    n = 1024
    tile_size = 16
    a = CUDA.rand(Float32, n) .+ 0.1f0  # Ensure positive
    b = CUDA.zeros(Float32, n)

    ct.launch(vsqrt_1d, cld(n, tile_size), a, b)

    @test Array(b) ≈ sqrt.(Array(a)) rtol=1e-5
end

end

@testset "reduction operations" begin

@testset "reduce_sum along axis 1" begin
    function reduce_sum_kernel(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,1})
        pid = ct.bid(0)
        tile = ct.load(a, (pid, 0), (1, 128))  # Load 1x128 tile
        sums = ct.reduce_sum(tile, 1)           # Sum along axis 1 -> (1,)
        ct.store(b, pid, sums)
        return
    end

    m, n = 64, 128
    a = CUDA.rand(Float32, m, n)
    b = CUDA.zeros(Float32, m)

    ct.launch(reduce_sum_kernel, m, a, b)

    # Each row should be summed
    a_cpu = Array(a)
    b_cpu = Array(b)
    for i in 1:m
        @test b_cpu[i] ≈ sum(a_cpu[i, :]) rtol=1e-3
    end
end

@testset "reduce_sum along axis 0" begin
    function reduce_sum_axis0_kernel(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,1})
        pid = ct.bid(0)
        tile = ct.load(a, (0, pid), (64, 1))   # Load 64x1 tile
        sums = ct.reduce_sum(tile, 0)           # Sum along axis 0 -> (1,)
        ct.store(b, pid, sums)
        return
    end

    m, n = 64, 128
    a = CUDA.rand(Float32, m, n)
    b = CUDA.zeros(Float32, n)

    ct.launch(reduce_sum_axis0_kernel, n, a, b)

    # Each column should be summed
    a_cpu = Array(a)
    b_cpu = Array(b)
    for j in 1:n
        @test b_cpu[j] ≈ sum(a_cpu[:, j]) rtol=1e-3
    end
end

@testset "reduce_max along axis 1" begin
    function reduce_max_kernel(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,1})
        pid = ct.bid(0)
        tile = ct.load(a, (pid, 0), (1, 128))  # Load 1x128 tile
        maxes = ct.reduce_max(tile, 1)          # Max along axis 1 -> (1,)
        ct.store(b, pid, maxes)
        return
    end

    m, n = 64, 128
    a = CUDA.rand(Float32, m, n)
    b = CUDA.zeros(Float32, m)

    ct.launch(reduce_max_kernel, m, a, b)

    # Each row should have its max
    a_cpu = Array(a)
    b_cpu = Array(b)
    for i in 1:m
        @test b_cpu[i] ≈ maximum(a_cpu[i, :]) rtol=1e-5
    end
end

end

@testset "scalar-tile operations" begin

@testset "tile / scalar" begin
    function div_by_scalar(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1})
        pid = ct.bid(0)
        tile = ct.load(a, pid, (16,))
        result = tile / 2.0f0
        ct.store(b, pid, result)
        return
    end

    n = 1024
    tile_size = 16
    a = CUDA.rand(Float32, n)
    b = CUDA.zeros(Float32, n)

    ct.launch(div_by_scalar, cld(n, tile_size), a, b)

    @test Array(b) ≈ Array(a) ./ 2.0f0 rtol=1e-5
end

@testset "tile / integer" begin
    function div_by_int(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1})
        pid = ct.bid(0)
        tile = ct.load(a, pid, (16,))
        result = tile / 4
        ct.store(b, pid, result)
        return
    end

    n = 1024
    tile_size = 16
    a = CUDA.rand(Float32, n)
    b = CUDA.zeros(Float32, n)

    ct.launch(div_by_int, cld(n, tile_size), a, b)

    @test Array(b) ≈ Array(a) ./ 4.0f0 rtol=1e-5
end

@testset "scalar / tile" begin
    function scalar_div_tile_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1})
        pid = ct.bid(0)
        tile = ct.load(a, pid, (16,))
        result = 1.0f0 / tile
        ct.store(b, pid, result)
        return
    end

    n = 1024
    tile_size = 16
    a = CUDA.rand(Float32, n) .+ 0.1f0  # Ensure non-zero
    b = CUDA.zeros(Float32, n)

    ct.launch(scalar_div_tile_kernel, cld(n, tile_size), a, b)

    @test Array(b) ≈ 1.0f0 ./ Array(a) rtol=1e-5
end

@testset "tile + scalar" begin
    function add_scalar(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1})
        pid = ct.bid(0)
        tile = ct.load(a, pid, (16,))
        result = tile + 3.5f0
        ct.store(b, pid, result)
        return
    end

    n = 1024
    tile_size = 16
    a = CUDA.rand(Float32, n)
    b = CUDA.zeros(Float32, n)

    ct.launch(add_scalar, cld(n, tile_size), a, b)

    @test Array(b) ≈ Array(a) .+ 3.5f0 rtol=1e-5
end

@testset "scalar + tile" begin
    function scalar_add_tile_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1})
        pid = ct.bid(0)
        tile = ct.load(a, pid, (16,))
        result = 2.5f0 + tile
        ct.store(b, pid, result)
        return
    end

    n = 1024
    tile_size = 16
    a = CUDA.rand(Float32, n)
    b = CUDA.zeros(Float32, n)

    ct.launch(scalar_add_tile_kernel, cld(n, tile_size), a, b)

    @test Array(b) ≈ 2.5f0 .+ Array(a) rtol=1e-5
end

@testset "tile - scalar" begin
    function sub_scalar(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1})
        pid = ct.bid(0)
        tile = ct.load(a, pid, (16,))
        result = tile - 1.5f0
        ct.store(b, pid, result)
        return
    end

    n = 1024
    tile_size = 16
    a = CUDA.rand(Float32, n)
    b = CUDA.zeros(Float32, n)

    ct.launch(sub_scalar, cld(n, tile_size), a, b)

    @test Array(b) ≈ Array(a) .- 1.5f0 rtol=1e-5
end

@testset "scalar - tile" begin
    function scalar_sub_tile_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1})
        pid = ct.bid(0)
        tile = ct.load(a, pid, (16,))
        result = 5.0f0 - tile
        ct.store(b, pid, result)
        return
    end

    n = 1024
    tile_size = 16
    a = CUDA.rand(Float32, n)
    b = CUDA.zeros(Float32, n)

    ct.launch(scalar_sub_tile_kernel, cld(n, tile_size), a, b)

    @test Array(b) ≈ 5.0f0 .- Array(a) rtol=1e-5
end

@testset "tile * scalar" begin
    function mul_scalar(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1})
        pid = ct.bid(0)
        tile = ct.load(a, pid, (16,))
        result = tile * 2.5f0
        ct.store(b, pid, result)
        return
    end

    n = 1024
    tile_size = 16
    a = CUDA.rand(Float32, n)
    b = CUDA.zeros(Float32, n)

    ct.launch(mul_scalar, cld(n, tile_size), a, b)

    @test Array(b) ≈ Array(a) .* 2.5f0 rtol=1e-5
end

@testset "scalar * tile" begin
    function scalar_mul_tile_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1})
        pid = ct.bid(0)
        tile = ct.load(a, pid, (16,))
        result = 3.0f0 * tile
        ct.store(b, pid, result)
        return
    end

    n = 1024
    tile_size = 16
    a = CUDA.rand(Float32, n)
    b = CUDA.zeros(Float32, n)

    ct.launch(scalar_mul_tile_kernel, cld(n, tile_size), a, b)

    @test Array(b) ≈ 3.0f0 .* Array(a) rtol=1e-5
end

end

@testset "tile broadcasting" begin

@testset "1D broadcast: (1,) + (128,)" begin
    # Test broadcasting a single-element tile to a larger tile
    function broadcast_1d_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                                  c::ct.TileArray{Float32,1})
        pid = ct.bid(0)
        # Load scalar-like tile (1 element)
        scalar_tile = ct.load(a, 0, (1,))
        # Load full tile (128 elements)
        full_tile = ct.load(b, pid, (128,))
        # Broadcast add: (1,) + (128,) -> (128,)
        result = scalar_tile + full_tile
        ct.store(c, pid, result)
        return
    end

    n = 1024
    tile_size = 128
    a = CUDA.rand(Float32, 1)  # Single element
    b = CUDA.rand(Float32, n)
    c = CUDA.zeros(Float32, n)

    ct.launch(broadcast_1d_kernel, cld(n, tile_size), a, b, c)

    # Each output element should be a[1] + b[i]
    a_cpu = Array(a)
    b_cpu = Array(b)
    c_cpu = Array(c)
    @test c_cpu ≈ a_cpu[1] .+ b_cpu rtol=1e-5
end

@testset "2D broadcast: (1, 128) + (64, 1)" begin
    # Test broadcasting 2D tiles with complementary shapes
    function broadcast_2d_kernel(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,2},
                                  c::ct.TileArray{Float32,2})
        # Load row tile (1, 128) and column tile (64, 1)
        row_tile = ct.load(a, (0, 0), (1, 128))
        col_tile = ct.load(b, (0, 0), (64, 1))
        # Broadcast add: (1, 128) + (64, 1) -> (64, 128)
        result = row_tile + col_tile
        ct.store(c, (0, 0), result)
        return
    end

    m, n = 64, 128
    a = CUDA.rand(Float32, 1, n)   # Row vector
    b = CUDA.rand(Float32, m, 1)   # Column vector
    c = CUDA.zeros(Float32, m, n)

    ct.launch(broadcast_2d_kernel, 1, a, b, c)

    # Result should be outer sum: c[i,j] = a[1,j] + b[i,1]
    a_cpu = Array(a)
    b_cpu = Array(b)
    c_cpu = Array(c)
    expected = a_cpu .+ b_cpu  # Julia broadcasting
    @test c_cpu ≈ expected rtol=1e-5
end

@testset "broadcast mul: (4, 1) * (1, 8)" begin
    function broadcast_mul_kernel(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,2},
                                   c::ct.TileArray{Float32,2})
        col_tile = ct.load(a, (0, 0), (4, 1))
        row_tile = ct.load(b, (0, 0), (1, 8))
        # Broadcast multiply: (4, 1) * (1, 8) -> (4, 8)
        result = col_tile * row_tile
        ct.store(c, (0, 0), result)
        return
    end

    a = CUDA.rand(Float32, 4, 1)
    b = CUDA.rand(Float32, 1, 8)
    c = CUDA.zeros(Float32, 4, 8)

    ct.launch(broadcast_mul_kernel, 1, a, b, c)

    a_cpu = Array(a)
    b_cpu = Array(b)
    c_cpu = Array(c)
    expected = a_cpu .* b_cpu  # Outer product
    @test c_cpu ≈ expected rtol=1e-5
end

@testset "broadcast sub: (128,) - (1,)" begin
    function broadcast_sub_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                                   c::ct.TileArray{Float32,1})
        pid = ct.bid(0)
        full_tile = ct.load(a, pid, (128,))
        scalar_tile = ct.load(b, 0, (1,))
        # Broadcast subtract: (128,) - (1,) -> (128,)
        result = full_tile - scalar_tile
        ct.store(c, pid, result)
        return
    end

    n = 1024
    tile_size = 128
    a = CUDA.rand(Float32, n)
    b = CUDA.rand(Float32, 1)  # Single element
    c = CUDA.zeros(Float32, n)

    ct.launch(broadcast_sub_kernel, cld(n, tile_size), a, b, c)

    a_cpu = Array(a)
    b_cpu = Array(b)
    c_cpu = Array(c)
    @test c_cpu ≈ a_cpu .- b_cpu[1] rtol=1e-5
end

@testset "broadcast div: (64, 128) / (1, 128)" begin
    # Divide each row by a scaling vector
    function broadcast_div_kernel(a::ct.TileArray{Float32,2}, scale::ct.TileArray{Float32,2},
                                   c::ct.TileArray{Float32,2})
        data = ct.load(a, (0, 0), (64, 128))
        scale_row = ct.load(scale, (0, 0), (1, 128))
        # Broadcast divide: (64, 128) / (1, 128) -> (64, 128)
        result = data / scale_row
        ct.store(c, (0, 0), result)
        return
    end

    m, n = 64, 128
    a = CUDA.rand(Float32, m, n)
    scale = CUDA.rand(Float32, 1, n) .+ 0.1f0  # Non-zero scale factors
    c = CUDA.zeros(Float32, m, n)

    ct.launch(broadcast_div_kernel, 1, a, scale, c)

    a_cpu = Array(a)
    scale_cpu = Array(scale)
    c_cpu = Array(c)
    expected = a_cpu ./ scale_cpu
    @test c_cpu ≈ expected rtol=1e-5
end

end
