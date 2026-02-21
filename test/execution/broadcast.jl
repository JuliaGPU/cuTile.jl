using CUDA

@testset "scalar-tile operations" begin

for (name, kernel_expr, cpu_expr) in [
    ("tile / scalar",   :(tile / 2.0f0),    :(Array(a) ./ 2.0f0)),
    ("tile / integer",  :(tile / 4),         :(Array(a) ./ 4.0f0)),
    ("scalar ./ tile",  :(1.0f0 ./ tile),    :(1.0f0 ./ Array(a))),
    ("tile .+ scalar",  :(tile .+ 3.5f0),    :(Array(a) .+ 3.5f0)),
    ("scalar .+ tile",  :(2.5f0 .+ tile),    :(2.5f0 .+ Array(a))),
    ("tile .- scalar",  :(tile .- 1.5f0),    :(Array(a) .- 1.5f0)),
    ("scalar .- tile",  :(5.0f0 .- tile),    :(5.0f0 .- Array(a))),
    ("tile * scalar",   :(tile * 2.5f0),     :(Array(a) .* 2.5f0)),
    ("scalar * tile",   :(3.0f0 * tile),     :(3.0f0 .* Array(a))),
]
    sym = Symbol("scalar_tile_", replace(name, r"[^a-zA-Z0-9]" => "_"))
    @eval @testset $name begin
        function $sym(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1})
            pid = ct.bid(1)
            tile = ct.load(a, pid, (16,))
            ct.store(b, pid, $kernel_expr)
            return
        end
        a = CUDA.rand(Float32, 1024) .+ 0.1f0
        b = CUDA.zeros(Float32, 1024)
        ct.launch($sym, cld(1024, 16), a, b)
        @test Array(b) ≈ $cpu_expr
    end
end

end

@testset "tile broadcasting" begin

@testset "1D broadcast: (1,) .+ (128,)" begin
    # Test broadcasting a single-element tile to a larger tile
    function broadcast_1d_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                                  c::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        # Load scalar-like tile (1 element)
        scalar_tile = ct.load(a, 1, (1,))
        # Load full tile (128 elements)
        full_tile = ct.load(b, pid, (128,))
        # Broadcast add: (1,) .+ (128,) -> (128,)
        result = scalar_tile .+ full_tile
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
    @test c_cpu ≈ a_cpu[1] .+ b_cpu
end

@testset "2D broadcast: (1, 128) .+ (64, 1)" begin
    # Test broadcasting 2D tiles with complementary shapes
    function broadcast_2d_kernel(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,2},
                                  c::ct.TileArray{Float32,2})
        # Load row tile (1, 128) and column tile (64, 1)
        row_tile = ct.load(a, (1, 1), (1, 128))
        col_tile = ct.load(b, (1, 1), (64, 1))
        # Broadcast add: (1, 128) .+ (64, 1) -> (64, 128)
        result = row_tile .+ col_tile
        ct.store(c, (1, 1), result)
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
    @test c_cpu ≈ expected
end

@testset "broadcast mul: (4, 1) .* (1, 8)" begin
    function broadcast_mul_kernel(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,2},
                                   c::ct.TileArray{Float32,2})
        col_tile = ct.load(a, (1, 1), (4, 1))
        row_tile = ct.load(b, (1, 1), (1, 8))
        # Broadcast multiply: (4, 1) .* (1, 8) -> (4, 8)
        result = col_tile .* row_tile
        ct.store(c, (1, 1), result)
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
    @test c_cpu ≈ expected
end

@testset "broadcast sub: (128,) .- (1,)" begin
    function broadcast_sub_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                                   c::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        full_tile = ct.load(a, pid, (128,))
        scalar_tile = ct.load(b, 1, (1,))
        # Broadcast subtract: (128,) .- (1,) -> (128,)
        result = full_tile .- scalar_tile
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
    @test c_cpu ≈ a_cpu .- b_cpu[1]
end

@testset "broadcast div: (64, 128) ./ (1, 128)" begin
    # Divide each row by a scaling vector
    function broadcast_div_kernel(a::ct.TileArray{Float32,2}, scale::ct.TileArray{Float32,2},
                                   c::ct.TileArray{Float32,2})
        data = ct.load(a, (1, 1), (64, 128))
        scale_row = ct.load(scale, (1, 1), (1, 128))
        # Broadcast divide: (64, 128) ./ (1, 128) -> (64, 128)
        result = data ./ scale_row
        ct.store(c, (1, 1), result)
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
    @test c_cpu ≈ expected
end

@testset "explicit broadcast_to" begin
    # Test ct.broadcast_to() for explicit shape broadcasting
    function broadcast_to_kernel(a::ct.TileArray{Float32,2}, c::ct.TileArray{Float32,2})
        # Load a row tile (1, 128)
        row_tile = ct.load(a, (1, 1), (1, 128))
        # Explicitly broadcast to (64, 128)
        expanded = ct.broadcast_to(row_tile, (64, 128))
        ct.store(c, (1, 1), expanded)
        return
    end

    m, n = 64, 128
    a = CUDA.rand(Float32, 1, n)
    c = CUDA.zeros(Float32, m, n)

    ct.launch(broadcast_to_kernel, 1, a, c)

    a_cpu = Array(a)
    c_cpu = Array(c)
    # Each row of c should equal the single row of a
    for i in 1:m
        @test c_cpu[i, :] ≈ a_cpu[1, :]
    end
end

@testset "1D-to-2D broadcast: (64,) .+ (64, 128)" begin
    function broadcast_1d_2d_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,2},
                                     c::ct.TileArray{Float32,2})
        col = ct.load(a, 1, (64,))           # 1D: (64,)
        tile = ct.load(b, (1, 1), (64, 128))  # 2D: (64, 128)
        result = col .+ tile                   # broadcast (64,) → (64, 1) → (64, 128)
        ct.store(c, (1, 1), result)
        return
    end

    m, n = 64, 128
    a = CUDA.rand(Float32, m)
    b = CUDA.rand(Float32, m, n)
    c = CUDA.zeros(Float32, m, n)

    ct.launch(broadcast_1d_2d_kernel, 1, a, b, c)

    expected = Array(a) .+ Array(b)  # Julia: (64,) broadcasts along dim 1
    @test Array(c) ≈ expected
end

end

@testset "comparison operations" begin

for (name, op1, op2) in [
    ("< and >",   :<,  :>),
    ("<= and >=", :<=, :>=),
]
    sym = Symbol("cmp_", replace(name, r"[^a-zA-Z0-9]" => "_"))
    @eval @testset "float $($name)" begin
        function $sym(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                      out1::ct.TileArray{Float32,1}, out2::ct.TileArray{Float32,1})
            pid = ct.bid(1)
            ta = ct.load(a, pid, (16,))
            tb = ct.load(b, pid, (16,))
            ct.store(out1, pid, ct.where(broadcast($op1, ta, tb), 1.0f0, 0.0f0))
            ct.store(out2, pid, ct.where(broadcast($op2, ta, tb), 1.0f0, 0.0f0))
            return
        end
        n = 1024
        a = CUDA.rand(Float32, n)
        b = CUDA.rand(Float32, n)
        out1 = CUDA.zeros(Float32, n)
        out2 = CUDA.zeros(Float32, n)
        ct.launch($sym, cld(n, 16), a, b, out1, out2)
        @test Array(out1) ≈ Float32.(broadcast($op1, Array(a), Array(b)))
        @test Array(out2) ≈ Float32.(broadcast($op2, Array(a), Array(b)))
    end
end

@testset "float .== and .!=" begin
    function cmp_eq_ne_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                              out_eq::ct.TileArray{Float32,1}, out_ne::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        ta = ct.load(a, pid, (16,))
        tb = ct.load(b, pid, (16,))
        ct.store(out_eq, pid, ct.where(ta .== tb, 1.0f0, 0.0f0))
        ct.store(out_ne, pid, ct.where(ta .!= tb, 1.0f0, 0.0f0))
        return
    end

    n = 1024
    # Use integer-valued floats so equality is meaningful
    a = CUDA.fill(Float32(1), n)
    b = CUDA.fill(Float32(1), n)
    # Set half to different values
    CUDA.@allowscalar b[1:512] .= 2.0f0
    out_eq = CUDA.zeros(Float32, n)
    out_ne = CUDA.zeros(Float32, n)

    ct.launch(cmp_eq_ne_kernel, cld(n, 16), a, b, out_eq, out_ne)

    @test Array(out_eq) ≈ Float32.(Array(a) .== Array(b))
    @test Array(out_ne) ≈ Float32.(Array(a) .!= Array(b))
end

@testset "tile vs scalar comparison" begin
    function cmp_scalar_kernel(a::ct.TileArray{Float32,1},
                               out::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        ta = ct.load(a, pid, (16,))
        ct.store(out, pid, ct.where(ta .> 0.5f0, 1.0f0, 0.0f0))
        return
    end

    n = 1024
    a = CUDA.rand(Float32, n)
    out = CUDA.zeros(Float32, n)

    ct.launch(cmp_scalar_kernel, cld(n, 16), a, out)

    @test Array(out) ≈ Float32.(Array(a) .> 0.5f0)
end

end

@testset "power operations" begin

@testset "tile .^ tile" begin
    function pow_tt_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                           c::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        ta = ct.load(a, pid, (16,))
        tb = ct.load(b, pid, (16,))
        ct.store(c, pid, ta .^ tb)
        return
    end

    n = 1024
    a = CUDA.rand(Float32, n) .+ 0.5f0  # Ensure positive base
    b = CUDA.rand(Float32, n) .+ 0.5f0
    c = CUDA.zeros(Float32, n)

    ct.launch(pow_tt_kernel, cld(n, 16), a, b, c)

    @test Array(c) ≈ Array(a) .^ Array(b) rtol=1e-4
end

@testset "tile .^ scalar" begin
    function pow_ts_kernel(a::ct.TileArray{Float32,1}, c::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        ta = ct.load(a, pid, (16,))
        ct.store(c, pid, ta .^ 2.0f0)
        return
    end

    n = 1024
    a = CUDA.rand(Float32, n) .+ 0.1f0
    c = CUDA.zeros(Float32, n)

    ct.launch(pow_ts_kernel, cld(n, 16), a, c)

    @test Array(c) ≈ Array(a) .^ 2.0f0 rtol=1e-4
end

end

@testset "where / ifelse broadcasting" begin

@testset "where same-shape" begin
    function where_same_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                               c::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        ta = ct.load(a, pid, (16,))
        tb = ct.load(b, pid, (16,))
        mask = ta .> tb
        result = ct.where(mask, ta, tb)
        ct.store(c, pid, result)
        return
    end

    n = 1024
    a = CUDA.rand(Float32, n)
    b = CUDA.rand(Float32, n)
    c = CUDA.zeros(Float32, n)

    ct.launch(where_same_kernel, cld(n, 16), a, b, c)

    @test Array(c) ≈ ifelse.(Array(a) .> Array(b), Array(a), Array(b)) rtol=1e-5
end

@testset "where with scalar y" begin
    function where_scalar_y_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        ta = ct.load(a, pid, (16,))
        mask = ta .> 0.5f0
        result = ct.where(mask, ta, 0.0f0)
        ct.store(b, pid, result)
        return
    end

    n = 1024
    a = CUDA.rand(Float32, n)
    b = CUDA.zeros(Float32, n)

    ct.launch(where_scalar_y_kernel, cld(n, 16), a, b)

    @test Array(b) ≈ ifelse.(Array(a) .> 0.5f0, Array(a), 0.0f0) rtol=1e-5
end

@testset "where with scalar x" begin
    function where_scalar_x_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        ta = ct.load(a, pid, (16,))
        mask = ta .> 0.5f0
        result = ct.where(mask, 1.0f0, ta)
        ct.store(b, pid, result)
        return
    end

    n = 1024
    a = CUDA.rand(Float32, n)
    b = CUDA.zeros(Float32, n)

    ct.launch(where_scalar_x_kernel, cld(n, 16), a, b)

    @test Array(b) ≈ ifelse.(Array(a) .> 0.5f0, 1.0f0, Array(a)) rtol=1e-5
end

@testset "where with broadcasting" begin
    function where_broadcast_kernel(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,2})
        mask = ct.load(a, (1, 1), (1, 128))  # (1, 128) mask
        tile = ct.load(a, (1, 1), (64, 128))  # (64, 128) tile
        result = ct.where(mask .> 0.5f0, tile, 0.0f0)
        ct.store(b, (1, 1), result)
        return
    end

    m, n = 64, 128
    a = CUDA.rand(Float32, m, n)
    b = CUDA.zeros(Float32, m, n)

    ct.launch(where_broadcast_kernel, 1, a, b)

    a_cpu = Array(a)
    mask_cpu = a_cpu[1:1, :] .> 0.5f0
    expected = ifelse.(mask_cpu, a_cpu, 0.0f0)
    @test Array(b) ≈ expected rtol=1e-5
end

@testset "ifelse. same-shape" begin
    function ifelse_same_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                                c::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        ta = ct.load(a, pid, (16,))
        tb = ct.load(b, pid, (16,))
        result = ifelse.(ta .> tb, ta, tb)
        ct.store(c, pid, result)
        return
    end

    n = 1024
    a = CUDA.rand(Float32, n)
    b = CUDA.rand(Float32, n)
    c = CUDA.zeros(Float32, n)

    ct.launch(ifelse_same_kernel, cld(n, 16), a, b, c)

    @test Array(c) ≈ ifelse.(Array(a) .> Array(b), Array(a), Array(b)) rtol=1e-5
end

@testset "ifelse. with scalar y" begin
    function ifelse_scalar_y_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        ta = ct.load(a, pid, (16,))
        result = ifelse.(ta .> 0.5f0, ta, 0.0f0)
        ct.store(b, pid, result)
        return
    end

    n = 1024
    a = CUDA.rand(Float32, n)
    b = CUDA.zeros(Float32, n)

    ct.launch(ifelse_scalar_y_kernel, cld(n, 16), a, b)

    @test Array(b) ≈ ifelse.(Array(a) .> 0.5f0, Array(a), 0.0f0) rtol=1e-5
end

@testset "ifelse. with both scalars" begin
    function ifelse_both_scalar_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        ta = ct.load(a, pid, (16,))
        result = ifelse.(ta .> 0.5f0, 1.0f0, 0.0f0)
        ct.store(b, pid, result)
        return
    end

    n = 1024
    a = CUDA.rand(Float32, n)
    b = CUDA.zeros(Float32, n)

    ct.launch(ifelse_both_scalar_kernel, cld(n, 16), a, b)

    @test Array(b) ≈ ifelse.(Array(a) .> 0.5f0, 1.0f0, 0.0f0) rtol=1e-5
end

@testset "ifelse. with broadcasting shapes" begin
    function ifelse_broadcast_kernel(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,2})
        col_mask = ct.load(a, (1, 1), (64, 1))  # (64, 1) column
        tile = ct.load(a, (1, 1), (64, 128))     # (64, 128) tile
        result = ifelse.(col_mask .> 0.5f0, tile, 0.0f0)
        ct.store(b, (1, 1), result)
        return
    end

    m, n = 64, 128
    a = CUDA.rand(Float32, m, n)
    b = CUDA.zeros(Float32, m, n)

    ct.launch(ifelse_broadcast_kernel, 1, a, b)

    a_cpu = Array(a)
    mask_cpu = a_cpu[:, 1:1] .> 0.5f0
    expected = ifelse.(mask_cpu, a_cpu, 0.0f0)
    @test Array(b) ≈ expected rtol=1e-5
end

end # where / ifelse broadcasting

@testset "max / min broadcasting" begin

@testset "max. float tile-tile" begin
    function max_float_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                              c::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        ta = ct.load(a, pid, (16,))
        tb = ct.load(b, pid, (16,))
        result = max.(ta, tb)
        ct.store(c, pid, result)
        return
    end

    n = 1024
    a = CUDA.rand(Float32, n)
    b = CUDA.rand(Float32, n)
    c = CUDA.zeros(Float32, n)

    ct.launch(max_float_kernel, cld(n, 16), a, b, c)

    @test Array(c) ≈ max.(Array(a), Array(b)) rtol=1e-5
end

@testset "min. float tile-tile" begin
    function min_float_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                              c::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        ta = ct.load(a, pid, (16,))
        tb = ct.load(b, pid, (16,))
        result = min.(ta, tb)
        ct.store(c, pid, result)
        return
    end

    n = 1024
    a = CUDA.rand(Float32, n)
    b = CUDA.rand(Float32, n)
    c = CUDA.zeros(Float32, n)

    ct.launch(min_float_kernel, cld(n, 16), a, b, c)

    @test Array(c) ≈ min.(Array(a), Array(b)) rtol=1e-5
end

@testset "max. float tile-scalar (ReLU)" begin
    function relu_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        ta = ct.load(a, pid, (16,))
        result = max.(ta, 0.0f0)
        ct.store(b, pid, result)
        return
    end

    n = 1024
    a = CUDA.rand(Float32, n) .- 0.5f0  # Mix of positive and negative
    b = CUDA.zeros(Float32, n)

    ct.launch(relu_kernel, cld(n, 16), a, b)

    @test Array(b) ≈ max.(Array(a), 0.0f0) rtol=1e-5
end

@testset "min. float tile-scalar" begin
    function clamp_max_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        ta = ct.load(a, pid, (16,))
        result = min.(ta, 1.0f0)
        ct.store(b, pid, result)
        return
    end

    n = 1024
    a = CUDA.rand(Float32, n) .* 2.0f0  # Values in [0, 2]
    b = CUDA.zeros(Float32, n)

    ct.launch(clamp_max_kernel, cld(n, 16), a, b)

    @test Array(b) ≈ min.(Array(a), 1.0f0) rtol=1e-5
end

@testset "max. integer tile-tile (signed)" begin
    function max_int_kernel(a::ct.TileArray{Int32,1}, b::ct.TileArray{Int32,1},
                            c::ct.TileArray{Int32,1})
        pid = ct.bid(1)
        ta = ct.load(a, pid, (16,))
        tb = ct.load(b, pid, (16,))
        result = max.(ta, tb)
        ct.store(c, pid, result)
        return
    end

    n = 1024
    a = CuArray(rand(Int32(-100):Int32(100), n))
    b = CuArray(rand(Int32(-100):Int32(100), n))
    c = CUDA.zeros(Int32, n)

    ct.launch(max_int_kernel, cld(n, 16), a, b, c)

    @test Array(c) == max.(Array(a), Array(b))
end

@testset "min. integer tile-tile (signed)" begin
    function min_int_kernel(a::ct.TileArray{Int32,1}, b::ct.TileArray{Int32,1},
                            c::ct.TileArray{Int32,1})
        pid = ct.bid(1)
        ta = ct.load(a, pid, (16,))
        tb = ct.load(b, pid, (16,))
        result = min.(ta, tb)
        ct.store(c, pid, result)
        return
    end

    n = 1024
    a = CuArray(rand(Int32(-100):Int32(100), n))
    b = CuArray(rand(Int32(-100):Int32(100), n))
    c = CUDA.zeros(Int32, n)

    ct.launch(min_int_kernel, cld(n, 16), a, b, c)

    @test Array(c) == min.(Array(a), Array(b))
end

@testset "max. broadcasting: (64,1) vs (1,128)" begin
    function max_broadcast_kernel(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,2},
                                  c::ct.TileArray{Float32,2})
        col_tile = ct.load(a, (1, 1), (64, 1))
        row_tile = ct.load(b, (1, 1), (1, 128))
        result = max.(col_tile, row_tile)
        ct.store(c, (1, 1), result)
        return
    end

    m, n = 64, 128
    a = CUDA.rand(Float32, m, 1)
    b = CUDA.rand(Float32, 1, n)
    c = CUDA.zeros(Float32, m, n)

    ct.launch(max_broadcast_kernel, 1, a, b, c)

    @test Array(c) ≈ max.(Array(a), Array(b)) rtol=1e-5
end

end # max / min broadcasting

@testset "fma broadcasting" begin

@testset "fma. same-shape" begin
    function fma_same_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                             c::ct.TileArray{Float32,1}, d::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        ta = ct.load(a, pid, (16,))
        tb = ct.load(b, pid, (16,))
        tc = ct.load(c, pid, (16,))
        result = fma.(ta, tb, tc)
        ct.store(d, pid, result)
        return
    end

    n = 1024
    a = CUDA.rand(Float32, n)
    b = CUDA.rand(Float32, n)
    c = CUDA.rand(Float32, n)
    d = CUDA.zeros(Float32, n)

    ct.launch(fma_same_kernel, cld(n, 16), a, b, c, d)

    @test Array(d) ≈ fma.(Array(a), Array(b), Array(c)) rtol=1e-5
end

@testset "fma. with scalar c" begin
    function fma_scalar_c_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                                 c::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        ta = ct.load(a, pid, (16,))
        tb = ct.load(b, pid, (16,))
        result = fma.(ta, tb, 1.0f0)
        ct.store(c, pid, result)
        return
    end

    n = 1024
    a = CUDA.rand(Float32, n)
    b = CUDA.rand(Float32, n)
    c = CUDA.zeros(Float32, n)

    ct.launch(fma_scalar_c_kernel, cld(n, 16), a, b, c)

    @test Array(c) ≈ fma.(Array(a), Array(b), 1.0f0) rtol=1e-5
end

@testset "fma. with broadcasting bias" begin
    function fma_broadcast_kernel(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,2},
                                  bias::ct.TileArray{Float32,2}, c::ct.TileArray{Float32,2})
        ta = ct.load(a, (1, 1), (64, 128))
        tb = ct.load(b, (1, 1), (64, 128))
        tbias = ct.load(bias, (1, 1), (1, 128))  # (1, 128) bias row
        result = fma.(ta, tb, tbias)
        ct.store(c, (1, 1), result)
        return
    end

    m, n = 64, 128
    a = CUDA.rand(Float32, m, n)
    b = CUDA.rand(Float32, m, n)
    bias = CUDA.rand(Float32, 1, n)
    c = CUDA.zeros(Float32, m, n)

    ct.launch(fma_broadcast_kernel, 1, a, b, bias, c)

    @test Array(c) ≈ fma.(Array(a), Array(b), Array(bias)) rtol=1e-5
end

end # fma broadcasting

@testset "type argument broadcasting" begin

@testset "convert.(Float16, tile)" begin
    function convert_f16_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float16,1})
        pid = ct.bid(1)
        tile = ct.load(a, pid, (16,))
        ct.store(b, pid, convert.(Float16, tile))
        return
    end

    n = 1024
    a = CUDA.rand(Float32, n)
    b = CUDA.zeros(Float16, n)

    ct.launch(convert_f16_kernel, cld(n, 16), a, b)

    @test Array(b) == Float16.(Array(a))
end

@testset "convert.(Float32, float16_tile)" begin
    function convert_f32_kernel(a::ct.TileArray{Float16,1}, b::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        tile = ct.load(a, pid, (16,))
        ct.store(b, pid, convert.(Float32, tile))
        return
    end

    n = 1024
    a = CUDA.rand(Float16, n)
    b = CUDA.zeros(Float32, n)

    ct.launch(convert_f32_kernel, cld(n, 16), a, b)

    @test Array(b) == Float32.(Array(a))
end

@testset "unsafe_trunc.(Int32, float_tile)" begin
    function unsafe_trunc_i32_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Int32,1})
        pid = ct.bid(1)
        tile = ct.load(a, pid, (16,))
        ct.store(b, pid, unsafe_trunc.(Int32, tile))
        return
    end

    n = 1024
    a = CuArray(Float32.(rand(-100:100, n)) .+ 0.7f0)
    b = CUDA.zeros(Int32, n)

    ct.launch(unsafe_trunc_i32_kernel, cld(n, 16), a, b)

    @test Array(b) == unsafe_trunc.(Int32, Array(a))
end

end # type argument broadcasting

@testset "multi-arg map" begin
    @testset "binary map(+, ...)" begin
        function map_add_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                                c::ct.TileArray{Float32,1})
            pid = ct.bid(1)
            ta = ct.load(a, pid, (16,))
            tb = ct.load(b, pid, (16,))
            ct.store(c, pid, map(+, ta, tb))
            return
        end

        n = 1024
        a = CUDA.rand(Float32, n)
        b = CUDA.rand(Float32, n)
        c = CUDA.zeros(Float32, n)
        ct.launch(map_add_kernel, cld(n, 16), a, b, c)
        @test Array(c) ≈ Array(a) + Array(b)
    end

    @testset "ternary map(fma, ...)" begin
        function map_fma_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                                c::ct.TileArray{Float32,1}, d::ct.TileArray{Float32,1})
            pid = ct.bid(1)
            ta = ct.load(a, pid, (16,))
            tb = ct.load(b, pid, (16,))
            tc = ct.load(c, pid, (16,))
            ct.store(d, pid, map(fma, ta, tb, tc))
            return
        end

        n = 1024
        a = CUDA.rand(Float32, n)
        b = CUDA.rand(Float32, n)
        c = CUDA.rand(Float32, n)
        d = CUDA.zeros(Float32, n)
        ct.launch(map_fma_kernel, cld(n, 16), a, b, c, d)
        @test Array(d) ≈ fma.(Array(a), Array(b), Array(c))
    end

    @testset "nested broadcast a .+ b .* c" begin
        function nested_bc_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                                  c::ct.TileArray{Float32,1}, d::ct.TileArray{Float32,1})
            pid = ct.bid(1)
            ta = ct.load(a, pid, (16,))
            tb = ct.load(b, pid, (16,))
            tc = ct.load(c, pid, (16,))
            ct.store(d, pid, ta .+ tb .* tc)
            return
        end

        n = 1024
        a = CUDA.rand(Float32, n)
        b = CUDA.rand(Float32, n)
        c = CUDA.rand(Float32, n)
        d = CUDA.zeros(Float32, n)
        ct.launch(nested_bc_kernel, cld(n, 16), a, b, c, d)
        @test Array(d) ≈ Array(a) .+ Array(b) .* Array(c)
    end

    @testset "ifelse broadcast" begin
        function ifelse_bc_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                                  c::ct.TileArray{Float32,1})
            pid = ct.bid(1)
            ta = ct.load(a, pid, (16,))
            tb = ct.load(b, pid, (16,))
            mask = ta .> tb
            ct.store(c, pid, ifelse.(mask, ta, tb))
            return
        end

        n = 1024
        a = CUDA.rand(Float32, n)
        b = CUDA.rand(Float32, n)
        c = CUDA.zeros(Float32, n)
        ct.launch(ifelse_bc_kernel, cld(n, 16), a, b, c)
        @test Array(c) ≈ max.(Array(a), Array(b))
    end
end
