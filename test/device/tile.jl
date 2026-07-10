# operations on tiles

using CUDA

@testset "1D vector add" begin
    function vadd_1d(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                     c::ct.TileArray{Float32,1})
        pid = ct.bid(1)
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

    @cuda backend=cuTile blocks=cld(n, tile_size) vadd_1d(a, b, c)

    @test Array(c) ≈ Array(a) + Array(b)
end

@testset "1D vector sub" begin
    function vsub_1d(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                     c::ct.TileArray{Float32,1})
        pid = ct.bid(1)
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

    @cuda backend=cuTile blocks=cld(n, tile_size) vsub_1d(a, b, c)

    @test Array(c) ≈ Array(a) - Array(b)
end

@testset "1D vector mul" begin
    function vmul_1d(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                     c::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        tile_a = ct.load(a, pid, (16,))
        tile_b = ct.load(b, pid, (16,))
        ct.store(c, pid, tile_a .* tile_b)
        return
    end

    n = 1024
    tile_size = 16
    a = CUDA.rand(Float32, n)
    b = CUDA.rand(Float32, n)
    c = CUDA.zeros(Float32, n)

    @cuda backend=cuTile blocks=cld(n, tile_size) vmul_1d(a, b, c)

    @test Array(c) ≈ Array(a) .* Array(b)
end

@testset "1D vector div" begin
    function vdiv_1d(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                     c::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        tile_a = ct.load(a, pid, (16,))
        tile_b = ct.load(b, pid, (16,))
        ct.store(c, pid, tile_a ./ tile_b)
        return
    end

    n = 1024
    tile_size = 16
    a = CUDA.rand(Float32, n)
    b = CUDA.rand(Float32, n) .+ 0.1f0  # Ensure non-zero
    c = CUDA.zeros(Float32, n)

    @cuda backend=cuTile blocks=cld(n, tile_size) vdiv_1d(a, b, c)

    @test Array(c) ≈ Array(a) ./ Array(b)
end

@testset "2D matrix add" begin
    function madd_2d(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,2},
                     c::ct.TileArray{Float32,2})
        bidx = ct.bid(1)
        bidy = ct.bid(2)
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

    @cuda backend=cuTile blocks=(cld(m, tile_x), cld(n, tile_y)) madd_2d(a, b, c)

    @test Array(c) ≈ Array(a) + Array(b)
end

@testset "4D tensor add" begin
    # 4D loads require TileArray with explicit sizes (grid only provides 3D)
    function tadd_4d(a::ct.TileArray{Float32,4}, b::ct.TileArray{Float32,4},
                     c::ct.TileArray{Float32,4})
        bidx = ct.bid(1)
        bidy = ct.bid(2)
        bidz = ct.bid(3)
        # Load 4D tiles - 4th dimension index is fixed at 1
        tile_a = ct.load(a, (bidx, bidy, bidz, 1), (4, 4, 4, 2))
        tile_b = ct.load(b, (bidx, bidy, bidz, 1), (4, 4, 4, 2))
        ct.store(c, (bidx, bidy, bidz, 1), tile_a + tile_b)
        return
    end

    # Array shape: (d1, d2, d3, d4) with tile shape (4, 4, 4, 2)
    d1, d2, d3, d4 = 16, 16, 8, 2
    tile_1, tile_2, tile_3, tile_4 = 4, 4, 4, 2
    a = CUDA.rand(Float32, d1, d2, d3, d4)
    b = CUDA.rand(Float32, d1, d2, d3, d4)
    c = CUDA.zeros(Float32, d1, d2, d3, d4)

    grid = (cld(d1, tile_1), cld(d2, tile_2), cld(d3, tile_3))
    @cuda backend=cuTile blocks=grid tadd_4d(a, b, c)

    @test Array(c) ≈ Array(a) + Array(b)
end

for (op, name) in [
    (:sqrt,  "sqrt"),  (:abs, "abs"),   (:cos, "cos"),   (:sin, "sin"),
    (:exp,   "exp"),   (:log, "log"),   (:ceil, "ceil"), (:floor, "floor"),
]
    @eval @testset "1D $($name)" begin
        function $(Symbol("vmath_$(name)"))(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1})
            pid = ct.bid(1)
            tile = ct.load(a, pid, (16,))
            ct.store(b, pid, $op.(tile))
            return
        end
        a = CUDA.rand(Float32, 1024) .+ 0.1f0
        b = CUDA.zeros(Float32, 1024)
        @cuda backend=cuTile blocks=cld(1024, 16) $(Symbol("vmath_$(name)"))(a, b)
        @test Array(b) ≈ $op.(Array(a)) rtol=1e-4
    end
end

@testset "full with runtime value" begin
    function full_runtime_kernel(src::ct.TileArray{Float32,1}, dst::ct.TileArray{Float32,1})
        # Load a single-element tile to get a runtime scalar
        val = ct.load(src, 1, (1,))
        pid = ct.bid(1)
        tile = fill(val, (16,))
        ct.store(dst, pid, tile)
        return
    end

    n = 1024
    src = CUDA.fill(3.14f0, 1)
    dst = CUDA.zeros(Float32, n)

    @cuda backend=cuTile blocks=cld(n, 16) full_runtime_kernel(src, dst)

    @test all(Array(dst) .≈ 3.14f0)
end

@testset "rank mismatch load/store" begin
    @testset "1D shape on 2D array" begin
        function copy_1d_2d(src::ct.TileArray{Float32,2}, dst::ct.TileArray{Float32,2})
            bid = ct.bid(1)
            tile = ct.load(src, (bid, 1), (16,))
            ct.store(dst, (bid, 1), tile)
            return
        end

        m = 64
        src = CUDA.rand(Float32, m, 1)
        dst = CUDA.zeros(Float32, m, 1)

        @cuda backend=cuTile blocks=cld(m, 16) copy_1d_2d(src, dst)

        @test Array(dst) ≈ Array(src)
    end

    @testset "2D shape on 4D array" begin
        function copy_2d_4d(src::ct.TileArray{Float32,4}, dst::ct.TileArray{Float32,4})
            bidx = ct.bid(1)
            bidy = ct.bid(2)
            tile = ct.load(src, (bidx, bidy, 1, 1), (4, 4))
            ct.store(dst, (bidx, bidy, 1, 1), tile)
            return
        end

        d1, d2 = 16, 16
        src = CUDA.rand(Float32, d1, d2, 1, 1)
        dst = CUDA.zeros(Float32, d1, d2, 1, 1)

        @cuda backend=cuTile blocks=(cld(d1, 4), cld(d2, 4)) copy_2d_4d(src, dst)

        @test Array(dst) ≈ Array(src)
    end
end

@testset "scalar store" begin
    function scalar_store_1d(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1}, tileSz::Int)
        tile = ct.load(a, ct.bid(1), (tileSz,))
        ct.store(b, ct.bid(1), sum(tile))
        return nothing
    end

    sz = 32; N = 1024
    a = CUDA.rand(Float32, N)
    b = CUDA.zeros(Float32, cld(N, sz))
    @cuda backend=cuTile blocks=cld(N, sz) scalar_store_1d(a, b, ct.Constant(sz))

    a_cpu = reshape(Array(a), sz, :)
    @test Array(b) ≈ vec(sum(a_cpu; dims=1)) rtol=1e-3
end

@testset "transpose" begin
    function transpose_kernel(x::ct.TileArray{Float32,2}, y::ct.TileArray{Float32,2})
        bidx = ct.bid(1)
        bidy = ct.bid(2)
        tile = ct.load(x, (bidx, bidy), (32, 32))
        transposed = transpose(tile)
        ct.store(y, (bidy, bidx), transposed)
        return
    end

    m, n = 256, 128
    tile_size = 32
    x = CUDA.rand(Float32, m, n)
    y = CUDA.zeros(Float32, n, m)

    @cuda backend=cuTile blocks=(cld(m, tile_size), cld(n, tile_size)) transpose_kernel(x, y)

    @test Array(y) ≈ transpose(Array(x))
end

@testset "reshape" begin
    @testset "2D -> 1D reshape preserves elements" begin
        function reshape_2d_to_1d_kernel(x::ct.TileArray{Float32,2}, y::ct.TileArray{Float32,1})
            bid = ct.bid(1)
            # Load a 4x8 tile
            tile = ct.load(x, (bid, 1), (4, 8))
            # Reshape to 32 elements (flat)
            reshaped = reshape(tile, (32,))
            ct.store(y, bid, reshaped)
            return
        end

        m, n = 64, 8
        x = CUDA.rand(Float32, m, n)
        # Each of the m/4 = 16 blocks produces 32 elements
        y = CUDA.zeros(Float32, m * n)

        @cuda backend=cuTile blocks=cld(m, 4) reshape_2d_to_1d_kernel(x, y)

        # Verify all elements are preserved (same multiset)
        x_cpu = Array(x)
        y_cpu = Array(y)
        for bid in 0:(cld(m, 4)-1)
            row_start = bid * 4 + 1
            row_end = row_start + 3
            input_elements = sort(vec(x_cpu[row_start:row_end, 1:8]))
            output_elements = sort(y_cpu[(bid*32+1):((bid+1)*32)])
            @test output_elements ≈ input_elements
        end
    end

    @testset "1D -> 2D reshape preserves elements" begin
        function reshape_1d_to_2d_kernel(x::ct.TileArray{Float32,1}, y::ct.TileArray{Float32,2})
            bid = ct.bid(1)
            # Load 32 elements
            tile = ct.load(x, bid, (32,))
            # Reshape to 4x8
            reshaped = reshape(tile, (4, 8))
            ct.store(y, (bid, 1), reshaped)
            return
        end

        n = 512
        x = CUDA.rand(Float32, n)
        m_out = n ÷ 8
        y = CUDA.zeros(Float32, m_out, 8)

        @cuda backend=cuTile blocks=cld(n, 32) reshape_1d_to_2d_kernel(x, y)

        # Verify all elements are preserved (same multiset)
        x_cpu = Array(x)
        y_cpu = Array(y)
        for bid in 0:(cld(n, 32)-1)
            start_idx = bid * 32 + 1
            input_elements = sort(x_cpu[start_idx:(start_idx+31)])
            row_start = bid * 4 + 1
            output_elements = sort(vec(y_cpu[row_start:(row_start+3), 1:8]))
            @test output_elements ≈ input_elements
        end
    end

    @testset "reshape roundtrip preserves data" begin
        function reshape_roundtrip_kernel(x::ct.TileArray{Float32,2}, y::ct.TileArray{Float32,2})
            bid = ct.bid(1)
            # Load 8x4 tile
            tile = ct.load(x, (bid, 1), (8, 4))
            # Reshape to 32, then back to 8x4
            flat = reshape(tile, (32,))
            back = reshape(flat, (8, 4))
            ct.store(y, (bid, 1), back)
            return
        end

        m, n = 64, 4
        x = CUDA.rand(Float32, m, n)
        y = CUDA.zeros(Float32, m, n)

        @cuda backend=cuTile blocks=cld(m, 8) reshape_roundtrip_kernel(x, y)

        @test Array(y) ≈ Array(x)
    end
end

@testset "reshape column-major semantics" begin
    # These tests verify that reshape matches Julia's column-major reshape behavior,
    # not just that elements are preserved (which would pass even with wrong ordering).
    # Note: tile shapes must be powers of 2.

    @testset "1D → 2D matches Julia reshape exactly" begin
        function reshape_1d_to_2d_exact_kernel(x::ct.TileArray{Float32,1}, y::ct.TileArray{Float32,2},
                                               n::Int, shape::NTuple{2,Int})
            bid = ct.bid(1)
            tile = ct.load(x, bid, (n,))
            reshaped = reshape(tile, shape)
            ct.store(y, (bid, 1), reshaped)
            return
        end

        n = 32
        shape = (4, 8)
        # Sequential values to detect any reordering
        x = CuArray(Float32.(1:n))
        y = CUDA.zeros(Float32, shape)

        @cuda backend=cuTile reshape_1d_to_2d_exact_kernel(x, y, ct.Constant(n), ct.Constant(shape))

        # Must match Julia's column-major reshape exactly (not just same elements)
        expected = reshape(Float32.(1:n), shape)
        @test Array(y) ≈ expected
    end

    @testset "2D → 1D matches Julia vec exactly" begin
        function reshape_2d_to_1d_exact_kernel(x::ct.TileArray{Float32,2}, y::ct.TileArray{Float32,1},
                                               shape::NTuple{2,Int}, n::Int)
            bid = ct.bid(1)
            tile = ct.load(x, (bid, 1), shape)
            reshaped = reshape(tile, (n,))
            ct.store(y, bid, reshaped)
            return
        end

        shape = (4, 8)
        n = prod(shape)
        # Create 2D array with sequential column-major values
        x = CuArray(Float32.(reshape(1:n, shape)))
        y = CUDA.zeros(Float32, n)

        @cuda backend=cuTile reshape_2d_to_1d_exact_kernel(x, y, ct.Constant(shape), ct.Constant(n))

        # Flattening should give column-major order: 1,2,3,4,...,32
        expected = vec(Float32.(reshape(1:n, shape)))
        @test Array(y) ≈ expected
    end

    @testset "2D → 2D reshape matches Julia reshape exactly" begin
        function reshape_2d_to_2d_exact_kernel(x::ct.TileArray{Float32,2}, y::ct.TileArray{Float32,2},
                                               src_shape::NTuple{2,Int},
                                               tgt_shape::NTuple{2,Int})
            bid = ct.bid(1)
            tile = ct.load(x, (bid, 1), src_shape)
            reshaped = reshape(tile, tgt_shape)
            ct.store(y, (bid, 1), reshaped)
            return
        end

        src_shape = (4, 8)
        tgt_shape = (8, 4)
        n = prod(src_shape)
        x = CuArray(Float32.(reshape(1:n, src_shape)))
        y = CUDA.zeros(Float32, tgt_shape)

        @cuda backend=cuTile reshape_2d_to_2d_exact_kernel(x, y, ct.Constant(src_shape), ct.Constant(tgt_shape))

        expected = reshape(Float32.(reshape(1:n, src_shape)), tgt_shape)
        @test Array(y) ≈ expected
    end

    @testset "3D → 2D reshape matches Julia reshape exactly" begin
        function reshape_3d_to_2d_exact_kernel(x::ct.TileArray{Float32,3}, y::ct.TileArray{Float32,2},
                                               src_shape::NTuple{3,Int},
                                               tgt_shape::NTuple{2,Int})
            bid = ct.bid(1)
            tile = ct.load(x, (bid, 1, 1), src_shape)
            reshaped = reshape(tile, tgt_shape)
            ct.store(y, (bid, 1), reshaped)
            return
        end

        src_shape = (2, 4, 4)
        tgt_shape = (8, 4)
        n = prod(src_shape)
        x = CuArray(Float32.(reshape(1:n, src_shape)))
        y = CUDA.zeros(Float32, tgt_shape)

        @cuda backend=cuTile reshape_3d_to_2d_exact_kernel(x, y, ct.Constant(src_shape), ct.Constant(tgt_shape))

        expected = reshape(Float32.(reshape(1:n, src_shape)), tgt_shape)
        @test Array(y) ≈ expected
    end

    @testset "3D reshape round-trip with packing dim D=$D" for D in [2, 4]
        # This is the atom_packing pattern: (BS, N, 2) → (BS, N*2/D, D) → (BS, N, 2)
        function reshape_roundtrip_3d_kernel(x::ct.TileArray{Float32,3}, y::ct.TileArray{Float32,3},
                                             orig_shape::NTuple{3,Int},
                                             packed_shape::NTuple{3,Int})
            bid = ct.bid(1)
            tile = ct.load(x, (bid, 1, 1), orig_shape)
            packed = reshape(tile, packed_shape)
            unpacked = reshape(packed, orig_shape)
            ct.store(y, (bid, 1, 1), unpacked)
            return
        end

        BS, N = 1, 8
        orig_shape = (BS, N, 2)
        packed_shape = (BS, N * 2 ÷ D, D)

        # Sequential values to detect any reordering
        x = CuArray(Float32.(reshape(1:prod(orig_shape), orig_shape)))
        y = CUDA.zeros(Float32, orig_shape)

        @cuda backend=cuTile reshape_roundtrip_3d_kernel(x, y, ct.Constant(orig_shape), ct.Constant(packed_shape))

        # Round-trip must preserve exact data, not just same elements
        @test Array(y) ≈ Array(x)
    end

    @testset "2D → 1D → 2D round-trip preserves exact layout" begin
        function reshape_2d_1d_2d_kernel(x::ct.TileArray{Float32,2}, y::ct.TileArray{Float32,2},
                                         shape::NTuple{2,Int})
            bid = ct.bid(1)
            tile = ct.load(x, (bid, 1), shape)
            flat = reshape(tile, (prod(shape),))
            back = reshape(flat, shape)
            ct.store(y, (bid, 1), back)
            return
        end

        shape = (4, 8)
        x = CuArray(Float32.(reshape(1:prod(shape), shape)))
        y = CUDA.zeros(Float32, shape)

        @cuda backend=cuTile reshape_2d_1d_2d_kernel(x, y, ct.Constant(shape))

        @test Array(y) ≈ Array(x)
    end
end

@testset "repeat" begin
    @testset "1D outer repeat matches Base.repeat" begin
        function repeat_1d_kernel(x::ct.TileArray{Float32,1}, y::ct.TileArray{Float32,1})
            bid = ct.bid(1)
            tile = ct.load(x, bid, (8,))
            ct.store(y, bid, repeat(tile, 4))  # (8,) -> (32,)
            return
        end

        x = CuArray(Float32.(1:8))
        y = CUDA.zeros(Float32, 32)

        @cuda backend=cuTile repeat_1d_kernel(x, y)

        @test Array(y) ≈ repeat(Float32.(1:8), 4)
    end

    @testset "2D outer repeat matches Base.repeat" begin
        function repeat_2d_kernel(x::ct.TileArray{Float32,2}, y::ct.TileArray{Float32,2})
            bid = ct.bid(1)
            tile = ct.load(x, (bid, 1), (2, 4))
            ct.store(y, (bid, 1), repeat(tile, 2, 2))  # (2, 4) -> (4, 8)
            return
        end

        src = Float32.(reshape(1:8, 2, 4))
        x = CuArray(src)
        y = CUDA.zeros(Float32, 4, 8)

        @cuda backend=cuTile repeat_2d_kernel(x, y)

        @test Array(y) ≈ repeat(src, 2, 2)
    end

    @testset "2D inner repeat matches Base.repeat" begin
        function repeat_inner_kernel(x::ct.TileArray{Float32,2}, y::ct.TileArray{Float32,2})
            bid = ct.bid(1)
            tile = ct.load(x, (bid, 1), (2, 4))
            ct.store(y, (bid, 1), repeat(tile; inner=(2, 1)))  # (2, 4) -> (4, 4)
            return
        end

        src = Float32.(reshape(1:8, 2, 4))
        x = CuArray(src)
        y = CUDA.zeros(Float32, 4, 4)

        @cuda backend=cuTile repeat_inner_kernel(x, y)

        @test Array(y) ≈ repeat(src; inner=(2, 1))
    end

    @testset "2D combined inner+outer repeat matches Base.repeat" begin
        function repeat_inner_outer_kernel(x::ct.TileArray{Float32,2}, y::ct.TileArray{Float32,2})
            bid = ct.bid(1)
            tile = ct.load(x, (bid, 1), (2, 4))
            ct.store(y, (bid, 1), repeat(tile; inner=(2, 1), outer=(1, 2)))  # (2, 4) -> (4, 8)
            return
        end

        src = Float32.(reshape(1:8, 2, 4))
        x = CuArray(src)
        y = CUDA.zeros(Float32, 4, 8)

        @cuda backend=cuTile repeat_inner_outer_kernel(x, y)

        @test Array(y) ≈ repeat(src; inner=(2, 1), outer=(1, 2))
    end
end

@testset "permutedims" begin
    @testset "2D permutedims (transpose-like)" begin
        function permute_2d_kernel(x::ct.TileArray{Float32,2}, y::ct.TileArray{Float32,2})
            bid = ct.bid(1)
            # Load 8x4 tile
            tile = ct.load(x, (bid, 1), (8, 4))
            # Permute with (2, 1) to swap dimensions: (8, 4) -> (4, 8)
            permuted = permutedims(tile, (2, 1))
            ct.store(y, (bid, 1), permuted)
            return
        end

        m, n = 64, 4
        x = CUDA.rand(Float32, m, n)
        y = CUDA.zeros(Float32, cld(m, 8) * 4, 8)

        @cuda backend=cuTile blocks=cld(m, 8) permute_2d_kernel(x, y)

        # Verify permutedims matches transpose
        x_cpu = Array(x)
        y_cpu = Array(y)
        for bid in 0:(cld(m, 8)-1)
            row_start = bid * 8 + 1
            input_tile = x_cpu[row_start:(row_start+7), 1:4]
            out_row_start = bid * 4 + 1
            output_tile = y_cpu[out_row_start:(out_row_start+3), 1:8]
            # Compare sorted values since memory layouts may differ
            @test sort(vec(output_tile)) ≈ sort(vec(transpose(input_tile)))
        end
    end

    @testset "permutedims roundtrip preserves data" begin
        function permute_roundtrip_kernel(x::ct.TileArray{Float32,2}, y::ct.TileArray{Float32,2})
            bid = ct.bid(1)
            # Load 4x8 tile
            tile = ct.load(x, (bid, 1), (4, 8))
            # Permute with (2, 1), then back with (2, 1)
            permuted = permutedims(tile, (2, 1))  # (4, 8) -> (8, 4)
            back = permutedims(permuted, (2, 1))  # (8, 4) -> (4, 8)
            ct.store(y, (bid, 1), back)
            return
        end

        m, n = 64, 8
        x = CUDA.rand(Float32, m, n)
        y = CUDA.zeros(Float32, m, n)

        @cuda backend=cuTile blocks=cld(m, 4) permute_roundtrip_kernel(x, y)

        @test Array(y) ≈ Array(x)
    end
end

@testset "strided" begin
    @testset "PermutedDimsArray" begin
        function copy_kernel_2d(
            src::ct.TileArray{Float32, 2}, dst::ct.TileArray{Float32, 2},
            tile_x::Int, tile_y::Int
        )
            bid_x = ct.bid(1)
            bid_y = ct.bid(2)
            tile = ct.load(src, (bid_x, bid_y), (tile_x, tile_y))
            ct.store(dst, (bid_x, bid_y), tile)
            return
        end

        m, n = 64, 32
        tm, tn = 16, 16
        A = CuArray(Float32.(reshape(1:n*m, n, m)))
        P = PermutedDimsArray(A, (2, 1))
        out = CUDA.zeros(Float32, m, n)

        grid = (cld(m, tm), cld(n, tn))
        @cuda backend=cuTile blocks=grid copy_kernel_2d(P, out, ct.Constant(tm), ct.Constant(tn))

        @test out == permutedims(A, (2, 1))
    end

    @testset "load with order=(2,1)" begin
        function order_load_kernel(
            src::ct.TileArray{Float32, 2}, dst::ct.TileArray{Float32, 2},
            t::Int
        )
            bid_x = ct.bid(1)
            bid_y = ct.bid(2)
            tile = ct.load(src, (bid_x, bid_y), (t, t); order=(2, 1))
            ct.store(dst, (bid_x, bid_y), tile)
            return
        end

        n = 64; t = 16
        src = CuArray(Float32.(reshape(1:n*n, n, n)))
        dst = CUDA.zeros(Float32, n, n)

        @cuda backend=cuTile blocks=(cld(n, t), cld(n, t)) order_load_kernel(src, dst, ct.Constant(t))

        @test Array(dst) ≈ transpose(Array(src))
    end

    @testset "store with order=(2,1)" begin
        function order_store_kernel(
            src::ct.TileArray{Float32, 2}, dst::ct.TileArray{Float32, 2},
            t::Int
        )
            bid_x = ct.bid(1)
            bid_y = ct.bid(2)
            tile = ct.load(src, (bid_x, bid_y), (t, t))
            ct.store(dst, (bid_x, bid_y), tile; order=(2, 1))
            return
        end

        n = 64; t = 16
        src = CuArray(Float32.(reshape(1:n*n, n, n)))
        dst = CUDA.zeros(Float32, n, n)

        @cuda backend=cuTile blocks=(cld(n, t), cld(n, t)) order_store_kernel(src, dst, ct.Constant(t))

        @test Array(dst) ≈ transpose(Array(src))
    end
end

@testset "extract" begin
    @testset "extract identity full shape" begin
        function extract_identity_kernel(x::ct.TileArray{Float32,2}, y::ct.TileArray{Float32,2})
            bid = ct.bid(1)
            # Load 4x8 tile
            tile = ct.load(x, (bid, 1), (4, 8))
            # Extract the full tile: a single slice, index (1, 1)
            extracted = ct.extract(tile, (1, 1), (4, 8))
            ct.store(y, (bid, 1), extracted)
            return
        end

        m, n = 64, 8
        x = CUDA.rand(Float32, m, n)
        y = CUDA.zeros(Float32, m, n)

        @cuda backend=cuTile blocks=cld(m, 4) extract_identity_kernel(x, y)

        # Full extract at (0,0) should preserve data
        @test Array(y) ≈ Array(x)
    end

    @testset "extract (1,1) smaller shape" begin
        function extract_smaller_kernel(x::ct.TileArray{Float32,2}, y::ct.TileArray{Float32,2})
            bid = ct.bid(1)
            # Load 8x8 tile
            tile = ct.load(x, (bid, 1), (8, 8))
            # Extract 4x4 at (1, 1) - top-left corner
            extracted = ct.extract(tile, (1, 1), (4, 4))
            ct.store(y, (bid, 1), extracted)
            return
        end

        m, n = 64, 8
        x = CUDA.rand(Float32, m, n)
        y = CUDA.zeros(Float32, cld(m, 8) * 4, 4)

        @cuda backend=cuTile blocks=cld(m, 8) extract_smaller_kernel(x, y)

        # Verify elements are preserved for top-left 4x4
        x_cpu = Array(x)
        y_cpu = Array(y)
        for bid in 0:(cld(m, 8)-1)
            input_start = bid * 8 + 1
            input_slice = x_cpu[input_start:(input_start+3), 1:4]
            output_start = bid * 4 + 1
            output_slice = y_cpu[output_start:(output_start+3), 1:4]
            @test sort(vec(output_slice)) ≈ sort(vec(input_slice))
        end
    end

    @testset "extract with slice indices" begin
        # Extract uses SLICE INDICES, not offsets!
        # For shape (8,8) -> (4,4): valid indices are {1,2} x {1,2}
        # Index (2, 1) extracts rows 5-8 (the second slice in first dimension)

        function extract_all_quadrants_kernel(x::ct.TileArray{Float32,2},
                                              y0::ct.TileArray{Float32,2},
                                              y1::ct.TileArray{Float32,2},
                                              y2::ct.TileArray{Float32,2},
                                              y3::ct.TileArray{Float32,2})
            bid = ct.bid(1)
            tile = ct.load(x, (bid, 1), (8, 8))
            # Extract all 4 quadrants
            q0 = ct.extract(tile, (1, 1), (4, 4))  # Top-left
            q1 = ct.extract(tile, (2, 1), (4, 4))  # Bottom-left
            q2 = ct.extract(tile, (1, 2), (4, 4))  # Top-right
            q3 = ct.extract(tile, (2, 2), (4, 4))  # Bottom-right
            ct.store(y0, (bid, 1), q0)
            ct.store(y1, (bid, 1), q1)
            ct.store(y2, (bid, 1), q2)
            ct.store(y3, (bid, 1), q3)
            return
        end

        # Create input with different values in each quadrant
        x = CUDA.zeros(Float32, 8, 8)
        x[1:4, 1:4] .= 1.0f0   # TL
        x[5:8, 1:4] .= 2.0f0   # BL
        x[1:4, 5:8] .= 3.0f0   # TR
        x[5:8, 5:8] .= 4.0f0   # BR

        y0 = CUDA.zeros(Float32, 4, 4)
        y1 = CUDA.zeros(Float32, 4, 4)
        y2 = CUDA.zeros(Float32, 4, 4)
        y3 = CUDA.zeros(Float32, 4, 4)

        @cuda backend=cuTile extract_all_quadrants_kernel(x, y0, y1, y2, y3)

        @test all(Array(y0) .≈ 1.0f0)  # Top-left = 1
        @test all(Array(y1) .≈ 2.0f0)  # Bottom-left = 2
        @test all(Array(y2) .≈ 3.0f0)  # Top-right = 3
        @test all(Array(y3) .≈ 4.0f0)  # Bottom-right = 4
    end

    @testset "extract real/imag pattern (FFT)" begin
        # This is the pattern used in FFT: shape (BS, N, 2) -> (BS, N, 1)
        # Real at slice index 1, imag at slice index 2

        function extract_real_imag_kernel(x_ri::ct.TileArray{Float32,3},
                                          y_real::ct.TileArray{Float32,3},
                                          y_imag::ct.TileArray{Float32,3})
            bid = ct.bid(1)
            tile = ct.load(x_ri, (bid, 1, 1), (2, 8, 2))  # (BS, N, real/imag)
            # Extract real (slice 1) and imag (slice 2) in last dimension
            real_part = ct.extract(tile, (1, 1, 1), (2, 8, 1))
            imag_part = ct.extract(tile, (1, 1, 2), (2, 8, 1))
            ct.store(y_real, (bid, 1, 1), real_part)
            ct.store(y_imag, (bid, 1, 1), imag_part)
            return
        end

        # Create input: real=1.0, imag=2.0
        x = CUDA.zeros(Float32, 2, 8, 2)
        x[:, :, 1] .= 1.0f0  # real
        x[:, :, 2] .= 2.0f0  # imag

        y_real = CUDA.zeros(Float32, 2, 8, 1)
        y_imag = CUDA.zeros(Float32, 2, 8, 1)

        @cuda backend=cuTile extract_real_imag_kernel(x, y_real, y_imag)

        @test all(Array(y_real) .≈ 1.0f0)  # Real component
        @test all(Array(y_imag) .≈ 2.0f0)  # Imag component
    end
end

@testset "scalar tile getindex" begin
    function tile_getindex_kernel(x::ct.TileArray{Float32,1}, y::ct.TileArray{Float32,1})
        tile = ct.load(x, 1, (8,))
        scalar = tile[3]  # Extract 3rd element
        ct.store(y, 1, ct.broadcast_to(ct.Tile(scalar), (8,)))
        return
    end
    host_x = zeros(Float32, 8)
    host_x[3] = 42.0f0
    x = CuArray(host_x)
    y = CUDA.zeros(Float32, 8)
    @cuda backend=cuTile tile_getindex_kernel(x, y)
    @test all(Array(y) .≈ 42.0f0)
end

@testset "scalar tile setindex" begin
    function tile_setindex_kernel(x::ct.TileArray{Float32,1}, y::ct.TileArray{Float32,1})
        tile = ct.load(x, 1, (8,))
        new_tile = Base.setindex(tile, 0.0f0, 3)
        ct.store(y, 1, new_tile)
        return
    end
    x = CuArray(Float32.(1:8))
    y = CUDA.zeros(Float32, 8)
    @cuda backend=cuTile tile_setindex_kernel(x, y)
    expected = Float32.(1:8)
    expected[3] = 0.0f0
    @test Array(y) ≈ expected
end

@testset "cat" begin
    @testset "cat along last axis (axis -1)" begin
        function cat_last_axis_kernel(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,2},
                                      c::ct.TileArray{Float32,2})
            bid = ct.bid(1)
            # Load two (4, 4) tiles
            tile_a = ct.load(a, (bid, 1), (4, 4))
            tile_b = ct.load(b, (bid, 1), (4, 4))
            # Concatenate along last axis -> (4, 8)
            combined = ct.cat((tile_a, tile_b), Val(-1))
            ct.store(c, (bid, 1), combined)
            return
        end

        m, n = 64, 4
        a = CUDA.rand(Float32, m, n)
        b = CUDA.rand(Float32, m, n)
        c = CUDA.zeros(Float32, m, 8)

        @cuda backend=cuTile blocks=cld(m, 4) cat_last_axis_kernel(a, b, c)

        # Verify concatenation: c[:, 1:4] should match a, c[:, 5:8] should match b
        c_cpu = Array(c)
        a_cpu = Array(a)
        b_cpu = Array(b)

        # Due to memory layout, verify elements are preserved by checking sorted values
        for bid in 0:(cld(m, 4)-1)
            start_row = bid * 4 + 1
            input_a = a_cpu[start_row:(start_row+3), :]
            input_b = b_cpu[start_row:(start_row+3), :]
            output = c_cpu[start_row:(start_row+3), :]

            # Combined output should contain all elements from both inputs
            expected = sort(vcat(vec(input_a), vec(input_b)))
            actual = sort(vec(output))
            @test actual ≈ expected
        end
    end

    @testset "cat along first axis (axis 1)" begin
        function cat_first_axis_kernel(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,2},
                                       c::ct.TileArray{Float32,2})
            bid = ct.bid(1)
            # Load two (4, 4) tiles
            tile_a = ct.load(a, (bid, 1), (4, 4))
            tile_b = ct.load(b, (bid, 1), (4, 4))
            # Concatenate along first axis -> (8, 4)
            combined = ct.cat((tile_a, tile_b), Val(1))
            ct.store(c, (bid, 1), combined)
            return
        end

        m, n = 32, 4
        a = CUDA.rand(Float32, m, n)
        b = CUDA.rand(Float32, m, n)
        c = CUDA.zeros(Float32, m * 2, n)

        @cuda backend=cuTile blocks=cld(m, 4) cat_first_axis_kernel(a, b, c)

        # Verify concatenation: elements from both inputs should be preserved
        c_cpu = Array(c)
        a_cpu = Array(a)
        b_cpu = Array(b)

        for bid in 0:(cld(m, 4)-1)
            start_a = bid * 4 + 1
            start_c = bid * 8 + 1
            input_a = a_cpu[start_a:(start_a+3), :]
            input_b = b_cpu[start_a:(start_a+3), :]
            output = c_cpu[start_c:(start_c+7), :]

            # Combined output should contain all elements from both inputs
            expected = sort(vcat(vec(input_a), vec(input_b)))
            actual = sort(vec(output))
            @test actual ≈ expected
        end
    end

    @testset "cat roundtrip (extract then cat)" begin
        # This tests cat as the inverse of extract: extract splits, cat joins
        function extract_cat_roundtrip_kernel(x::ct.TileArray{Float32,2}, y::ct.TileArray{Float32,2})
            bid = ct.bid(1)
            # Load 4x8 tile
            tile = ct.load(x, (bid, 1), (4, 8))
            # Extract two 4x4 halves
            left = ct.extract(tile, (1, 1), (4, 4))   # rows 1-4, cols 1-4
            right = ct.extract(tile, (1, 2), (4, 4))  # rows 1-4, cols 5-8
            # Cat them back together along last axis
            combined = ct.cat((left, right), Val(-1))
            ct.store(y, (bid, 1), combined)
            return
        end

        m, n = 64, 8
        x = CUDA.rand(Float32, m, n)
        y = CUDA.zeros(Float32, m, n)

        @cuda backend=cuTile blocks=cld(m, 4) extract_cat_roundtrip_kernel(x, y)

        # Output should match input (roundtrip)
        x_cpu = Array(x)
        y_cpu = Array(y)

        for bid in 0:(cld(m, 4)-1)
            start_row = bid * 4 + 1
            input = x_cpu[start_row:(start_row+3), :]
            output = y_cpu[start_row:(start_row+3), :]

            @test output ≈ input
        end
    end
end

@testset "matmul" begin
    @testset "basic matmul" begin
        function matmul_kernel(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,2},
                               c::ct.TileArray{Float32,2})
            bidx = ct.bid(1)
            bidy = ct.bid(2)
            # Load tiles: a is (M, K), b is (K, N)
            tile_a = ct.load(a, (bidx, 1), (32, 16))
            tile_b = ct.load(b, (1, bidy), (16, 32))
            # matmul: c = a @ b (using * operator)
            result = tile_a * tile_b
            ct.store(c, (bidx, bidy), result)
            return
        end

        M, K, N = 64, 16, 64
        a = CUDA.rand(Float32, M, K)
        b = CUDA.rand(Float32, K, N)
        c = CUDA.zeros(Float32, M, N)

        grid_x = cld(M, 32)
        grid_y = cld(N, 32)
        @cuda backend=cuTile blocks=(grid_x, grid_y, 1) matmul_kernel(a, b, c)

        # Verify against CPU reference
        a_cpu = Array(a)
        b_cpu = Array(b)
        c_cpu = Array(c)
        c_ref = a_cpu * b_cpu

        @test c_cpu ≈ c_ref
    end

    @testset "vec-mat outer product" begin
        function outer_product_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,2},
                                      c::ct.TileArray{Float32,2})
            bidx = ct.bid(1)
            tile_a = ct.load(a, bidx, (16,))
            tile_b = ct.load(b, bidx, (1, 16))
            result = tile_a * tile_b
            ct.store(c, bidx, result)
            return
        end

        M, N = 16, 16
        a = CUDA.rand(Float32, M)
        b = CUDA.rand(Float32, 1, N)
        c = CUDA.zeros(Float32, M, N)

        @cuda backend=cuTile outer_product_kernel(a, b, c)

        a_cpu = Array(a)
        b_cpu = Array(b)
        c_cpu = Array(c)
        # Julia: (M,) * (1, N) = (M, 1) * (1, N) = (M, N)
        c_ref = reshape(a_cpu, :, 1) * b_cpu

        @test c_cpu ≈ c_ref
    end

    @testset "mat-vec" begin
        function matvec_kernel(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,1},
                               c::ct.TileArray{Float32,1})
            bidx = ct.bid(1)
            tile_a = ct.load(a, bidx, (32, 16))
            tile_b = ct.load(b, bidx, (16,))
            result = tile_a * tile_b
            ct.store(c, bidx, result)
            return
        end

        M, K = 32, 16
        a = CUDA.rand(Float32, M, K)
        b = CUDA.rand(Float32, K)
        c = CUDA.zeros(Float32, M)

        @cuda backend=cuTile matvec_kernel(a, b, c)

        a_cpu = Array(a)
        b_cpu = Array(b)
        c_cpu = Array(c)
        c_ref = a_cpu * b_cpu

        @test c_cpu ≈ c_ref
    end

    @testset "batched matmul with trailing batch broadcast" begin
        function batched_matmul_kernel(a::ct.TileArray{Float32,3}, b::ct.TileArray{Float32,3},
                                       c::ct.TileArray{Float32,3})
            bidx = ct.bid(1)
            tile_a = ct.load(a, bidx, (32, 16, 1))
            tile_b = ct.load(b, bidx, (16, 32, 4))
            result = tile_a * tile_b
            ct.store(c, bidx, result)
            return
        end

        M, K, N, B = 32, 16, 32, 4
        a = CUDA.rand(Float32, M, K, 1)
        b = CUDA.rand(Float32, K, N, B)
        c = CUDA.zeros(Float32, M, N, B)

        @cuda backend=cuTile batched_matmul_kernel(a, b, c)

        a_cpu = Array(a)
        b_cpu = Array(b)
        c_cpu = Array(c)
        # Reference: broadcast batch dim and matmul per batch
        a_bc = repeat(a_cpu, 1, 1, B)
        c_ref = similar(c_cpu)
        for i in 1:B
            c_ref[:, :, i] = a_bc[:, :, i] * b_cpu[:, :, i]
        end

        @test c_cpu ≈ c_ref
    end

    @testset "multi-block vec-mat" begin
        function multi_vecmat_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,2},
                                     c::ct.TileArray{Float32,2})
            bidx = ct.bid(1)
            bidy = ct.bid(2)
            tile_a = ct.load(a, bidx, (16,))
            tile_b = ct.load(b, (1, bidy), (1, 32))
            result = tile_a * tile_b
            ct.store(c, (bidx, bidy), result)
            return
        end

        M, N = 64, 128
        a = CUDA.rand(Float32, M)
        b = CUDA.rand(Float32, 1, N)
        c = CUDA.zeros(Float32, M, N)

        # Each block handles a 16-row chunk of a and a 32-col chunk of b
        @cuda backend=cuTile blocks=(cld(M, 16), cld(N, 32)) multi_vecmat_kernel(a, b, c)

        a_cpu = Array(a)
        b_cpu = Array(b)
        c_cpu = Array(c)
        c_ref = reshape(a_cpu, :, 1) * b_cpu

        @test c_cpu ≈ c_ref
    end

    @testset "multi-block mat-vec" begin
        function multi_matvec_kernel(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,1},
                                     c::ct.TileArray{Float32,1})
            bidx = ct.bid(1)
            tile_a = ct.load(a, bidx, (32, 16))
            tile_b = ct.load(b, 1, (16,))
            result = tile_a * tile_b
            ct.store(c, bidx, result)
            return
        end

        M, K = 128, 16
        a = CUDA.rand(Float32, M, K)
        b = CUDA.rand(Float32, K)
        c = CUDA.zeros(Float32, M)

        @cuda backend=cuTile blocks=cld(M, 32) multi_matvec_kernel(a, b, c)

        a_cpu = Array(a)
        b_cpu = Array(b)
        c_cpu = Array(c)
        c_ref = a_cpu * b_cpu

        @test c_cpu ≈ c_ref
    end

    @testset "batched mat-vec (3D × 1D) errors" begin
        @test_throws cuTile.CodegenErrors begin
            ct.code_tiled(Tuple{ct.TileArray{Float32,3,ct.ArraySpec{3}(16,true)},
                                ct.TileArray{Float32,1,ct.ArraySpec{1}(16,true)}}) do a, b
                bidx = ct.bid(1)
                tile_a = ct.load(a, bidx, (16, 8, 4))
                tile_b = ct.load(b, bidx, (8,))
                tile_a * tile_b
                return
            end
        end
    end

    @testset "4D batched matmul (2 batch dims)" begin
        function batched_4d_kernel(a::ct.TileArray{Float32,4}, b::ct.TileArray{Float32,4},
                                   c::ct.TileArray{Float32,4})
            bidx = ct.bid(1)
            tile_a = ct.load(a, bidx, (16, 8, 2, 4))
            tile_b = ct.load(b, bidx, (8, 16, 2, 4))
            result = tile_a * tile_b
            ct.store(c, bidx, result)
            return
        end

        M, K, N, B1, B2 = 16, 8, 16, 2, 4
        a = CUDA.rand(Float32, M, K, B1, B2)
        b = CUDA.rand(Float32, K, N, B1, B2)
        c = CUDA.zeros(Float32, M, N, B1, B2)

        @cuda backend=cuTile batched_4d_kernel(a, b, c)

        a_cpu = Array(a)
        b_cpu = Array(b)
        c_cpu = Array(c)
        c_ref = similar(c_cpu)
        for j in 1:B2, i in 1:B1
            c_ref[:, :, i, j] = a_cpu[:, :, i, j] * b_cpu[:, :, i, j]
        end

        @test c_cpu ≈ c_ref
    end

    @testset "mixed dtype matmul (Float16 × Float32 with explicit convert)" begin
        function mixed_matmul_kernel(a::ct.TileArray{Float16,2}, b::ct.TileArray{Float32,2},
                                     c::ct.TileArray{Float32,2})
            bidx = ct.bid(1)
            tile_a = ct.load(a, bidx, (32, 16))
            tile_b = ct.load(b, bidx, (16, 32))
            # MmaFOp requires matching input types; caller must convert
            result = convert(ct.Tile{Float32}, tile_a) * tile_b
            ct.store(c, bidx, result)
            return
        end

        M, K, N = 32, 16, 32
        a = CUDA.rand(Float16, M, K)
        b = CUDA.rand(Float32, K, N)
        c = CUDA.zeros(Float32, M, N)

        @cuda backend=cuTile mixed_matmul_kernel(a, b, c)

        a_cpu = Array(a)
        b_cpu = Array(b)
        c_cpu = Array(c)
        c_ref = Float32.(a_cpu) * b_cpu

        @test c_cpu ≈ c_ref rtol=1e-2
    end

    @testset "muladd mat-vec accumulated" begin
        function matvec_muladd_kernel(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,1},
                                      bias::ct.TileArray{Float32,1}, c::ct.TileArray{Float32,1})
            bidx = ct.bid(1)
            tile_a = ct.load(a, bidx, (32, 16))
            tile_b = ct.load(b, 1, (16,))
            tile_bias = ct.load(bias, bidx, (32,))
            result = muladd(tile_a, tile_b, tile_bias)
            ct.store(c, bidx, result)
            return
        end

        M, K = 64, 16
        a = CUDA.rand(Float32, M, K)
        b = CUDA.rand(Float32, K)
        bias = CUDA.rand(Float32, M)
        c = CUDA.zeros(Float32, M)

        @cuda backend=cuTile blocks=cld(M, 32) matvec_muladd_kernel(a, b, bias, c)

        a_cpu = Array(a)
        b_cpu = Array(b)
        bias_cpu = Array(bias)
        c_cpu = Array(c)
        c_ref = a_cpu * b_cpu + bias_cpu

        @test c_cpu ≈ c_ref
    end

    @testset "muladd matmul accumulated" begin
        function matmul_muladd_kernel(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,2},
                                      bias::ct.TileArray{Float32,2}, c::ct.TileArray{Float32,2})
            bidx = ct.bid(1)
            tile_a = ct.load(a, bidx, (32, 16))
            tile_b = ct.load(b, bidx, (16, 32))
            tile_bias = ct.load(bias, bidx, (32, 32))
            result = muladd(tile_a, tile_b, tile_bias)
            ct.store(c, bidx, result)
            return
        end

        M, K, N = 32, 16, 32
        a = CUDA.rand(Float32, M, K)
        b = CUDA.rand(Float32, K, N)
        bias = CUDA.rand(Float32, M, N)
        c = CUDA.zeros(Float32, M, N)

        @cuda backend=cuTile matmul_muladd_kernel(a, b, bias, c)

        a_cpu = Array(a)
        b_cpu = Array(b)
        bias_cpu = Array(bias)
        c_cpu = Array(c)
        c_ref = a_cpu * b_cpu + bias_cpu

        @test c_cpu ≈ c_ref
    end
end

@testset "reinterpret matches Base.reinterpret" begin
    @testset "2D narrowing" begin
        function u16_to_u8(a::ct.TileArray{UInt16,2}, b::ct.TileArray{UInt8,2})
            pid = ct.bid(1)
            ct.store(b, pid, reinterpret(UInt8, ct.load(a, pid, (2, 4))))
            return
        end
        let M = reshape(UInt16[0x0201, 0x0403, 0x0605, 0x0807,
                            0x0a09, 0x0c0b, 0x0e0d, 0x100f], 2, 4)
            a = CuArray(M)
            b = CUDA.zeros(UInt8, 4, 4)
            @cuda backend=cuTile blocks=1 u16_to_u8(a, b)
            @test Array(b) == Array(reinterpret(UInt8, M))
        end
    end

    @testset "1D widening" begin
        function u8_to_u16(a::ct.TileArray{UInt8,1}, b::ct.TileArray{UInt16,1})
            pid = ct.bid(1)
            ct.store(b, pid, reinterpret(UInt16, ct.load(a, pid, (8,))))
            return
        end
        let v = UInt8[0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]
            a = CuArray(v)
            b = CUDA.zeros(UInt16, 4)
            @cuda backend=cuTile blocks=1 u8_to_u16(a, b)
            @test Array(b) == reinterpret(UInt16, v)
        end
    end

    @testset "narrowing: reshape argument drops dim" begin
        function u8_reshape_u16(a::ct.TileArray{UInt8,2}, b::ct.TileArray{UInt16,1})
            pid = ct.bid(1)
            ct.store(b, pid, reinterpret(reshape, UInt16, ct.load(a, pid, (2, 4))))
            return
        end
        let M = reshape(UInt8[0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08], 2, 4)
            a = CuArray(M)
            b = CUDA.zeros(UInt16, 4)
            @cuda backend=cuTile blocks=1 u8_reshape_u16(a, b)
            @test Array(b) == reinterpret(reshape, UInt16, M)
        end
    end

    @testset "widening: reshape argument inserts dim" begin
        function u16_reshape_u8(a::ct.TileArray{UInt16,1}, b::ct.TileArray{UInt8,2})
            pid = ct.bid(1)
            ct.store(b, pid, reinterpret(reshape, UInt8, ct.load(a, pid, (4,))))
            return
        end
        let M = UInt16[0x0201, 0x0403, 0x0605, 0x0807]
            a = CuArray(M)
            b = CUDA.zeros(UInt8, 2, 4)
            @cuda backend=cuTile blocks=1 u16_reshape_u8(a, b)
            @test Array(b) == reinterpret(reshape, UInt8, M)
        end
    end

    @testset "Equal-with round-trip preserves values and shape" begin
        function u32_to_f32(a::ct.TileArray{UInt32,1}, b::ct.TileArray{Float32,1})
            pid = ct.bid(1)
            ct.store(b, pid, reinterpret(Float32, ct.load(a, pid, (16,))))
            return
        end
        let v = rand(UInt32, 16)
            a = CuArray(v)
            b = CUDA.zeros(Float32, 16)
            @cuda backend=cuTile blocks=1 u32_to_f32(a, b)
            @test reinterpret(UInt32, Array(b)) == v   # bit-exact (avoids NaN ≠ NaN)
        end
    end

    @testset "Int32 to UInt32" begin
        function i32_to_u32_2d(a::ct.TileArray{Int32,2}, b::ct.TileArray{UInt32,2})
            pid = ct.bid(1)
            ct.store(b, pid, reinterpret(UInt32, ct.load(a, pid, (4, 4))))
            return
        end
        let M = reshape(Int32.(-8:7), 4, 4)
            a = CuArray(M)
            b = CUDA.zeros(UInt32, 4, 4)
            @cuda backend=cuTile blocks=1 i32_to_u32_2d(a, b)
            @test Array(b) == reinterpret(UInt32, M)
        end
    end
end
