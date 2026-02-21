using CUDA

@testset "launch" begin

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

    ct.launch(vadd_1d, cld(n, tile_size), a, b, c)

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

    ct.launch(vsub_1d, cld(n, tile_size), a, b, c)

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

    ct.launch(vmul_1d, cld(n, tile_size), a, b, c)

    @test Array(c) ≈ Array(a) .* Array(b)
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

    ct.launch(madd_2d, (cld(m, tile_x), cld(n, tile_y)), a, b, c)

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
    ct.launch(tadd_4d, grid, a, b, c)

    @test Array(c) ≈ Array(a) + Array(b)
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

        ct.launch(copy_1d_2d, cld(m, 16), src, dst)

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

        ct.launch(copy_2d_4d, (cld(d1, 4), cld(d2, 4)), src, dst)

        @test Array(dst) ≈ Array(src)
    end
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

    ct.launch(transpose_kernel, (cld(m, tile_size), cld(n, tile_size)), x, y)

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

        ct.launch(reshape_2d_to_1d_kernel, cld(m, 4), x, y)

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

        ct.launch(reshape_1d_to_2d_kernel, cld(n, 32), x, y)

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

        ct.launch(reshape_roundtrip_kernel, cld(m, 8), x, y)

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

        ct.launch(reshape_1d_to_2d_exact_kernel, 1, x, y, ct.Constant(n), ct.Constant(shape))

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

        ct.launch(reshape_2d_to_1d_exact_kernel, 1, x, y, ct.Constant(shape), ct.Constant(n))

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

        ct.launch(reshape_2d_to_2d_exact_kernel, 1, x, y,
                  ct.Constant(src_shape), ct.Constant(tgt_shape))

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

        ct.launch(reshape_3d_to_2d_exact_kernel, 1, x, y,
                  ct.Constant(src_shape), ct.Constant(tgt_shape))

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

        ct.launch(reshape_roundtrip_3d_kernel, 1, x, y,
                  ct.Constant(orig_shape), ct.Constant(packed_shape))

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

        ct.launch(reshape_2d_1d_2d_kernel, 1, x, y, ct.Constant(shape))

        @test Array(y) ≈ Array(x)
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

        ct.launch(permute_2d_kernel, cld(m, 8), x, y)

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

        ct.launch(permute_roundtrip_kernel, cld(m, 4), x, y)

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
        ct.launch(copy_kernel_2d, grid, P, out, ct.Constant(tm), ct.Constant(tn))

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

        ct.launch(order_load_kernel, (cld(n, t), cld(n, t)), src, dst, ct.Constant(t))

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

        ct.launch(order_store_kernel, (cld(n, t), cld(n, t)), src, dst, ct.Constant(t))

        @test Array(dst) ≈ transpose(Array(src))
    end
end

@testset "extract" begin
    @testset "extract identity (0,0) full shape" begin
        function extract_identity_kernel(x::ct.TileArray{Float32,2}, y::ct.TileArray{Float32,2})
            bid = ct.bid(1)
            # Load 4x8 tile
            tile = ct.load(x, (bid, 1), (4, 8))
            # Extract the full tile starting at (0, 0)
            extracted = ct.extract(tile, (2, 2), (4, 8))
            ct.store(y, (bid, 1), extracted)
            return
        end

        m, n = 64, 8
        x = CUDA.rand(Float32, m, n)
        y = CUDA.zeros(Float32, m, n)

        ct.launch(extract_identity_kernel, cld(m, 4), x, y)

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

        ct.launch(extract_smaller_kernel, cld(m, 8), x, y)

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

        ct.launch(extract_all_quadrants_kernel, 1, x, y0, y1, y2, y3)

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

        ct.launch(extract_real_imag_kernel, 1, x, y_real, y_imag)

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
    ct.launch(tile_getindex_kernel, 1, x, y)
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
    ct.launch(tile_setindex_kernel, 1, x, y)
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

        ct.launch(cat_last_axis_kernel, cld(m, 4), a, b, c)

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

        ct.launch(cat_first_axis_kernel, cld(m, 4), a, b, c)

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

        ct.launch(extract_cat_roundtrip_kernel, cld(m, 4), x, y)

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
        ct.launch(matmul_kernel, (grid_x, grid_y, 1), a, b, c)

        # Verify against CPU reference
        a_cpu = Array(a)
        b_cpu = Array(b)
        c_cpu = Array(c)
        c_ref = a_cpu * b_cpu

        @test c_cpu ≈ c_ref
    end
end

end

@testset "Constant parameters" begin

@testset "1D with Constant tile size" begin
    function vadd_const_tile(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                             c::ct.TileArray{Float32,1}, tile::Int)
        pid = ct.bid(1)
        tile_a = ct.load(a, pid, (tile,))
        tile_b = ct.load(b, pid, (tile,))
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
                              tx::Int, ty::Int)
        bidx = ct.bid(1)
        bidy = ct.bid(2)
        tile_a = ct.load(a, (bidx, bidy), (tx, ty))
        tile_b = ct.load(b, (bidx, bidy), (tx, ty))
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
        pid = ct.bid(1)
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
        pid = ct.bid(1)
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

@testset "BFloat16" begin
    function vadd_bf16(a::ct.TileArray{ct.BFloat16,1}, b::ct.TileArray{ct.BFloat16,1},
                      c::ct.TileArray{ct.BFloat16,1})
        pid = ct.bid(1)
        tile_a = ct.load(a, pid, (16,))
        tile_b = ct.load(b, pid, (16,))
        ct.store(c, pid, tile_a + tile_b)
        return
    end

    n = 1024
    tile_size = 16
    a = CUDA.rand(ct.BFloat16, n)
    b = CUDA.rand(ct.BFloat16, n)
    c = CUDA.zeros(ct.BFloat16, n)

    ct.launch(vadd_bf16, cld(n, tile_size), a, b, c)

    @test Array(c) ≈ Array(a) + Array(b)
end

end

@testset "compilation cache" begin
    function cached_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1})
        pid = ct.bid(1)
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
        pid = ct.bid(1)
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

@testset "math operations" begin

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

    ct.launch(vdiv_1d, cld(n, tile_size), a, b, c)

    @test Array(c) ≈ Array(a) ./ Array(b)
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
        ct.launch($(Symbol("vmath_$(name)")), cld(1024, 16), a, b)
        @test Array(b) ≈ $op.(Array(a)) rtol=1e-4
    end
end

end

const _EXEC_TEST_GLOBAL_CONST = Float32(1 / log(2))

@testset "global constant arithmetic" begin
    # Regression test for issue #77: scalar × global constant failed during codegen.
    function global_const_arith_kernel(a::ct.TileArray{Float32,1},
                                       b::ct.TileArray{Float32,1},
                                       scale::Float32)
        pid = ct.bid(1)
        tile = ct.load(a, pid, (16,))
        total_scale = scale * _EXEC_TEST_GLOBAL_CONST
        ct.store(b, pid, tile .* total_scale)
        return
    end

    n = 1024
    tile_size = 16
    scale = 2.5f0
    a = CUDA.rand(Float32, n)
    b = CUDA.zeros(Float32, n)

    ct.launch(global_const_arith_kernel, cld(n, tile_size), a, b, scale)

    @test Array(b) ≈ Array(a) .* (scale * _EXEC_TEST_GLOBAL_CONST)
end

@testset "kernel name with !" begin
    function kernel!()
        return
    end
    ct.launch(kernel!, 1)
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

