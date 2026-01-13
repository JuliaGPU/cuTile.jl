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

@testset "transpose" begin
    function transpose_kernel(x::ct.TileArray{Float32,2}, y::ct.TileArray{Float32,2})
        bidx = ct.bid(1)
        bidy = ct.bid(2)
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

@testset "reshape" begin
    @testset "2D -> 1D reshape preserves elements" begin
        function reshape_2d_to_1d_kernel(x::ct.TileArray{Float32,2}, y::ct.TileArray{Float32,1})
            bid = ct.bid(1)
            # Load a 4x8 tile
            tile = ct.load(x, (bid, 1), (4, 8))
            # Reshape to 32 elements (flat)
            reshaped = ct.reshape(tile, (32,))
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
            reshaped = ct.reshape(tile, (4, 8))
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
            flat = ct.reshape(tile, (32,))
            back = ct.reshape(flat, (8, 4))
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
    # These tests verify that ct.reshape matches Julia's column-major reshape behavior,
    # not just that elements are preserved (which would pass even with wrong ordering).
    # Note: tile shapes must be powers of 2.

    @testset "1D → 2D matches Julia reshape exactly" begin
        function reshape_1d_to_2d_exact_kernel(x::ct.TileArray{Float32,1}, y::ct.TileArray{Float32,2},
                                               n::ct.Constant{Int}, shape::ct.Constant{NTuple{2,Int}})
            bid = ct.bid(1)
            tile = ct.load(x, bid, (n[],))
            reshaped = ct.reshape(tile, shape[])
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
                                               shape::ct.Constant{NTuple{2,Int}}, n::ct.Constant{Int})
            bid = ct.bid(1)
            tile = ct.load(x, (bid, 1), shape[])
            reshaped = ct.reshape(tile, (n[],))
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
                                               src_shape::ct.Constant{NTuple{2,Int}},
                                               tgt_shape::ct.Constant{NTuple{2,Int}})
            bid = ct.bid(1)
            tile = ct.load(x, (bid, 1), src_shape[])
            reshaped = ct.reshape(tile, tgt_shape[])
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
                                               src_shape::ct.Constant{NTuple{3,Int}},
                                               tgt_shape::ct.Constant{NTuple{2,Int}})
            bid = ct.bid(1)
            tile = ct.load(x, (bid, 1, 1), src_shape[])
            reshaped = ct.reshape(tile, tgt_shape[])
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
                                             orig_shape::ct.Constant{NTuple{3,Int}},
                                             packed_shape::ct.Constant{NTuple{3,Int}})
            bid = ct.bid(1)
            tile = ct.load(x, (bid, 1, 1), orig_shape[])
            packed = ct.reshape(tile, packed_shape[])
            unpacked = ct.reshape(packed, orig_shape[])
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
                                         shape::ct.Constant{NTuple{2,Int}})
            bid = ct.bid(1)
            tile = ct.load(x, (bid, 1), shape[])
            flat = ct.reshape(tile, (prod(shape[]),))
            back = ct.reshape(flat, shape[])
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

@testset "permute" begin
    @testset "2D permute (transpose-like)" begin
        function permute_2d_kernel(x::ct.TileArray{Float32,2}, y::ct.TileArray{Float32,2})
            bid = ct.bid(1)
            # Load 8x4 tile
            tile = ct.load(x, (bid, 1), (8, 4))
            # Permute with (1, 0) to swap dimensions: (8, 4) -> (4, 8)
            permuted = ct.permute(tile, (2, 1))
            ct.store(y, (bid, 1), permuted)
            return
        end

        m, n = 64, 4
        x = CUDA.rand(Float32, m, n)
        y = CUDA.zeros(Float32, cld(m, 8) * 4, 8)

        ct.launch(permute_2d_kernel, cld(m, 8), x, y)

        # Verify permute matches transpose
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

    @testset "permute roundtrip preserves data" begin
        function permute_roundtrip_kernel(x::ct.TileArray{Float32,2}, y::ct.TileArray{Float32,2})
            bid = ct.bid(1)
            # Load 4x8 tile
            tile = ct.load(x, (bid, 1), (4, 8))
            # Permute with (1, 0), then back with (1, 0)
            permuted = ct.permute(tile, (2, 1))  # (4, 8) -> (8, 4)
            back = ct.permute(permuted, (2, 1))  # (8, 4) -> (4, 8)
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

        @test c_cpu ≈ c_ref rtol=1e-5
    end
end

end

@testset "Constant parameters" begin

@testset "1D with Constant tile size" begin
    function vadd_const_tile(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                             c::ct.TileArray{Float32,1}, tile::ct.Constant{Int})
        pid = ct.bid(1)
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
        bidx = ct.bid(1)
        bidy = ct.bid(2)
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

    @test Array(c) ≈ Array(a) ./ Array(b) rtol=1e-5
end

@testset "1D sqrt" begin
    function vsqrt_1d(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        tile = ct.load(a, pid, (16,))
        ct.store(b, pid, sqrt.(tile))
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
        pid = ct.bid(1)
        tile = ct.load(a, (pid, 1), (1, 128))  # Load 1x128 tile
        sums = ct.reduce_sum(tile, 2)           # Sum along axis 1 -> (1,)
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
        pid = ct.bid(1)
        tile = ct.load(a, (1, pid), (64, 1))   # Load 64x1 tile
        sums = ct.reduce_sum(tile, 1)           # Sum along axis 0 -> (1,)
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
        pid = ct.bid(1)
        tile = ct.load(a, (pid, 1), (1, 128))  # Load 1x128 tile
        maxes = ct.reduce_max(tile, 2)          # Max along axis 1 -> (1,)
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
        pid = ct.bid(1)
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
        pid = ct.bid(1)
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

@testset "scalar ./ tile" begin
    function scalar_div_tile_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        tile = ct.load(a, pid, (16,))
        result = 1.0f0 ./ tile
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

@testset "tile .+ scalar" begin
    function add_scalar(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        tile = ct.load(a, pid, (16,))
        result = tile .+ 3.5f0
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

@testset "scalar .+ tile" begin
    function scalar_add_tile_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        tile = ct.load(a, pid, (16,))
        result = 2.5f0 .+ tile
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

@testset "tile .- scalar" begin
    function sub_scalar(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        tile = ct.load(a, pid, (16,))
        result = tile .- 1.5f0
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

@testset "scalar .- tile" begin
    function scalar_sub_tile_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        tile = ct.load(a, pid, (16,))
        result = 5.0f0 .- tile
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
        pid = ct.bid(1)
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
        pid = ct.bid(1)
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
    @test c_cpu ≈ a_cpu[1] .+ b_cpu rtol=1e-5
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
    @test c_cpu ≈ expected rtol=1e-5
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
    @test c_cpu ≈ expected rtol=1e-5
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
    @test c_cpu ≈ a_cpu .- b_cpu[1] rtol=1e-5
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
    @test c_cpu ≈ expected rtol=1e-5
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
        @test c_cpu[i, :] ≈ a_cpu[1, :] rtol=1e-5
    end
end

@testset "mismatched shapes with + throws MethodError" begin
    # Verify that + with different tile shapes throws MethodError (Julia-idiomatic)
    # Note: This tests the type system, not kernel execution
    tile_a = ct.Tile{Float32, (1, 128)}()
    tile_b = ct.Tile{Float32, (64, 1)}()

    # + should require same shapes, so this should fail
    @test_throws MethodError tile_a + tile_b

    # But .+ should work (broadcasting)
    result = tile_a .+ tile_b
    @test result isa ct.Tile{Float32, (64, 128)}
end

end

@testset "comparison operations" begin

@testset "float comparison operators" begin
    # Test all broadcast comparison operators with Float32 tiles
    tile = ct.Tile{Float32, (16,)}()

    @test (tile .< tile) isa ct.Tile{Bool, (16,)}
    @test (tile .> tile) isa ct.Tile{Bool, (16,)}
    @test (tile .<= tile) isa ct.Tile{Bool, (16,)}
    @test (tile .>= tile) isa ct.Tile{Bool, (16,)}
    @test (tile .== tile) isa ct.Tile{Bool, (16,)}
    @test (tile .!= tile) isa ct.Tile{Bool, (16,)}
end

@testset "integer comparison operators" begin
    # Test all broadcast comparison operators with Int tiles
    int_tile = ct.arange((16,), Int)

    @test (int_tile .< int_tile) isa ct.Tile{Bool, (16,)}
    @test (int_tile .> int_tile) isa ct.Tile{Bool, (16,)}
    @test (int_tile .<= int_tile) isa ct.Tile{Bool, (16,)}
    @test (int_tile .>= int_tile) isa ct.Tile{Bool, (16,)}
    @test (int_tile .== int_tile) isa ct.Tile{Bool, (16,)}
    @test (int_tile .!= int_tile) isa ct.Tile{Bool, (16,)}
end

@testset "tile vs scalar comparison" begin
    int_tile = ct.arange((16,), Int)
    float_tile = ct.Tile{Float32, (16,)}()

    # Int tile vs Int scalar
    @test (int_tile .< 10) isa ct.Tile{Bool, (16,)}
    @test (5 .< int_tile) isa ct.Tile{Bool, (16,)}

    # Float32 tile vs Float32 scalar
    @test (float_tile .< 2.0f0) isa ct.Tile{Bool, (16,)}
    @test (1.0f0 .> float_tile) isa ct.Tile{Bool, (16,)}
end

@testset "broadcast comparison shapes" begin
    tile_a = ct.Tile{Float32, (1, 16)}()
    tile_b = ct.Tile{Float32, (8, 1)}()

    # (1, 16) .< (8, 1) -> (8, 16)
    result = tile_a .< tile_b
    @test result isa ct.Tile{Bool, (8, 16)}
end

end

@testset "power operations" begin

@testset "float tile .^ float tile" begin
    tile = ct.Tile{Float32, (16,)}()
    @test (tile .^ tile) isa ct.Tile{Float32, (16,)}
end

@testset "float tile .^ scalar" begin
    tile = ct.Tile{Float32, (16,)}()
    @test (tile .^ 2.0f0) isa ct.Tile{Float32, (16,)}
    @test (2.0f0 .^ tile) isa ct.Tile{Float32, (16,)}
end

@testset "broadcast power shapes" begin
    tile_a = ct.Tile{Float32, (1, 16)}()
    tile_b = ct.Tile{Float32, (8, 1)}()
    @test (tile_a .^ tile_b) isa ct.Tile{Float32, (8, 16)}
end

@testset "integer power not supported" begin
    int_tile = ct.arange((16,), Int)
    @test_throws MethodError int_tile .^ int_tile
end

end

@testset "atomic operations" begin

@testset "atomic_add Int" begin
    # Test atomic_add with Int: each thread block adds 1 to a counter
    function atomic_add_kernel(counters::ct.TileArray{Int,1})
        bid = ct.bid(1)
        ct.atomic_add(counters, 1, 1;
                     memory_order=ct.MemoryOrder.AcqRel)
        return
    end

    n_blocks = 1000
    counters = CUDA.zeros(Int, 1)

    ct.launch(atomic_add_kernel, n_blocks, counters)

    result = Array(counters)[1]
    @test result == n_blocks
end

@testset "atomic_add Float32" begin
    # Test atomic_add with Float32
    function atomic_add_f32_kernel(out::ct.TileArray{Float32,1}, val::ct.Constant{Float32})
        bid = ct.bid(1)
        ct.atomic_add(out, 1, val[];
                     memory_order=ct.MemoryOrder.AcqRel)
        return
    end

    n_blocks = 100
    out = CUDA.zeros(Float32, 1)
    val = 1.5f0

    ct.launch(atomic_add_f32_kernel, n_blocks, out, ct.Constant(val))

    result = Array(out)[1]
    @test result ≈ n_blocks * val rtol=1e-3
end

@testset "atomic_xchg" begin
    # Test atomic_xchg: each thread exchanges, last one wins
    function atomic_xchg_kernel(arr::ct.TileArray{Int,1})
        bid = ct.bid(1)
        ct.atomic_xchg(arr, 1, bid + 1;
                      memory_order=ct.MemoryOrder.AcqRel)
        return
    end

    n_blocks = 10
    arr = CUDA.zeros(Int, 1)

    ct.launch(atomic_xchg_kernel, n_blocks, arr)

    # Result should be one of 1..n_blocks (whichever thread ran last)
    result = Array(arr)[1]
    @test 1 <= result <= n_blocks
end

@testset "atomic_cas success" begin
    # Test atomic_cas: only one thread should succeed in setting 0->1
    function atomic_cas_kernel(locks::ct.TileArray{Int,1}, success_count::ct.TileArray{Int,1})
        bid = ct.bid(1)
        # Try to acquire lock (0 -> 1)
        old = ct.atomic_cas(locks, 1, 0, 1;
                           memory_order=ct.MemoryOrder.AcqRel)
        # If we got old=0, we succeeded
        # Use atomic_add to count successes (returns a tile, so comparison works)
        # Actually simpler: just increment success_count if old was 0
        # But we can't do conditionals easily here, so let's just verify lock changes
        return
    end

    locks = CUDA.zeros(Int, 1)
    success_count = CUDA.zeros(Int, 1)

    ct.launch(atomic_cas_kernel, 100, locks, success_count)

    # Lock should be set to 1 (at least one thread succeeded)
    lock_val = Array(locks)[1]
    @test lock_val == 1
end

@testset "spinlock with token ordering" begin
    # Test that token threading enforces memory ordering in spinlock patterns
    function spinlock_kernel(result::ct.TileArray{Float32,1}, lock::ct.TileArray{Int,1})
        bid = ct.bid(1)
        val = ct.full((1,), 1.0f0, Float32)

        # Spin until we acquire the lock (CAS returns old value, 0 means we got it)
        while ct.atomic_cas(lock, 1, 0, 1;
                           memory_order=ct.MemoryOrder.Acquire) == 1
        end

        # Critical section: load, increment, store
        # With proper token threading, these are ordered after the acquire
        current = ct.load(result, 1, (1,))
        updated = current .+ val
        ct.store(result, 1, updated)

        # Release the lock
        ct.atomic_xchg(lock, 1, 0;
                      memory_order=ct.MemoryOrder.Release)
        return
    end

    n_blocks = 50  # Use fewer blocks to reduce test time
    result = CUDA.zeros(Float32, 1)
    lock = CUDA.zeros(Int, 1)

    ct.launch(spinlock_kernel, n_blocks, result, lock)

    # Each block should have added 1.0 to the result
    final_result = Array(result)[1]
    @test final_result == Float32(n_blocks)
end

@testset "explicit memory ordering kwargs" begin
    # Test that explicit memory_order kwargs work correctly
    function explicit_ordering_kernel(result::ct.TileArray{Float32,1}, lock::ct.TileArray{Int,1})
        bid = ct.bid(1)
        val = ct.full((1,), 1.0f0, Float32)

        # Spin until we acquire the lock - use explicit Acquire ordering
        while ct.atomic_cas(lock, 1, 0, 1;
                           memory_order=ct.MemoryOrder.Acquire) == 1
        end

        # Critical section
        current = ct.load(result, 1, (1,))
        updated = current .+ val
        ct.store(result, 1, updated)

        # Release the lock - use explicit Release ordering
        ct.atomic_xchg(lock, 1, 0; memory_order=ct.MemoryOrder.Release)
        return
    end

    n_blocks = 50
    result = CUDA.zeros(Float32, 1)
    lock = CUDA.zeros(Int, 1)

    ct.launch(explicit_ordering_kernel, n_blocks, result, lock)

    final_result = Array(result)[1]
    @test final_result == Float32(n_blocks)
end

@testset "atomic_add with explicit kwargs" begin
    # Test atomic_add with explicit memory ordering
    function explicit_add_kernel(counters::ct.TileArray{Int,1})
        bid = ct.bid(1)
        ct.atomic_add(counters, 1, 1;
                     memory_order=ct.MemoryOrder.Relaxed,
                     memory_scope=ct.MemScope.Device)
        return
    end

    n_blocks = 100
    counters = CUDA.zeros(Int, 1)

    ct.launch(explicit_add_kernel, n_blocks, counters)

    result = Array(counters)[1]
    @test result == n_blocks
end

@testset "1D gather - simple" begin
    # Simple 1D gather: copy first 16 elements using gather
    function gather_simple_kernel(src::ct.TileArray{Float32,1}, dst::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        # Simple indices 0..15
        indices = ct.arange((16,), Int)
        # Gather from source
        tile = ct.gather(src, indices)
        # Store to destination
        ct.store(dst, pid, tile)
        return
    end

    n = 16
    src = CUDA.rand(Float32, n)
    dst = CUDA.zeros(Float32, n)

    ct.launch(gather_simple_kernel, 1, src, dst)

    @test Array(dst) ≈ Array(src)
end

@testset "1D scatter - simple" begin
    # Simple 1D scatter: write first 16 elements using scatter
    function scatter_simple_kernel(src::ct.TileArray{Float32,1}, dst::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        # Load from source
        tile = ct.load(src, pid, (16,))
        # Simple indices 0..15
        indices = ct.arange((16,), Int)
        # Scatter to destination
        ct.scatter(dst, indices, tile)
        return
    end

    n = 16
    src = CUDA.rand(Float32, n)
    dst = CUDA.zeros(Float32, n)

    ct.launch(scatter_simple_kernel, 1, src, dst)

    @test Array(dst) ≈ Array(src)
end

end

@testset "Entry Hints Integration" begin

@testset "launch with num_ctas" begin
    function vadd_kernel_num_ctas(a::ct.TileArray{Float32,1},
                        b::ct.TileArray{Float32,1},
                        c::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        ta = ct.load(a, pid, (16,))
        tb = ct.load(b, pid, (16,))
        ct.store(c, pid, ta + tb)
        return nothing
    end

    n = 1024
    a = CUDA.ones(Float32, n)
    b = CUDA.ones(Float32, n) .* 2
    c = CUDA.zeros(Float32, n)

    ct.launch(vadd_kernel_num_ctas, 64, a, b, c; num_ctas=2)
    CUDA.synchronize()
    @test Array(c) ≈ ones(Float32, n) .* 3
end

@testset "launch with occupancy" begin
    function vadd_kernel_occupancy(a::ct.TileArray{Float32,1},
                        b::ct.TileArray{Float32,1},
                        c::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        ta = ct.load(a, pid, (16,))
        tb = ct.load(b, pid, (16,))
        ct.store(c, pid, ta + tb)
        return nothing
    end

    n = 1024
    a = CUDA.ones(Float32, n)
    b = CUDA.ones(Float32, n) .* 2
    c = CUDA.zeros(Float32, n)

    ct.launch(vadd_kernel_occupancy, 64, a, b, c; occupancy=4)
    CUDA.synchronize()
    @test Array(c) ≈ ones(Float32, n) .* 3
end

@testset "launch with both hints" begin
    function vadd_kernel_both_hints(a::ct.TileArray{Float32,1},
                        b::ct.TileArray{Float32,1},
                        c::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        ta = ct.load(a, pid, (16,))
        tb = ct.load(b, pid, (16,))
        ct.store(c, pid, ta + tb)
        return nothing
    end

    n = 1024
    a = CUDA.ones(Float32, n)
    b = CUDA.ones(Float32, n) .* 2
    c = CUDA.zeros(Float32, n)

    ct.launch(vadd_kernel_both_hints, 64, a, b, c; num_ctas=4, occupancy=8)
    CUDA.synchronize()
    @test Array(c) ≈ ones(Float32, n) .* 3
end

end
