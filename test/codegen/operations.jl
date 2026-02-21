#=============================================================================
 Operations
=============================================================================#

@testset "Operations" begin
    # Common ArraySpecs for tests
    spec1d = ct.ArraySpec{1}(16, true)
    spec2d = ct.ArraySpec{2}(16, true)
    spec3d = ct.ArraySpec{3}(16, true)

    #=========================================================================
     8.3 Core
    =========================================================================#
    @testset "Core" begin
        # TODO: cat - tile concatenation
        # TODO: extract - extract subtile
        # TODO: get_global - global variable access
        # TODO: global - global variable definition
        # TODO: mmai - integer matrix multiply-accumulate
        # TODO: offset - tile offset computation
        # TODO: pack - pack tiles
        @testset "scan" begin
            # Forward scan - float and integer types
            for (T, spec, op_check) in [
                (Float32, spec1d, "addf"),
                (Int32, spec1d, "addi"),
            ]
                @test @filecheck begin
                    @check_label "entry"
                    code_tiled(Tuple{ct.TileArray{T,1,spec}}) do a
                        pid = ct.bid(1)
                        tile = ct.load(a, pid, (16,))
                        @check "scan"
                        @check op_check
                        Base.donotdelete(cumsum(tile; dims=1))
                        return
                    end
                end
            end

            # 2D scan along different axes
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,2,spec2d}}) do a
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (4, 8))
                    @check "scan"
                    Base.donotdelete(cumsum(tile; dims=1))
                    @check "scan"
                    Base.donotdelete(cumsum(tile; dims=2))
                    return
                end
            end

            # Reverse scan
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,2,spec2d}}) do a
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (4, 8))
                    @check "scan"
                    Base.donotdelete(cumsum(tile; dims=1, rev=true))
                    return
                end
            end

            # cumsum convenience
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,2,spec2d}}) do a
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (4, 8))
                    @check "scan"
                    Base.donotdelete(cumsum(tile; dims=2))
                    return
                end
            end

            # cumprod
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,2,spec2d}}) do a
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (4, 8))
                    @check "scan"
                    @check "mulf"
                    Base.donotdelete(cumprod(tile; dims=2))
                    return
                end
            end
        end
        # TODO: unpack - unpack tiles

        @testset "reshape" begin
            # 2D -> 1D reshape (emits pre-permute for column-major conversion)
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,2,spec2d}, ct.TileArray{Float32,1,spec1d}}) do a, b
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (4, 8))
                    @check "permute"   # pre-permute for 2D source
                    @check "reshape"
                    reshaped = reshape(tile, (32,))
                    ct.store(b, pid, reshaped)
                    return
                end
            end

            # 1D -> 2D reshape (emits post-permute for column-major conversion)
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,2,spec2d}}) do a, b
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (64,))
                    @check "reshape"
                    @check "permute"   # post-permute for 2D result
                    reshaped = reshape(tile, (8, 8))
                    ct.store(b, pid, reshaped)
                    return
                end
            end

            # 3D -> 2D reshape (emits pre-permute and post-permute)
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,3,spec3d}, ct.TileArray{Float32,2,spec2d}}) do a, b
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (2, 4, 8))
                    @check "permute"   # pre-permute for 3D source
                    @check "reshape"
                    @check "permute"   # post-permute for 2D result
                    reshaped = reshape(tile, (2, 32))
                    ct.store(b, pid, reshaped)
                    return
                end
            end

            # 1D -> 1D same-shape reshape is a no-op
            @test @filecheck begin
                @check_label "entry"
                @check_not "permute"
                @check_not "reshape"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}}) do a
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (32,))
                    reshaped = reshape(tile, (32,))
                    ct.store(a, pid, reshaped)
                    return
                end
            end

            # 2D -> 2D reshape (different shape, emits both permutes)
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,2,spec2d}}) do a
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (4, 8))
                    @check "permute"   # pre-permute
                    @check "reshape"
                    @check "permute"   # post-permute
                    reshaped = reshape(tile, (8, 4))
                    ct.store(a, pid, reshaped)
                    return
                end
            end
        end

        @testset "dropdims" begin
            # dropdims on dim 1: (1, 8) -> dropdims(; dims=2) -> (8,)
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,2,spec2d}, ct.TileArray{Float32,1,spec1d}}) do a, b
                    pid = ct.bid(1)
                    tile = ct.load(a, (pid, 1), (1, 8))
                    @check "reshape"
                    squeezed = dropdims(tile; dims=1)
                    ct.store(b, pid, squeezed)
                    return
                end
            end

            # dropdims on dim 2: (8, 1) -> dropdims(; dims=2) -> (8,)
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,2,spec2d}, ct.TileArray{Float32,1,spec1d}}) do a, b
                    pid = ct.bid(1)
                    tile = ct.load(a, (1, pid), (8, 1))
                    @check "reshape"
                    squeezed = dropdims(tile; dims=2)
                    ct.store(b, pid, squeezed)
                    return
                end
            end
        end

        @testset "permutedims" begin
            # 2D permutedims with explicit perm (same as transpose)
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,2,spec2d}, ct.TileArray{Float32,2,spec2d}}) do a, b
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (4, 8))
                    @check "permute"
                    permuted = permutedims(tile, (2, 1))
                    ct.store(b, pid, permuted)
                    return
                end
            end

            # 3D permutedims
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,3,spec3d}, ct.TileArray{Float32,3,spec3d}}) do a, b
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (2, 4, 8))
                    @check "permute"
                    # (2,4,8) with perm (3,1,2) -> (8,2,4)
                    permuted = permutedims(tile, (3, 1, 2))
                    ct.store(b, pid, permuted)
                    return
                end
            end

            # 3D identity permutedims
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,3,spec3d}, ct.TileArray{Float32,3,spec3d}}) do a, b
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (2, 4, 8))
                    @check "permute"
                    # (2,4,8) with perm (1,2,3) -> (2,4,8) (identity)
                    permuted = permutedims(tile, (1, 2, 3))
                    ct.store(b, pid, permuted)
                    return
                end
            end

            # 2D permutedims with no-arg (defaults to transpose)
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,2,spec2d}, ct.TileArray{Float32,2,spec2d}}) do a, b
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (4, 8))
                    @check "permute"
                    permuted = permutedims(tile)  # defaults to (2, 1)
                    ct.store(b, pid, permuted)
                    return
                end
            end

            # 1D permutedims (reshape to row)
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,2,spec2d}}) do a, b
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (32,))
                    @check "reshape"
                    row = permutedims(tile)  # (32,) -> (1, 32)
                    ct.store(b, pid, row)
                    return
                end
            end
        end

        @testset "extract" begin
            # Extract slice from 2D tile
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,2,spec2d}, ct.TileArray{Float32,2,spec2d}}) do a, b
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (4, 8))
                    @check "extract"
                    # Extract 2x4 slice starting at (2, 3) (1-indexed)
                    extracted = ct.extract(tile, (2, 3), (2, 4))
                    ct.store(b, pid, extracted)
                    return
                end
            end

            # Scalar tile getindex
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}}) do a
                    tile = ct.load(a, 1, (8,))
                    @check "extract"
                    # extract produces tile<1xf32>, reshape to scalar tile<f32>
                    @check "tile<1xf32> -> tile<f32>"
                    scalar = tile[3]
                    ct.store(a, 1, ct.broadcast_to(ct.Tile(scalar), (8,)))
                    return
                end
            end

            # Scalar tile setindex (functional)
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}}) do a
                    tile = ct.load(a, 1, (8,))
                    @check "iota"
                    @check "select"
                    new_tile = Base.setindex(tile, 0.0f0, 3)
                    ct.store(a, 1, new_tile)
                    return
                end
            end

            # Extract slice from 3D tile (FFT real/imag pattern)
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,3,spec3d}, ct.TileArray{Float32,3,spec3d}}) do a, b
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (2, 4, 2))  # (batch, N, real/imag)
                    @check "extract"
                    # Extract real component (index 1 in last dimension, 1-indexed)
                    real_part = ct.extract(tile, (1, 1, 1), (2, 4, 1))
                    ct.store(b, pid, real_part)
                    return
                end
            end
        end

        @testset "cat" begin
            # Concatenate along last axis (axis -1)
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,2,spec2d}, ct.TileArray{Float32,2,spec2d}}) do a, b
                    pid = ct.bid(1)
                    tile1 = ct.load(a, pid, (4, 4))
                    tile2 = ct.load(b, pid, (4, 4))
                    @check "cat"
                    combined = ct.cat((tile1, tile2), Val(-1))  # -> (4, 8)
                    ct.store(a, pid, combined)
                    return
                end
            end

            # Concatenate along first axis (axis 0)
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,2,spec2d}, ct.TileArray{Float32,2,spec2d}}) do a, b
                    pid = ct.bid(1)
                    tile1 = ct.load(a, pid, (4, 8))
                    tile2 = ct.load(b, pid, (4, 8))
                    @check "cat"
                    combined = ct.cat((tile1, tile2), Val(1))  # -> (8, 8)
                    ct.store(a, pid, combined)
                    return
                end
            end
        end

        @testset "broadcast" begin
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,2,spec2d}, ct.TileArray{Float32,2,spec2d}}) do a, b
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (1, 16))
                    @check "broadcast"
                    expanded = ct.broadcast_to(tile, (32, 16))
                    ct.store(b, pid, expanded)
                    return
                end
            end
        end

        @testset "cmpf" begin
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}}) do a, b, c
                    pid = ct.bid(1)
                    tile_a = ct.load(a, pid, (16,))
                    tile_b = ct.load(b, pid, (16,))
                    @check "cmpf"
                    mask = tile_a .< tile_b
                    result = ct.where(mask, tile_a, tile_b)
                    ct.store(c, pid, result)
                    return
                end
            end
        end

        @testset "cmpi" begin
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Int32,1,spec1d}, ct.TileArray{Int32,1,spec1d}, ct.TileArray{Int32,1,spec1d}}) do a, b, c
                    pid = ct.bid(1)
                    tile_a = ct.load(a, pid, (16,))
                    tile_b = ct.load(b, pid, (16,))
                    @check "cmpi"
                    mask = tile_a .< tile_b
                    result = ct.where(mask, tile_a, tile_b)
                    ct.store(c, pid, result)
                    return
                end
            end
        end

        @testset "mixed-type integer comparison" begin
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Int64,1,spec1d}}) do out
                    a = ct.arange((16,), Int64)
                    b = ct.arange((16,), Int32)
                    # Should promote Int32 to Int64 and compare
                    @check "exti"
                    @check "cmpi"
                    @check "select"
                    result = a .< b
                    # Use same-typed operands for where to avoid Union type
                    b_promoted = convert(ct.Tile{Int64}, b)
                    selected = ct.where(result, a, b_promoted)
                    ct.store(out, Int32(0), selected)
                    return
                end
            end
        end

        @testset "constant" begin
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}}) do a
                    pid = ct.bid(1)
                    @check "constant"
                    tile = ct.full((16,), 0.0f0, Float32)
                    ct.store(a, pid, tile)
                    return
                end
            end
        end

        @testset "get_num_tile_blocks" begin
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}}) do a
                    @check "get_num_tile_blocks"
                    nb = ct.num_blocks(1)
                    Base.donotdelete(nb)
                    return
                end
            end
        end

        @testset "get_tile_block_id" begin
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}}) do a
                    @check "get_tile_block_id"
                    pid = ct.bid(1)
                    Base.donotdelete(pid)
                    return
                end
            end
        end

        @testset "iota" begin
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Int32,1,spec1d}}) do a
                    pid = ct.bid(1)
                    @check "iota"
                    tile = ct.arange((16,), Int32)
                    ct.store(a, pid, tile)
                    return
                end
            end
        end

        @testset "mmaf" begin
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,2,spec2d}, ct.TileArray{Float32,2,spec2d}, ct.TileArray{Float32,2,spec2d}}) do a, b, c
                    bidx = ct.bid(1)
                    bidy = ct.bid(2)
                    tile_a = ct.load(a, bidx, (32, 16))
                    tile_b = ct.load(b, bidy, (16, 32))
                    acc = ct.full((32, 32), 0.0f0, Float32)
                    @check "mma"
                    result = muladd(tile_a, tile_b, acc)
                    ct.store(c, (bidx, bidy), result)
                    return
                end
            end
        end

        @testset "matmul" begin
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,2,spec2d}, ct.TileArray{Float32,2,spec2d}, ct.TileArray{Float32,2,spec2d}}) do a, b, c
                    bidx = ct.bid(1)
                    bidy = ct.bid(2)
                    tile_a = ct.load(a, bidx, (32, 16))
                    tile_b = ct.load(b, bidy, (16, 32))
                    # matmul via * operator = mma with zero accumulator
                    @check "mma"
                    result = tile_a * tile_b
                    ct.store(c, (bidx, bidy), result)
                    return
                end
            end
        end

        @testset "permute" begin
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,2,spec2d}, ct.TileArray{Float32,2,spec2d}}) do a, b
                    bidx = ct.bid(1)
                    bidy = ct.bid(2)
                    tile = ct.load(a, (bidx, bidy), (32, 64))
                    @check "permute"
                    transposed = transpose(tile)
                    ct.store(b, (bidy, bidx), transposed)
                    return
                end
            end
        end

        @testset "reduce" begin
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,2,spec2d}, ct.TileArray{Float32,1,spec1d}}) do a, b
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (4, 16))
                    @check "reduce"
                    @check "addf"
                    sums = sum(tile; dims=2)
                    ct.store(b, pid, sums)
                    return
                end
            end

            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,2,spec2d}, ct.TileArray{Float32,1,spec1d}}) do a, b
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (4, 16))
                    @check "reduce"
                    @check "maxf"
                    maxes = maximum(tile; dims=2)
                    ct.store(b, pid, maxes)
                    return
                end
            end

            # minimum
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,2,spec2d}, ct.TileArray{Float32,1,spec1d}}) do a, b
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (4, 16))
                    @check "reduce"
                    @check "minf"
                    mins = minimum(tile; dims=2)
                    ct.store(b, pid, mins)
                    return
                end
            end

            # prod
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,2,spec2d}}) do a
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (4, 16))
                    @check "reduce"
                    @check "mulf"
                    Base.donotdelete(prod(tile; dims=2))
                    return
                end
            end

            # Integer/unsigned reduce
            for (T, op, op_check) in [
                (Int32,  tile -> sum(tile; dims=2), "addi"),
                (Int32,  tile -> maximum(tile; dims=2), "maxi"),
                (UInt32, tile -> sum(tile; dims=2), "addi"),
                (UInt32, tile -> maximum(tile; dims=2), "maxi"),
            ]
                @test @filecheck begin
                    @check_label "entry"
                    code_tiled(Tuple{ct.TileArray{T,2,spec2d}}) do a
                        pid = ct.bid(1)
                        tile = ct.load(a, pid, (4, 16))
                        @check "reduce"
                        @check op_check
                        Base.donotdelete(op(tile))
                        return
                    end
                end
            end

            # Generic reduce with explicit function
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,2,spec2d}}) do a
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (4, 16))
                    @check "reduce"
                    @check "addf"
                    Base.donotdelete(reduce(+, tile; dims=2, init=0.0f0))
                    return
                end
            end

            # Reduce with custom combiner (lambda)
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,2,spec2d}}) do a
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (4, 16))
                    @check "reduce"
                    @check "addf"
                    Base.donotdelete(reduce((a, b) -> a + b, tile; dims=2, init=0.0f0))
                    return
                end
            end

            # any
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,2,spec2d}}) do a
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (4, 16))
                    mask = tile .> 0.0f0
                    @check "reduce"
                    @check "ori"
                    Base.donotdelete(any(mask; dims=2))
                    return
                end
            end

            # all
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,2,spec2d}}) do a
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (4, 16))
                    mask = tile .> 0.0f0
                    @check "reduce"
                    @check "andi"
                    Base.donotdelete(all(mask; dims=2))
                    return
                end
            end

            # count
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,2,spec2d}}) do a
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (4, 16))
                    @check "exti"
                    @check "reduce"
                    @check "addi"
                    Base.donotdelete(count(tile .> 0.0f0; dims=2))
                    return
                end
            end

            # argmax
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,2,spec2d}}) do a
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (4, 16))
                    @check "iota"
                    @check "reduce"
                    @check "cmpf"
                    @check "select"
                    Base.donotdelete(argmax(tile; dims=2))
                    return
                end
            end

            # argmin
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Int32,2,spec2d}}) do a
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (4, 16))
                    @check "iota"
                    @check "reduce"
                    @check "cmpi"
                    @check "select"
                    Base.donotdelete(argmin(tile; dims=2))
                    return
                end
            end
        end

        @testset "map" begin
            # map(abs, tile) should emit AbsFOp with correct shaped type
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,2,spec2d}}) do a
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (4, 16))
                    @check "absf"
                    Base.donotdelete(map(abs, tile))
                    return
                end
            end

            # map(x -> x * x, tile) should emit MulFOp
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,2,spec2d}}) do a
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (4, 16))
                    @check "mulf"
                    Base.donotdelete(map(x -> x * x, tile))
                    return
                end
            end

            # mapreduce(abs, +, tile) should emit AbsFOp then ReduceOp
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,2,spec2d}}) do a
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (4, 16))
                    @check "absf"
                    @check "reduce"
                    @check "addf"
                    Base.donotdelete(mapreduce(abs, +, tile; dims=2, init=0.0f0))
                    return
                end
            end

            # mapreduce(x -> x * x, +, tile) should emit MulFOp then ReduceOp
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,2,spec2d}}) do a
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (4, 16))
                    @check "mulf"
                    @check "reduce"
                    @check "addf"
                    Base.donotdelete(mapreduce(x -> x * x, +, tile; dims=2, init=0.0f0))
                    return
                end
            end
        end

        @testset "select" begin
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}}) do a, b, c
                    pid = ct.bid(1)
                    tile_a = ct.load(a, pid, (16,))
                    tile_b = ct.load(b, pid, (16,))
                    mask = tile_a .> tile_b
                    @check "select"
                    result = ct.where(mask, tile_a, tile_b)
                    ct.store(c, pid, result)
                    return
                end
            end
        end
    end

    #=========================================================================
     8.4 Conversions
    =========================================================================#
    @testset "Conversions" begin
        # TODO: bitcast - reinterpret bits as different type
        # TODO: exti - sign/zero extend integer
        # TODO: ftoi - float to integer
        # TODO: itof - integer to float
        # TODO: int_to_ptr - integer to pointer
        # TODO: ptr_to_int - pointer to integer
        # TODO: ptr_to_ptr - pointer cast
        # TODO: trunci - truncate integer

        @testset "ftof" begin
            # Float32 -> Float16
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float16,1,spec1d}}) do a, b
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (16,))
                    @check "ftof"
                    converted = convert(ct.Tile{Float16}, tile)
                    ct.store(b, pid, converted)
                    return
                end
            end

            # Float32 -> TFloat32
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}}) do a, b
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (16,))
                    @check "ftof"
                    converted = convert(ct.Tile{ct.TFloat32}, tile)
                    ct.store(b, pid, convert(ct.Tile{Float32}, converted))
                    return
                end
            end

            # Float32 -> BFloat16
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}}) do a, b
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (16,))
                    @check "ftof"
                    converted = convert(ct.Tile{ct.BFloat16}, tile)
                    ct.store(b, pid, convert(ct.Tile{Float32}, converted))
                    return
                end
            end

            # Broadcasting syntax: Float16.(tile)
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float16,1,spec1d}}) do a, b
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (16,))
                    @check "ftof"
                    ct.store(b, pid, Float16.(tile))
                    return
                end
            end

            # Broadcasting syntax: BFloat16.(tile)
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}}) do a, b
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (16,))
                    @check "ftof"
                    @check "ftof"
                    ct.store(b, pid, Float32.(ct.BFloat16.(tile)))
                    return
                end
            end

            # Broadcasting syntax: TFloat32.(tile)
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}}) do a, b
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (16,))
                    @check "ftof"
                    @check "ftof"
                    ct.store(b, pid, Float32.(ct.TFloat32.(tile)))
                    return
                end
            end
        end

        @testset "Type broadcasting" begin
            # convert.(Float16, tile) — Type arg via TypeRef
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float16,1,spec1d}}) do a, b
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (16,))
                    @check "ftof"
                    ct.store(b, pid, convert.(Float16, tile))
                    return
                end
            end

            # convert.(Float32, float16_tile) — upcast via Type arg
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float16,1,spec1d}, ct.TileArray{Float32,1,spec1d}}) do a, b
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (16,))
                    @check "ftof"
                    ct.store(b, pid, convert.(Float32, tile))
                    return
                end
            end

            # unsafe_trunc.(Int32, float32_tile) — ftoi via Type arg
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Int32,1,spec1d}}) do a, b
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (16,))
                    @check "ftoi"
                    ct.store(b, pid, unsafe_trunc.(Int32, tile))
                    return
                end
            end

        end
    end

    #=========================================================================
     8.5 Control Flow
    =========================================================================#
    @testset "Control Flow" begin
        @testset "@assert with message" begin
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}}) do a
                    bid = ct.bid(1)
                    @check "cmpi"
                    ct.@assert bid > Int32(0) "bid must be positive"
                    @check "assert"
                    @check "bid must be positive"
                    tile = ct.load(a, bid, (16,))
                    ct.store(a, bid, tile)
                    return
                end
            end
        end

        @testset "@assert without message" begin
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}}) do a
                    bid = ct.bid(1)
                    @check "cmpi"
                    ct.@assert bid > Int32(0)
                    @check "assert"
                    # Auto-generated message from the expression
                    @check "bid > Int32(0)"
                    tile = ct.load(a, bid, (16,))
                    ct.store(a, bid, tile)
                    return
                end
            end
        end

        @testset "if with empty branch" begin
            # Empty if branches must emit YieldOp to satisfy MLIR block terminator requirements
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Int,1,spec1d}, ct.TileArray{Int,1,spec1d}}) do counter, lock
                    result = ct.atomic_cas(lock, 1, 0, 1)
                    @check "if"
                    if result == 0
                        ct.atomic_add(counter, 1, 1)
                    else
                        # Empty else - must still emit YieldOp
                    end
                    return
                end
            end
        end

        @testset "for" begin
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}, Int32}) do a, b, n
                    pid = ct.bid(1)
                    acc = ct.full((16,), 0.0f0, Float32)
                    @check "for"
                    k = Int32(1)
                    while k <= n
                        tile = ct.load(a, (pid - Int32(1)) * n + k, (16,))
                        acc = acc + tile
                        k += Int32(1)
                    end
                    ct.store(b, pid, acc)
                    return
                end
            end
        end

        @testset "loop" begin
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Int,1,spec1d}, ct.TileArray{Float32,1,spec1d}}) do locks, data
                    bid = ct.bid(1)
                    @check "loop"
                    # Spinloop - unbounded iteration
                    while ct.atomic_cas(locks, bid, 0, 1;
                                       memory_order=ct.MemoryOrder.Acquire) == 1
                    end
                    tile = ct.load(data, bid, (16,))
                    ct.store(data, bid, tile)
                    ct.atomic_xchg(locks, bid, 0;
                                  memory_order=ct.MemoryOrder.Release)
                    return
                end
            end
        end

        @testset "return" begin
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}}) do a
                    @check "return"
                    return
                end
            end
        end
    end

    #=========================================================================
     8.6 Memory
    =========================================================================#
    @testset "Memory" begin
        # TODO: join_tokens - join multiple memory tokens
        # TODO: load_ptr_tko - direct pointer load (vs view-based)
        # TODO: store_ptr_tko - direct pointer store (vs view-based)

        @testset "TileArray load/store" begin
            # 1D
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}}) do a, b
                    pid = ct.bid(1)
                    @check "load_view_tko"
                    tile = ct.load(a, pid, (16,))
                    @check "store_view_tko"
                    ct.store(b, pid, tile)
                    return
                end
            end

            # 2D
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,2,spec2d}, ct.TileArray{Float32,2,spec2d}}) do a, b
                    bidx = ct.bid(1)
                    bidy = ct.bid(2)
                    @check "load_view_tko"
                    tile = ct.load(a, (bidx, bidy), (32, 32))
                    @check "store_view_tko"
                    ct.store(b, (bidx, bidy), tile)
                    return
                end
            end
        end

        @testset "make_token" begin
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}}) do a
                    @check "make_token"
                    return
                end
            end
        end
    end

    #=========================================================================
     8.7 Floating Point
    =========================================================================#
    @testset "Floating Point" begin
        @testset "unary float math" begin
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}}) do a
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (16,))
                    @check "absf"
                    Base.donotdelete(abs.(tile))
                    @check "ceil"
                    Base.donotdelete(ceil.(tile))
                    @check "cos"
                    Base.donotdelete(cos.(tile))
                    @check "cosh"
                    Base.donotdelete(cosh.(tile))
                    @check "floor"
                    Base.donotdelete(floor.(tile))
                    @check "sin"
                    Base.donotdelete(sin.(tile))
                    @check "sinh"
                    Base.donotdelete(sinh.(tile))
                    @check "tan"
                    Base.donotdelete(tan.(tile))
                    @check "tanh"
                    Base.donotdelete(tanh.(tile))
                    @check "sqrt"
                    Base.donotdelete(sqrt.(tile))
                    @check "rsqrt"
                    Base.donotdelete(ct.rsqrt.(tile))
                    @check "exp"
                    Base.donotdelete(exp.(tile))
                    @check "exp2"
                    Base.donotdelete(exp2.(tile))
                    @check "log"
                    Base.donotdelete(log.(tile))
                    @check "log2"
                    Base.donotdelete(log2.(tile))
                    return
                end
            end
        end

        @testset "fma" begin
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}}) do a, b, c, d
                    pid = ct.bid(1)
                    tile_a = ct.load(a, pid, (16,))
                    tile_b = ct.load(b, pid, (16,))
                    tile_c = ct.load(c, pid, (16,))
                    @check "fma"
                    result = fma.(tile_a, tile_b, tile_c)
                    ct.store(d, pid, result)
                    return
                end
            end
        end

        @testset "multi-arg map" begin
            # Binary map → addf
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}}) do a
                    pid = ct.bid(1)
                    tile_a = ct.load(a, pid, (16,))
                    tile_b = ct.load(a, pid, (16,))
                    @check "addf"
                    result = map(+, tile_a, tile_b)
                    ct.store(a, pid, result)
                    return
                end
            end

            # Broadcasting via .+ → broadcast + addf
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,2,spec2d}}) do a
                    pid = ct.bid(1)
                    row = ct.load(a, pid, (1, 16))
                    col = ct.load(a, pid, (32, 1))
                    @check "broadcast"
                    @check "addf"
                    result = row .+ col
                    Base.donotdelete(result)
                    return
                end
            end

            # Ternary map → fma
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}}) do a
                    pid = ct.bid(1)
                    ta = ct.load(a, pid, (16,))
                    tb = ct.load(a, pid, (16,))
                    tc = ct.load(a, pid, (16,))
                    @check "fma"
                    result = map(fma, ta, tb, tc)
                    ct.store(a, pid, result)
                    return
                end
            end

            # map(max, ...) → maxf
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}}) do a
                    pid = ct.bid(1)
                    ta = ct.load(a, pid, (16,))
                    tb = ct.load(a, pid, (16,))
                    @check "maxf"
                    result = map(max, ta, tb)
                    ct.store(a, pid, result)
                    return
                end
            end
        end

        @testset "nested broadcast" begin
            # a .+ b .* c → mulf then addf (no explicit broadcasted needed)
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}}) do a
                    pid = ct.bid(1)
                    ta = ct.load(a, pid, (16,))
                    tb = ct.load(a, pid, (16,))
                    tc = ct.load(a, pid, (16,))
                    @check "mulf"
                    @check "addf"
                    result = ta .+ tb .* tc
                    ct.store(a, pid, result)
                    return
                end
            end
        end

        @testset "remf" begin
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}}) do a, b, c
                    pid = ct.bid(1)
                    tile_a = ct.load(a, pid, (16,))
                    tile_b = ct.load(b, pid, (16,))
                    @check "remf"
                    result = rem.(tile_a, tile_b)
                    ct.store(c, pid, result)
                    return
                end
            end
        end

        @testset "binary float math" begin
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}}) do a, b
                    pid = ct.bid(1)
                    ta = ct.load(a, pid, (16,))
                    tb = ct.load(b, pid, (16,))
                    @check "addf"
                    Base.donotdelete(ta + tb)
                    @check "subf"
                    Base.donotdelete(ta - tb)
                    @check "mulf"
                    Base.donotdelete(ta .* tb)
                    @check "divf"
                    Base.donotdelete(ta ./ tb)
                    return
                end
            end
        end

        @testset "scalar broadcast" begin
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}}) do a
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (16,))
                    @check "broadcast"
                    @check "addf"
                    Base.donotdelete(tile .+ 1.0f0)
                    @check "broadcast"
                    @check "subf"
                    Base.donotdelete(1.0f0 .- tile)
                    @check "broadcast"
                    @check "mulf"
                    Base.donotdelete(tile .* 2.0f0)
                    @check "broadcast"
                    @check "divf"
                    Base.donotdelete(tile ./ 2.0f0)
                    return
                end
            end
        end

        @testset "power operations" begin
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}}) do a
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (16,))
                    @check "pow"
                    Base.donotdelete(tile .^ tile)
                    return
                end
            end

            # scalar exponent
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}}) do a
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (16,))
                    @check "broadcast"
                    @check "pow"
                    Base.donotdelete(tile .^ 2.0f0)
                    return
                end
            end
        end

        @testset "scalar math functions" begin
            # Test scalar math functions via overlays (sin, exp, sqrt, etc. on scalars)
            # Note: We pass scalar args to avoid constant folding at compile time
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}, Float32}) do a, b, x
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (16,))
                    @check "sin"
                    scale = sin(x)
                    @check "broadcast"
                    @check "mulf"
                    result = tile .* scale
                    ct.store(b, pid, result)
                    return
                end
            end

            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}, Float32}) do a, b, x
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (16,))
                    @check "exp"
                    scale = exp(x)
                    @check "broadcast"
                    result = tile .* scale
                    ct.store(b, pid, result)
                    return
                end
            end

            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}, Float32}) do a, b, x
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (16,))
                    @check "sqrt"
                    scale = sqrt(x)
                    @check "broadcast"
                    result = tile .* scale
                    ct.store(b, pid, result)
                    return
                end
            end

            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}, Float32}) do a, b, x
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (16,))
                    @check "rsqrt"
                    scale = ct.rsqrt(x)
                    @check "broadcast"
                    result = tile .* scale
                    ct.store(b, pid, result)
                    return
                end
            end
        end
    end

    #=========================================================================
     8.8 Integer
    =========================================================================#
    @testset "Integer" begin
        spec_i32 = ct.ArraySpec{1}(16, true)

        @testset "unary" begin
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Int32,1,spec_i32}}) do a
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (16,))
                    @check "absi"
                    Base.donotdelete(abs.(tile))
                    @check "addi"
                    Base.donotdelete(tile + tile)
                    return
                end
            end
        end

        @testset "binary" begin
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Int32,1,spec_i32}, ct.TileArray{Int32,1,spec_i32}}) do a, b
                    pid = ct.bid(1)
                    ta = ct.load(a, pid, (16,))
                    tb = ct.load(b, pid, (16,))
                    @check "mulhii"
                    Base.donotdelete(ct.mul_hi.(ta, tb))
                    @check "maxi"
                    Base.donotdelete(max.(ta, tb))
                    @check "mini"
                    Base.donotdelete(min.(ta, tb))
                    return
                end
            end
        end

        # TODO: divi, remi tests need tile-level operations to avoid DCE
        # The scalar intrinsics work but their results get eliminated if unused.
    end

    #=========================================================================
     8.9 Bitwise
    =========================================================================#
    # TODO: andi - bitwise AND

    #=========================================================================
     8.10 Atomics
    =========================================================================#
    @testset "Atomics" begin
        @testset "atomic_cas_tko" begin
            spec = ct.ArraySpec{1}(16, true)
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Int32,1,spec}}) do locks
                    bid = ct.bid(1)
                    @check "atomic_cas_tko"
                    old = ct.atomic_cas(locks, bid, Int32(0), Int32(1);
                                        memory_order=ct.MemoryOrder.Acquire)
                    return
                end
            end
        end

        @testset "atomic_rmw_tko" begin
            spec = ct.ArraySpec{1}(16, true)
            # xchg
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Int32,1,spec}}) do locks
                    bid = ct.bid(1)
                    @check "atomic_rmw_tko"
                    ct.atomic_xchg(locks, bid, Int32(0);
                                  memory_order=ct.MemoryOrder.Release)
                    return
                end
            end

            # add
            spec_f32 = ct.ArraySpec{1}(16, true)
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec_f32}}) do counters
                    bid = ct.bid(1)
                    @check "atomic_rmw_tko"
                    ct.atomic_add(counters, bid, 1.0f0)
                    return
                end
            end
        end
    end

    #=========================================================================
     8.11 Views
    =========================================================================#
    @testset "Views" begin
        # TODO: get_index_space_shape - get partition index space shape
        # TODO: get_tensor_shape - get tensor shape

        @testset "load_view_tko / store_view_tko" begin
            spec = ct.ArraySpec{1}(16, true)
            # 1D
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec}, ct.TileArray{Float32,1,spec}}) do a, b
                    pid = ct.bid(1)
                    @check "load_view_tko"
                    tile = ct.load(a, pid, (16,))
                    @check "store_view_tko"
                    ct.store(b, pid, tile)
                    return
                end
            end

            # 2D
            spec2d = ct.ArraySpec{2}(16, true)
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,2,spec2d}, ct.TileArray{Float32,2,spec2d}}) do a, b
                    bidx = ct.bid(1)
                    bidy = ct.bid(2)
                    @check "load_view_tko"
                    tile = ct.load(a, (bidx, bidy), (32, 32))
                    @check "store_view_tko"
                    ct.store(b, (bidx, bidy), tile)
                    return
                end
            end

            # 4D (requires TileArray with explicit sizes since grid only provides 3D)
            # Mixed Int32/Int64 indices are promoted to Int64 by _sub1
            spec4d = ct.ArraySpec{4}(16, true)
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,4,spec4d}, ct.TileArray{Float32,4,spec4d}}) do a, b
                    bidx = ct.bid(1)
                    bidy = ct.bid(2)
                    bidz = ct.bid(3)
                    @check "load_view_tko"
                    tile = ct.load(a, (bidx, bidy, bidz, 1), (2, 4, 4, 4))
                    @check "store_view_tko"
                    ct.store(b, (bidx, bidy, bidz, 1), tile)
                    return
                end
            end

            # with padding_mode
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec}, ct.TileArray{Float32,1,spec}}) do a, b
                    pid = ct.bid(1)
                    @check "load_view_tko"
                    tile = ct.load(a, pid, (16,); padding_mode=ct.PaddingMode.Zero)
                    ct.store(b, pid, tile)
                    return
                end
            end

            # with order (dim_map)
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,2,spec2d}, ct.TileArray{Float32,2,spec2d}}) do a, b
                    pid = ct.bid(1)
                    @check "dim_map=[1, 0]"
                    @check "load_view_tko"
                    tile = ct.load(a, (pid, 1), (4, 8); order=(2, 1))
                    @check "dim_map=[1, 0]"
                    @check "store_view_tko"
                    ct.store(b, (pid, 1), tile; order=(2, 1))
                    return
                end
            end

            # default order has no dim_map in output
            @test @filecheck begin
                @check_label "entry"
                @check_not "dim_map"
                code_tiled(Tuple{ct.TileArray{Float32,2,spec2d}, ct.TileArray{Float32,2,spec2d}}) do a, b
                    pid = ct.bid(1)
                    tile = ct.load(a, (pid, 1), (4, 8))
                    ct.store(b, (pid, 1), tile)
                    return
                end
            end
        end

        @testset "rank mismatch load/store" begin
            # 1D shape on 2D array: should pad shape to (16, 1) internally
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,2,spec2d}, ct.TileArray{Float32,2,spec2d}}) do a, b
                    pid = ct.bid(1)
                    @check "load_view_tko"
                    tile = ct.load(a, (pid, 1), (16,))
                    @check "store_view_tko"
                    ct.store(b, (pid, 1), tile)
                    return
                end
            end
        end

        @testset "TileArray scalar getindex" begin
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}}) do lengths, out
                    bid = ct.bid(1)
                    @check "make_partition_view"
                    @check "load_view_tko"
                    @check "reshape"
                    len = lengths[bid]
                    ct.store(out, bid, ct.broadcast_to(ct.Tile(len), (16,)))
                    return
                end
            end
        end

        @testset "num_tiles helper" begin
            spec = ct.ArraySpec{2}(16, true)
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,2,spec}, ct.TileArray{Float32,2,spec}}) do a, b
                    @check "make_tensor_view"
                    @check "make_partition_view"
                    @check "get_index_space_shape"
                    num = ct.num_tiles(a, 1, (32, 32))
                    # Use num as a tile index to prevent DCE
                    @check "load_view_tko"
                    tile = ct.load(a, (num, Int32(0)), (32, 32))
                    @check "store_view_tko"
                    ct.store(b, (Int32(0), Int32(0)), tile)
                    return
                end
            end
        end
    end

    #=========================================================================
     8.12 Miscellaneous
    =========================================================================#
    # TODO: assume - optimization hint
    # TODO: print_tko - debug printing
end
