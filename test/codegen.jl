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
        # TODO: scan - parallel scan/prefix sum
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
                    reshaped = ct.reshape(tile, (32,))
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
                    reshaped = ct.reshape(tile, (8, 8))
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
                    reshaped = ct.reshape(tile, (2, 32))
                    ct.store(b, pid, reshaped)
                    return
                end
            end

            # 1D -> 1D reshape (no permutes needed - optimization)
            @test @filecheck begin
                @check_label "entry"
                @check_not "permute"   # should NOT have permute for 1D->1D
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}}) do a
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (32,))
                    @check "reshape"
                    reshaped = ct.reshape(tile, (32,))
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
                    reshaped = ct.reshape(tile, (8, 4))
                    ct.store(a, pid, reshaped)
                    return
                end
            end
        end

        @testset "permute" begin
            # 2D permute (same as transpose)
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,2,spec2d}, ct.TileArray{Float32,2,spec2d}}) do a, b
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (4, 8))
                    @check "permute"
                    permuted = ct.permute(tile, (2, 1))
                    ct.store(b, pid, permuted)
                    return
                end
            end

            # 3D permute
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,3,spec3d}, ct.TileArray{Float32,3,spec3d}}) do a, b
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (2, 4, 8))
                    @check "permute"
                    # (2,4,8) with perm (3,1,2) -> (8,2,4)
                    permuted = ct.permute(tile, (3, 1, 2))
                    ct.store(b, pid, permuted)
                    return
                end
            end

            # 3D identity permute
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,3,spec3d}, ct.TileArray{Float32,3,spec3d}}) do a, b
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (2, 4, 8))
                    @check "permute"
                    # (2,4,8) with perm (1,2,3) -> (2,4,8) (identity)
                    permuted = ct.permute(tile, (1, 2, 3))
                    ct.store(b, pid, permuted)
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
                code_tiled(Tuple{}) do
                    a = ct.arange((16,), Int64)
                    b = ct.arange((16,), Int32)
                    # Should promote Int32 to Int64 and compare
                    @check "exti"
                    @check "cmpi"
                    result = a .< b
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
                    transposed = ct.transpose(tile)
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
                    sums = ct.reduce_sum(tile, 2)
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
                    maxes = ct.reduce_max(tile, 2)
                    ct.store(b, pid, maxes)
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
                    converted = ct.astype(tile, Float16)
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
                    ct.store(b, pid, ct.astype(converted, Float32))
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
                    ct.store(b, pid, ct.astype(converted, Float32))
                    return
                end
            end
        end
    end

    #=========================================================================
     8.5 Control Flow
    =========================================================================#
    @testset "Control Flow" begin
        # TODO: assert - runtime assertion

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
        @testset "absf" begin
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}}) do a, b
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (16,))
                    @check "absf"
                    result = abs.(tile)
                    ct.store(b, pid, result)
                    return
                end
            end
        end

        @testset "ceil" begin
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}}) do a, b
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (16,))
                    @check "ceil"
                    result = ceil.(tile)
                    ct.store(b, pid, result)
                    return
                end
            end
        end

        @testset "cos" begin
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}}) do a, b
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (16,))
                    @check "cos"
                    result = cos.(tile)
                    ct.store(b, pid, result)
                    return
                end
            end
        end

        @testset "cosh" begin
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}}) do a, b
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (16,))
                    @check "cosh"
                    result = cosh.(tile)
                    ct.store(b, pid, result)
                    return
                end
            end
        end

        @testset "floor" begin
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}}) do a, b
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (16,))
                    @check "floor"
                    result = floor.(tile)
                    ct.store(b, pid, result)
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

        @testset "sin" begin
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}}) do a, b
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (16,))
                    @check "sin"
                    result = sin.(tile)
                    ct.store(b, pid, result)
                    return
                end
            end
        end

        @testset "sinh" begin
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}}) do a, b
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (16,))
                    @check "sinh"
                    result = sinh.(tile)
                    ct.store(b, pid, result)
                    return
                end
            end
        end

        @testset "tan" begin
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}}) do a, b
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (16,))
                    @check "tan"
                    result = tan.(tile)
                    ct.store(b, pid, result)
                    return
                end
            end
        end

        @testset "tanh" begin
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}}) do a, b
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (16,))
                    @check "tanh"
                    result = tanh.(tile)
                    ct.store(b, pid, result)
                    return
                end
            end
        end

        @testset "addf" begin
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}}) do a, b, c
                    pid = ct.bid(1)
                    tile_a = ct.load(a, pid, (16,))
                    tile_b = ct.load(b, pid, (16,))
                    @check "addf"
                    result = tile_a + tile_b
                    ct.store(c, pid, result)
                    return
                end
            end
        end

        @testset "subf" begin
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}}) do a, b, c
                    pid = ct.bid(1)
                    tile_a = ct.load(a, pid, (16,))
                    tile_b = ct.load(b, pid, (16,))
                    @check "subf"
                    result = tile_a - tile_b
                    ct.store(c, pid, result)
                    return
                end
            end
        end

        @testset "mulf" begin
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}}) do a, b, c
                    pid = ct.bid(1)
                    tile_a = ct.load(a, pid, (16,))
                    tile_b = ct.load(b, pid, (16,))
                    @check "mulf"
                    result = tile_a .* tile_b
                    ct.store(c, pid, result)
                    return
                end
            end
        end

        @testset "divf" begin
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}}) do a, b, c
                    pid = ct.bid(1)
                    tile_a = ct.load(a, pid, (16,))
                    tile_b = ct.load(b, pid, (16,))
                    @check "divf"
                    result = tile_a ./ tile_b
                    ct.store(c, pid, result)
                    return
                end
            end
        end

        @testset "sqrt" begin
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}}) do a, b
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (16,))
                    @check "sqrt"
                    result = sqrt.(tile)
                    ct.store(b, pid, result)
                    return
                end
            end
        end

        @testset "rsqrt" begin
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}}) do a, b
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (16,))
                    @check "rsqrt"
                    result = ct.rsqrt.(tile)
                    ct.store(b, pid, result)
                    return
                end
            end
        end

        @testset "exp" begin
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}}) do a, b
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (16,))
                    @check "exp"
                    result = exp.(tile)
                    ct.store(b, pid, result)
                    return
                end
            end
        end

        @testset "exp2" begin
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}}) do a, b
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (16,))
                    @check "exp2"
                    result = exp2.(tile)
                    ct.store(b, pid, result)
                    return
                end
            end
        end

        @testset "log" begin
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}}) do a, b
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (16,))
                    @check "log"
                    result = log.(tile)
                    ct.store(b, pid, result)
                    return
                end
            end
        end

        @testset "log2" begin
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}}) do a, b
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (16,))
                    @check "log2"
                    result = log2.(tile)
                    ct.store(b, pid, result)
                    return
                end
            end
        end

        @testset "scalar broadcast" begin
            # tile + scalar
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}}) do a, b
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (16,))
                    @check "broadcast"
                    @check "addf"
                    result = tile .+ 1.0f0
                    ct.store(b, pid, result)
                    return
                end
            end

            # scalar - tile
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}}) do a, b
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (16,))
                    @check "broadcast"
                    @check "subf"
                    result = 1.0f0 .- tile
                    ct.store(b, pid, result)
                    return
                end
            end

            # tile * scalar
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}}) do a, b
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (16,))
                    @check "broadcast"
                    @check "mulf"
                    result = tile .* 2.0f0
                    ct.store(b, pid, result)
                    return
                end
            end

            # tile / scalar
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}}) do a, b
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (16,))
                    @check "broadcast"
                    @check "divf"
                    result = tile ./ 2.0f0
                    ct.store(b, pid, result)
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
        @testset "absi" begin
            spec_i32 = ct.ArraySpec{1}(16, true)
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Int32,1,spec_i32}, ct.TileArray{Int32,1,spec_i32}}) do a, b
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (16,))
                    @check "absi"
                    result = abs.(tile)
                    ct.store(b, pid, result)
                    return
                end
            end
        end

        @testset "mulhii" begin
            spec_i32 = ct.ArraySpec{1}(16, true)
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Int32,1,spec_i32}, ct.TileArray{Int32,1,spec_i32}, ct.TileArray{Int32,1,spec_i32}}) do a, b, c
                    pid = ct.bid(1)
                    tile_a = ct.load(a, pid, (16,))
                    tile_b = ct.load(b, pid, (16,))
                    @check "mulhii"
                    result = ct.mul_hi.(tile_a, tile_b)
                    ct.store(c, pid, result)
                    return
                end
            end
        end

        @testset "addi" begin
            spec_i32 = ct.ArraySpec{1}(16, true)
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Int32,1,spec_i32}, ct.TileArray{Int32,1,spec_i32}}) do a, b
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (16,))
                    @check "addi"
                    result = tile + tile
                    ct.store(b, pid, result)
                    return
                end
            end
        end

        # TODO: divi, mini, remi tests need tile-level operations to avoid DCE
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
        end

        @testset "num_tiles helper" begin
            spec = ct.ArraySpec{2}(16, true)
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,2,spec}}) do a
                    @check "make_tensor_view"
                    @check "make_partition_view"
                    @check "get_index_space_shape"
                    num = ct.num_tiles(a, 1, (32, 32))
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

#=============================================================================
 Type Support
=============================================================================#

@testset "Type Support" begin
    spec = ct.ArraySpec{1}(16, true)

    @testset "Float32" begin
        @test @filecheck begin
            @check_label "entry"
            code_tiled(Tuple{ct.TileArray{Float32,1,spec}, ct.TileArray{Float32,1,spec}}) do a, b
                pid = ct.bid(1)
                tile = ct.load(a, pid, (16,))
                ct.store(b, pid, tile)
                return
            end
        end
    end

    @testset "Float64" begin
        @test @filecheck begin
            @check_label "entry"
            code_tiled(Tuple{ct.TileArray{Float64,1,spec}, ct.TileArray{Float64,1,spec}}) do a, b
                pid = ct.bid(1)
                tile = ct.load(a, pid, (16,))
                @check "addf"
                result = tile + tile
                ct.store(b, pid, result)
                return
            end
        end
    end

    @testset "Float16" begin
        @test @filecheck begin
            @check_label "entry"
            code_tiled(Tuple{ct.TileArray{Float16,1,spec}, ct.TileArray{Float16,1,spec}}) do a, b
                pid = ct.bid(1)
                tile = ct.load(a, pid, (16,))
                @check "addf"
                result = tile + tile
                ct.store(b, pid, result)
                return
            end
        end
    end

    @testset "Int32" begin
        @test @filecheck begin
            @check_label "entry"
            code_tiled(Tuple{ct.TileArray{Int32,1,spec}, ct.TileArray{Int32,1,spec}}) do a, b
                pid = ct.bid(1)
                tile = ct.load(a, pid, (16,))
                @check "addi"
                result = tile + tile
                ct.store(b, pid, result)
                return
            end
        end
    end
end

#=============================================================================
 Integration Tests
=============================================================================#

@testset "Integration" begin
    @testset "vector add kernel" begin
        spec = ct.ArraySpec{1}(16, true)
        @test @filecheck begin
            @check_label "entry"
            code_tiled(Tuple{ct.TileArray{Float32,1,spec}, ct.TileArray{Float32,1,spec}, ct.TileArray{Float32,1,spec}, ct.Constant{Int,16}}) do a, b, c, tile
                @check "get_tile_block_id"
                bid = ct.bid(1)
                @check "load_view_tko"
                a_tile = ct.load(a, bid, (tile[],))
                @check "load_view_tko"
                b_tile = ct.load(b, bid, (tile[],))
                @check "addf"
                result = a_tile + b_tile
                @check "store_view_tko"
                ct.store(c, bid, result)
                @check "return"
                return
            end
        end
    end

    @testset "transpose kernel" begin
        spec = ct.ArraySpec{2}(16, true)
        @test @filecheck begin
            @check_label "entry"
            code_tiled(Tuple{ct.TileArray{Float32,2,spec}, ct.TileArray{Float32,2,spec}, ct.Constant{Int,32}, ct.Constant{Int,32}}) do x, y, tm, tn
                @check "get_tile_block_id"
                bidx = ct.bid(1)
                bidy = ct.bid(2)
                @check "load_view_tko"
                input_tile = ct.load(x, (bidx, bidy), (tm[], tn[]))
                @check "permute"
                transposed_tile = ct.transpose(input_tile)
                @check "store_view_tko"
                ct.store(y, (bidy, bidx), transposed_tile)
                @check "return"
                return
            end
        end
    end

    @testset "matmul reduction loop" begin
        spec = ct.ArraySpec{2}(16, true)
        @test @filecheck begin
            @check_label "entry"
            code_tiled(Tuple{ct.TileArray{Float32,2,spec}, ct.TileArray{Float32,2,spec}, ct.TileArray{Float32,2,spec}, ct.Constant{Int,32}, ct.Constant{Int,32}, ct.Constant{Int,16}}) do A, B, C, tm, tn, tk
                bid = ct.bid(1)
                num_k = ct.num_tiles(A, 2, (tm[], tk[]))
                acc = ct.full((tm[], tn[]), zero(Float32), Float32)
                # NOTE: Uses while-loop pattern because Julia's for-loop generates
                # complex iterator IR with PhiNodes that isn't fully supported.
                # The structurizer upgrades this counting while-loop to a ForOp.
                @check "for"
                k = Int32(1)
                while k <= num_k
                    @check "load_view_tko"
                    a = ct.load(A, (bid, k), (tm[], tk[]); padding_mode=ct.PaddingMode.Zero)
                    @check "load_view_tko"
                    b = ct.load(B, (k, bid), (tk[], tn[]); padding_mode=ct.PaddingMode.Zero)
                    @check "mma"
                    acc = muladd(a, b, acc)
                    k += Int32(1)
                end
                @check "store_view_tko"
                ct.store(C, (bid, bid), acc)
                return
            end
        end
    end

    @testset "layernorm forward pattern (multiple sequential for loops)" begin
        # This test captures the pattern from layer_norm_fwd:
        # Multiple sequential for loops (mean accumulation, then output pass)
        spec = ct.ArraySpec{2}(16, true)
        spec1d = ct.ArraySpec{1}(16, true)
        @test @filecheck begin
            @check_label "entry"
            code_tiled(Tuple{ct.TileArray{Float32,2,spec}, ct.TileArray{Float32,2,spec},
                           ct.TileArray{Float32,1,spec1d}, ct.Constant{Int,16}}) do X, Y, Sum, TILE_N
                bid_m = ct.bid(1)
                num_tiles = ct.num_tiles(X, 2, (1, TILE_N[]))

                # First for loop: compute sum
                @check "for"
                acc = ct.full((1, TILE_N[]), 0.0f0, Float32)
                j = Int32(1)
                while j <= num_tiles
                    tx = ct.load(X, (bid_m, j), (1, TILE_N[]); padding_mode=ct.PaddingMode.Zero)
                    acc = acc .+ tx
                    j += Int32(1)
                end
                @check "reduce"
                sum_val = ct.reduce_sum(acc, 2)
                ct.store(Sum, bid_m, sum_val)

                # Second for loop: scale output by sum
                @check "for"
                j = Int32(1)
                while j <= num_tiles
                    tx = ct.load(X, (bid_m, j), (1, TILE_N[]); padding_mode=ct.PaddingMode.Zero)
                    ty = tx .* sum_val
                    ct.store(Y, (bid_m, j), ty)
                    j += Int32(1)
                end
                return
            end
        end
    end

    @testset "layernorm backward pattern (atomic spinloop)" begin
        # This test captures the actual pattern from layer_norm_bwd_dx_partial_dwdb:
        # A for loop iterating over tiles, with a spinloop inside for atomic accumulation
        spec = ct.ArraySpec{1}(16, true)
        spec2d = ct.ArraySpec{2}(16, true)
        @test @filecheck begin
            @check_label "entry"
            code_tiled(Tuple{ct.TileArray{Float32,2,spec2d}, ct.TileArray{Float32,2,spec2d},
                           ct.TileArray{Int32,1,spec}, Int32, ct.Constant{Int,16}}) do DW, Partial, Locks, group_bid, TILE_N
                bid = ct.bid(1)
                num_tiles = ct.num_tiles(DW, 2, (1, TILE_N[]))

                @check "for"
                j = Int32(1)
                while j <= num_tiles
                    # Load and compute partial result
                    partial = ct.load(Partial, (bid, j), (1, TILE_N[]); padding_mode=ct.PaddingMode.Zero)

                    @check "loop"
                    # Acquire spinlock (nested inside for loop)
                    while ct.atomic_cas(Locks, group_bid, Int32(0), Int32(1);
                                       memory_order=ct.MemoryOrder.Acquire) == Int32(1)
                        # spin
                    end

                    # Critical section: accumulate
                    @check "load_view_tko"
                    acc = ct.load(DW, (group_bid, j), (1, TILE_N[]); padding_mode=ct.PaddingMode.Zero)
                    @check "addf"
                    acc = acc .+ partial
                    @check "store_view_tko"
                    ct.store(DW, (group_bid, j), acc)

                    # Release spinlock
                    @check "atomic_rmw_tko"
                    ct.atomic_xchg(Locks, group_bid, Int32(0);
                                  memory_order=ct.MemoryOrder.Release)

                    j += Int32(1)
                end
                return
            end
        end
    end

    @testset "nested spinloop uses correct loop index (regression test)" begin
        # This test catches a bug where nested while loops inside for loops
        # shadow the for loop's induction variable, causing incorrect indexing.
        # The bug: store uses (group_bid, group_bid) instead of (group_bid, loopIdx)
        spec = ct.ArraySpec{2}(16, true)
        spec1d = ct.ArraySpec{1}(16, true)
        @test @filecheck begin
            @check_label "entry"
            @check "for %loopIdx in"
            @check "loop iter_values"
            # The store MUST use a column index derived from loopIdx, not the spinloop result
            # After 1→0 index conversion, the store uses (loopIdx - 1), captured as [[IDX]]
            @check "[[IDX:%.+]] = subi %loopIdx"
            @check "store_view_tko{{.*}}[%{{[^,]+}}, [[IDX]]]"
            code_tiled(Tuple{ct.TileArray{Float32,2,spec}, ct.TileArray{Int32,1,spec1d},
                           Int32, ct.Constant{Int,4}, ct.Constant{Int,4}}) do DB, Locks, num_iters, GROUP_SIZE_M, TILE_N
                bid_m = ct.bid(1)
                # group_bid_m: 1-indexed group ID
                group_bid_m = ((bid_m - Int32(1)) % Int32(GROUP_SIZE_M[])) + Int32(1)

                j = Int32(1)
                while j <= num_iters
                    # Nested spinloop - this must not shadow loopIdx
                    while ct.atomic_cas(Locks, group_bid_m, Int32(0), Int32(1);
                                       memory_order=ct.MemoryOrder.Acquire) == Int32(1)
                    end

                    val = ct.full((1, TILE_N[]), 1.0f0, Float32)
                    ct.store(DB, (group_bid_m, j), val)

                    ct.atomic_xchg(Locks, group_bid_m, Int32(0);
                                  memory_order=ct.MemoryOrder.Release)

                    j += Int32(1)
                end
                return
            end
        end
    end

    @testset "nested spinloop captures correct outer variable (regression test)" begin
        # This test catches a bug where nested while loops inside for loops
        # capture the for loop's induction variable instead of the correct outer variable.
        # The bug: spinloop uses loopIdx for atomic_cas instead of group_bid_m, causing hangs.
        #
        # The inner loop should capture %iterArg0 (group_bid_m), NOT %loopIdx.
        # Bug produces: loop iter_values(%arg9 = %loopIdx, ...)
        # Correct:      loop iter_values(%arg9 = %iterArg0, ...)
        spec = ct.ArraySpec{2}(16, true)
        spec1d = ct.ArraySpec{1}(16, true)
        @test begin
            tir = code_tiled(Tuple{ct.TileArray{Float32,2,spec}, ct.TileArray{Int32,1,spec1d},
                               Int32, ct.Constant{Int,4}, ct.Constant{Int,4}}) do DB, Locks, num_iters, GROUP_SIZE_M, TILE_N
                bid_m = ct.bid(1)
                # group_bid_m: 1-indexed group ID
                group_bid_m = ((bid_m - Int32(1)) % Int32(GROUP_SIZE_M[])) + Int32(1)

                j = Int32(1)
                while j <= num_iters
                    # Spinloop should use group_bid_m for the lock, not j
                    while ct.atomic_cas(Locks, group_bid_m, Int32(0), Int32(1);
                                       memory_order=ct.MemoryOrder.Acquire) == Int32(1)
                    end

                    val = ct.full((1, TILE_N[]), 1.0f0, Float32)
                    ct.store(DB, (group_bid_m, j), val)

                    ct.atomic_xchg(Locks, group_bid_m, Int32(0);
                                  memory_order=ct.MemoryOrder.Release)

                    j += Int32(1)
                end
                return
            end
            tir_str = string(tir)
            # The inner loop must NOT capture %loopIdx - it should capture %iterArg0
            # Bug: "loop iter_values(%arg9 = %loopIdx"
            !occursin(r"loop iter_values\([^)]*= %loopIdx", tir_str)
        end
    end

    @testset "nested while loops compile correctly (regression test)" begin
        # Regression test: Nested while loops must compile without errors.
        # Previously, nested WhileOp caused "operand index out of bounds" errors
        # during bytecode parsing due to value ID conflicts in nested regions.
        spec1d = ct.ArraySpec{1}(16, true)
        @test @filecheck begin
            @check_label "entry"
            @check "loop iter_values"
            @check "loop iter_values"
            code_tiled(Tuple{ct.TileArray{Int32,1,spec1d}, ct.TileArray{Int32,1,spec1d}}) do Locks1, Locks2
                idx = ct.bid(1)

                # Outer spinloop
                while ct.atomic_cas(Locks1, idx, Int32(0), Int32(1);
                                   memory_order=ct.MemoryOrder.Acquire) == Int32(1)
                    # Inner spinloop
                    while ct.atomic_cas(Locks2, idx, Int32(0), Int32(1);
                                       memory_order=ct.MemoryOrder.Acquire) == Int32(1)
                    end
                end

                return
            end
        end
    end

    @testset "counting while loop with nested control flow upgrades to for (regression test)" begin
        # Regression test: A counting while loop (j = 0; while j < n; j += 1) should be
        # upgraded to ForOp even when it contains nested control flow (like inner loops).
        # Previously, the for-loop pattern detection only searched for the step expression
        # in the immediate body block, missing it when nested control flow was present.
        spec1d = ct.ArraySpec{1}(16, true)
        @test @filecheck begin
            @check_label "entry"
            @check "for %loopIdx in"
            @check "loop iter_values"
            code_tiled(Tuple{ct.TileArray{Int32,1,spec1d}, Int32}) do Locks, num_iters
                idx = ct.bid(1)

                j = Int32(1)
                while j <= num_iters
                    # Inner spinloop - the presence of this nested loop shouldn't prevent
                    # the outer loop from being detected as a for-loop pattern
                    while ct.atomic_cas(Locks, idx, Int32(0), Int32(1);
                                       memory_order=ct.MemoryOrder.Acquire) == Int32(1)
                    end
                    j += Int32(1)
                end

                return
            end
        end
    end

    @testset "Multiple loop results (regression test)" begin
        # Regression test: A while loop with multiple iter_args must generate
        # different result indices (%for#0, %for#1, etc.) for each result.
        # Previously, all loop results resolved to %for#0, causing incorrect code.
        TILE_M = 32
        TILE_N = 1024

        # Use ArraySpec with shape_div_by to match real CuArray behavior
        spec2d = ct.ArraySpec{2}(128, true, (0, 4), (32, 32))
        spec1d = ct.ArraySpec{1}(128, true, (0,), (32,))

        @test @filecheck begin
            @check_label "entry"
            # The for loop should have multiple results
            @check "for %loopIdx in"
            # We should see both %for#0 and %for#1 used (not the same one twice)
            @check "reduce %for#1"
            @check "reduce %for#0"
            code_tiled(Tuple{ct.TileArray{Float32, 2, spec2d}, ct.TileArray{Float32, 2, spec2d},
                           ct.TileArray{Float32, 1, spec1d}, ct.TileArray{Float32, 1, spec1d},
                           ct.Constant{Int, TILE_M}, ct.Constant{Int, TILE_N}}) do DW, DB, FINAL_DW, FINAL_DB, _TILE_M, _TILE_N
                bid_n = ct.bid(1)
                num_tiles = ct.num_tiles(DW, 1, (_TILE_M[], _TILE_N[]))

                dw = ct.zeros((_TILE_M[], _TILE_N[]), Float32)
                db = ct.zeros((_TILE_M[], _TILE_N[]), Float32)
                i = Int32(1)
                while i <= num_tiles
                    dw = dw .+ ct.load(DW, (i, bid_n), (_TILE_M[], _TILE_N[]); padding_mode=ct.PaddingMode.Zero)
                    db = db .+ ct.load(DB, (i, bid_n), (_TILE_M[], _TILE_N[]); padding_mode=ct.PaddingMode.Zero)
                    i += Int32(1)
                end

                sum_dw = ct.reduce_sum(dw, 1)
                sum_db = ct.reduce_sum(db, 1)

                ct.store(FINAL_DW, bid_n, sum_dw)
                ct.store(FINAL_DB, bid_n, sum_db)
                return
            end
        end
    end

    @testset "sequential for loops with shared accumulator value" begin
        # Regression test: Two sequential for loops where the second loop both:
        # 1. Uses a value computed from the first loop's reduction
        # 2. Has its own accumulator (loop-carried value)
        #
        # This pattern appears in LayerNorm forward pass where:
        # - First loop computes mean/variance
        # - Second loop normalizes using those computed values while accumulating
        #
        # Test: Sequential for loops where the second loop uses a value computed from
        # the first loop's result AND has its own loop-carried accumulator.
        # This exercises correct SSA index storage across multiple ForOps.
        spec = ct.ArraySpec{1}(16, true)
        @test @filecheck begin
            @check_label "entry"
            code_tiled(Tuple{ct.TileArray{Float32,1,spec}, ct.TileArray{Float32,1,spec},
                           ct.Constant{Int,16}}) do out, inp, TILE_N
                bid = ct.bid(1)
                num_tiles = ct.num_tiles(inp, 1, (TILE_N[],))

                # First loop: accumulate and reduce
                @check "for"
                acc = ct.zeros((TILE_N[],), Float32)
                i = Int32(1)
                while i <= num_tiles
                    tile = ct.load(inp, i, (TILE_N[],); padding_mode=ct.PaddingMode.Zero)
                    acc = acc .+ tile
                    i += Int32(1)
                end
                @check "reduce"
                sum_val = ct.reduce_sum(acc, 1)

                # Second loop: use sum_val AND accumulate
                @check "for"
                acc2 = ct.zeros((TILE_N[],), Float32)
                i = Int32(1)
                while i <= num_tiles
                    tile = ct.load(inp, i, (TILE_N[],); padding_mode=ct.PaddingMode.Zero)
                    @check "subf"
                    acc2 = acc2 .+ (tile .- sum_val)  # Uses sum_val from first loop
                    i += Int32(1)
                end
                @check "reduce"
                @check "store_view_tko"
                ct.store(out, bid, ct.reduce_sum(acc2, 1))

                return
            end
        end
    end

    #=========================================================================
     Gather/Scatter Operations
    =========================================================================#
    @testset "gather/scatter" begin
        spec = ct.ArraySpec{1}(16, true)

        @testset "1D gather" begin
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec}, ct.TileArray{Float32,1,spec}}) do a, b
                    pid = ct.bid(1)
                    # Create index tile (simple: just use arange)
                    @check "iota"
                    indices = ct.arange((16,), Int32)
                    # Gather from array
                    @check "offset"
                    @check "load_ptr_tko"
                    tile = ct.gather(a, indices)
                    ct.store(b, pid, tile)
                    return
                end
            end
        end

        @testset "1D scatter" begin
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec}, ct.TileArray{Float32,1,spec}}) do a, b
                    pid = ct.bid(1)
                    # Load tile
                    tile = ct.load(a, pid, (16,))
                    # Create index tile (simple: just use arange)
                    @check "iota"
                    indices = ct.arange((16,), Int32)
                    # Scatter to array
                    @check "offset"
                    @check "store_ptr_tko"
                    ct.scatter(b, indices, tile)
                    return
                end
            end
        end

        @testset "1D gather with Int indices" begin
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec}, ct.TileArray{Float32,1,spec}}) do a, b
                    pid = ct.bid(1)
                    # Use Int (Int64) to test type conversion
                    @check "iota"
                    indices = ct.arange((16,), Int)
                    # Should convert to Int32 internally
                    @check "trunci"
                    @check "offset"
                    @check "load_ptr_tko"
                    tile = ct.gather(a, indices)
                    ct.store(b, pid, tile)
                    return
                end
            end
        end

        @testset "1D scatter with Int indices" begin
            @test @filecheck begin
                @check_label "entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec}, ct.TileArray{Float32,1,spec}}) do a, b
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (16,))
                    # Use Int (Int64) to test type conversion
                    @check "iota"
                    indices = ct.arange((16,), Int)
                    # Should convert to Int32 internally
                    @check "trunci"
                    @check "offset"
                    @check "store_ptr_tko"
                    ct.scatter(b, indices, tile)
                    return
                end
            end
        end
    end

    #=========================================================================
     Type Validation
    =========================================================================#
    @testset "type validation" begin
        spec = ct.ArraySpec{1}(16, true)

        @testset "binary op type mismatch errors in Julia" begin
            # This should fail with a Julia error (not tileiras), since the intrinsic
            # is invoked with mismatched types (Int32 + Int64)
            @test_throws ErrorException code_tiled(Tuple{ct.TileArray{Float32,1,spec}}) do a
                pid = ct.bid(1)  # Int32
                # Force type mismatch by calling addi with different types
                result = ct.Intrinsics.addi(pid, Int64(1))
                return
            end
        end
    end

    #=========================================================================
     Tile Shape Validation
    =========================================================================#
    @testset "tile shape validation" begin
        spec = ct.ArraySpec{1}(16, true)
        spec2d = ct.ArraySpec{2}(16, true)

        @testset "non-power-of-2 load shape rejected" begin
            @test_throws "load: tile dimension 1 must be a power of 2, got 3" begin
                code_tiled(Tuple{ct.TileArray{Float32,1,spec}}) do a
                    ct.load(a, ct.bid(1), (3,))
                end
            end
        end

        @testset "non-power-of-2 full shape rejected" begin
            @test_throws "full: tile dimension 1 must be a power of 2, got 5" begin
                code_tiled(Tuple{}) do
                    ct.full((5,), 0.0f0, Float32)
                end
            end
        end

        @testset "non-power-of-2 arange shape rejected" begin
            @test_throws "arange: tile dimension 1 must be a power of 2, got 7" begin
                code_tiled(Tuple{}) do
                    ct.arange((7,), Int32)
                end
            end
        end

        @testset "non-power-of-2 reshape target rejected" begin
            @test_throws "reshape: tile dimension 1 must be a power of 2, got 3" begin
                code_tiled(Tuple{ct.TileArray{Float32,1,spec}}) do a
                    tile = ct.load(a, ct.bid(1), (16,))
                    ct.reshape(tile, (3,))
                end
            end
        end

        @testset "zero dimension rejected" begin
            @test_throws "load: tile dimension 1 must be positive, got 0" begin
                code_tiled(Tuple{ct.TileArray{Float32,1,spec}}) do a
                    ct.load(a, ct.bid(1), (0,))
                end
            end
        end

        @testset "negative dimension rejected" begin
            @test_throws "full: tile dimension 1 must be positive, got -4" begin
                code_tiled(Tuple{}) do
                    ct.full((-4,), 0.0f0, Float32)
                end
            end
        end

        @testset "valid power-of-2 shapes accepted" begin
            # These should not throw - test a few key sizes
            code_tiled(Tuple{ct.TileArray{Float32,1,spec}}) do a
                tile = ct.load(a, ct.bid(1), (16,))
                ct.store(a, ct.bid(1), tile)
                return
            end
            code_tiled(Tuple{ct.TileArray{Float32,1,spec}}) do a
                tile = ct.load(a, ct.bid(1), (32,))
                ct.store(a, ct.bid(1), tile)
                return
            end
            code_tiled(Tuple{ct.TileArray{Float32,1,spec}}) do a
                tile = ct.load(a, ct.bid(1), (128,))
                ct.store(a, ct.bid(1), tile)
                return
            end
        end

        @testset "multi-dim: all dimensions must be pow2" begin
            @test_throws "load: tile dimension 2 must be a power of 2, got 3" begin
                code_tiled(Tuple{ct.TileArray{Float32,2,spec2d}}) do a
                    ct.load(a, (ct.bid(1), 1), (4, 3))
                end
            end
        end
    end

    #=========================================================================
     Constant Folding
    =========================================================================#
    @testset "constant folding" begin
        spec = ct.ArraySpec{1}(16, true)

        # XXX: This test verifies that store() returns the tile to enable constant
        # folding. If this test fails after removing `return tile` from store(),
        # Julia's optimizer will emit subi operations for constant index math.
        # See operations.jl store() for the workaround.
        @testset "store with constant index folds subtraction" begin
            @test @filecheck begin
                @check_label "entry"
                @check "load_view_tko"
                # Verify no subi appears between load and store - constant 1-1 should fold to 0
                @check_not "subi"
                @check "store_view_tko"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec}}) do a
                    idx = Int32(1)
                    tile = ct.load(a, idx, (16,))
                    ct.store(a, idx, tile)
                    return
                end
            end
        end
    end
end

#=============================================================================
 Entry Hints (kernel-level optimization hints)
=============================================================================#

@testset "Entry Hints" begin
    # Common ArraySpecs for tests
    spec1d = ct.ArraySpec{1}(16, true)

    @testset "num_ctas only" begin
        @test @filecheck begin
            @check "optimization_hints=<sm_100 = {num_cta_in_cga = 4}>"
            ct.code_tiled(Tuple{ct.TileArray{Float32, 1, spec1d}}; sm_arch="sm_100", num_ctas=4) do a
                pid = ct.bid(1)
                t = ct.load(a, pid, (16,))
                ct.store(a, pid, t)
                return nothing
            end
        end
    end

    @testset "occupancy only" begin
        @test @filecheck begin
            @check "optimization_hints=<sm_100 = {occupancy = 8}>"
            ct.code_tiled(Tuple{ct.TileArray{Float32, 1, spec1d}}; sm_arch="sm_100", occupancy=8) do a
                pid = ct.bid(1)
                t = ct.load(a, pid, (16,))
                ct.store(a, pid, t)
                return nothing
            end
        end
    end

    @testset "both hints" begin
        @test @filecheck begin
            @check "optimization_hints=<sm_120 = {num_cta_in_cga = 2, occupancy = 4}"
            ct.code_tiled(Tuple{ct.TileArray{Float32, 1, spec1d}}; sm_arch="sm_120", num_ctas=2, occupancy=4) do a
                pid = ct.bid(1)
                t = ct.load(a, pid, (16,))
                ct.store(a, pid, t)
                return nothing
            end
        end
    end

    @testset "no hints" begin
        @test @filecheck begin
            @check_not "optimization_hints"
            ct.code_tiled(Tuple{ct.TileArray{Float32, 1, spec1d}}; sm_arch="sm_100") do a
                pid = ct.bid(1)
                t = ct.load(a, pid, (16,))
                ct.store(a, pid, t)
                return nothing
            end
        end
    end

    @testset "architecture parameter" begin
        @test @filecheck begin
            @check "optimization_hints=<sm_120 = {num_cta_in_cga = 4}>"
            ct.code_tiled(Tuple{ct.TileArray{Float32, 1, spec1d}}; sm_arch="sm_120", num_ctas=4) do a
                pid = ct.bid(1)
                t = ct.load(a, pid, (16,))
                ct.store(a, pid, t)
                return nothing
            end
        end
    end

    @testset "num_ctas validation" begin
        # Too small
        @test_throws "num_ctas must be between 1 and 16" begin
            code_tiled((a) -> nothing, Tuple{ct.TileArray{Float32, 1, spec1d}}; sm_arch="sm_100", num_ctas=0)
        end

        # Too large
        @test_throws "num_ctas must be between 1 and 16" begin
            code_tiled((a) -> nothing, Tuple{ct.TileArray{Float32, 1, spec1d}}; sm_arch="sm_100", num_ctas=17)
        end

        # Not power of 2
        @test_throws "num_ctas must be a power of 2" begin
            code_tiled((a) -> nothing, Tuple{ct.TileArray{Float32, 1, spec1d}}; sm_arch="sm_100", num_ctas=3)
        end

        @test_throws "num_ctas must be a power of 2" begin
            code_tiled((a) -> nothing, Tuple{ct.TileArray{Float32, 1, spec1d}}; sm_arch="sm_100", num_ctas=5)
        end

        # Valid values should succeed
        for num_ctas in [1, 2, 4, 8, 16]
            bytecode = code_tiled((a) -> nothing, Tuple{ct.TileArray{Float32, 1, spec1d}}; sm_arch="sm_100", num_ctas)
            @test !isempty(bytecode)
        end
    end

    @testset "occupancy validation" begin
        # Too small
        @test_throws "occupancy must be between 1 and 32" begin
            code_tiled((a) -> nothing, Tuple{ct.TileArray{Float32, 1, spec1d}}; sm_arch="sm_100", occupancy=0)
        end

        # Too large
        @test_throws "occupancy must be between 1 and 32" begin
            code_tiled((a) -> nothing, Tuple{ct.TileArray{Float32, 1, spec1d}}; sm_arch="sm_100", occupancy=33)
        end

        # Valid boundaries
        bytecode1 = code_tiled((a) -> nothing, Tuple{ct.TileArray{Float32, 1, spec1d}}; sm_arch="sm_100", occupancy=1)
        @test !isempty(bytecode1)

        bytecode32 = code_tiled((a) -> nothing, Tuple{ct.TileArray{Float32, 1, spec1d}}; sm_arch="sm_100", occupancy=32)
        @test !isempty(bytecode32)
    end
end

#=============================================================================
 Load / Store Hints (operation-level optimization hints)
=============================================================================#

@testset "Load / Store Optimization Hints" begin
    # Common ArraySpecs for tests
    spec1d = ct.ArraySpec{1}(16, true)

    @testset "latency only on load" begin
        @test @filecheck begin
            @check "load_view_tko"
            @check "optimization_hints = <sm_120 = {latency = 5}>"
            ct.code_tiled(Tuple{ct.TileArray{Float32, 1, spec1d}}; sm_arch="sm_120") do a
                pid = ct.bid(1)
                t = ct.load(a, pid, (16,); latency=5)
                ct.store(a, pid, t)
                return nothing
            end
        end
    end

    @testset "allow_tma=false only on load" begin
        @test @filecheck begin
            @check "load_view_tko"
            @check "optimization_hints = <sm_120 = {allow_tma = false}>"
            ct.code_tiled(Tuple{ct.TileArray{Float32, 1, spec1d}}; sm_arch="sm_120") do a
                pid = ct.bid(1)
                t = ct.load(a, pid, (16,); allow_tma=false)
                ct.store(a, pid, t)
                return nothing
            end
        end
    end

    @testset "both hints on load" begin
        @test @filecheck begin
            @check "load_view_tko"
            @check "optimization_hints = <sm_120 = {allow_tma = false, latency = 7}>"
            ct.code_tiled(Tuple{ct.TileArray{Float32, 1, spec1d}}; sm_arch="sm_120") do a
                pid = ct.bid(1)
                t = ct.load(a, pid, (16,); latency=7, allow_tma=false)
                ct.store(a, pid, t)
                return nothing
            end
        end
    end

    @testset "latency only on store" begin
        @test @filecheck begin
            @check "store_view_tko"
            @check "optimization_hints = <sm_120 = {latency = 3}>"
            ct.code_tiled(Tuple{ct.TileArray{Float32, 1, spec1d}}; sm_arch="sm_120") do a
                pid = ct.bid(1)
                t = ct.load(a, pid, (16,))
                ct.store(a, pid, t; latency=3)
                return nothing
            end
        end
    end

    @testset "allow_tma=false only on store" begin
        @test @filecheck begin
            @check "store_view_tko"
            @check "optimization_hints = <sm_120 = {allow_tma = false}>"
            ct.code_tiled(Tuple{ct.TileArray{Float32, 1, spec1d}}; sm_arch="sm_120") do a
                pid = ct.bid(1)
                t = ct.load(a, pid, (16,))
                ct.store(a, pid, t; allow_tma=false)
                return nothing
            end
        end
    end

    @testset "both hints on store" begin
        @test @filecheck begin
            @check "store_view_tko"
            @check "optimization_hints = <sm_120 = {allow_tma = false, latency = 2}>"
            ct.code_tiled(Tuple{ct.TileArray{Float32, 1, spec1d}}; sm_arch="sm_120") do a
                pid = ct.bid(1)
                t = ct.load(a, pid, (16,))
                ct.store(a, pid, t; allow_tma=false, latency=2)
                return nothing
            end
        end
    end

    @testset "latency validation" begin
        @test_throws "latency must be between 1 and 10" begin
            code_tiled(Tuple{ct.TileArray{Float32, 1, spec1d}}; sm_arch="sm_120") do a
                pid = ct.bid(1)
                ct.load(a, pid, (16,); latency=11)
            end
        end

        bytecode1 = code_tiled(Tuple{ct.TileArray{Float32, 1, spec1d}}; sm_arch="sm_120") do a
            pid = ct.bid(1)
            t = ct.load(a, pid, (16,); latency=8)
            ct.store(a, pid, t)
            return nothing
        end
        @test !isempty(bytecode1)
    end

    @testset "multiple operations with mixed hints" begin
        @test @filecheck begin
            # First load with latency
            @check "load_view_tko"
            @check "optimization_hints = <sm_120 = {latency = 5}>"
            # Second load with allow_tma=false
            @check "load_view_tko"
            @check "optimization_hints = <sm_120 = {allow_tma = false}>"
            # Third load with no hints
            @check "load_view_tko"
            @check_not "optimization_hints"
            ct.code_tiled(Tuple{ct.TileArray{Float32, 1, spec1d},
                               ct.TileArray{Float32, 1, spec1d},
                               ct.TileArray{Float32, 1, spec1d}}; sm_arch="sm_120") do a, b, c
                pid = ct.bid(1)
                t1 = ct.load(a, pid, (16,); latency=5)
                t2 = ct.load(b, pid, (16,); allow_tma=false)
                t3 = ct.load(c, pid, (16,))
                result = t1 + t2 + t3
                ct.store(a, pid, result)
                return nothing
            end
        end
    end

    # Pointer-based operations (gather/scatter) with latency hints
    @testset "gather with latency hint" begin
        @test @filecheck begin
            @check "load_ptr_tko"
            @check "optimization_hints = <sm_120 = {latency = 3}>"
            code_tiled(Tuple{ct.TileArray{Float32, 1, spec1d}, ct.TileArray{Float32, 1, spec1d}}; sm_arch="sm_120") do a, b
                pid = ct.bid(1)
                indices = ct.arange((16,), Int32)
                tile = ct.gather(a, indices; latency=3)
                ct.store(b, pid, tile)
                return nothing
            end
        end
    end

    @testset "scatter with latency hint" begin
        @test @filecheck begin
            @check "store_ptr_tko"
            @check "optimization_hints = <sm_120 = {latency = 5}>"
            code_tiled(Tuple{ct.TileArray{Float32, 1, spec1d}, ct.TileArray{Float32, 1, spec1d}}; sm_arch="sm_120") do a, b
                pid = ct.bid(1)
                tile = ct.load(a, pid, (16,))
                indices = ct.arange((16,), Int32)
                ct.scatter(b, indices, tile; latency=5)
                return nothing
            end
        end
    end
end
