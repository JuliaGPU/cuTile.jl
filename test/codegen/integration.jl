
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
                a_tile = ct.load(a, bid, (tile,))
                @check "load_view_tko"
                b_tile = ct.load(b, bid, (tile,))
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
                input_tile = ct.load(x, (bidx, bidy), (tm, tn))
                @check "permute"
                transposed_tile = transpose(input_tile)
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
                num_k = ct.num_tiles(A, 2, (tm, tk))
                acc = ct.full((tm, tn), zero(Float32), Float32)
                # NOTE: Uses while-loop pattern because Julia's for-loop generates
                # complex iterator IR with PhiNodes that isn't fully supported.
                # The structurizer upgrades this counting while-loop to a ForOp.
                @check "for"
                k = Int32(1)
                while k <= num_k
                    @check "load_view_tko"
                    a = ct.load(A, (bid, k), (tm, tk); padding_mode=ct.PaddingMode.Zero)
                    @check "load_view_tko"
                    b = ct.load(B, (k, bid), (tk, tn); padding_mode=ct.PaddingMode.Zero)
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
                num_tiles = ct.num_tiles(X, 2, (1, TILE_N))

                # First for loop: compute sum
                @check "for"
                acc = ct.full((1, TILE_N), 0.0f0, Float32)
                j = Int32(1)
                while j <= num_tiles
                    tx = ct.load(X, (bid_m, j), (1, TILE_N); padding_mode=ct.PaddingMode.Zero)
                    acc = acc .+ tx
                    j += Int32(1)
                end
                @check "reduce"
                sum_val = sum(acc; dims=2)
                ct.store(Sum, bid_m, sum_val)

                # Second for loop: scale output by sum
                @check "for"
                j = Int32(1)
                while j <= num_tiles
                    tx = ct.load(X, (bid_m, j), (1, TILE_N); padding_mode=ct.PaddingMode.Zero)
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
                num_tiles = ct.num_tiles(DW, 2, (1, TILE_N))

                @check "for"
                j = Int32(1)
                while j <= num_tiles
                    # Load and compute partial result
                    partial = ct.load(Partial, (bid, j), (1, TILE_N); padding_mode=ct.PaddingMode.Zero)

                    @check "loop"
                    # Acquire spinlock (nested inside for loop)
                    while ct.atomic_cas(Locks, group_bid, Int32(0), Int32(1);
                                       memory_order=ct.MemoryOrder.Acquire) == Int32(1)
                        # spin
                    end

                    # Critical section: accumulate
                    @check "load_view_tko"
                    acc = ct.load(DW, (group_bid, j), (1, TILE_N); padding_mode=ct.PaddingMode.Zero)
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
            # After 1→0 index conversion, the store uses (loopIdx - 1)
            @check "[[IDX:%.+]] = subi %loopIdx"
            @check "store_view_tko{{.*}}[%{{[^,]+}}, [[IDX]]]"
            code_tiled(Tuple{ct.TileArray{Float32,2,spec}, ct.TileArray{Int32,1,spec1d},
                           Int32, ct.Constant{Int,4}, ct.Constant{Int,4}}) do DB, Locks, num_iters, GROUP_SIZE_M, TILE_N
                bid_m = ct.bid(1)
                # group_bid_m: 1-indexed group ID
                group_bid_m = ((bid_m - Int32(1)) % Int32(GROUP_SIZE_M)) + Int32(1)

                j = Int32(1)
                while j <= num_iters
                    # Nested spinloop - this must not shadow loopIdx
                    while ct.atomic_cas(Locks, group_bid_m, Int32(0), Int32(1);
                                       memory_order=ct.MemoryOrder.Acquire) == Int32(1)
                    end

                    val = ct.full((1, TILE_N), 1.0f0, Float32)
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
        @test @filecheck begin
            # The inner loop must NOT capture %loopIdx - it should capture %iterArg0
            # Bug: "loop iter_values(%arg9 = %loopIdx"
            @check_not "iter_values({{.*}}= %loopIdx"
            ct.code_tiled(Tuple{ct.TileArray{Float32,2,spec}, ct.TileArray{Int32,1,spec1d},
                               Int32, ct.Constant{Int,4}, ct.Constant{Int,4}}) do DB, Locks, num_iters, GROUP_SIZE_M, TILE_N
                bid_m = ct.bid(1)
                # group_bid_m: 1-indexed group ID
                group_bid_m = ((bid_m - Int32(1)) % Int32(GROUP_SIZE_M)) + Int32(1)

                j = Int32(1)
                while j <= num_iters
                    # Spinloop should use group_bid_m for the lock, not j
                    while ct.atomic_cas(Locks, group_bid_m, Int32(0), Int32(1);
                                       memory_order=ct.MemoryOrder.Acquire) == Int32(1)
                    end

                    val = ct.full((1, TILE_N), 1.0f0, Float32)
                    ct.store(DB, (group_bid_m, j), val)

                    ct.atomic_xchg(Locks, group_bid_m, Int32(0);
                                  memory_order=ct.MemoryOrder.Release)

                    j += Int32(1)
                end
                return
            end
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
        spec2d = ct.ArraySpec{2}(128, true, (4, 0), (32, 32))
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
                num_tiles = ct.num_tiles(DW, 2, (_TILE_N, _TILE_M))

                dw = ct.zeros((_TILE_N, _TILE_M), Float32)
                db = ct.zeros((_TILE_N, _TILE_M), Float32)
                i = Int32(1)
                while i <= num_tiles
                    dw = dw .+ ct.load(DW, (bid_n, i), (_TILE_N, _TILE_M); padding_mode=ct.PaddingMode.Zero)
                    db = db .+ ct.load(DB, (bid_n, i), (_TILE_N, _TILE_M); padding_mode=ct.PaddingMode.Zero)
                    i += Int32(1)
                end

                sum_dw = sum(dw; dims=2)
                sum_db = sum(db; dims=2)

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
                num_tiles = ct.num_tiles(inp, 1, (TILE_N,))

                # First loop: accumulate and reduce
                @check "for"
                acc = ct.zeros((TILE_N,), Float32)
                i = Int32(1)
                while i <= num_tiles
                    tile = ct.load(inp, i, (TILE_N,); padding_mode=ct.PaddingMode.Zero)
                    acc = acc .+ tile
                    i += Int32(1)
                end
                @check "reduce"
                sum_val = sum(acc; dims=1)

                # Second loop: use sum_val AND accumulate
                @check "for"
                acc2 = ct.zeros((TILE_N,), Float32)
                i = Int32(1)
                while i <= num_tiles
                    tile = ct.load(inp, i, (TILE_N,); padding_mode=ct.PaddingMode.Zero)
                    @check "subf"
                    acc2 = acc2 .+ (tile .- sum_val)  # Uses sum_val from first loop
                    i += Int32(1)
                end
                @check "reduce"
                @check "store_view_tko"
                ct.store(out, bid, sum(acc2; dims=1))

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
            # This should fail with an IRError, since the intrinsic
            # is invoked with mismatched types (Int32 + Int64)
            @test_throws ct.IRError code_tiled(Tuple{ct.TileArray{Float32,1,spec}}) do a
                pid = ct.bid(1)  # Int32
                # Force type mismatch by calling addi with different types
                result = ct.Intrinsics.addi(pid, Int64(1))
                return
            end
        end
    end

    @testset "method error detection" begin
        spec = ct.ArraySpec{1}(16, true)

        isdefined(Core, :throw_methoderror) &&
        @testset "mismatched tile shapes with + produces MethodError" begin
            spec2d = ct.ArraySpec{2}(16, true)
            @test_throws "MethodError during Tile IR compilation" begin
                code_tiled(Tuple{ct.TileArray{Float32,2,spec2d}}) do a
                    pid = ct.bid(1)
                    tile_a = ct.load(a, pid, (4, 8))
                    tile_b = ct.load(a, pid, (8, 4))
                    Base.donotdelete(tile_a + tile_b)
                    return
                end
            end
        end

        isdefined(Core, :throw_methoderror) &&
        @testset "no matching method produces MethodError" begin
            only_ints(x::Int) = x
            @test_throws "MethodError during Tile IR compilation" begin
                code_tiled(Tuple{ct.TileArray{Float32,1,spec}}) do a
                    tile = ct.load(a, ct.bid(1), (16,))
                    only_ints(tile)
                    return
                end
            end
        end

        @testset "unsupported function produces clear error" begin
            @test_throws "Unsupported function call during Tile IR compilation" begin
                code_tiled(Tuple{ct.TileArray{Float32,1,spec}}) do a
                    tile = ct.load(a, ct.bid(1), (16,))
                    print(tile)
                    return
                end
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
                    reshape(tile, (3,))
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
            code_tiled(devnull, a -> (ct.store(a, ct.bid(1), ct.load(a, ct.bid(1), (16,))); return),
                       Tuple{ct.TileArray{Float32,1,spec}})
            code_tiled(devnull, a -> (ct.store(a, ct.bid(1), ct.load(a, ct.bid(1), (32,))); return),
                       Tuple{ct.TileArray{Float32,1,spec}})
            code_tiled(devnull, a -> (ct.store(a, ct.bid(1), ct.load(a, ct.bid(1), (128,))); return),
                       Tuple{ct.TileArray{Float32,1,spec}})
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

        @testset "float constant addition folds through addf" begin
            @test @filecheck begin
                @check_label "entry"
                @check_not "addf"
                @check "constant <f32: 5"
                @check "mulf"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec}}) do a
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (16,))
                    scale = 2.0f0 + 3.0f0
                    Base.donotdelete(tile .* scale)
                    return
                end
            end
        end

        @testset "integer constant subtraction folds through subi" begin
            @test @filecheck begin
                @check_label "entry"
                @check_not "subi"
                @check "load_view"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec}}) do a
                    idx = Int32(5) - Int32(2)
                    tile = ct.load(a, idx, (16,))
                    Base.donotdelete(tile)
                    return
                end
            end
        end

        @testset "float constant multiplication folds through mulf" begin
            @test @filecheck begin
                @check_label "entry"
                @check "constant <f32: 6"
                @check "broadcast"
                @check "mulf"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec}}) do a
                    pid = ct.bid(1)
                    tile = ct.load(a, pid, (16,))
                    scale = 2.0f0 * 3.0f0
                    Base.donotdelete(tile .* scale)
                    return
                end
            end
        end
    end
end

#=============================================================================
 External Constants (GlobalRef handling)
=============================================================================#

# Constants defined outside the kernel (module-level `const`) appear as GlobalRef
# nodes in Julia IR. These must emit proper ConstantOp for numeric types,
# not ghost values (which produce nothing in the bytecode).

const _CODEGEN_TEST_FLOAT32 = Float32(1 / log(2))
const _CODEGEN_TEST_FLOAT64 = 3.14159

@testset "External Constants" begin
    spec1d = ct.ArraySpec{1}(16, true)

    @testset "external Float32 constant in arithmetic" begin
        # Bug 1: GlobalRef for Float32 must emit ConstantOp, not a ghost value.
        # Previously, emit_value!(ctx, ::GlobalRef) wrapped all values as ghosts,
        # causing MulFOp to receive `nothing` instead of a bytecode Value.
        @test @filecheck begin
            @check_label "entry"
            code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}}) do a
                pid = ct.bid(1)
                tile = ct.load(a, pid, (16,))
                @check "constant <f32"
                @check "mulf"
                Base.donotdelete(tile * _CODEGEN_TEST_FLOAT32)
                return
            end
        end
    end

    @testset "external Float64 constant in arithmetic" begin
        @test @filecheck begin
            @check_label "entry"
            code_tiled(Tuple{ct.TileArray{Float64,1,spec1d}}) do a
                pid = ct.bid(1)
                tile = ct.load(a, pid, (16,))
                @check "constant <f64"
                @check "mulf"
                Base.donotdelete(tile * _CODEGEN_TEST_FLOAT64)
                return
            end
        end
    end

    @testset "external constant assigned to local variable" begin
        # Bug 2: GlobalRef on RHS of assignment in emit_rhs! returned nothing.
        # Using a local variable forces Julia to emit an assignment from the GlobalRef.
        @test @filecheck begin
            @check_label "entry"
            code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}}) do a
                pid = ct.bid(1)
                tile = ct.load(a, pid, (16,))
                local_const = _CODEGEN_TEST_FLOAT32
                @check "constant <f32"
                @check "mulf"
                Base.donotdelete(tile * local_const)
                return
            end
        end
    end

    @testset "scalar arg multiplied by external constant" begin
        # Regression test for issue #77: scalar × global constant failed
        # because encode_MulFOp! received Nothing from the ghost-wrapped GlobalRef.
        @test @filecheck begin
            @check_label "entry"
            code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, Float32}) do a, scale
                pid = ct.bid(1)
                tile = ct.load(a, pid, (16,))
                @check "constant <f32"
                @check "mulf"
                total_scale = scale * _CODEGEN_TEST_FLOAT32
                @check "broadcast"
                @check "mulf"
                Base.donotdelete(tile .* total_scale)
                return
            end
        end
    end

    @testset "external constant as first operand in scalar addition" begin
        # Regression test for issue #77: global constant used in scalar arithmetic
        # must emit a ConstantOp, not a ghost value. Tests GlobalRef as the first
        # operand (LHS) to cover both operand positions.
        @test @filecheck begin
            @check_label "entry"
            code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, Float32}) do a, offset
                pid = ct.bid(1)
                tile = ct.load(a, pid, (16,))
                @check "constant <f32"
                @check "addf"
                total = _CODEGEN_TEST_FLOAT32 + offset
                @check "broadcast"
                @check "addf"
                Base.donotdelete(tile .+ total)
                return
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
            @test @filecheck begin
                @check "num_cta_in_cga = $(num_ctas)"
                ct.code_tiled((a) -> nothing, Tuple{ct.TileArray{Float32, 1, spec1d}}; sm_arch="sm_100", num_ctas)
            end
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
        @test @filecheck begin
            @check "occupancy = 1"
            ct.code_tiled((a) -> nothing, Tuple{ct.TileArray{Float32, 1, spec1d}}; sm_arch="sm_100", occupancy=1)
        end

        @test @filecheck begin
            @check "occupancy = 32"
            ct.code_tiled((a) -> nothing, Tuple{ct.TileArray{Float32, 1, spec1d}}; sm_arch="sm_100", occupancy=32)
        end
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

        @test @filecheck begin
            @check "optimization_hints = <sm_120 = {latency = 8}>"
            ct.code_tiled(Tuple{ct.TileArray{Float32, 1, spec1d}}; sm_arch="sm_120") do a
                pid = ct.bid(1)
                t = ct.load(a, pid, (16,); latency=8)
                ct.store(a, pid, t)
                return nothing
            end
        end
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

#=============================================================================
 Ghost Type Arguments
=============================================================================#

@testset "Non-Constant Ghost Type Arguments" begin
    spec = ct.ArraySpec{1}(16, true)

    @testset "Nothing argument" begin
        @test @filecheck begin
            @check_label "entry"
            @check "load_view_tko"
            @check "store_view_tko"
            code_tiled(Tuple{ct.TileArray{Float32,1,spec}, ct.TileArray{Float32,1,spec}, Nothing}) do a, b, _
                pid = ct.bid(1)
                tile = ct.load(a, pid, (16,))
                ct.store(b, pid, tile)
                return
            end
        end
    end

    @testset "Val argument" begin
        @test @filecheck begin
            @check_label "entry"
            @check "load_view_tko"
            @check "store_view_tko"
            code_tiled(Tuple{ct.TileArray{Float32,1,spec}, ct.TileArray{Float32,1,spec}, Val{16}}) do a, b, _
                pid = ct.bid(1)
                tile = ct.load(a, pid, (16,))
                ct.store(b, pid, tile)
                return
            end
        end
    end
end
