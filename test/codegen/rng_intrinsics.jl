# RNG state-threading pass tests
#
# Verifies that `rng_counter()` / `rng_advance(n)` placeholder intrinsics
# lower cleanly to concrete SSA arithmetic by `rng_state_pass!`. These tests
# exercise the pass directly via `Intrinsics.rng_*`; the user-facing `rand`
# API (Phase 3) sits on top of these.

@testset "rng_state_pass!" begin
    spec1d = ct.ArraySpec{1}(16, true)

    @testset "straight-line: two counter reads and one advance" begin
        @test @filecheck begin
            @check_label "entry"
            # rng_counter() twice with an rng_advance(16) between → one addi
            # with constant 16. Const-folding collapses 0 + 16 = 16, so the
            # output store sees a constant-16 tile.
            code_tiled(Tuple{ct.TileArray{UInt32,1,spec1d}}) do a
                pid = ct.bid(1)
                v = ct.Intrinsics.rng_counter(0)
                ct.Intrinsics.rng_advance(0, 16)
                v2 = ct.Intrinsics.rng_counter(0)
                x = ct.Intrinsics.addi(v, v2)
                @check "constant <i32: 16>"
                @check "store_view_tko"
                ct.store(a, pid, x)
                return
            end
        end
    end

    @testset "loop: advance in body adds a carry" begin
        # `for i in 1:n` with rng_advance(1) per iteration should produce a
        # ForOp with one iter_values carry initialized to 0, body doing
        # `addi %iterArg, 1` and continuing with the bumped value.
        @test @filecheck begin
            @check_label "entry"
            code_tiled(Tuple{ct.TileArray{UInt32,1,spec1d}, Int32}) do a, n
                pid = ct.bid(1)
                for i in Int32(1):n
                    ct.Intrinsics.rng_advance(0, 1)
                end
                v = ct.Intrinsics.rng_counter(0)
                @check "iter_values"
                @check "addi"
                @check "continue"
                ct.store(a, pid, v)
                return
            end
        end
    end

    @testset "if-op: advance in one arm yields counter from both" begin
        # `if cond; rng_advance(1); end` advances only in the then-arm. The
        # pass must add a yield to BOTH arms so the merged IfOp result has a
        # well-defined counter value.
        @test @filecheck begin
            @check_label "entry"
            code_tiled(Tuple{ct.TileArray{UInt32,1,spec1d}, Int32}) do a, cond
                pid = ct.bid(1)
                if cond > Int32(0)
                    ct.Intrinsics.rng_advance(0, 1)
                end
                v = ct.Intrinsics.rng_counter(0)
                # Both arms must yield the counter → look for two yield sites.
                @check "yield"
                @check "yield"
                ct.store(a, pid, v)
                return
            end
        end
    end

    @testset "two RNG handles produce independent counters" begin
        # Two handles (ids 1 and 2) in one kernel must each get their own
        # counter SSA chain. Independently advancing h=1 by 16 and h=2 by
        # 32 and reading each should yield distinct constants (16 and 32);
        # a single-slot implementation would merge to 48.
        @test @filecheck begin
            @check_label "entry"
            code_tiled(Tuple{ct.TileArray{UInt32,1,spec1d}, ct.TileArray{UInt32,1,spec1d}}) do a, b
                pid = ct.bid(1)
                ct.Intrinsics.rng_advance(1, 16)
                ct.Intrinsics.rng_advance(2, 32)
                v1 = ct.Intrinsics.rng_counter(1)
                v2 = ct.Intrinsics.rng_counter(2)
                @check "constant <i32: 16>"
                @check "constant <i32: 32>"
                ct.store(a, pid, v1)
                ct.store(b, pid, v2)
                return
            end
        end
    end

    @testset "no RNG usage → pass is a no-op" begin
        # A kernel that never touches the RNG intrinsics must compile
        # identically to what it would have without the pass.
        @test @filecheck begin
            @check_label "entry"
            @check_not "rng_counter"
            @check_not "rng_advance"
            code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}}) do a
                pid = ct.bid(1)
                tile = ct.load(a, pid, (16,))
                ct.store(a, pid, tile)
                return
            end
        end
    end
end
