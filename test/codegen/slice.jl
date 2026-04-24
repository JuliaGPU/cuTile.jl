# Tests for @view / slice on TileArrays.
#
# These tests FileCheck-match the core IR shape produced by the slicing path:
# - a `subi` for the new size (stop - start)
# - a `muli` for start * stride
# - an `offset` for base + offset
# - a follow-up `make_tensor_view` on the derived pointer

spec1d = ct.ArraySpec{1}(16, true)
spec2d = ct.ArraySpec{2}(16, true)

@testset "slice — 1D single axis" begin
    # Static literal bounds: `stop - start` is emitted as `subi` at the Tile IR
    # level inside the intrinsic, then the usual `muli`/`offset`/make_tensor_view
    # pipeline runs.
    @test @filecheck begin
        @check_label "entry"
        code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}}) do a
            sub = @view a[3:10]
            t = ct.load(sub, 1, (4,))
            ct.store(sub, 1, t)
            return
        end
        @check "subi"
        @check "muli"
        @check "offset"
        @check "make_tensor_view"
    end

    # Dynamic bounds: start and stop are runtime values; the intrinsic emits
    # `subi(stop, start)` for the new axis size.
    @test @filecheck begin
        @check_label "entry"
        code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, Int32, Int32}) do a, i, j
            sub = @view a[i:j]
            t = ct.load(sub, 1, (4,))
            ct.store(sub, 1, t)
            return
        end
        @check "subi"
        @check "muli"
        @check "offset"
        @check "make_tensor_view"
    end
end

@testset "slice — 2D single axis" begin
    # Slice along axis 1; axis 2 is full (`:`).
    @test @filecheck begin
        @check_label "entry"
        code_tiled(Tuple{ct.TileArray{Float32,2,spec2d}, Int32, Int32}) do a, i, j
            sub = @view a[i:j, :]
            t = ct.load(sub, (1, 1), (4, 4))
            ct.store(sub, (1, 1), t)
            return
        end
        @check "subi"
        @check "muli"
        @check "offset"
        @check "make_tensor_view"
    end

    # Slice along axis 2; axis 1 is full (`:`).
    @test @filecheck begin
        @check_label "entry"
        code_tiled(Tuple{ct.TileArray{Float32,2,spec2d}, Int32, Int32}) do a, i, j
            sub = @view a[:, i:j]
            t = ct.load(sub, (1, 1), (4, 4))
            ct.store(sub, (1, 1), t)
            return
        end
        @check "subi"
        @check "muli"
        @check "offset"
        @check "make_tensor_view"
    end
end

@testset "slice — 2D multi-axis (chained)" begin
    # Both axes sliced. Emits two offsets (one per axis).
    @test @filecheck begin
        @check_label "entry"
        code_tiled(Tuple{ct.TileArray{Float32,2,spec2d},
                         Int32, Int32, Int32, Int32}) do a, i, j, k, l
            sub = @view a[i:j, k:l]
            t = ct.load(sub, (1, 1), (4, 4))
            ct.store(sub, (1, 1), t)
            return
        end
        @check "offset"
        @check "offset"
        @check "make_tensor_view"
    end
end

@testset "slice — divisibility annotations (Phase 3)" begin
    # For spec1d {alignment=16}, @view A[3:10]:
    #   start_0 = 3-1 = 2, stop = 10; new size = stop - start = 8.
    #   new_size divby = gcd(2, 10) = 2  (loose — see Gap 1 in divisibility.jl).
    #   new_base divby = gcd(16, 2 * stride * sizeof(T)) with stride=1, elem=4 → gcd(16, 8) = 8.
    @test @filecheck begin
        @check_label "entry"
        code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}}) do a
            sub = @view a[3:10]
            t = ct.load(sub, 1, (4,))
            ct.store(sub, 1, t)
            return
        end
        @check "assume div_by<2>"   # new_size = subi(stop=10, start=2); loose gcd bound
        @check "offset"
        @check "assume div_by<8>"   # on derived new_base pointer
        @check "make_tensor_view"
    end

    # Once Gap 1 (constant folding for subi(const, const)) lands, the new_size
    # should tighten to the exact value abs(stop - start).
    @test_broken @filecheck begin
        @check_label "entry"
        code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}}) do a
            sub = @view a[3:10]
            t = ct.load(sub, 1, (4,))
            ct.store(sub, 1, t)
            return
        end
        @check "assume div_by<8>"   # tight: abs(10 - 2) = 8  (blocked on Gap 1)
        @check "offset"
        @check "assume div_by<8>"
        @check "make_tensor_view"
    end

    # Dynamic slice with divby-annotated bounds (e.g. @view A[bid*TILE : (bid+1)*TILE])
    # should get a divby=TILE annotation on the derived size. Blocked on Gap 1 or
    # Gap 2 in src/compiler/passes/divisibility.jl (constant folding + ArraySpec
    # seeding). Flip this to @test once either lands.
    @test_broken begin
        # Placeholder for when dynamic-bounds divby propagation lands.
        false
    end
end
