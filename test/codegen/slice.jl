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
    # Static literal bounds: Julia inference folds `stop - start` to a literal
    # size, so the IR shows `muli`/`offset`/`make_tensor_view` but no explicit
    # `subi` for the size computation.
    @test @filecheck begin
        @check_label "entry"
        code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}}) do a
            sub = @view a[3:10]
            t = ct.load(sub, 1, (4,))
            ct.store(sub, 1, t)
            return
        end
        @check "muli"
        @check "offset"
        @check "make_tensor_view"
    end

    # Dynamic bounds: start and (stop - start) are runtime values. `subi` is
    # present at the Julia-IR level (stop - start) → passed to the intrinsic.
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

@testset "slice — ct.slice explicit call" begin
    # The 0-indexed half-open form exposed as ct.slice(arr, axis, start, stop).
    @test @filecheck begin
        @check_label "entry"
        code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, Int32, Int32}) do a, i, j
            sub = ct.slice(a, 1, i, j)
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

@testset "slice — divisibility annotations (Phase 3)" begin
    # For spec1d {alignment=16}, @view A[3:10]:
    #   start_0 = 3-1 = 2, size = 10-2 = 8 (Julia inference folds to literal).
    #   new_size divby = abs_divby(8) = 8.
    #   new_base divby = gcd(16, 2 * stride * sizeof(T)) with stride=1, elem=4 → gcd(16, 8) = 8.
    @test @filecheck begin
        @check_label "entry"
        code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}}) do a
            sub = @view a[3:10]
            t = ct.load(sub, 1, (4,))
            ct.store(sub, 1, t)
            return
        end
        @check "assume div_by<8>"   # on derived new_size literal 8
        @check "offset"
        @check "assume div_by<8>"   # on derived new_base pointer
        @check "make_tensor_view"
    end

    # TODO: dynamic slice with divby-annotated bounds (e.g. @view A[bid*TILE : (bid+1)*TILE])
    # should get a divby=TILE annotation on the derived size. Blocked on Gap 1 or
    # Gap 2 in src/compiler/passes/divisibility.jl (constant folding + ArraySpec
    # seeding). Flip this to @test once either lands.
    @test_broken begin
        # Placeholder for when dynamic-bounds divby propagation lands.
        false
    end
end
