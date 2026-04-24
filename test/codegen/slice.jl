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

@testset "slice — divisibility annotations" begin
    # For spec1d {alignment=16}, @view A[3:10]:
    #   start_0 = 2, stop = 10; new_size = subi(stop, start).
    #   new_size divby = gcd(2, 10) = 2  (loose — const operands aren't folded).
    #   new_base divby = gcd(16, 2 * stride * sizeof(T)) = gcd(16, 8) = 8.
    @test @filecheck begin
        @check_label "entry"
        code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}}) do a
            sub = @view a[3:10]
            t = ct.load(sub, 1, (4,))
            ct.store(sub, 1, t)
            return
        end
        @check "assume div_by<2>"
        @check "offset"
        @check "assume div_by<8>"
        @check "make_tensor_view"
    end

    # Tightens once a const-fold pass reaches the subi(stop, start): div_by<8>.
    @test_broken @filecheck begin
        @check_label "entry"
        code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}}) do a
            sub = @view a[3:10]
            t = ct.load(sub, 1, (4,))
            ct.store(sub, 1, t)
            return
        end
        @check "assume div_by<8>"
        @check "offset"
        @check "assume div_by<8>"
        @check "make_tensor_view"
    end

    # Dynamic-bounds divby propagation (ArraySpec seeding into derived values).
    @test_broken false
end
