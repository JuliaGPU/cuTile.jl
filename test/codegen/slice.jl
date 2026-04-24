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
    # Static bounds fold start/stop to constants; the IR still carries subi/muli/offset.
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

    # Dynamic bounds: start/stop are runtime kernel args.
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
