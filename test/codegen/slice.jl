# Tests for @view / slice on TileArrays.
#
# FileCheck-matches the core IR shape produced by the slicing path:
# - a `subi` for the new size (stop - start) — unless const-folded
# - a `muli` for start * stride
# - an `offset` for base + offset
# - a follow-up `make_tensor_view` on the derived pointer

spec1d = ct.ArraySpec{1}(16, true)
spec2d = ct.ArraySpec{2}(16, true)

@testset "slice — 1D single axis" begin
    # Static literal bounds: the arithmetic is emitted via language-level
    # `-`/`*`/`offset` intrinsics, so constant operands get folded. `3 - 1`
    # and `10 - 2` fold to `2` and `8`; `muli` and `offset` remain because the
    # stride and base pointer are runtime kernel parameters.
    @test @filecheck begin
        @check_label "entry"
        code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}}) do a
            sub = @view a[3:10]
            t = ct.load(sub, 1, (4,))
            ct.store(sub, 1, t)
            return
        end
        @check_not "subi"  # folded: 3 - 1 and 10 - 2
        @check "constant <i32: 2>"  # folded start_0
        @check "muli"
        @check "offset"
        @check "constant <i32: 8>"  # folded new_size
        @check "make_tensor_view"
    end

    # Dynamic bounds: start and stop are runtime values; the pipeline emits
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

@testset "slice — bounds asserts" begin
    # Static literal bounds that are valid: `start >= 1` folds to `true`, so
    # the AssertOp is elided by the assert intrinsic's Const(true) fast path.
    @test @filecheck begin
        @check_label "entry"
        code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}}) do a
            sub = @view a[3:10]
            t = ct.load(sub, 1, (4,))
            ct.store(sub, 1, t)
            return
        end
        @check_not "assert {{.*}}slice start"
    end

    # Dynamic bounds lower to a runtime AssertOp on `start >= 1`. The stop
    # bound is not asserted: Julia's `unitrange_last` already clamps
    # `last(r) >= first(r) - 1`, so any such assert would always pass.
    @test @filecheck begin
        @check_label "entry"
        code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, Int32, Int32}) do a, i, j
            sub = @view a[i:j]
            t = ct.load(sub, 1, (4,))
            ct.store(sub, 1, t)
            return
        end
        @check "assert {{.*}}slice start must be"
        @check_not "assert {{.*}}slice stop"
    end

    # Bad static literal: `start < 1` folds to `true`, so the AssertOp is
    # emitted with an `i1 false` condition — the kernel will always abort at
    # runtime with the slice-start message when this point is reached.
    # (We don't fold `Const(false)` to a compile error because the assert may
    # be sitting in a conditional branch that isn't taken; see the comment in
    # `emit_intrinsic!(::typeof(Intrinsics.assert), ...)`.)
    @test @filecheck begin
        @check_label "entry"
        code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}}) do a
            sub = @view a[0:10]
            t = ct.load(sub, 1, (4,))
            ct.store(sub, 1, t)
            return
        end
        @check "constant <i1: false>"
        @check "assert {{.*}}slice start must be"
    end
end

