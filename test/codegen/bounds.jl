# Codegen tests for `BoundsAnalysis` and the bounded-predicate
# emission path through `analyze_assume_info`. The analysis tracks
# integer-valued SSA values to a closed `[lo, hi]` interval; the
# aggregator intersects the result with the structural lower bound
# (sizes/strides ≥ 0) and emits a sharper `Bounded(lo, hi)` predicate
# where the dataflow has information.
#
# Today the dataflow doesn't add information at `make_tensor_view`
# operands for typical kernels (sizes come from `getfield(arg, :sizes)`
# which is `top`, or literal tuples that the pass skips), so the
# observable IR is byte-identical to the prior hardcoded path. The
# real consumers are downstream — e.g. `no_wrap_pass!` on integer
# arithmetic — and the analysis is exercised directly below.

@testset "bounds — kernel-arg sizes still get Bounded(0, ?)" begin
    # Round-trip check: the structural lower bound `0` is preserved
    # when the dataflow has nothing else to add.
    spec = ct.ArraySpec{1}(16, true)
    @test @filecheck begin
        @check_label "entry"
        code_tiled(Tuple{ct.TileArray{Float32,1,spec}}) do a
            ct.store(a, 1, ct.load(a, 1, (16,)))
            return
        end
        @check "assume bounded<0, ?>"
    end
end

@testset "bounds — slice operand keeps Bounded(0, ?) when both endpoints unknown" begin
    # `a[i:j]` with scalar args produces `subi(stop, start_0)` whose
    # bounds analysis is `top`. After intersection with structural
    # `[0, ∞)`, we still emit `bounded<0, ?>`.
    spec = ct.ArraySpec{1}(16, true)
    @test @filecheck begin
        @check_label "entry"
        code_tiled(Tuple{ct.TileArray{Float32,1,spec}, Int32, Int32}) do a, i, j
            sub = @view a[i:j]
            ct.store(sub, 1, ct.load(sub, 1, (16,)))
            return
        end
        @check "assume bounded<0, ?>"
    end
end

#=============================================================================
 Direct-API tests for the analysis lattice & transfer rules. These
 bypass the codegen path and exercise `analyze_bounds` on a small SCI
 directly — they pin the lattice semantics (widening, interval
 arithmetic, ForOp IV seeding, getfield bounds) so future changes have
 to keep producing the same facts.
=============================================================================#

@testset "bounds — lattice basics" begin
    a = cuTile.BoundsAnalysis()

    # Bottom acts as identity in tmerge.
    @test cuTile.tmerge(a, cuTile.bottom(a), cuTile.IntRange(0, 5)) ==
          cuTile.IntRange(0, 5)
    @test cuTile.tmerge(a, cuTile.IntRange(0, 5), cuTile.bottom(a)) ==
          cuTile.IntRange(0, 5)

    # Aggressive widening: any disagreement opens the offending endpoint.
    @test cuTile.tmerge(a, cuTile.IntRange(0, 5), cuTile.IntRange(0, 5)) ==
          cuTile.IntRange(0, 5)
    @test cuTile.tmerge(a, cuTile.IntRange(0, 5), cuTile.IntRange(0, 6)) ==
          cuTile.IntRange(0, nothing)
    @test cuTile.tmerge(a, cuTile.IntRange(0, 5), cuTile.IntRange(-1, 5)) ==
          cuTile.IntRange(nothing, 5)
    @test cuTile.tmerge(a, cuTile.IntRange(0, 5), cuTile.IntRange(-1, 6)) ==
          cuTile.TOP_RANGE
end

@testset "bounds — interval arithmetic primitives" begin
    a = cuTile.BoundsAnalysis()

    # Addition: [a, b] + [c, d] = [a+c, b+d].
    @test cuTile.range_add(cuTile.IntRange(0, 5), cuTile.IntRange(2, 3)) ==
          cuTile.IntRange(2, 8)
    @test cuTile.range_add(cuTile.IntRange(0, nothing), cuTile.IntRange(2, 3)) ==
          cuTile.IntRange(2, nothing)

    # Subtraction: [a, b] - [c, d] = [a-d, b-c] (worst-case interval).
    @test cuTile.range_sub(cuTile.IntRange(5, 10), cuTile.IntRange(2, 3)) ==
          cuTile.IntRange(2, 8)

    # Negation flips and swaps endpoints.
    @test cuTile.range_neg(cuTile.IntRange(2, 8)) == cuTile.IntRange(-8, -2)
    @test cuTile.range_neg(cuTile.IntRange(2, nothing)) ==
          cuTile.IntRange(nothing, -2)

    # Non-negative multiplication: [a, b] * [c, d] = [a*c, b*d].
    @test cuTile.range_mul(cuTile.IntRange(2, 4), cuTile.IntRange(3, 5)) ==
          cuTile.IntRange(6, 20)
    # Mul with one zero endpoint stays at zero.
    @test cuTile.range_mul(cuTile.IntRange(0, 5), cuTile.IntRange(3, 7)) ==
          cuTile.IntRange(0, 35)
    # Signed mixed signs falls back to top (we don't elaborate the
    # general case yet).
    @test cuTile.range_mul(cuTile.IntRange(-2, 4), cuTile.IntRange(3, 5)) ==
          cuTile.TOP_RANGE
end

@testset "bounds — combine narrows structural with dataflow" begin
    a = cuTile.BoundsAnalysis()
    nonneg = cuTile.IntRange(0, nothing)

    # Sharper dataflow info wins: structural [0, ∞) ∩ [5, 10] = [5, 10].
    @test cuTile.combine_bound(nonneg, cuTile.IntRange(5, 10)) ==
          cuTile.IntRange(5, 10)

    # Negative dataflow lower is clamped to structural 0.
    @test cuTile.combine_bound(nonneg, cuTile.IntRange(-3, 10)) ==
          cuTile.IntRange(0, 10)

    # Top dataflow falls through to structural.
    @test cuTile.combine_bound(nonneg, cuTile.TOP_RANGE) == nonneg
end
