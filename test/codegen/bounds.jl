# Codegen tests for `BoundsAnalysis` and the bounded-predicate
# emission path through `op_predicates` (analysis/assume.jl). The
# analysis tracks integer-valued SSA values to a closed `[lo, hi]`
# interval; `op_predicates` intersects the result with the structural
# lower bound (sizes/strides ≥ 0) and emits a sharper `Bounded(lo, hi)`
# predicate where the dataflow has information.
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
    @test cuTile.range_neg(cuTile.IntRange(typemin(Int), -1)) ==
          cuTile.IntRange(1, nothing)

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

using Core: SSAValue, Argument, ReturnNode
using IRStructurizer: StructuredIRCode, Block, BlockArgument, ForOp, ContinueOp

@testset "bounds — ForOp IV upper bound is sound for step > 1" begin
    mk_for(lower, upper, step) = begin
        iv = BlockArgument(1, Int)
        body = Block()
        body.terminator = ContinueOp(Any[])
        fop = ForOp(lower, upper, step, iv, body, Any[])
        entry = Block()
        push!(entry, 1, fop, Nothing)
        entry.terminator = ReturnNode(nothing)
        sci = StructuredIRCode(Any[Any, Int], Any[], entry, 1)
        (cuTile.analyze_bounds(sci), iv)
    end

    # 0:4:<10 visits 0, 4, 8 — the last IV overshoots `upper - step`
    # (= 6), so only the exclusive bound `upper - 1` is sound.
    r, iv = mk_for(0, 10, 4)
    @test r[iv] == cuTile.IntRange(0, 9)

    # Trip length divides the step: the sharp `upper - step` applies.
    r, iv = mk_for(0, 12, 4)
    @test r[iv] == cuTile.IntRange(0, 8)

    # Step 1: exclusive upper bound minus one, as before.
    r, iv = mk_for(0, 10, 1)
    @test r[iv] == cuTile.IntRange(0, 9)

    # Unknown step: `iv < upper` alone still bounds the IV.
    r, iv = mk_for(0, 10, Argument(2))
    @test r[iv] == cuTile.IntRange(0, 9)
end

@testset "bounds — exti consults signedness" begin
    mk_exti(scalar, T, s) = begin
        S = typeof(scalar)
        entry = Block()
        push!(entry, 1, Expr(:call, cuTile.Intrinsics.constant,
                             QuoteNode(()), scalar, S), S)
        push!(entry, 2, Expr(:call, cuTile.Intrinsics.exti,
                             SSAValue(1), T, QuoteNode(s)), T)
        entry.terminator = ReturnNode(SSAValue(2))
        sci = StructuredIRCode(Any[Any], Any[], entry, 2)
        cuTile.analyze_bounds(sci)
    end

    # Sign-extension preserves the mathematical value.
    r = mk_exti(Int32(-5), Int64, cuTile.Signedness.Signed)
    @test cuTile.bounds(r, SSAValue(2)) == cuTile.IntRange(-5, -5)

    # Zero-extension of a possibly-negative value reinterprets the bits
    # (Int32(-5) zexts to 2^32 - 5) — the range must not pass through.
    r = mk_exti(Int32(-5), Int64, cuTile.Signedness.Unsigned)
    @test cuTile.bounds(r, SSAValue(2)) == cuTile.TOP_RANGE

    # Zero-extension of a provably non-negative value stays exact.
    r = mk_exti(Int32(5), Int64, cuTile.Signedness.Unsigned)
    @test cuTile.bounds(r, SSAValue(2)) == cuTile.IntRange(5, 5)

    # Sign-extension reinterprets the high bit of an unsigned source.
    r = mk_exti(UInt8(250), Int32, cuTile.Signedness.Signed)
    @test cuTile.bounds(r, SSAValue(2)) == cuTile.TOP_RANGE

    # Values below the signed maximum have the same interpretation.
    r = mk_exti(UInt8(5), Int32, cuTile.Signedness.Signed)
    @test cuTile.bounds(r, SSAValue(2)) == cuTile.IntRange(5, 5)

    # The destination interpretation can also change the value.
    r = mk_exti(Int8(-5), UInt32, cuTile.Signedness.Signed)
    @test cuTile.bounds(r, SSAValue(2)) == cuTile.TOP_RANGE
    r = mk_exti(Int8(5), UInt32, cuTile.Signedness.Signed)
    @test cuTile.bounds(r, SSAValue(2)) == cuTile.IntRange(5, 5)
end

@testset "bounds — bitcast consults source and destination signedness" begin
    function bitcast_sci(scalar, T)
        S = typeof(scalar)
        entry = Block()
        push!(entry, 1, Expr(:call, cuTile.Intrinsics.constant,
                             QuoteNode(()), scalar, S), S)
        push!(entry, 2, Expr(:call, cuTile.Intrinsics.bitcast,
                             SSAValue(1), T), T)
        entry.terminator = ReturnNode(SSAValue(2))
        StructuredIRCode(Any[Any], Any[], entry, 2)
    end

    r = cuTile.analyze_bounds(bitcast_sci(Int32(-1), UInt32))
    @test cuTile.bounds(r, SSAValue(2)) == cuTile.TOP_RANGE

    r = cuTile.analyze_bounds(bitcast_sci(Int32(5), UInt32))
    @test cuTile.bounds(r, SSAValue(2)) == cuTile.IntRange(5, 5)

    r = cuTile.analyze_bounds(bitcast_sci(UInt8(250), Int8))
    @test cuTile.bounds(r, SSAValue(2)) == cuTile.TOP_RANGE

    r = cuTile.analyze_bounds(bitcast_sci(UInt8(5), Int8))
    @test cuTile.bounds(r, SSAValue(2)) == cuTile.IntRange(5, 5)

    r = cuTile.analyze_bounds(bitcast_sci(Int32(-1), Float32))
    @test cuTile.bounds(r, SSAValue(2)) == cuTile.TOP_RANGE

    # The unknown bitcast range must not justify a no-wrap flag on an add
    # that wraps from typemax(UInt32) to zero.
    sci = bitcast_sci(Int32(-1), UInt32)
    entry = sci.entry
    push!(entry, 3, Expr(:call, cuTile.Intrinsics.constant,
                         QuoteNode(()), UInt32(1), UInt32), UInt32)
    push!(entry, 4, Expr(:call, cuTile.Intrinsics.addi,
                         SSAValue(2), SSAValue(3)), UInt32)
    entry.terminator = ReturnNode(SSAValue(4))
    r = cuTile.analyze_bounds(sci)
    @test cuTile.bounds(r, SSAValue(4)) == cuTile.TOP_RANGE
    cuTile.no_wrap_pass!(sci, r)
    @test length(entry.body[4].stmt.args) == 3
end

@testset "bounds — arithmetic clamps to the result element width" begin
    mk_add(T) = begin
        entry = Block()
        push!(entry, 1, Expr(:call, cuTile.Intrinsics.constant,
                             QuoteNode(()), typemax(Int32), Int32), Int32)
        push!(entry, 2, Expr(:call, cuTile.Intrinsics.addi,
                             SSAValue(1), SSAValue(1)), T)
        entry.terminator = ReturnNode(SSAValue(2))
        sci = StructuredIRCode(Any[Any], Any[], entry, 2)
        cuTile.analyze_bounds(sci)
    end

    # i32 + i32 wraps at runtime: the exact sum 2^32 - 2 escapes Int32,
    # so the wrapped value can land anywhere.
    r = mk_add(Int32)
    @test cuTile.bounds(r, SSAValue(2)) == cuTile.TOP_RANGE

    # The same sum fits an Int64 destination and is kept exact.
    r = mk_add(Int64)
    @test cuTile.bounds(r, SSAValue(2)) ==
          cuTile.IntRange(2 * Int(typemax(Int32)), 2 * Int(typemax(Int32)))
end

@testset "bounds — host Int overflow widens to top" begin
    entry = Block()
    push!(entry, 1, Expr(:call, cuTile.Intrinsics.assume, Argument(2),
                         QuoteNode(cuTile.Bounded(typemin(Int), -1))), Int)
    push!(entry, 2, Expr(:call, cuTile.Intrinsics.addi,
                         SSAValue(1), SSAValue(1)), Int)
    push!(entry, 3, Expr(:call, cuTile.Intrinsics.addi,
                         SSAValue(2), 2), Int)
    entry.terminator = ReturnNode(SSAValue(3))
    sci = StructuredIRCode(Any[Any, Int], Any[], entry, 3)

    r = cuTile.analyze_bounds(sci)
    @test cuTile.bounds(r, SSAValue(2)) == cuTile.TOP_RANGE
    @test cuTile.bounds(r, SSAValue(3)) == cuTile.TOP_RANGE

    cuTile.no_wrap_pass!(sci, r)
    @test length(entry.body[3].stmt.args) == 3
end

@testset "bounds — assume(Bounded) refines through a QuoteNode predicate" begin
    entry = Block()
    push!(entry, 1, Expr(:call, cuTile.Intrinsics.assume, Argument(2),
                         QuoteNode(cuTile.Bounded(0, 63))), Int)
    entry.terminator = ReturnNode(SSAValue(1))
    sci = StructuredIRCode(Any[Any, Int], Any[], entry, 1)

    r = cuTile.analyze_bounds(sci)
    @test cuTile.bounds(r, SSAValue(1)) == cuTile.IntRange(0, 63)
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
