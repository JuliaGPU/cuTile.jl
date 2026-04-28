# Codegen tests for `no_wrap_pass!`: attaches `IntegerOverflow.{NSW,
# NUW, NW}` flags to `addi`/`subi`/`muli` ops when the bounds analysis
# proves the result fits in the destination width without wrap. The
# flag is forwarded by the addi/subi/muli emitters as the `overflow`
# kwarg of the corresponding `encode_*Op!` and surfaces in the Tile
# IR text as `overflow<nsw>` / `overflow<nuw>` / `overflow<no_wrap>`.

@testset "no_wrap — muli by literal zero is no_wrap" begin
    # `0 × anything = 0` is provably non-wrapping regardless of the
    # other operand's range. The slice path emits `muli(start_0,
    # stride)` where `start_0 == 0` for `a[1:N]`; the resulting muli
    # picks up the `no_wrap` flag.
    spec = ct.ArraySpec{1}(16, true)
    @test @filecheck begin
        @check_label "entry"
        code_tiled(Tuple{ct.TileArray{Float32,1,spec}}) do a
            sub = @view a[1:64]
            ct.store(sub, 1, ct.load(sub, 1, (16,)))
            return
        end
        @check "muli {{.*}} overflow<no_wrap>"
    end
end

@testset "no_wrap — runtime-arg subi gets no flag" begin
    # `j - i` with both args scalar parameters has top operand ranges,
    # so the pass can't prove no-wrap and leaves the op unflagged.
    spec = ct.ArraySpec{1}(16, true)
    @test @filecheck begin
        @check_label "entry"
        code_tiled(Tuple{ct.TileArray{Float32,1,spec}, Int32, Int32}) do a, i, j
            sub = @view a[i:j]
            ct.store(sub, 1, ct.load(sub, 1, (16,)))
            return
        end
        @check "subi"
        @check_not "subi {{.*}}overflow"
    end
end

#=============================================================================
 Direct-API tests for `prove_no_wrap`. Pin the lattice → flag mapping
 so future changes to the bounds analysis don't silently regress
 no-wrap inference.
=============================================================================#

@testset "no_wrap — prove_no_wrap on Int32 width" begin
    using cuTile: prove_no_wrap, IntRange
    using cuTile: Intrinsics

    # 32-bit signed range: [-2^31, 2^31 - 1]; unsigned: [0, 2^32 - 1].

    # Both operands fit and result is non-negative & small → NW.
    @test prove_no_wrap(Intrinsics.addi, 32, IntRange(0, 100), IntRange(0, 100)) ==
          cuTile.IntegerOverflow.NW

    # Result fits signed but lower bound is negative → NSW only.
    @test prove_no_wrap(Intrinsics.subi, 32, IntRange(0, 100), IntRange(0, 200)) ==
          cuTile.IntegerOverflow.NSW

    # 0 × anything: NW.
    @test prove_no_wrap(Intrinsics.muli, 32, IntRange(0, 0), IntRange(nothing, nothing)) ==
          cuTile.IntegerOverflow.NW
    @test prove_no_wrap(Intrinsics.muli, 32, IntRange(nothing, nothing), IntRange(0, 0)) ==
          cuTile.IntegerOverflow.NW

    # Operand range too wide to prove non-wrap → None.
    @test prove_no_wrap(Intrinsics.addi, 32,
                        IntRange(nothing, nothing), IntRange(0, 100)) ==
          cuTile.IntegerOverflow.None

    # Mul that would overflow Int32 → None even when ranges are
    # finite. 2^16 * 2^16 = 2^32 > Int32 max.
    @test prove_no_wrap(Intrinsics.muli, 32,
                        IntRange(0, 1 << 16), IntRange(0, 1 << 16)) ==
          cuTile.IntegerOverflow.None
end
