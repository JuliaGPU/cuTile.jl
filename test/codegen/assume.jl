# Codegen tests for the consumer-driven `AssumeOp` emission path
# (`make_tensor_view`, `load_ptr_tko`, `store_ptr_tko`) plus the
# kernel-arg-entry wrap. Chains are derived on demand by `op_predicates`
# / `arg_chain` (analysis/assume.jl) from the TileArray-type `ArraySpec`
# combined with the divisibility (analysis/divisibility.jl) and bounds
# (analysis/bounds.jl) dataflow analyses, so derived TileArrays
# (slices, permutes, reshapes) get assumes too — recovering
# through-arithmetic facts that the conservative `sliced_arraytype`
# etc. drop.

@testset "assume — kernel-arg alignment" begin
    spec1d = ct.ArraySpec{1}(128, true)
    @test @filecheck begin
        @check_label "entry"
        code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}}) do a
            ct.store(a, 1, ct.load(a, 1, (16,)))
            return
        end
        @check "assume div_by<128>"
    end

    # alignment=0 → no DivBy on the pointer (still bounded(0) on size).
    spec_unaligned = ct.ArraySpec{1}(0, false)
    @test @filecheck begin
        @check_label "entry"
        code_tiled(Tuple{ct.TileArray{Float32,1,spec_unaligned}}) do a
            ct.store(a, 1, ct.load(a, 1, (16,)))
            return
        end
        @check_not "assume div_by"
        @check "assume bounded<0, ?>"
    end
end

@testset "assume — per-axis shape divisibility" begin
    # stride_div_by left at 0 on the contiguous axis (consistent with
    # stride[1]=1); the test only asserts on shape facts.
    spec2d = ct.ArraySpec{2}(16, true, (0, 0), (16, 8))
    @test @filecheck begin
        @check_label "entry"
        code_tiled(Tuple{ct.TileArray{Float32,2,spec2d}}) do a
            ct.store(a, (1, 1), ct.load(a, (1, 1), (16, 16)))
            return
        end
        @check "assume div_by<16>"     # ptr alignment
        @check "assume bounded<0, ?>"  # size[1] bound
        @check "assume div_by<16>"     # size[1] DivBy
        @check "assume bounded<0, ?>"  # size[2] bound
        @check "assume div_by<8>"      # size[2] DivBy
    end
end

@testset "assume — strides skip the contiguous axis" begin
    # contiguous=true: stride[1]=1 statically; that operand never enters
    # the bytecode signature, so no assume is emitted for it.
    spec2d = ct.ArraySpec{2}(16, true, (0, 0), (16, 8))
    @test @filecheck begin
        @check_label "entry"
        code_tiled(Tuple{ct.TileArray{Float32,2,spec2d}}) do a
            ct.store(a, (1, 1), ct.load(a, (1, 1), (16, 16)))
            return
        end
        @check "make_tensor_view"
        @check "strides=[?,1]"
    end
end

@testset "assume — slice recovers ptr alignment via dataflow" begin
    # Source has 128-byte ptr alignment and stride_div_by[1]=4 (so the
    # slice offset is `start * stride` divisible by 4 elements = 16 bytes).
    # The slice's TileArray type has alignment=0 (conservative), but the
    # divisibility dataflow recovers gcd(128, 16) = 16 on the offset ptr.
    # Uses contiguous=false: `stride_div_by[1]>1` is only physically
    # consistent with a non-unit stride.
    spec1d = ct.ArraySpec{1}(128, false, (4,), (16,))
    @test @filecheck begin
        @check_label "entry"
        code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, Int32, Int32}) do a, i, j
            sub = @view a[i:j]
            ct.store(sub, 1, ct.load(sub, 1, (16,)))
            return
        end
        # The offset-derived ptr (used by the slice's make_tensor_view)
        # gets a div_by recovered from the dataflow.
        @check "offset"
        @check "assume div_by<16>"
    end
end

@testset "assume — literal slice's size is a static constant" begin
    # Source has shape_div_by[1]=16. Slicing with a literal range
    # (e.g. `1:64`) folds the new axis size to a compile-time constant,
    # so `make_tensor_view`'s shape operand is the literal `64` directly
    # — no `assume bounded<0,?>` / `assume div_by<…>` wrap is needed
    # because the literal already carries full information.
    spec1d = ct.ArraySpec{1}(128, true, (0,), (16,))
    @test @filecheck begin
        @check_label "entry"
        code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}}) do a
            sub = @view a[1:64]
            ct.store(sub, 1, ct.load(sub, 1, (16,)))
            return
        end
        @check "constant <i32: 64>"
        @check "make_tensor_view"
    end
end

@testset "assume — literal slice with zero offset preserves full alignment" begin
    # `a[1:64]` has start_0 == 0, so offset == 0 bytes; the dataflow
    # treats the literal `0` as ∞-divisible, so `gcd(spec.alignment, 0)`
    # == spec.alignment. The slice ptr inherits the source's full
    # 128-byte alignment. Uses contiguous=false: `stride_div_by[1]>1` is
    # only physically consistent with a non-unit stride.
    spec1d = ct.ArraySpec{1}(128, false, (4,), (16,))
    @test @filecheck begin
        @check_label "entry"
        code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}}) do a
            sub = @view a[1:64]
            ct.store(sub, 1, ct.load(sub, 1, (16,)))
            return
        end
        @check "offset"
        @check "assume div_by<128>"
    end
end

@testset "assume — slice ptr alignment from source alone" begin
    # Source has 16-byte alignment but no stride_div_by. The slice offset
    # has unknown divisibility; gcd(16, off_bytes=4) = 4 is the best we
    # can prove from the source alignment alone.
    spec1d = ct.ArraySpec{1}(16, true, (0,), (0,))
    @test @filecheck begin
        @check_label "entry"
        code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, Int32, Int32}) do a, i, j
            sub = @view a[i:j]
            ct.store(sub, 1, ct.load(sub, 1, (16,)))
            return
        end
        @check "offset"
        @check "assume div_by<4>"
        @check "assume bounded<0, ?>"
    end
end

@testset "assume — non-TileArray args don't get assumes" begin
    spec1d = ct.ArraySpec{1}(16, true)
    @test @filecheck begin
        @check_label "entry"
        code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, Int32}) do a, n
            ct.store(a, n, ct.load(a, n, (16,)))
            return
        end
        @check "assume div_by<16>"
        @check "assume bounded<0, ?>"
    end
end

@testset "assume — kernel-arg ptr wrap survives offset for gather/scatter" begin
    # Pure-gather/scatter kernel: no MTV consumes the kernel-arg ptr, so
    # the only path that can attach `spec.alignment` to the base pointer
    # is the kernel-arg-entry wrap (`apply_arg_assume_predicates!`). That
    # base-alignment fact still flows through `reshape` → `broadcast` →
    # `offset` to the gather/scatter consumer, but we deliberately do NOT
    # emit a per-element divby assume on the resulting tile-of-pointers:
    # the natural pointee alignment is vacuous for tileiras and stamping
    # it onto the IR defeats wide-store coalescing. Mirrors cuTile
    # Python's `PointerOffset` rule.
    spec1d = ct.ArraySpec{1}(128, true)
    @test @filecheck begin
        @check_label "entry"
        code_tiled(Tuple{ct.TileArray{Float32,1,spec1d},
                         ct.TileArray{Float32,1,spec1d}}) do a, b
            indices = ct.arange(16)
            tile = ct.gather(a, indices)
            ct.scatter(b, indices, tile)
            return
        end
        # Base alignment on each kernel-arg ptr (entry wrap).
        @check "assume div_by<128>"
        @check "assume div_by<128>"
        # No divby assume on the post-offset tile-of-pointers — tileiras
        # walks the offset SSA itself.
        @check_not "assume div_by"
        @check "load_ptr_tko"
        @check_not "assume div_by"
        @check "store_ptr_tko"
    end
end

@testset "assume — shared ptr Value is wrapped once across consumers" begin
    # Two MTVs (`ct.load` + `ct.store`) plus a gather all source from
    # the same kernel-arg ptr. The entry wrap puts one `assume div_by<128>`
    # on it; the per-`Value` cache (`ctx.assume_wrapped`) ensures the
    # MTV consumer wraps don't re-emit the same predicate on the same
    # source. The post-offset gather ptr is a tile-of-pointers and the
    # `PointerOffset` divby rule deliberately suppresses any assume there.
    spec1d = ct.ArraySpec{1}(128, true)
    @test @filecheck begin
        @check_label "entry"
        code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}}) do a
            tile = ct.load(a, 1, (16,))
            indices = ct.arange(16)
            tile2 = ct.gather(a, indices)
            ct.store(a, 1, tile + tile2)
            return
        end
        # Exactly one `assume div_by<128>` despite three consumers of
        # the same kernel-arg ptr.
        @check "assume div_by<128>"
        @check_not "assume div_by<128>"
    end
end

@testset "assume — user assume_divisible_by" begin
    spec1d = ct.ArraySpec{1}(16, true)
    @test @filecheck begin
        @check_label "entry"
        code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}}) do a
            n = ct.bid(1) * Int32(128)
            n = ct.assume_divisible_by(n, 128)
            @check "muli"
            @check "assume div_by<128>"
            ct.store(a, n, ct.load(a, n, (64,)))
            return
        end
    end

    # Non-positive divisor is rejected
    @test_throws "assume: DivBy requires a positive divisor, got 0" begin
        code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}}) do a
            n = ct.assume_divisible_by(ct.bid(1), 0)
            ct.store(a, n, ct.load(a, n, (64,)))
            return
        end
    end


    @test_throws "contradicts a known constant" begin
        code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}}) do a
            n = ct.assume_divisible_by(Int32(5), 4)
            ct.store(a, n, ct.load(a, n, (64,)))
            return
        end
    end

    @test @filecheck begin
        @check_label "entry"
        code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}}) do a
            n = ct.assume_divisible_by(Int32(8), 4)
            @check_not "assume div_by<4>"
            ct.store(a, n, ct.load(a, n, (64,)))
            return
        end
    end
end
