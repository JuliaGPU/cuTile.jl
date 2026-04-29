# Codegen tests for the `analyze_assume_info` aggregator + the
# `make_tensor_view` codegen path that wraps each operand `Value` with
# `encode_AssumeOp!`. Facts come from the TileArray-type `ArraySpec`
# plus the divisibility dataflow analysis (analysis/divisibility.jl)
# and bounds dataflow analysis (analysis/bounds.jl), so derived
# TileArrays (slices, permutes, reshapes) get assumes too — recovering
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
    spec2d = ct.ArraySpec{2}(16, true, (4, 0), (16, 8))
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
    spec2d = ct.ArraySpec{2}(16, true, (4, 0), (16, 8))
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
    spec1d = ct.ArraySpec{1}(128, true, (4,), (16,))
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

@testset "assume — literal slice with zero offset preserves full alignment" begin
    # `a[1:64]` has start_0 == 0, so offset == 0 bytes; the dataflow
    # treats the literal `0` as ∞-divisible, so `gcd(spec.alignment, 0)`
    # == spec.alignment. The slice ptr inherits the source's full
    # 128-byte alignment.
    spec1d = ct.ArraySpec{1}(128, true, (4,), (16,))
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
