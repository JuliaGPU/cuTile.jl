# Codegen tests for `permutedims` / `transpose` / `reshape` on TileArrays.
# These operations derive a new TileArray with adjusted sizes/strides
# without touching memory; the new aggregate flows through a fresh
# `make_tensor_view` with the permuted/recomputed strides.

spec1d = ct.ArraySpec{1}(16, true)
spec2d = ct.ArraySpec{2}(16, true)

@testset "permutedims — 2D (2, 1)" begin
    # Source: contiguous (strides=[?,1]). After permutedims with (2,1),
    # contiguity flag drops (new stride[1] is the old stride[2], not 1) —
    # the new tensor_view is fully dynamic.
    @test @filecheck begin
        @check_label "entry"
        code_tiled(Tuple{ct.TileArray{Float32,2,spec2d}}) do a
            b = permutedims(a, (2, 1))
            t = ct.load(b, (1, 1), (4, 4))
            ct.store(a, (1, 1), t)
            return
        end
        @check "make_tensor_view"
        # Two tensor_views: the source (strides=[?,1]) and the permuted
        # one (strides=[?,?] — fully dynamic).
        @check "strides=[?,?]"
    end
end

@testset "transpose — 2D" begin
    # transpose(arr) === permutedims(arr, (2, 1)).
    @test @filecheck begin
        @check_label "entry"
        code_tiled(Tuple{ct.TileArray{Float32,2,spec2d}}) do a
            b = transpose(a)
            t = ct.load(b, (1, 1), (4, 4))
            ct.store(a, (1, 1), t)
            return
        end
        @check "make_tensor_view"
        @check "strides=[?,?]"
    end
end

@testset "reshape — 1D → 2D contiguous" begin
    # Reshape requires `Spec.contiguous`; with literal new_shape `(4, 4)`,
    # the new strides fold to constants 1 and 4. The contiguity assert
    # folds to `true` and is elided.
    @test @filecheck begin
        @check_label "entry"
        code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}}) do a
            b = reshape(a, (4, 4))
            t = ct.load(b, (1, 1), (4, 4))
            ct.store(a, 1, ct.reshape(t, (16,)))
            return
        end
        @check "constant <i32: 4>"
        @check "make_tensor_view"
        @check "strides=[?,1]"
        @check_not "assert {{.*}}reshape"
    end
end

@testset "reshape — non-contiguous source emits failing assert" begin
    # Non-contiguous spec: the contiguity check folds to `false`, so an
    # unconditional `assert <i1: false>, msg` is emitted — the kernel will
    # always abort at this point if reached. (We can't test the runtime
    # abort itself; cuTile doesn't catch device-side exceptions.)
    spec_noncontig = ct.ArraySpec{2}(0, false)
    @test @filecheck begin
        @check_label "entry"
        code_tiled(Tuple{ct.TileArray{Float32,2,spec_noncontig}}) do a
            b = reshape(a, (16,))
            t = ct.load(b, 1, (16,))
            ct.store(a, (1, 1), ct.reshape(t, (4, 4)))
            return
        end
        @check "constant <i1: false>"
        @check "assert {{.*}}reshape: TileArray must be contiguous"
    end
end

@testset "eachtile — partition and strided views" begin
    # Equal shape and step preserves the pre-v13.3 PartitionView path.
    @test @filecheck begin
        @check_label "entry"
        @check "make_partition_view"
        @check_not "make_strided_view"
        code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}};
                   bytecode_version=v"13.2") do a, b
            src = eachtile(a, (8,))
            dst = eachtile(b, (8,); step=(8,))
            dst[1] = src[1]
            return
        end
    end

    # Tile IR is row-major, so Julia shape/step `(8,4)`/`(3,2)` are reversed
    # in the emitted StridedView type.
    @test @filecheck begin
        @check_label "entry"
        @check "make_strided_view"
        @check "tile=(4x8), traversal_strides=[2,3]"
        @check "load_view_tko"
        @check "store_view_tko"
        code_tiled(Tuple{ct.TileArray{Float32,2,spec2d}, ct.TileArray{Float32,2,spec2d}};
                   bytecode_version=v"13.3") do a, b
            src = eachtile(a, (8, 4); step=(3, 2), padding_mode=ct.PaddingMode.Zero)
            dst = eachtile(b, (8, 4); step=(3, 2))
            tile = ct.load(src, (2, 3); check_bounds=true, latency=3, allow_tma=false)
            ct.store(dst, (2, 3), tile; check_bounds=true, latency=3, allow_tma=false)
            return
        end
    end

    @test_throws "v13.3+" code_tiled(
        Tuple{ct.TileArray{Float32,1,spec1d}}; bytecode_version=v"13.2") do a
            tiles = eachtile(a, (8,); step=(4,))
            ct.store(tiles, 1, ct.load(tiles, 1))
            return
        end
end

@testset "eachtile — device size queries the backend index space" begin
    # `size(tiles, d)` computes `cld` on the host, but in kernels it is
    # overlaid to defer to `get_index_space_shape` (matching cuTile Python's
    # `num_tiles` lowering). Feed the result into a load index so it isn't
    # DCE'd.
    @test @filecheck begin
        @check_label "entry"
        @check "make_strided_view"
        @check "get_index_space_shape"
        @check "load_view_tko"
        @check "store_view_tko"
        code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}}; bytecode_version=v"13.3") do a
            tiles = eachtile(a, (8,); step=(4,))
            n = size(tiles, 1)
            ct.store(a, 1, ct.load(tiles, n))
            return
        end
    end
end
