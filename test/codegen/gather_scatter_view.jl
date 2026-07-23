# The low-level intrinsic tests live separately from the Julia `view` surface:
# this commit proves the Tile IR carrier, dimension conversion, mixed-rank
# index encoding, and conservative token behavior on their own.

spec2d = ct.ArraySpec{2}(16, true)
AT2d = ct.TileArray{Float32,2,spec2d}

@testset "GatherScatterView — row-major dimension conversion" begin
    @test @filecheck begin
        @check_label "entry"
        # Julia shape (4, 8) becomes Tile IR tile=(8x4). Julia dim 1 maps to
        # Tile IR dim 1 after the column-major/row-major reversal.
        @check "gather_scatter_view<tile=(8x4), padding_value = zero{{.*}}sparse_dim=1"
        @check "tile<4xi32>"
        code_tiled(Tuple{AT2d}; bytecode_version=v"13.3") do a
            tv = ct.Intrinsics.make_tensor_view(typeof(a), a.ptr, a.sizes, a.strides)
            view = ct.Intrinsics.make_gather_scatter_view(tv, (4, 8), 1, ct.PaddingMode.Zero)
            rows = ct.arange(4; start=0)
            tile = ct.Intrinsics.load_gather_scatter_view(
                view, nothing, nothing, (rows, Int32(0)), true)
            Base.donotdelete(tile)
            return
        end
    end
end

@testset "GatherScatterView — sparse stores retain loop tokens" begin
    @test @filecheck begin
        @check_label "entry"
        @check "gather_scatter_view<tile=(4x4){{.*}}sparse_dim=0"
        @check "iter_values"
        @check "store_view_tko"
        code_tiled(Tuple{AT2d, Int32}; bytecode_version=v"13.3") do a, n
            tv = ct.Intrinsics.make_tensor_view(typeof(a), a.ptr, a.sizes, a.strides)
            view = ct.Intrinsics.make_gather_scatter_view(
                tv, (4, 4), 2, ct.PaddingMode.Undetermined)
            cols = ct.arange(4; start=0)
            for _ in 1:n
                tile = ct.load(a, (1, 1), (4, 4))
                ct.Intrinsics.store_gather_scatter_view(
                    view, tile, nothing, nothing, (Int32(0), cols), true)
            end
            return
        end
    end
end

@testset "GatherScatterView — Julia view surface" begin
    @test @filecheck begin
        @check_label "entry"
        @check "assert"
        @check "gather_scatter_view<tile=(4x4), padding_value = zero{{.*}}sparse_dim=1"
        @check "load_view_tko"
        code_tiled(Tuple{AT2d, AT2d, Int32}; bytecode_version=v"13.3") do a, b, col_start
            rows = ct.arange(4; start=1, step=2)
            gather_view = @view a[rows, col_start:col_start + Int32(3)]
            tile = ct.load(gather_view, (4, 4); padding_mode=ct.PaddingMode.Zero)
            ct.store(b, (1, 1), tile)
            return
        end
    end

    # `@views` lowers to Base.maybeview, which must reach the same descriptor.
    @test @filecheck begin
        @check_label "entry"
        @check "gather_scatter_view<tile=(4x4){{.*}}sparse_dim=1"
        code_tiled(Tuple{AT2d}; bytecode_version=v"13.3") do a
            rows = ct.arange(4)
            gather_view = @views a[rows, Int32(1):Int32(4)]
            tile = ct.load(gather_view, (4, 4))
            Base.donotdelete(tile)
            return
        end
    end

    # Julia dimension 2 becomes Tile IR sparse_dim 0 and reverses the mixed
    # scalar/tile index tuple exactly once.
    @test @filecheck begin
        @check_label "entry"
        @check "gather_scatter_view<tile=(8x4){{.*}}sparse_dim=0"
        @check "tile<8xi32>"
        code_tiled(Tuple{AT2d}; bytecode_version=v"13.3") do a
            cols = ct.arange(8)
            gather_view = view(a, Int32(1):Int32(4), cols)
            tile = ct.load(gather_view, (4, 8))
            Base.donotdelete(tile)
            return
        end
    end

    # A `:` dense dimension starts at element 1 (constant 0); its extent comes
    # from the load shape rather than the range length.
    @test @filecheck begin
        @check_label "entry"
        @check "gather_scatter_view<tile=(8x4), padding_value = zero{{.*}}sparse_dim=1"
        @check "constant <i32: 0>"
        @check "load_view_tko"
        code_tiled(Tuple{AT2d}; bytecode_version=v"13.3") do a
            rows = ct.arange(4)
            gather_view = @view a[rows, :]
            tile = ct.load(gather_view, (4, 8); padding_mode=ct.PaddingMode.Zero)
            Base.donotdelete(tile)
            return
        end
    end

    # The pre-existing affine view route is still more specific than the
    # GatherScatterView fallback.
    @test @filecheck begin
        @check_label "entry"
        @check "make_partition_view"
        @check_not "gather_scatter_view"
        code_tiled(Tuple{AT2d}; bytecode_version=v"13.3") do a
            affine = view(a, Int32(1):Int32(4), Int32(1):Int32(4))
            tile = ct.load(affine, (1, 1), (4, 4))
            Base.donotdelete(tile)
            return
        end
    end

    @test_throws "tile dimension" code_tiled(Tuple{AT2d}; bytecode_version=v"13.3") do a
        rows = ct.arange(4)
        gather_view = view(a, rows, Int32(1):Int32(3))
        Base.donotdelete(ct.load(gather_view, (4, 3)))
        return
    end
    @test_throws "sparse index length" code_tiled(Tuple{AT2d}; bytecode_version=v"13.3") do a
        rows = ct.arange(8)
        gather_view = view(a, rows, Int32(1):Int32(4))
        Base.donotdelete(ct.load(gather_view, (4, 4)))
        return
    end
    # A load/store shape whose rank differs from the view is a compile-time error.
    @test_throws "shape rank" code_tiled(Tuple{AT2d}; bytecode_version=v"13.3") do a
        rows = ct.arange(4)
        gather_view = view(a, rows, Int32(1):Int32(4))
        Base.donotdelete(ct.load(gather_view, (4, 4, 4)))
        return
    end
end
