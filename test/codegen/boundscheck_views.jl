# Tile memory bounds handling is distinct from Julia's protective bounds checks:
# `@inbounds` must not discard a partial tile's padding or store clipping.

const view_spec = ct.ArraySpec{1}(64, true)
const ViewArray = ct.TileArray{Float32, 1, Int32, view_spec}
const view2_spec = ct.ArraySpec{2}(64, true)
const ViewArray2 = ct.TileArray{Float32, 2, Int32, view2_spec}

direct_plain(a) = (ct.store(a, ct.bid(1),
                            ct.load(a, ct.bid(1), (16,); padding_mode=ct.PaddingMode.Zero)); return)
direct_inbounds(a) = (@inbounds ct.store(a, ct.bid(1),
                                          ct.load(a, ct.bid(1), (16,); padding_mode=ct.PaddingMode.Zero)); return)
direct_explicit_checked(a) = (@inbounds ct.store(a, ct.bid(1),
                                                   ct.load(a, ct.bid(1), (16,);
                                                           padding_mode=ct.PaddingMode.Zero,
                                                           check_bounds=true)); return)
direct_explicit_unchecked(a) = (ct.store(a, ct.bid(1),
                                          ct.load(a, ct.bid(1), (16,);
                                                  padding_mode=ct.PaddingMode.Zero,
                                                  check_bounds=false);
                                check_bounds=false); return)

function tiled_plain(a)
    tiles = ct.eachtile(a, (16,); padding_mode=ct.PaddingMode.Zero)
    tiles[ct.bid(1)] = tiles[ct.bid(1)]
    return
end
function tiled_inbounds(a)
    @inbounds begin
        tiles = ct.eachtile(a, (16,); padding_mode=ct.PaddingMode.Zero)
        tiles[ct.bid(1)] = tiles[ct.bid(1)]
    end
    return
end
function strided_inbounds(a)
    @inbounds begin
        tiles = ct.eachtile(a, (16,); step=(32,), padding_mode=ct.PaddingMode.Zero)
        tiles[ct.bid(1)] = tiles[ct.bid(1)]
    end
    return
end

function gather_scatter_plain(a)
    rows = ct.arange(4)
    view = @view a[rows, Int32(1):Int32(4)]
    tile = ct.load(view, (4, 4); padding_mode=ct.PaddingMode.Zero)
    ct.store(view, tile)
    return
end
function gather_scatter_inbounds(a)
    @inbounds begin
        rows = ct.arange(4)
        view = @view a[rows, Int32(1):Int32(4)]
        tile = ct.load(view, (4, 4); padding_mode=ct.PaddingMode.Zero)
        ct.store(view, tile)
    end
    return
end

scalar_plain(a) = (a[1] = a[2]; return)
scalar_inbounds(a) = (@inbounds (a[1] = a[2]); return)

function view_ir(@nospecialize(f), @nospecialize(argtypes); bytecode_version)
    sprint(io -> code_tiled(io, f, argtypes; bytecode_version))
end

unchecked_view_ops(f, argtypes; bytecode_version) =
    count(line -> occursin("inbounds = [true", line),
          split(view_ir(f, argtypes; bytecode_version), '\n'))

has_padding(f, argtypes; bytecode_version) =
    occursin("padding_value = zero", view_ir(f, argtypes; bytecode_version))

@testset "Tile views do not inherit @inbounds" begin
    one_dim = Tuple{ViewArray}
    two_dim = Tuple{ViewArray2}

    # v13.3 cannot encode unchecked views, but `@inbounds` is irrelevant to
    # Tile memory operations: all of these must stay checked and padded.
    for (f, argtypes) in ((direct_plain, one_dim), (direct_inbounds, one_dim),
                          (direct_explicit_checked, one_dim),
                          (tiled_plain, one_dim), (tiled_inbounds, one_dim),
                          (strided_inbounds, one_dim),
                          (gather_scatter_plain, two_dim),
                          (gather_scatter_inbounds, two_dim),
                          (scalar_plain, one_dim), (scalar_inbounds, one_dim))
        @test unchecked_view_ops(f, argtypes; bytecode_version=v"13.3") == 0
    end

    for f in (direct_plain, direct_inbounds, direct_explicit_checked,
              tiled_plain, tiled_inbounds, strided_inbounds)
        @test has_padding(f, one_dim; bytecode_version=v"13.3")
    end
    for f in (gather_scatter_plain, gather_scatter_inbounds)
        @test has_padding(f, two_dim; bytecode_version=v"13.3")
    end

    if ct.bytecode_version() >= v"13.4"
        # An explicit false is the only way to select the v13.4 `inbounds`
        # encoding. It also drops padding, which Tile IR rejects on an
        # unchecked view.
        @test unchecked_view_ops(direct_explicit_unchecked, one_dim;
                                 bytecode_version=ct.bytecode_version()) == 2
        @test !has_padding(direct_explicit_unchecked, one_dim;
                           bytecode_version=ct.bytecode_version())
    end
end
