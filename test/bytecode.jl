make_builder(version) = let strings = cuTile.StringTable(), constants = cuTile.ConstantTable()
    types = cuTile.TypeTable(; version)
    cuTile.CodeBuilder(strings, constants, types; version)
end

@testset "Tile IR v13.4 encodings" begin
    @test v"13.4" in cuTile.SUPPORTED_BYTECODE_VERSIONS

    tt33 = cuTile.TypeTable(; version=v"13.3")
    tt34 = cuTile.TypeTable(; version=v"13.4")
    cuTile.pointer_type!(tt33, cuTile.I32(tt33))
    cuTile.pointer_type!(tt34, cuTile.I32(tt34))
    @test last(cuTile.items(tt33)).first == UInt8[0x0c, 0x01]
    @test last(cuTile.items(tt34)).first == UInt8[0x0c, 0x00, 0x01]

    cb33 = make_builder(v"13.3")
    cb34 = make_builder(v"13.4")
    cuTile.encode_FToIOp!(cb33, cuTile.TypeId(1), cuTile.Value(0))
    cuTile.encode_FToIOp!(cb34, cuTile.TypeId(1), cuTile.Value(0))
    @test cb33.buf == UInt8[43, 1, 1, 6, 0]
    @test cb34.buf == UInt8[43, 1, 0, 1, 6, 0]

    cb33 = make_builder(v"13.3")
    cb34 = make_builder(v"13.4")
    args = (cuTile.TypeId(1), cuTile.TypeId(2), cuTile.Value(3),
            [cuTile.Value(4), cuTile.Value(5)])
    cuTile.encode_LoadViewTkoOp!(cb33, args...)
    cuTile.encode_LoadViewTkoOp!(cb34, args...; inbounds=[true, true])
    @test cb33.buf == UInt8[62, 2, 1, 2, 0, 0, 3, 2, 4, 5]
    @test cb34.buf == UInt8[62, 2, 1, 2, 0, 0, 2, 1, 1, 3, 2, 4, 5]
    @test_throws "requires Tile IR v13.4+" begin
        cb = make_builder(v"13.3")
        cuTile.encode_LoadViewTkoOp!(cb, args...; inbounds=[true, true])
    end

    cb = make_builder(v"13.4")
    result = cuTile.encode_InsertOp!(cb, cuTile.TypeId(1), cuTile.Value(0),
                                    cuTile.Value(1), [cuTile.Value(2), cuTile.Value(3)])
    @test result == cuTile.Value(0)
    @test cb.buf == UInt8[118, 1, 1, 4, 0, 1, 2, 3]
    @test_throws "requires Tile IR v13.4+" begin
        cb = make_builder(v"13.3")
        cuTile.encode_InsertOp!(cb, cuTile.TypeId(1), cuTile.Value(0),
                               cuTile.Value(1), cuTile.Value[])
    end
end

@testset "Tile IR v13.3 StridedView encodings" begin
    tt = cuTile.TypeTable(; version=v"13.3")
    tensor_view = cuTile.TypeId(7)
    strided = cuTile.strided_view_type!(
        tt, cuTile.RowMajorShape([4, 8]), cuTile.RowMajorShape([2, 3]),
        tensor_view, [0, 1], cuTile.PaddingValue.Zero)
    @test strided == cuTile.TypeId(2)
    @test last(cuTile.items(tt)).first == UInt8[
        0x15, 0x01,
        0x02, 0x04, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00,
        0x02, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
        0x07,
        0x02, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
        0x00,
    ]
    @test_throws "v13.3+" cuTile.strided_view_type!(
        cuTile.TypeTable(; version=v"13.2"), cuTile.RowMajorShape([4]),
        cuTile.RowMajorShape([2]), tensor_view, [0], cuTile.PaddingValue.Missing)

    cb = make_builder(v"13.3")
    @test cuTile.encode_MakeStridedViewOp!(cb, cuTile.TypeId(5), cuTile.Value(2)) == cuTile.Value(0)
    @test cb.buf == UInt8[116, 5, 2]
    @test_throws "v13.3+" cuTile.encode_MakeStridedViewOp!(
        make_builder(v"13.2"), cuTile.TypeId(5), cuTile.Value(2))
end
