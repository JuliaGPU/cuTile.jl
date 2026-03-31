spec = ct.ArraySpec{1}(16, true)
TT3 = Tuple{ct.TileArray{Float32,1,spec}, ct.TileArray{Float32,1,spec}, ct.TileArray{Float32,1,spec}}

function reflect_vadd(a, b, c)
    pid = ct.bid(1)
    tile_a = ct.load(a; index=pid, shape=(16,))
    tile_b = ct.load(b; index=pid, shape=(16,))
    ct.store(c; index=pid, tile=tile_a + tile_b)
    return
end

@testset "code_typed" begin
    @test @filecheck begin
        @check "get_tile_block_id"
        @check "load_partition_view"
        @check "addf"
        @check "store_partition_view"
        ct.code_typed(reflect_vadd, TT3)
    end
end

@testset "code_structured" begin
    @testset "optimize=false" begin
        @test @filecheck begin
            @check "StructuredIRCode"
            @check "get_tile_block_id"
            # Core intrinsics survive without optimization
            @check "Base.add_int"
            @check "addf"
            ct.code_structured(reflect_vadd, TT3; optimize=false)
        end
    end

    @testset "optimize=true" begin
        @test @filecheck begin
            @check "StructuredIRCode"
            # Token ordering inserts make_token
            @check "MakeTokenNode"
            @check "get_tile_block_id"
            @check "addf"
            # Core intrinsics lowered by normalize pass
            @check_not "Base.add_int"
            ct.code_structured(reflect_vadd, TT3)
        end
    end
end

@testset "Constant args" begin
    const_spec = ct.ArraySpec{1}(128, true, (0,), (32,))
    ConstTT = Tuple{ct.TileArray{Float32,1,const_spec}, ct.TileArray{Float32,1,const_spec},
                    ct.TileArray{Float32,1,const_spec}, ct.Constant{Int64, 16}}

    function reflect_const_vadd(a, b, c, tile_size::Int)
        pid = ct.bid(1)
        tile_a = ct.load(a; index=pid, shape=(tile_size,))
        tile_b = ct.load(b; index=pid, shape=(tile_size,))
        ct.store(c; index=pid, tile=tile_a + tile_b)
        return
    end

    @testset "code_typed" begin
        @test @filecheck begin
            # Constant folded: shape=(16,) appears as literal tuple
            @check "make_partition_view"
            @check "(16,)"
            @check "Tuple{16}"
            ct.code_typed(reflect_const_vadd, ConstTT)
        end
    end

    @testset "code_structured" begin
        @test @filecheck begin
            @check "make_partition_view"
            @check "(16,)"
            @check "Tuple{16}"
            ct.code_structured(reflect_const_vadd, ConstTT; optimize=false)
        end
    end
end
