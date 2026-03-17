@testset "Tile" begin
    @test eltype(ct.Tile{Float32, Tuple{16}}) == Float32
    @test eltype(ct.Tile{Float64, Tuple{32, 32}}) == Float64
    @test size(ct.Tile{Float32, Tuple{16}}) == (16,)
    @test size(ct.Tile{Float32, Tuple{32, 32}}) == (32, 32)
    @test size(ct.Tile{Float32, Tuple{8, 16}}, 1) == 8
    @test size(ct.Tile{Float32, Tuple{8, 16}}, 2) == 16
    @test ndims(ct.Tile{Float32, Tuple{16}}) == 1
    @test ndims(ct.Tile{Float32, Tuple{32, 32}}) == 2

    # length tests
    @test length(ct.Tile{Float32, Tuple{16}}) == 16
    @test length(ct.Tile{Float32, Tuple{16, 32}}) == 512

    # similar_type tests
    @test ct.similar_type(ct.Tile{Float32, Tuple{16}}, Float64) == ct.Tile{Float64, Tuple{16}}
    @test ct.similar_type(ct.Tile{Float32, Tuple{16}}, Int32, (8, 8)) == ct.Tile{Int32, Tuple{8, 8}}
    @test ct.similar_type(Float32, Int32) == Int32  # fallback
end

@testset "TileArray" begin
    @test eltype(ct.TileArray{Float32, 2}) == Float32
    @test eltype(ct.TileArray{Float64, 3}) == Float64
    @test ndims(ct.TileArray{Float32, 2}) == 2
    @test ndims(ct.TileArray{Int32, 1}) == 1

    # array_spec tests
    spec = ct.ArraySpec{2}(128, true)
    TA = ct.TileArray{Float32, 2, spec}
    @test ct.array_spec(TA) === spec
    @test ct.array_spec(ct.TileArray{Float32, 2}) === nothing
end

@testset "PartitionView" begin
    PV = cuTile.PartitionView{Float32, 2, Tuple{16, 32}}
    @test eltype(PV) == Float32
    @test ndims(PV) == 2
    @test size(PV) == (16, 32)
    @test size(PV, 1) == 16
    @test size(PV, 2) == 32
end

@testset "TensorView" begin
    @test eltype(cuTile.TensorView{Float32, 2}) == Float32
    @test eltype(cuTile.TensorView{Float64, 3}) == Float64
    @test ndims(cuTile.TensorView{Float32, 2}) == 2
    @test ndims(cuTile.TensorView{Int32, 1}) == 1
end

@testset "ByTarget" begin
    # Construction with pairs
    bt = ct.ByTarget(v"10.0" => 2, v"12.0" => 4)
    @test bt isa ct.ByTarget{Int}
    @test bt.default === nothing

    # Construction with default
    bt_d = ct.ByTarget(v"10.0" => 2; default=1)
    @test bt_d.default === Some(1)

    # resolve: exact match
    @test cuTile.resolve(bt, v"10.0") == 2
    @test cuTile.resolve(bt, v"12.0") == 4

    # resolve: no match, no default → nothing
    @test cuTile.resolve(bt, v"9.0") === nothing

    # resolve: no match, has default → default
    @test cuTile.resolve(bt_d, v"9.0") == 1

    # resolve pass-through for plain values
    @test cuTile.resolve(42, v"10.0") == 42
    @test cuTile.resolve(nothing, v"10.0") === nothing
end

@testset "validate_hint" begin
    # num_ctas: valid powers of 2 in [1, 16]
    for v in (1, 2, 4, 8, 16)
        cuTile.validate_hint(:num_ctas, v)  # should not throw
    end
    @test_throws ArgumentError cuTile.validate_hint(:num_ctas, 3)
    @test_throws ArgumentError cuTile.validate_hint(:num_ctas, 0)
    @test_throws ArgumentError cuTile.validate_hint(:num_ctas, 32)

    # occupancy: [1, 32]
    cuTile.validate_hint(:occupancy, 1)
    cuTile.validate_hint(:occupancy, 32)
    @test_throws ArgumentError cuTile.validate_hint(:occupancy, 0)
    @test_throws ArgumentError cuTile.validate_hint(:occupancy, 33)

    # opt_level: [0, 3]
    for v in 0:3
        cuTile.validate_hint(:opt_level, v)
    end
    @test_throws ArgumentError cuTile.validate_hint(:opt_level, -1)
    @test_throws ArgumentError cuTile.validate_hint(:opt_level, 4)

    # nothing is always valid (means "no override")
    cuTile.validate_hint(:num_ctas, nothing)
    cuTile.validate_hint(:occupancy, nothing)
    cuTile.validate_hint(:opt_level, nothing)
end

@testset "format_sm_arch" begin
    @test cuTile.format_sm_arch(v"10.0") == "sm_100"
    @test cuTile.format_sm_arch(v"12.0") == "sm_120"
    @test cuTile.format_sm_arch(v"9.0-a") == "sm_90a"
    @test cuTile.format_sm_arch(v"8.0") == "sm_80"
    @test_throws ArgumentError cuTile.format_sm_arch(v"10.0.1")
end

@testset "@compiler_options validation" begin
    # Invalid num_ctas (not power of 2) should throw at definition time
    @test_throws "num_ctas must be" @eval function _test_bad_ctas(a::ct.TileArray{Float32,1})
        ct.@compiler_options num_ctas=3
        return
    end

    # Invalid occupancy (out of range) should throw at definition time
    @test_throws "occupancy must be" @eval function _test_bad_occ(a::ct.TileArray{Float32,1})
        ct.@compiler_options occupancy=64
        return
    end

    # Invalid opt_level should throw at definition time
    @test_throws "opt_level must be" @eval function _test_bad_opt(a::ct.TileArray{Float32,1})
        ct.@compiler_options opt_level=5
        return
    end

    # ByTarget with invalid inner value should throw
    @test_throws "num_ctas must be" @eval function _test_bad_bt(a::ct.TileArray{Float32,1})
        ct.@compiler_options num_ctas=ct.ByTarget(v"10.0" => 3)
        return
    end

    # Valid plain hints should work fine
    @eval function _test_good_hints(a::ct.TileArray{Float32,1})
        ct.@compiler_options num_ctas=2 occupancy=8 opt_level=2
        return
    end

    # Valid ByTarget should work fine
    @eval function _test_good_bt(a::ct.TileArray{Float32,1})
        ct.@compiler_options num_ctas=ct.ByTarget(v"10.0" => 2, v"12.0" => 4; default=1)
        return
    end
end
