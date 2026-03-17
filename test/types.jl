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
