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

@testset "mismatched shapes with + throws MethodError" begin
    tile_a = ct.Tile{Float32, Tuple{1, 128}}()
    tile_b = ct.Tile{Float32, Tuple{64, 1}}()

    # + should require same shapes, so this should fail
    @test_throws MethodError tile_a + tile_b

    # But .+ should work (broadcasting)
    result = tile_a .+ tile_b
    @test result isa ct.Tile{Float32, Tuple{64, 128}}
end

@testset "comparison operations" begin

@testset "float comparison operators" begin
    tile = ct.Tile{Float32, Tuple{16}}()

    @test (tile .< tile) isa ct.Tile{Bool, Tuple{16}}
    @test (tile .> tile) isa ct.Tile{Bool, Tuple{16}}
    @test (tile .<= tile) isa ct.Tile{Bool, Tuple{16}}
    @test (tile .>= tile) isa ct.Tile{Bool, Tuple{16}}
    @test (tile .== tile) isa ct.Tile{Bool, Tuple{16}}
    @test (tile .!= tile) isa ct.Tile{Bool, Tuple{16}}
end

@testset "integer comparison operators" begin
    int_tile = ct.arange((16,), Int)

    @test (int_tile .< int_tile) isa ct.Tile{Bool, Tuple{16}}
    @test (int_tile .> int_tile) isa ct.Tile{Bool, Tuple{16}}
    @test (int_tile .<= int_tile) isa ct.Tile{Bool, Tuple{16}}
    @test (int_tile .>= int_tile) isa ct.Tile{Bool, Tuple{16}}
    @test (int_tile .== int_tile) isa ct.Tile{Bool, Tuple{16}}
    @test (int_tile .!= int_tile) isa ct.Tile{Bool, Tuple{16}}
end

@testset "tile vs scalar comparison" begin
    int_tile = ct.arange((16,), Int)
    float_tile = ct.Tile{Float32, Tuple{16}}()

    @test (int_tile .< 10) isa ct.Tile{Bool, Tuple{16}}
    @test (5 .< int_tile) isa ct.Tile{Bool, Tuple{16}}

    @test (float_tile .< 2.0f0) isa ct.Tile{Bool, Tuple{16}}
    @test (1.0f0 .> float_tile) isa ct.Tile{Bool, Tuple{16}}
end

@testset "broadcast comparison shapes" begin
    tile_a = ct.Tile{Float32, Tuple{1, 16}}()
    tile_b = ct.Tile{Float32, Tuple{8, 1}}()

    result = tile_a .< tile_b
    @test result isa ct.Tile{Bool, Tuple{8, 16}}
end

end

@testset "power operations" begin

@testset "float tile .^ float tile" begin
    tile = ct.Tile{Float32, Tuple{16}}()
    @test (tile .^ tile) isa ct.Tile{Float32, Tuple{16}}
end

@testset "float tile .^ scalar" begin
    tile = ct.Tile{Float32, Tuple{16}}()
    @test (tile .^ 2.0f0) isa ct.Tile{Float32, Tuple{16}}
    @test (2.0f0 .^ tile) isa ct.Tile{Float32, Tuple{16}}
end

@testset "broadcast power shapes" begin
    tile_a = ct.Tile{Float32, Tuple{1, 16}}()
    tile_b = ct.Tile{Float32, Tuple{8, 1}}()
    @test (tile_a .^ tile_b) isa ct.Tile{Float32, Tuple{8, 16}}
end

@testset "integer power dispatches through generic broadcast" begin
    int_tile = ct.arange((16,), Int)
    # Generic copy→map accepts this (no MethodError), but it will fail
    # at codegen time since there's no ^ overlay for integers.
    @test (int_tile .^ int_tile) isa ct.Tile
end

end

@testset "multi-arg map" begin
    a = ct.Tile{Float32, Tuple{16}}()
    b = ct.Tile{Float32, Tuple{16}}()
    c = ct.Tile{Float32, Tuple{16}}()

    # Binary map
    @test map(+, a, b) isa ct.Tile{Float32, Tuple{16}}

    # Ternary map
    @test map(fma, a, b, c) isa ct.Tile{Float32, Tuple{16}}

    # Broadcasting goes through the .op path, not map directly
    @test (a .+ 1.0f0) isa ct.Tile{Float32, Tuple{16}}
    @test (1.0f0 .+ a) isa ct.Tile{Float32, Tuple{16}}

    # Broadcasting with different shapes goes through .op path
    row = ct.Tile{Float32, Tuple{4, 1}}()
    col = ct.Tile{Float32, Tuple{1, 16}}()
    @test (row .+ col) isa ct.Tile{Float32, Tuple{4, 16}}

    # Nested broadcast expression: a .+ b .* c
    @test (a .+ b .* c) isa ct.Tile{Float32, Tuple{16}}
end
