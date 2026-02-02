@testset "Tile" begin
    @test eltype(ct.Tile{Float32, (16,)}) == Float32
    @test eltype(ct.Tile{Float64, (32, 32)}) == Float64
    @test ct.tile_shape(ct.Tile{Float32, (16,)}) == (16,)
    @test ct.tile_shape(ct.Tile{Float32, (32, 32)}) == (32, 32)
end

@testset "mismatched shapes with + throws MethodError" begin
    tile_a = ct.Tile{Float32, (1, 128)}()
    tile_b = ct.Tile{Float32, (64, 1)}()

    # + should require same shapes, so this should fail
    @test_throws MethodError tile_a + tile_b

    # But .+ should work (broadcasting)
    result = tile_a .+ tile_b
    @test result isa ct.Tile{Float32, (64, 128)}
end

@testset "comparison operations" begin

@testset "float comparison operators" begin
    tile = ct.Tile{Float32, (16,)}()

    @test (tile .< tile) isa ct.Tile{Bool, (16,)}
    @test (tile .> tile) isa ct.Tile{Bool, (16,)}
    @test (tile .<= tile) isa ct.Tile{Bool, (16,)}
    @test (tile .>= tile) isa ct.Tile{Bool, (16,)}
    @test (tile .== tile) isa ct.Tile{Bool, (16,)}
    @test (tile .!= tile) isa ct.Tile{Bool, (16,)}
end

@testset "integer comparison operators" begin
    int_tile = ct.arange((16,), Int)

    @test (int_tile .< int_tile) isa ct.Tile{Bool, (16,)}
    @test (int_tile .> int_tile) isa ct.Tile{Bool, (16,)}
    @test (int_tile .<= int_tile) isa ct.Tile{Bool, (16,)}
    @test (int_tile .>= int_tile) isa ct.Tile{Bool, (16,)}
    @test (int_tile .== int_tile) isa ct.Tile{Bool, (16,)}
    @test (int_tile .!= int_tile) isa ct.Tile{Bool, (16,)}
end

@testset "tile vs scalar comparison" begin
    int_tile = ct.arange((16,), Int)
    float_tile = ct.Tile{Float32, (16,)}()

    @test (int_tile .< 10) isa ct.Tile{Bool, (16,)}
    @test (5 .< int_tile) isa ct.Tile{Bool, (16,)}

    @test (float_tile .< 2.0f0) isa ct.Tile{Bool, (16,)}
    @test (1.0f0 .> float_tile) isa ct.Tile{Bool, (16,)}
end

@testset "broadcast comparison shapes" begin
    tile_a = ct.Tile{Float32, (1, 16)}()
    tile_b = ct.Tile{Float32, (8, 1)}()

    result = tile_a .< tile_b
    @test result isa ct.Tile{Bool, (8, 16)}
end

end

@testset "power operations" begin

@testset "float tile .^ float tile" begin
    tile = ct.Tile{Float32, (16,)}()
    @test (tile .^ tile) isa ct.Tile{Float32, (16,)}
end

@testset "float tile .^ scalar" begin
    tile = ct.Tile{Float32, (16,)}()
    @test (tile .^ 2.0f0) isa ct.Tile{Float32, (16,)}
    @test (2.0f0 .^ tile) isa ct.Tile{Float32, (16,)}
end

@testset "broadcast power shapes" begin
    tile_a = ct.Tile{Float32, (1, 16)}()
    tile_b = ct.Tile{Float32, (8, 1)}()
    @test (tile_a .^ tile_b) isa ct.Tile{Float32, (8, 16)}
end

@testset "integer power dispatches through generic broadcast" begin
    int_tile = ct.arange((16,), Int)
    # Generic copy→map accepts this (no MethodError), but it will fail
    # at codegen time since there's no ^ overlay for integers.
    @test (int_tile .^ int_tile) isa ct.Tile
end

end

@testset "multi-arg map" begin
    a = ct.Tile{Float32, (16,)}()
    b = ct.Tile{Float32, (16,)}()
    c = ct.Tile{Float32, (16,)}()

    # Binary map
    @test map(+, a, b) isa ct.Tile{Float32, (16,)}

    # Ternary map
    @test map(fma, a, b, c) isa ct.Tile{Float32, (16,)}

    # Broadcasting goes through the .op path, not map directly
    @test (a .+ 1.0f0) isa ct.Tile{Float32, (16,)}
    @test (1.0f0 .+ a) isa ct.Tile{Float32, (16,)}

    # Broadcasting with different shapes goes through .op path
    row = ct.Tile{Float32, (4, 1)}()
    col = ct.Tile{Float32, (1, 16)}()
    @test (row .+ col) isa ct.Tile{Float32, (4, 16)}

    # Nested broadcast expression: a .+ b .* c
    @test (a .+ b .* c) isa ct.Tile{Float32, (16,)}
end
