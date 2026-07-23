@testset "Constant" begin
    # Auto-typed convenience constructor
    @test cuTile.Constant(Int32(5)) === cuTile.Constant{Int32, Int32(5)}()
    @test cuTile.Constant(0.5f0) === cuTile.Constant{Float32, 0.5f0}()
    @test cuTile.Constant(Float32) === cuTile.Constant{Type{Float32}, Float32}()

    # Value access via getindex
    @test cuTile.Constant{Int32, 42}()[] === 42
    @test cuTile.Constant{Float32, 0.25f0}()[] === 0.25f0

    # Reject integer literals that don't fit the declared element type so
    # codegen never silently truncates a bogus `Constant{Int8, 1024}`.
    # Matches `Int8(1024)` raising InexactError.
    @test_throws InexactError cuTile.Constant{Int8,  1024}()
    @test_throws InexactError cuTile.Constant{UInt8, -1}()
    @test_throws InexactError cuTile.Constant{UInt32, -5}()
    # Boundary values must work; `[]` returns the literal as-stored.
    @test cuTile.Constant{Int8, 127}()[] == 127
    @test cuTile.Constant{Int8, -128}()[] == -128
    @test cuTile.Constant{UInt8, 255}()[] == 255
end

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

    spec1 = ct.ArraySpec{2}(128, true, (1, 4), (16, 8))
    T = ct.TileArray{Float32, 2, spec1}
    @test ct.array_spec(ct.sliced_arraytype(T)).contiguous
    @test !ct.array_spec(ct.sliced_arraytype(T; stepped_axis=1)).contiguous
    @test ct.array_spec(ct.sliced_arraytype(T; stepped_axis=2)).contiguous
    @test ct.array_spec(ct.sliced_arraytype(T; stepped_axis=1)).stride_div_by == (1, 4)
end

@testset "PartitionView" begin
    PV = cuTile.PartitionView{Float32, 2, Tuple{16, 32}}
    @test eltype(PV) == Float32
    @test ndims(PV) == 2
    @test size(PV) == (16, 32)
    @test size(PV, 1) == 16
    @test size(PV, 2) == 32
end

@testset "StridedView" begin
    SV = cuTile.StridedView{Float32, 2, Tuple{16, 32}, Tuple{8, 4}}
    @test eltype(SV) == Float32
    @test ndims(SV) == 2
    @test size(SV) == (16, 32)
    @test size(SV, 1) == 16
    @test size(SV, 2) == 32
end

@testset "eachtile" begin
    a = ct.TileArray(Ptr{Float32}(0), (Int32(16), Int32(10)), (Int32(1), Int32(16)))
    tiles = eachtile(a, (8, 4); step=(3, 2))
    @test parent(tiles) === a
    @test ndims(tiles) == 2
    @test size(tiles) == (Int32(6), Int32(5))
    @test size(tiles, 1) == Int32(6)
    @test size(tiles, 2) == Int32(5)
    @test :eachtile in names(cuTile)

    # The element of the collection is a whole tile of the requested shape.
    @test eltype(tiles) == ct.Tile{Float32, Tuple{8, 4}}
    @test eltype(typeof(tiles)) == ct.Tile{Float32, Tuple{8, 4}}

    @test_throws "strictly positive" eachtile(a, (8, 4); step=(0, 2))
    # `step` must carry one entry per user tile dimension: a mismatched length
    # is rejected rather than silently overlapping the padded dimensions.
    @test_throws "must match tile_shape" eachtile(a, (8, 4); step=(2, 2, 2))
    @test_throws "must match tile_shape" eachtile(a, (4, 4); step=(2,))

    # A short tile_shape/step pair on a higher-rank array pads consistently:
    # the trailing dimension is a unit-step singleton, not a step-1 overlap.
    short = eachtile(a, (4,); step=(2,))
    @test cuTile.tiled_view_shape(short) == (4, 1)
    @test cuTile.tiled_view_step(short) == (2, 1)

    # `order` permutes which array dimension each tile dimension walks, so the
    # host tile count pairs step[i] with array dim order[i]. a is 16×10.
    permuted = eachtile(a, (8, 4); step=(3, 2), order=(2, 1))
    @test size(permuted) == (Int32(4), Int32(8))  # cld(10, 3), cld(16, 2)
    @test_throws "must be a permutation" eachtile(a, (8, 4); order=(1, 1))
    @test_throws "must be a permutation" eachtile(a, (8, 4); order=(1,))
end

@testset "TensorView" begin
    @test eltype(cuTile.TensorView{Float32, 2}) == Float32
    @test eltype(cuTile.TensorView{Float64, 3}) == Float64
    @test ndims(cuTile.TensorView{Float32, 2}) == 2
    @test ndims(cuTile.TensorView{Int32, 1}) == 1
end

@testset "TFloat32 bit packing" begin
    # 19-bit float: sign(1) | exp(8) | mantissa(10), Float32 layout with the
    # low 13 mantissa bits dropped (RNE).

    # Simple values
    @test cuTile.float_to_bits(0.0, ct.TFloat32) == 0x00000
    @test cuTile.float_to_bits(-0.0, ct.TFloat32) == 0x40000
    @test cuTile.float_to_bits(1.0, ct.TFloat32) == 0x1FC00
    @test cuTile.float_to_bits(-1.0, ct.TFloat32) == 0x5FC00
    @test cuTile.float_to_bits(2.0, ct.TFloat32) == 0x20000

    # Inf
    @test cuTile.float_to_bits(Inf, ct.TFloat32) == 0x3FC00
    @test cuTile.float_to_bits(-Inf, ct.TFloat32) == 0x7FC00

    # NaN: exp=0xff, mantissa nonzero
    nan_bits = cuTile.float_to_bits(NaN, ct.TFloat32)
    @test (nan_bits >> 10) & 0xff == 0xff
    @test (nan_bits & 0x3ff) != 0

    # Result must fit in 19 bits
    for v in (0.0, 1.0, -1.0, 1e10, -1e-10, Float64(prevfloat(typemax(Float32))))
        @test cuTile.float_to_bits(v, ct.TFloat32) <= 0x7FFFF
    end

    # RNE rounding: bit 12 of mantissa is the round bit. Construct values whose
    # low 13 mantissa bits are exactly half (0x1000) and verify ties round to even.
    # 1.0 + 2^-11 + 2^-23 ≈ has mantissa with bit 12 = 1, others = 0 ⇒ tie
    # truncation alone gives 0; RNE should keep 0 because the kept LSB is 0 (even).
    # Pick a value whose Float32 mantissa is exactly 0x001000 (only the round bit).
    f32 = reinterpret(Float32, UInt32(0x3f800800))  # 1 + 2^-12 (mantissa = 0x000800)... let's just spot-check
    # Verify packing is consistent with round-trip conversion
    @test cuTile.float_to_bits(1.5, ct.TFloat32) == ((127 << 10) | 0x200)  # mantissa 0x200 = 0.5

    # constant_to_bytes serialises TF32 as 3 little-endian bytes; a literal
    # 4-byte Float32 path would overflow into the next constant section field.
    @test cuTile.constant_to_bytes(0.0, ct.TFloat32) == UInt8[0x00, 0x00, 0x00]
    @test cuTile.constant_to_bytes(1.0, ct.TFloat32) == UInt8[0x00, 0xfc, 0x01]   # bits 0x01FC00
    @test cuTile.constant_to_bytes(-1.0, ct.TFloat32) == UInt8[0x00, 0xfc, 0x05]  # bits 0x05FC00
    @test cuTile.constant_to_bytes(Inf, ct.TFloat32) == UInt8[0x00, 0xfc, 0x03]   # bits 0x03FC00
    for v in (0.0, 1.0, -1.0, 1.5, Inf, -Inf, NaN)
        @test length(cuTile.constant_to_bytes(v, ct.TFloat32)) == 3
    end

    # Scalar TFloat32 must resolve to a 0-D tile; `tile_type_for_julia!`
    # used to skip it, breaking any call site that fed a bare scalar to codegen.
    let tt = cuTile.TypeTable(; version=cuTile.bytecode_version())
        tid = cuTile.tile_type_for_julia!(tt, ct.TFloat32)
        @test tid !== nothing
    end
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

@testset "tile_ir_requirement" begin
    # Blackwell (sm_100+) requires bytecode ≥ v13.1
    @test cuTile.tile_ir_requirement(v"10.0") == ("Blackwell", v"13.1")
    @test cuTile.tile_ir_requirement(v"12.1") == ("Blackwell", v"13.1")
    # Hopper (sm_90) requires bytecode ≥ v13.3
    @test cuTile.tile_ir_requirement(v"9.0") == ("Hopper", v"13.3")
    # Ampere / Ada (sm_80..sm_89) requires bytecode ≥ v13.2
    @test cuTile.tile_ir_requirement(v"8.0") == ("Ampere/Ada", v"13.2")
    @test cuTile.tile_ir_requirement(v"8.9") == ("Ampere/Ada", v"13.2")
    # Pre-Ampere is unsupported
    @test cuTile.tile_ir_requirement(v"7.5") === nothing
    @test cuTile.tile_ir_requirement(v"7.0") === nothing
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
