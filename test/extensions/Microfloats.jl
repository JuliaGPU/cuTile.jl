using CUDA
using Microfloats: Float8_E4M3FN, Float8_E5M2, Float8_E8M0FNU, Float4_E2M1FN

spec1d = ct.ArraySpec{1}(16, true)

@testset "codegen" begin

# Float32 -> Float8_E4M3FN (always available; 13.1+)
@test @filecheck begin
    @check_label "entry"
    code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}}) do a, b
        pid = ct.bid(1)
        tile = ct.load(a, pid, (16,))
        @check "ftof"
        converted = convert(ct.Tile{Float8_E4M3FN}, tile)
        ct.store(b, pid, convert(ct.Tile{Float32}, converted))
        return
    end
end

# Float32 -> Float8_E5M2 (always available; 13.1+)
@test @filecheck begin
    @check_label "entry"
    code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}}) do a, b
        pid = ct.bid(1)
        tile = ct.load(a, pid, (16,))
        @check "ftof"
        converted = convert(ct.Tile{Float8_E5M2}, tile)
        ct.store(b, pid, convert(ct.Tile{Float32}, converted))
        return
    end
end

# Float32 -> Float8_E8M0FNU works on bytecode 13.2+
@test @filecheck begin
    @check_label "entry"
    code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}};
               bytecode_version=v"13.2") do a, b
        pid = ct.bid(1)
        tile = ct.load(a, pid, (16,))
        @check "ftof"
        converted = convert(ct.Tile{Float8_E8M0FNU}, tile)
        ct.store(b, pid, convert(ct.Tile{Float32}, converted))
        return
    end
end

# Float8_E8M0FNU rejected on bytecode 13.1 with a clear version error
let kernel = (a, b) -> begin
        pid = ct.bid(1)
        tile = ct.load(a, pid, (16,))
        converted = convert(ct.Tile{Float8_E8M0FNU}, tile)
        ct.store(b, pid, convert(ct.Tile{Float32}, converted))
        return
    end
    @test_throws "v13.2+" code_tiled(devnull, kernel,
        Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}};
        bytecode_version=v"13.1")
end

# Float4_E2M1FN requires bytecode 13.3 — rejected at 13.2 with a clear error
let kernel = (a, b) -> begin
        pid = ct.bid(1)
        tile = ct.load(a, pid, (16,))
        converted = convert(ct.Tile{Float4_E2M1FN}, tile)
        ct.store(b, pid, convert(ct.Tile{Float32}, converted))
        return
    end
    @test_throws "v13.3+" code_tiled(devnull, kernel,
        Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}};
        bytecode_version=v"13.2")
end

end

@testset "execution" begin

# Round-trip Float32 → microfloat → Float32 on values exactly representable
# in the target type — result must match input bit-for-bit.
function rt_e4m3(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1})
    pid = ct.bid(1)
    tile = ct.load(a, pid, (16,))
    ct.store(b, pid, convert(ct.Tile{Float32}, convert(ct.Tile{Float8_E4M3FN}, tile)))
    return
end
function rt_e5m2(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1})
    pid = ct.bid(1)
    tile = ct.load(a, pid, (16,))
    ct.store(b, pid, convert(ct.Tile{Float32}, convert(ct.Tile{Float8_E5M2}, tile)))
    return
end
# Float8_E4M3FN / Float8_E5M2: 13.1+, always available
representable8 = Float32[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0,
                         16.0, 32.0, 64.0, 128.0, 256.0, -1.0, -2.0, -0.5]
let a = CuArray(representable8), b = CUDA.zeros(Float32, length(representable8))
    @cuda backend=cuTile blocks=1 rt_e4m3(a, b)
    @test Array(b) == representable8
    @cuda backend=cuTile blocks=1 rt_e5m2(a, b)
    @test Array(b) == representable8
end

# FMA in FP8: load Float32, convert to FP8, multiply-add in FP8, convert back.
# Uses inputs whose products and sums also stay representable, so the result
# is exact.
function fma_e4m3(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                  c::ct.TileArray{Float32,1}, d::ct.TileArray{Float32,1})
    pid = ct.bid(1)
    ta = convert(ct.Tile{Float8_E4M3FN}, ct.load(a, pid, (16,)))
    tb = convert(ct.Tile{Float8_E4M3FN}, ct.load(b, pid, (16,)))
    tc = convert(ct.Tile{Float8_E4M3FN}, ct.load(c, pid, (16,)))
    ct.store(d, pid, convert(ct.Tile{Float32}, muladd.(ta, tb, tc)))
    return
end
let av = Float32[1.0, 2.0, 0.5, 4.0, 1.5, 2.0, -1.0, -0.5, 3.0, 0.5, 1.0, 2.0, -2.0, 1.0, 0.5, 4.0],
    bv = Float32[2.0, 1.0, 4.0, 0.5, 2.0, 3.0,  2.0,  4.0, 1.0, 2.0, 1.0, 0.5,  2.0, 1.0, 2.0, 1.0],
    cv = Float32[0.0, 1.0, 0.0, 0.0, 1.0, 1.0,  0.0,  0.0, 1.0, 0.0, 0.0, 1.0,  0.0, 0.0, 1.0, 0.0]
    a, b, c = CuArray(av), CuArray(bv), CuArray(cv)
    d = CUDA.zeros(Float32, length(av))
    @cuda backend=cuTile blocks=1 fma_e4m3(a, b, c, d)
    @test Array(d) == av .* bv .+ cv
end

# Float8_E8M0FNU and Float4_E2M1FN: codegen for `ftof` lowers fine (see the
# codegen tests above) but tileiras refuses to lower a standalone f32 ↔
# microfloat conversion on Blackwell — these formats only have meaningful
# hardware paths as the scale/operand dtypes of block-scaled mma.

end
