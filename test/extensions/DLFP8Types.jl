using CUDA
using DLFP8Types: Float8_E4M3FN, Float8_E5M2

spec1d = ct.ArraySpec{1}(16, true)

@testset "codegen" begin

# Float32 -> Float8_E4M3FN
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

# Float32 -> Float8_E5M2
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

end

@testset "execution" begin

# Round-trip Float32 → FP8 → Float32 on values exactly representable in
# the target FP8 type — result must match input bit-for-bit.
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

representable = Float32[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0,
                        16.0, 32.0, 64.0, 128.0, 256.0, -1.0, -2.0, -0.5]
let a = CuArray(representable), b = CUDA.zeros(Float32, length(representable))
    @cuda backend=cuTile blocks=1 rt_e4m3(a, b)
    @test Array(b) == representable
    @cuda backend=cuTile blocks=1 rt_e5m2(a, b)
    @test Array(b) == representable
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

end
