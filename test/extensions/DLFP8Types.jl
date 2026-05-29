using CUDA
using DLFP8Types: Float8_E4M3FN, Float8_E5M2

spec1d = ct.ArraySpec{1}(16, true)
spec2d = ct.ArraySpec{2}(16, true)

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

# Non-scaled f8 matmul lowers to `mmaf` (f8 operands, f32 accumulator).
@test @filecheck begin
    @check_label "entry"
    code_tiled(Tuple{ct.TileArray{Float8_E4M3FN,2,spec2d}, ct.TileArray{Float8_E4M3FN,2,spec2d},
                     ct.TileArray{Float32,2,spec2d}}) do a, b, c
        ta = ct.load(a, (1, 1), (16, 16))
        tb = ct.load(b, (1, 1), (16, 16))
        @check "mmaf"
        ct.store(c, (1, 1), muladd(ta, tb, zeros(Float32, (16, 16))))
        return
    end
end

# `fast_acc=true` is an FP8-only hint; it still lowers to `mmaf` (13.3+).
@test @filecheck begin
    @check_label "entry"
    code_tiled(Tuple{ct.TileArray{Float8_E4M3FN,2,spec2d}, ct.TileArray{Float8_E4M3FN,2,spec2d},
                     ct.TileArray{Float32,2,spec2d}}; bytecode_version=v"13.3") do a, b, c
        ta = ct.load(a, (1, 1), (16, 16))
        tb = ct.load(b, (1, 1), (16, 16))
        @check "mmaf"
        ct.store(c, (1, 1), muladd(ta, tb, zeros(Float32, (16, 16)); fast_acc=true))
        return
    end
end

end

# Execution kernels are plain top-level functions, each defined next to the
# test that exercises it. Kernels parametric on accumulator dtype must stay at
# top level — defining them inside a testset scope boxes them into closures.

# Round-trip Float32 → FP8 → Float32 on values exactly representable in the
# target FP8 type — result must match input bit-for-bit.
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
# FMA in FP8: load Float32, convert to FP8, multiply-add in FP8, convert back.
# Inputs whose products and sums also stay representable, so the result is exact.
function fma_e4m3(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                  c::ct.TileArray{Float32,1}, d::ct.TileArray{Float32,1})
    pid = ct.bid(1)
    ta = convert(ct.Tile{Float8_E4M3FN}, ct.load(a, pid, (16,)))
    tb = convert(ct.Tile{Float8_E4M3FN}, ct.load(b, pid, (16,)))
    tc = convert(ct.Tile{Float8_E4M3FN}, ct.load(c, pid, (16,)))
    ct.store(d, pid, convert(ct.Tile{Float32}, muladd.(ta, tb, tc)))
    return
end
# Non-scaled FP8 matmul with both allowed accumulator dtypes (f16 and f32).
function mma_dl_fp8(A::ct.TileArray{Float8_E4M3FN,2}, B::ct.TileArray{Float8_E4M3FN,2},
                    C::ct.TileArray{Tacc,2}, D::ct.TileArray{Float32,2}) where {Tacc<:Union{Float16,Float32}}
    a = ct.load(A, (1, 1), (16, 16)); b = ct.load(B, (1, 1), (16, 16)); c = ct.load(C, (1, 1), (16, 16))
    ct.store(D, (1, 1), convert(ct.Tile{Float32}, muladd(a, b, c)))
    return
end
function mma_dl_fast(A::ct.TileArray{Float8_E4M3FN,2}, B::ct.TileArray{Float8_E4M3FN,2},
                     C::ct.TileArray{Float32,2}, D::ct.TileArray{Float32,2})
    a = ct.load(A, (1, 1), (16, 16)); b = ct.load(B, (1, 1), (16, 16)); c = ct.load(C, (1, 1), (16, 16))
    ct.store(D, (1, 1), convert(ct.Tile{Float32}, muladd(a, b, c; fast_acc=true)))
    return
end

# FP8 (e4m3/e5m2) conversions and matmul need Hopper (sm_90+).
@testset "execution" begin
if capability(device()) >= v"9"

representable = Float32[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0,
                        16.0, 32.0, 64.0, 128.0, 256.0, -1.0, -2.0, -0.5]
let a = CuArray(representable), b = CUDA.zeros(Float32, length(representable))
    @cuda backend=cuTile blocks=1 rt_e4m3(a, b)
    @test Array(b) == representable
    @cuda backend=cuTile blocks=1 rt_e5m2(a, b)
    @test Array(b) == representable
end

let av = Float32[1.0, 2.0, 0.5, 4.0, 1.5, 2.0, -1.0, -0.5, 3.0, 0.5, 1.0, 2.0, -2.0, 1.0, 0.5, 4.0],
    bv = Float32[2.0, 1.0, 4.0, 0.5, 2.0, 3.0,  2.0,  4.0, 1.0, 2.0, 1.0, 0.5,  2.0, 1.0, 2.0, 1.0],
    cv = Float32[0.0, 1.0, 0.0, 0.0, 1.0, 1.0,  0.0,  0.0, 1.0, 0.0, 0.0, 1.0,  0.0, 0.0, 1.0, 0.0]
    a, b, c = CuArray(av), CuArray(bv), CuArray(cv)
    d = CUDA.zeros(Float32, length(av))
    @cuda backend=cuTile blocks=1 fma_e4m3(a, b, c, d)
    @test Array(d) == av .* bv .+ cv
end

@testset "mma → $Tacc acc" for Tacc in (Float32, Float16)
    M = 16
    ah = Float8_E4M3FN.(Float32.(rand(0:2, M, M)) ./ 2)
    bh = Float8_E4M3FN.(Float32.(rand(0:2, M, M)) ./ 2)
    ch = Tacc.(Float32.(rand(0:2, M, M)))
    ref = Float32.(ah) * Float32.(bh) .+ Float32.(ch)
    D = CUDA.zeros(Float32, M, M)
    @cuda backend=cuTile blocks=1 mma_dl_fp8(CuArray(ah), CuArray(bh), CuArray(ch), D)
    @test Array(D) == ref
end

# fast_acc only has an effect on Hopper (sm_90); ignored elsewhere. So off
# Hopper we assert the exact result (the flag must ride through without
# perturbing the output); on Hopper we make no numeric claim.
@testset "mma fast_acc (exact off Hopper)" begin
    M = 16
    ah = Float8_E4M3FN.(Float32.(rand(0:2, M, M)) ./ 2)
    bh = Float8_E4M3FN.(Float32.(rand(0:2, M, M)) ./ 2)
    ch = Float32.(rand(0:2, M, M))
    ref = Float32.(ah) * Float32.(bh) .+ ch
    D = CUDA.zeros(Float32, M, M)
    @cuda backend=cuTile blocks=1 mma_dl_fast(CuArray(ah), CuArray(bh), CuArray(ch), D)
    @test (Array(D) == ref) || (v"9" <= capability(device()) < v"10")
end

end
end
