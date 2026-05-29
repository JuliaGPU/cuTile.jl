using CUDA
using Microfloats: Float8_E4M3FN, Float8_E5M2, Float8_E8M0FNU, Float4_E2M1FN

# Kernels are plain top-level functions (not closures), each defined next to the
# testset that exercises it. Kernels parametric on operand dtype must stay at
# top level — defining them inside a testset scope boxes them into closures.

# ftof round-trips
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
function rt_e8m0(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1})
    pid = ct.bid(1)
    tile = ct.load(a, pid, (16,))
    ct.store(b, pid, convert(ct.Tile{Float32}, convert(ct.Tile{Float8_E8M0FNU}, tile)))
    return
end
function rt_f4(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1})
    pid = ct.bid(1)
    tile = ct.load(a, pid, (16,))
    ct.store(b, pid, convert(ct.Tile{Float32}, convert(ct.Tile{Float4_E2M1FN}, tile)))
    return
end
function fma_e4m3(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                  c::ct.TileArray{Float32,1}, d::ct.TileArray{Float32,1})
    pid = ct.bid(1)
    ta = convert(ct.Tile{Float8_E4M3FN}, ct.load(a, pid, (16,)))
    tb = convert(ct.Tile{Float8_E4M3FN}, ct.load(b, pid, (16,)))
    tc = convert(ct.Tile{Float8_E4M3FN}, ct.load(c, pid, (16,)))
    ct.store(d, pid, convert(ct.Tile{Float32}, muladd.(ta, tb, tc)))
    return
end

# Standalone f32 → microfloat → f32 conversion round-trips exactly for every
# microfloat type on representable inputs. FP8 (e4m3/e5m2) needs Hopper (sm_90+);
# E8M0FNU and Float4_E2M1FN need Blackwell (sm_100+). E8M0 is exponent-only, so
# its representable values are exact powers of two.
@testset "ftof" begin
if capability(device()) >= v"9"
    # Round-trip Float32 → microfloat → Float32 on exactly-representable values.
    representable8 = Float32[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0,
                             16.0, 32.0, 64.0, 128.0, 256.0, -1.0, -2.0, -0.5]
    let a = CuArray(representable8), b = CUDA.zeros(Float32, length(representable8))
        @cuda backend=cuTile blocks=1 rt_e4m3(a, b)
        @test Array(b) == representable8
        @cuda backend=cuTile blocks=1 rt_e5m2(a, b)
        @test Array(b) == representable8
    end

    # FMA in FP8: load f32, convert to FP8, multiply-add in FP8, convert back.
    # Inputs whose products and sums stay representable, so the result is exact.
    let av = Float32[1.0, 2.0, 0.5, 4.0, 1.5, 2.0, -1.0, -0.5, 3.0, 0.5, 1.0, 2.0, -2.0, 1.0, 0.5, 4.0],
        bv = Float32[2.0, 1.0, 4.0, 0.5, 2.0, 3.0,  2.0,  4.0, 1.0, 2.0, 1.0, 0.5,  2.0, 1.0, 2.0, 1.0],
        cv = Float32[0.0, 1.0, 0.0, 0.0, 1.0, 1.0,  0.0,  0.0, 1.0, 0.0, 0.0, 1.0,  0.0, 0.0, 1.0, 0.0]
        a, b, c = CuArray(av), CuArray(bv), CuArray(cv)
        d = CUDA.zeros(Float32, length(av))
        @cuda backend=cuTile blocks=1 fma_e4m3(a, b, c, d)
        @test Array(d) == av .* bv .+ cv
    end
end
if capability(device()) >= v"10"
    # E8M0 round-trip: exponent-only, so representable values are powers of two.
    representable_e8m0 = Float32(2) .^ Float32[-8, -6, -4, -2, -1, 0, 1, 2,
                                               3, 4, 5, 6, 7, 8, 10, 16]
    let a = CuArray(representable_e8m0), b = CUDA.zeros(Float32, length(representable_e8m0))
        @cuda backend=cuTile blocks=1 rt_e8m0(a, b)
        @test Array(b) == representable_e8m0
    end

    # F4 round-trip via the standalone conversion path (no pack/reinterpret).
    representable4 = Float32[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                             -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, 0.5]
    let a = CuArray(representable4), b = CUDA.zeros(Float32, length(representable4))
        @cuda backend=cuTile blocks=1 rt_f4(a, b)
        @test Array(b) == representable4
    end
end
end

# reinterpret (pack / unpack)
function rt_pack(a::ct.TileArray{UInt8,1}, b::ct.TileArray{UInt8,1})
    pid = ct.bid(1)
    bytes = ct.load(a, pid, (8,))
    fp4 = reinterpret(Float4_E2M1FN, bytes)   # unpack: (8,) UInt8 → (16,) FP4
    ct.store(b, pid, reinterpret(UInt8, fp4)) # pack:   (16,) FP4 → (8,) UInt8
    return
end
function pack_fp4(src::ct.TileArray{Float32,1}, dst::ct.TileArray{UInt8,1})
    pid = ct.bid(1)
    vals = ct.load(src, pid, (16,))
    fp4 = convert(ct.Tile{Float4_E2M1FN}, vals)
    ct.store(dst, pid, reinterpret(UInt8, fp4))  # pack: (16,) FP4 → (8,) UInt8
    return
end
function unpack_fp4(src::ct.TileArray{UInt8,1}, dst::ct.TileArray{Float32,1})
    pid = ct.bid(1)
    bytes = ct.load(src, pid, (8,))
    fp4 = reinterpret(Float4_E2M1FN, bytes)      # unpack: (8,) UInt8 → (16,) FP4
    ct.store(dst, pid, convert(ct.Tile{Float32}, fp4))
    return
end
function rt_fp4_2d(src::ct.TileArray{Float32,2}, dst::ct.TileArray{Float32,2})
    pid = ct.bid(1)
    fp4 = convert(ct.Tile{Float4_E2M1FN}, ct.load(src, pid, (8, 2)))  # (8,2) FP4
    bytes = reinterpret(UInt8, fp4)                                   # (4,2) UInt8
    fp4b = reinterpret(Float4_E2M1FN, bytes)                         # (8,2) FP4
    ct.store(dst, pid, convert(ct.Tile{Float32}, fp4b))
    return
end

# Float4_E2M1FN requires Blackwell (sm_100+).
@testset "reinterpret" begin
if capability(device()) >= v"10"
    # Pure pack/unpack round-trip: UInt8 → FP4 → UInt8 must be a no-op.
    let av = UInt8[0x00, 0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde],
        b = CUDA.zeros(UInt8, 8)
        a = CuArray(av)
        @cuda backend=cuTile blocks=1 rt_pack(a, b)
        @test Array(b) == av
    end

    # Value round-trip through FP4 stored as UInt8 (all inputs representable).
    let representable4 = Float32[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                                 -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, 0.5]
        src = CuArray(representable4)
        packed = CUDA.zeros(UInt8, 8)
        out = CUDA.zeros(Float32, 16)
        @cuda backend=cuTile blocks=1 pack_fp4(src, packed)
        @cuda backend=cuTile blocks=1 unpack_fp4(packed, out)
        @test Array(out) == representable4
    end

    # N-D reinterpret: whole-tile `reinterpret` flattens, so it works on any
    # rank. (8,2) FP4 ↔ (4,2) UInt8 — the leading dim absorbs the 2× / ½.
    let m = reshape(Float32[0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -1.0,
                            -2.0, -3.0, -4.0, -6.0, 0.5, 1.0, 2.0, 3.0], 8, 2)
        src = CuArray(m)
        out = CUDA.zeros(Float32, 8, 2)
        @cuda backend=cuTile blocks=1 rt_fp4_2d(src, out)
        @test Array(out) == m
    end
end
end

# non-scaled FP8 matmul (f8 × f8 with f16 or f32 accumulator)
function mma_fp8(A::ct.TileArray{T,2}, B::ct.TileArray{T,2},
                 C::ct.TileArray{Tacc,2}, D::ct.TileArray{Float32,2}
                 ) where {T<:Union{Float8_E4M3FN,Float8_E5M2},Tacc<:Union{Float16,Float32}}
    a = ct.load(A, (1, 1), (16, 16)); b = ct.load(B, (1, 1), (16, 16)); c = ct.load(C, (1, 1), (16, 16))
    ct.store(D, (1, 1), convert(ct.Tile{Float32}, muladd(a, b, c)))
    return
end
function mma_e4m3_star(A::ct.TileArray{Float8_E4M3FN,2}, B::ct.TileArray{Float8_E4M3FN,2},
                       D::ct.TileArray{Float32,2})
    a = ct.load(A, (1, 1), (16, 16)); b = ct.load(B, (1, 1), (16, 16))
    ct.store(D, (1, 1), convert(ct.Tile{Float32}, a * b))
    return
end
function mma_e4m3_fast(A::ct.TileArray{Float8_E4M3FN,2}, B::ct.TileArray{Float8_E4M3FN,2},
                       C::ct.TileArray{Float32,2}, D::ct.TileArray{Float32,2})
    a = ct.load(A, (1, 1), (16, 16)); b = ct.load(B, (1, 1), (16, 16)); c = ct.load(C, (1, 1), (16, 16))
    ct.store(D, (1, 1), convert(ct.Tile{Float32}, muladd(a, b, c; fast_acc=true)))
    return
end

# Non-scaled FP8 matmul needs only Hopper (sm_90+); block-scaled mma (below) is
# the Blackwell-only variant.
@testset "mma" begin
if capability(device()) >= v"9"
    M, K, N = 16, 16, 16
    # f8 operands with both allowed accumulator dtypes (f16 and f32). Inputs are
    # in {0, 0.5, 1} and K = 16, so every partial sum is exactly representable
    # in f16 too — the f16-acc result matches the f32 reference bit-for-bit.
    @testset "$T × $T → $Tacc acc" for (T, Tacc) in (
            (Float8_E4M3FN, Float32), (Float8_E4M3FN, Float16),
            (Float8_E5M2,   Float32), (Float8_E5M2,   Float16))
        ah = T.(Float32.(rand(0:2, M, K)) ./ 2)
        bh = T.(Float32.(rand(0:2, K, N)) ./ 2)
        ch = Tacc.(Float32.(rand(0:2, M, N)))
        ref = Float32.(ah) * Float32.(bh) .+ Float32.(ch)
        D = CUDA.zeros(Float32, M, N)
        @cuda backend=cuTile blocks=1 mma_fp8(CuArray(ah), CuArray(bh), CuArray(ch), D)
        @test Array(D) == ref
    end

    @testset "* operator (auto acc → f8)" begin
        ah = Float8_E4M3FN.(Float32.(rand(0:2, M, M)) ./ 2)
        bh = Float8_E4M3FN.(Float32.(rand(0:2, M, M)) ./ 2)
        # f8 × f8 accumulates in f16, then downcasts back to f8.
        ref = Float32.(Float8_E4M3FN.(Float16.(Float32.(ah) * Float32.(bh))))
        D = CUDA.zeros(Float32, M, M)
        @cuda backend=cuTile blocks=1 mma_e4m3_star(CuArray(ah), CuArray(bh), D)
        @test Array(D) == ref
    end

    # fast_acc trades accumulator precision for throughput, but only has an
    # effect on Hopper (sm_90); it is silently ignored on every other arch. So
    # off Hopper we assert the exact result — the flag must ride through tileiras
    # and ptxas without perturbing the output. On Hopper we make no numeric
    # claim: fast accumulation may legitimately diverge there, and we have no
    # fast-accum reference to compare against.
    @testset "fast_acc (exact off Hopper)" begin
        ah = Float8_E4M3FN.(Float32.(rand(0:2, M, K)) ./ 2)
        bh = Float8_E4M3FN.(Float32.(rand(0:2, K, N)) ./ 2)
        ch = Float32.(rand(0:2, M, N))
        ref = Float32.(ah) * Float32.(bh) .+ ch
        D = CUDA.zeros(Float32, M, N)
        @cuda backend=cuTile blocks=1 mma_e4m3_fast(CuArray(ah), CuArray(bh), CuArray(ch), D)
        @test (Array(D) == ref) || (v"9" <= capability(device()) < v"10")
    end
end
end

# block-scaled mma (B = K ÷ K_s; here K = 64, K_s = 2 → B = 32)
function mma_scaled_2d(X::ct.TileArray{T,2}, XS::ct.TileArray{Float8_E8M0FNU,2},
                       Y::ct.TileArray{T,2}, YS::ct.TileArray{Float8_E8M0FNU,2},
                       Z::ct.TileArray{Float32,2}) where {T<:Union{Float8_E4M3FN,Float8_E5M2}}
    x  = ct.load(X,  (1, 1), (16, 64)); xs = ct.load(XS, (1, 1), (16, 2))
    y  = ct.load(Y,  (1, 1), (64, 16)); ys = ct.load(YS, (1, 1), (2, 16))
    ct.store(Z, (1, 1), ct.muladd_scaled(x, xs, y, ys, zeros(Float32, (16, 16))))
    return
end
function mma_scaled_matvec(X::ct.TileArray{Float8_E4M3FN,2}, XS::ct.TileArray{Float8_E8M0FNU,2},
                           Y::ct.TileArray{Float8_E4M3FN,1}, YS::ct.TileArray{Float8_E8M0FNU,1},
                           Z::ct.TileArray{Float32,1})
    x  = ct.load(X,  (1, 1), (16, 64)); xs = ct.load(XS, (1, 1), (16, 2))
    y  = ct.load(Y,  1, (64,));         ys = ct.load(YS, 1, (2,))
    ct.store(Z, 1, ct.muladd_scaled(x, xs, y, ys, zeros(Float32, (16,))))
    return
end
function mma_scaled_batched(X::ct.TileArray{Float8_E4M3FN,2}, XS::ct.TileArray{Float8_E8M0FNU,2},
                            Y::ct.TileArray{Float8_E4M3FN,3}, YS::ct.TileArray{Float8_E8M0FNU,3},
                            Z::ct.TileArray{Float32,3})
    x  = ct.load(X,  (1, 1),    (16, 64)); xs = ct.load(XS, (1, 1),    (16, 2))
    y  = ct.load(Y,  (1, 1, 1), (64, 16, 2)); ys = ct.load(YS, (1, 1, 1), (2, 16, 2))
    ct.store(Z, (1, 1, 1), ct.muladd_scaled(x, xs, y, ys, zeros(Float32, (16, 16, 2))))
    return
end
# FP4 operands have no direct sub-byte load: they arrive packed two-per-byte
# along the matmul-contiguous axis (K for X, N for Y, matching the row-major
# reference). cuTile's `reinterpret` doubles the *leading* (column-major) dim,
# so we load each operand with its packed axis leading, unpack, then transpose
# into the (M,K) / (K,N) orientation `muladd_scaled` expects:
#   X: bytes (K/2, M) → reinterpret (K, M) → transpose (M, K)
#   Y: bytes (N/2, K) → reinterpret (N, K) → transpose (K, N)
# MXFP4 takes an f8e8m0fnu scale (B = 32 → K_s = 2); NVFP4 an f8e4m3fn scale
# (B = 16 → K_s = 4). Operand unpacking is identical — only the scale differs.
function mma_scaled_mxfp4(X::ct.TileArray{UInt8,2}, XS::ct.TileArray{Float8_E8M0FNU,2},
                          Y::ct.TileArray{UInt8,2}, YS::ct.TileArray{Float8_E8M0FNU,2},
                          Z::ct.TileArray{Float32,2})
    x  = permutedims(reinterpret(Float4_E2M1FN, ct.load(X, (1, 1), (32, 16))), (2, 1))  # (M,K)
    y  = permutedims(reinterpret(Float4_E2M1FN, ct.load(Y, (1, 1), (8, 64))), (2, 1))   # (K,N)
    xs = ct.load(XS, (1, 1), (16, 2))
    ys = ct.load(YS, (1, 1), (2, 16))
    ct.store(Z, (1, 1), ct.muladd_scaled(x, xs, y, ys, zeros(Float32, (16, 16))))
    return
end
function mma_scaled_nvfp4(X::ct.TileArray{UInt8,2}, XS::ct.TileArray{Float8_E4M3FN,2},
                          Y::ct.TileArray{UInt8,2}, YS::ct.TileArray{Float8_E4M3FN,2},
                          Z::ct.TileArray{Float32,2})
    x  = permutedims(reinterpret(Float4_E2M1FN, ct.load(X, (1, 1), (32, 16))), (2, 1))  # (M,K)
    y  = permutedims(reinterpret(Float4_E2M1FN, ct.load(Y, (1, 1), (8, 64))), (2, 1))   # (K,N)
    xs = ct.load(XS, (1, 1), (16, 4))
    ys = ct.load(YS, (1, 1), (4, 16))
    ct.store(Z, (1, 1), ct.muladd_scaled(x, xs, y, ys, zeros(Float32, (16, 16))))
    return
end

# Expand a block-scale tile to per-element scales along the K dimension: each
# scale entry covers B contiguous K positions (matches Tile IR's broadcast).
expand_k(scale, B, ::Val{:cols}) = repeat(Float32.(scale), inner=(1, B))   # (M, K_s) → (M, K)
expand_k(scale, B, ::Val{:rows}) = repeat(Float32.(scale), inner=(B, 1))   # (K_s, N) → (K, N)

# Reference: (x ⊙ x_scale) * (y ⊙ y_scale) + acc, with acc = 0.
function mma_scaled_ref(xh, xs, yh, ys, B)
    (Float32.(xh) .* expand_k(xs, B, Val(:cols))) *
        (Float32.(yh) .* expand_k(ys, B, Val(:rows)))
end

# Low 4 bits of an FP4 value's byte storage (its E2M1 nibble).
f4_nibble(v) = reinterpret(UInt8, Float4_E2M1FN(v)) & 0x0f

# Pack an (R, C) FP4 matrix two-per-byte along the contiguous (second) axis →
# (C/2, R) bytes: low nibble = even column, high nibble = odd column. Matches
# X (M,K)→(K/2,M) and Y (K,N)→(N/2,K) in the kernels above.
pack_f4_bytes(vals) = UInt8[f4_nibble(vals[r, 2c-1]) | (f4_nibble(vals[r, 2c]) << 4)
                            for c in 1:(size(vals, 2) ÷ 2), r in 1:size(vals, 1)]

# All FP4-representable magnitudes, for sampling exact inputs.
const F4_VALUES = Float32[0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.5, -1, -1.5, -2, -3, -4, -6]

# Block-scaled mma is Blackwell-only (sm_100+), for both FP8 and FP4 operands.
@testset "mma_scaled" begin
if capability(device()) >= v"10"
    m, n, k, ks, B = 16, 16, 64, 2, 32

    # MXFP8: f8 operands with an f8e8m0fnu scale, B = 32. 2D × 2D, each f8 type.
    @testset "MXFP8 2D ($T)" for T in (Float8_E4M3FN, Float8_E5M2)
        xh = T.(Float32.(rand(0:2, m, k)) ./ 2)        # values in {0, 0.5, 1}
        yh = T.(Float32.(rand(0:2, k, n)) ./ 2)
        xs = Float8_E8M0FNU.(Float32(2) .^ rand(0:1, m, ks))   # scales in {1, 2}
        ys = Float8_E8M0FNU.(Float32(2) .^ rand(0:1, ks, n))
        ref = mma_scaled_ref(xh, xs, yh, ys, B)
        Z = CUDA.zeros(Float32, m, n)
        @cuda backend=cuTile blocks=1 mma_scaled_2d(CuArray(xh), CuArray(xs), CuArray(yh), CuArray(ys), Z)
        @test Array(Z) == ref
    end

    @testset "MXFP8 mat-vec" begin
        xh = Float8_E4M3FN.(Float32.(rand(0:2, m, k)) ./ 2)
        yh = Float8_E4M3FN.(Float32.(rand(0:2, k)) ./ 2)
        xs = Float8_E8M0FNU.(Float32(2) .^ rand(0:1, m, ks))
        ys = Float8_E8M0FNU.(Float32(2) .^ rand(0:1, ks))
        ref = (Float32.(xh) .* expand_k(xs, B, Val(:cols))) *
              (Float32.(yh) .* repeat(Float32.(ys), inner=(B,)))
        Z = CUDA.zeros(Float32, m)
        @cuda backend=cuTile blocks=1 mma_scaled_matvec(CuArray(xh), CuArray(xs), CuArray(yh), CuArray(ys), Z)
        @test Array(Z) == ref
    end

    @testset "MXFP8 batched (trailing batch broadcast)" begin
        bt = 2
        xh = Float8_E4M3FN.(Float32.(rand(0:2, m, k)) ./ 2)
        xs = Float8_E8M0FNU.(Float32(2) .^ rand(0:1, m, ks))
        yh = Float8_E4M3FN.(Float32.(rand(0:2, k, n, bt)) ./ 2)
        ys = Float8_E8M0FNU.(Float32(2) .^ rand(0:1, ks, n, bt))
        ref = stack(mma_scaled_ref(xh, xs, yh[:, :, bi], ys[:, :, bi], B) for bi in 1:bt)
        Z = CUDA.zeros(Float32, m, n, bt)
        @cuda backend=cuTile blocks=1 mma_scaled_batched(CuArray(xh), CuArray(xs), CuArray(yh), CuArray(ys), Z)
        @test Array(Z) == ref
    end

    # MXFP4: FP4 operands packed two-per-byte (unpacked + transposed in the
    # kernel) with an f8e8m0fnu scale, B = 32 → K_s = 2.
    @testset "MXFP4" begin
        xh = [rand(F4_VALUES) for _ in 1:m, _ in 1:k]   # (M, K)
        yh = [rand(F4_VALUES) for _ in 1:k, _ in 1:n]   # (K, N)
        xs = Float8_E8M0FNU.(Float32(2) .^ rand(0:1, m, ks))
        ys = Float8_E8M0FNU.(Float32(2) .^ rand(0:1, ks, n))
        ref = mma_scaled_ref(xh, xs, yh, ys, B)
        Z = CUDA.zeros(Float32, m, n)
        @cuda backend=cuTile blocks=1 mma_scaled_mxfp4(CuArray(pack_f4_bytes(xh)), CuArray(xs),
                                                       CuArray(pack_f4_bytes(yh)), CuArray(ys), Z)
        @test Array(Z) == ref
    end

    # NVFP4: same FP4 operands but an f8e4m3fn scale at B = 16 → K_s = 4.
    @testset "NVFP4" begin
        Bn, ksn = 16, k ÷ 16                            # B = 16 → K_s = 4
        xh = [rand(F4_VALUES) for _ in 1:m, _ in 1:k]   # (M, K)
        yh = [rand(F4_VALUES) for _ in 1:k, _ in 1:n]   # (K, N)
        xs = Float8_E4M3FN.(Float32(2) .^ rand(0:1, m, ksn))   # scales in {1, 2}, exact in e4m3
        ys = Float8_E4M3FN.(Float32(2) .^ rand(0:1, ksn, n))
        ref = mma_scaled_ref(xh, xs, yh, ys, Bn)
        Z = CUDA.zeros(Float32, m, n)
        @cuda backend=cuTile blocks=1 mma_scaled_nvfp4(CuArray(pack_f4_bytes(xh)), CuArray(xs),
                                                       CuArray(pack_f4_bytes(yh)), CuArray(ys), Z)
        @test Array(Z) == ref
    end
end
end
